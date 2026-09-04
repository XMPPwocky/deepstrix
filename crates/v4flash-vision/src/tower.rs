//! Vision tower (ViT + aligner) — weights resident on one HIP device.
//!
//! Pipeline (`Tower::encode`, all on `self.stream`, kernels in
//! `kernels/vit.hip`; f16 weights as stored in the mmproj, f16 activations
//! between kernels, f32 accumulation + f32 residual stream):
//!
//! ```text
//! patches f32 [n][588] --host--> f16 [n][608] (K zero-padded to a multiple of 32)
//!   gemm(+bias)                       -> x f32 [n][1024]
//!   32 × {
//!     rmsnorm(ln1)                    -> h f16
//!     gemm(fused qkv [3072][1024] +b) -> qkv f32 [n][3072]
//!     rope_split (2-D RoPE q,k)       -> q,k,v f16 [n][16][64]
//!     attention (bidirectional, 1/8)  -> o f16 [n][1024]
//!     gemm(attn_out +b, accumulate)   -> x += ...
//!     rmsnorm(ln2)                    -> h f16
//!     gemm(fused gate|up [5632][1024])-> gu f32 [n][5632]
//!     swiglu                          -> a f16 [n][2816]
//!     gemm(down, accumulate)          -> x += ...
//!   }
//!   rmsnorm(post_ln)                  -> h f16
//!   unfold (3×3, channel-major, zero pad) -> u f16 [n_llm][9216]
//!   gemm(mm.1 +b, GELU erf)           -> a f16 [n_llm][4096]
//!   gemm(mm.2 +b)                     -> out f32 [n_llm][4096]  --> host, place_rows
//! ```
//!
//! Weight-format choice: the f16 weights are kept as-is (889.9 MiB on the
//! device). On gfx1151 the fast GEMM is the RDNA3 f16 WMMA path (the tree's
//! Q8_0 / K-quant WMMA GEMMs are gfx12-only), so a Q8_0 requant would only
//! add error without unlocking anything; the iGPU has no int8 matrix rate
//! advantage over f16 WMMA at these shapes.
//!
//! ROOFLINE (gfx1151; measured 2026-09-04, `tests/tower_encode.rs`):
//! the encode is COMPUTE-bound, not weight-BW-bound. One image is 3.86
//! TFLOP at n=3108 while the weights are only 932 MB read once — 3.9 ms
//! at 240 GB/s, i.e. 0.9% of the 446 ms encode. Arithmetic intensity is
//! ~n FLOP/byte (3108), ~13x past the 246 FLOP/byte ridge point. So the
//! kernels are sized for compute, not for weight streaming, and a
//! persistent-weight/batched formulation would buy nothing.
//!
//!   compute ceiling: 59.4 TFLOP/s theoretical f16 WMMA
//!                    (40 CU x 512 FLOP/clk x 2.9 GHz);
//!                    27.0 TFLOP/s achieved by `vit_gemm` (45% of peak).
//!   n=3108: floor 167 ms vs 446 ms measured = 38% of ceiling.
//!     GEMMs      42-86% of their compute floor
//!     glue       ~75% of DRAM BW (rope_split, swiglu, rmsnorm: BW-bound)
//!     attention  19% of floor, 54% of the wall  <-- THE lever
//!
//! `vit_attention` is VALU-ISSUE bound: the ISA is already optimal per
//! instruction (`v_dot2acc_f32_f16` for the scores, `v_fma_mix_f32` for
//! PV, 74 VGPRs, no spill, occupancy not limited by LDS or registers),
//! but it retires only ~2.2 FLOP per lane-instruction. WMMA retires ~8x
//! that, so the only real lever is a WMMA score+PV rewrite, which needs
//! the P->V fragment permute that `docs`/memory record as a Pareto wall
//! on this ISA (the tree's own `fa2_hg_packed` sits at ~10% of matmul
//! peak; we are at 9.5%). Not attempted here.

use std::path::Path;
use std::time::Instant;

use color_eyre::eyre::{self, eyre, WrapErr};
use v4flash_core::kquants::f32_to_f16_bits;
use v4flash_hip::{Device, DeviceBuffer, Stream};

use crate::kernels::{VitKernels, FLAG_ACCUM, FLAG_GELU};
use crate::layout::ImageLayout;
use crate::mmproj::{MmprojHost, MmprojMeta, VitBlockHost};
use crate::preprocess::PreprocessedImage;
use crate::rope::vision_cos_sin;
use crate::{TokenType, ALIGNER_IN, PATCH_ELEMS, TEXT_DIM, VIT_DIM, VIT_FFN, VIT_HEAD_DIM, VIT_RMS_EPS, VIT_ROPE_DIM};

/// Patch-embed GEMM K, zero-padded from 588 to a multiple of 32.
pub const PATCH_K_PAD: usize = PATCH_ELEMS.div_ceil(32) * 32; // 608

/// One ViT block on the device: f16 bits (`u16`) row-major `[out][in]`,
/// q/k/v fused as `[3072][1024]` (+ `[3072]` bias), gate|up fused as
/// `[5632][1024]`.
pub struct VitBlockDev {
    pub ln1: DeviceBuffer<f32>,
    pub qkv_w: DeviceBuffer<u16>,
    pub qkv_b: DeviceBuffer<f32>,
    pub attn_out_w: DeviceBuffer<u16>,
    pub attn_out_b: DeviceBuffer<f32>,
    pub ln2: DeviceBuffer<f32>,
    pub gateup_w: DeviceBuffer<u16>,
    pub ffn_down_w: DeviceBuffer<u16>,
}

/// All mmproj weights on the device.
pub struct MmprojDev {
    /// `[1024][PATCH_K_PAD]`, columns 588.. zero.
    pub patch_embd_w: DeviceBuffer<u16>,
    pub patch_embd_b: DeviceBuffer<f32>,
    pub blocks: Vec<VitBlockDev>,
    pub post_ln: DeviceBuffer<f32>,
    pub mm1_w: DeviceBuffer<u16>,
    pub mm1_b: DeviceBuffer<f32>,
    pub mm2_w: DeviceBuffer<u16>,
    pub mm2_b: DeviceBuffer<f32>,
}

/// Activation scratch, grown on demand (`n_cap` patches, `llm_cap` aligner rows).
struct Workspace {
    n_cap: usize,
    llm_cap: usize,
    patches16: DeviceBuffer<u16>,
    x: DeviceBuffer<f32>,
    h16: DeviceBuffer<u16>,
    qkv: DeviceBuffer<f32>,
    q16: DeviceBuffer<u16>,
    k16: DeviceBuffer<u16>,
    v16: DeviceBuffer<u16>,
    o16: DeviceBuffer<u16>,
    gu: DeviceBuffer<f32>,
    a16: DeviceBuffer<u16>,
    cos: DeviceBuffer<f32>,
    sin: DeviceBuffer<f32>,
    unf16: DeviceBuffer<u16>,
    al16: DeviceBuffer<u16>,
    out32: DeviceBuffer<f32>,
}

impl Workspace {
    fn new(dev: i32, n_cap: usize, llm_cap: usize) -> eyre::Result<Self> {
        let b = |len: usize| DeviceBuffer::<u16>::new(dev, len);
        let f = |len: usize| DeviceBuffer::<f32>::new(dev, len);
        Ok(Workspace {
            n_cap,
            llm_cap,
            patches16: b(n_cap * PATCH_K_PAD)?,
            x: f(n_cap * VIT_DIM)?,
            h16: b(n_cap * VIT_DIM)?,
            qkv: f(n_cap * 3 * VIT_DIM)?,
            q16: b(n_cap * VIT_DIM)?,
            k16: b(n_cap * VIT_DIM)?,
            v16: b(n_cap * VIT_DIM)?,
            o16: b(n_cap * VIT_DIM)?,
            gu: f(n_cap * 2 * VIT_FFN)?,
            a16: b(n_cap * VIT_FFN)?,
            cos: f(n_cap * VIT_ROPE_DIM)?,
            sin: f(n_cap * VIT_ROPE_DIM)?,
            unf16: b(llm_cap * ALIGNER_IN)?,
            al16: b(llm_cap * TEXT_DIM)?,
            out32: f(llm_cap * TEXT_DIM)?,
        })
    }

    fn bytes(&self) -> usize {
        self.n_cap * (PATCH_K_PAD * 2 + VIT_DIM * 4 + VIT_DIM * 2 * 5 + 3 * VIT_DIM * 4 + 2 * VIT_FFN * 4 + VIT_FFN * 2 + VIT_ROPE_DIM * 8)
            + self.llm_cap * (ALIGNER_IN * 2 + TEXT_DIM * 2 + TEXT_DIM * 4)
    }
}

pub struct Tower {
    pub device: Device,
    pub meta: MmprojMeta,
    /// Host copy (raw f16 bits). `None` after [`Tower::drop_host`].
    pub host: Option<MmprojHost>,
    pub dev: MmprojDev,
    pub kernels: VitKernels,
    stream: Stream,
    ws: Option<Workspace>,
    /// Sentinel rows (host, f32 [4096]) — START, PAD, NEWLINE, END.
    sentinels: Sentinels,
    dev_bytes: usize,
    /// Wall time of the last [`Tower::encode`] (upload → readback), ms.
    pub last_encode_ms: f64,
    /// When set, [`Tower::encode_rows`] syncs the stream after every kernel
    /// and fills [`Tower::stage_ms`]. Adds a per-launch sync cost (~30 us ×
    /// 9 × 32) — use it for the roofline breakdown, not for e2e timing.
    pub profile: bool,
    /// Per-stage GPU time of the last profiled encode, ms, in pipeline order.
    pub stage_ms: Vec<(&'static str, f64)>,
}

/// Per-stage timer. `mark` syncs the stream, so it is only wired up when
/// [`Tower::profile`] is set.
#[derive(Default)]
struct Prof {
    stages: Vec<(&'static str, f64)>,
    last: Option<Instant>,
}

impl Prof {
    fn start(&mut self) {
        self.last = Some(Instant::now());
    }
    fn mark(&mut self, st: &Stream, name: &'static str) -> eyre::Result<()> {
        st.synchronize()?;
        let now = Instant::now();
        let ms = now.duration_since(self.last.unwrap_or(now)).as_secs_f64() * 1e3;
        self.last = Some(now);
        match self.stages.iter_mut().find(|(n, _)| *n == name) {
            Some((_, acc)) => *acc += ms,
            None => self.stages.push((name, ms)),
        }
        Ok(())
    }
}

/// `mark!(prof, stream, "name")` — no-op when profiling is off.
macro_rules! mark {
    ($p:expr, $st:expr, $name:literal) => {
        if let Some(p) = $p.as_deref_mut() {
            p.mark($st, $name)?;
        }
    };
}

#[derive(Clone)]
struct Sentinels {
    start: Vec<f32>,
    pad: Vec<f32>,
    newline: Vec<f32>,
    end: Vec<f32>,
}

fn upload<T: Copy>(device: Device, name: &str, src: &[T], tally: &mut usize) -> eyre::Result<DeviceBuffer<T>> {
    let mut b = DeviceBuffer::<T>::new(device.id, src.len()).wrap_err_with(|| format!("alloc {name}"))?;
    b.copy_from_host(src).wrap_err_with(|| format!("upload {name}"))?;
    *tally += std::mem::size_of_val(src);
    Ok(b)
}

fn concat<T: Copy>(parts: &[&[T]]) -> Vec<T> {
    let mut v = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for p in parts {
        v.extend_from_slice(p);
    }
    v
}

fn upload_block(device: Device, l: usize, b: &VitBlockHost, tally: &mut usize) -> eyre::Result<VitBlockDev> {
    let n = |s: &str| format!("v.blk.{l}.{s}");
    Ok(VitBlockDev {
        ln1: upload(device, &n("ln1"), &b.ln1, tally)?,
        qkv_w: upload(device, &n("attn_qkv.weight"), &concat(&[&b.attn_q_w, &b.attn_k_w, &b.attn_v_w]), tally)?,
        qkv_b: upload(device, &n("attn_qkv.bias"), &concat(&[&b.attn_q_b, &b.attn_k_b, &b.attn_v_b]), tally)?,
        attn_out_w: upload(device, &n("attn_out.weight"), &b.attn_out_w, tally)?,
        attn_out_b: upload(device, &n("attn_out.bias"), &b.attn_out_b, tally)?,
        ln2: upload(device, &n("ln2"), &b.ln2, tally)?,
        gateup_w: upload(device, &n("ffn_gateup.weight"), &concat(&[&b.ffn_gate_w, &b.ffn_up_w]), tally)?,
        ffn_down_w: upload(device, &n("ffn_down.weight"), &b.ffn_down_w, tally)?,
    })
}

/// `[1024][588]` → `[1024][PATCH_K_PAD]` with zero columns appended.
fn pad_patch_embd(w: &[u16]) -> Vec<u16> {
    let mut out = vec![0u16; VIT_DIM * PATCH_K_PAD];
    for r in 0..VIT_DIM {
        out[r * PATCH_K_PAD..r * PATCH_K_PAD + PATCH_ELEMS].copy_from_slice(&w[r * PATCH_ELEMS..(r + 1) * PATCH_ELEMS]);
    }
    out
}

impl Tower {
    /// Read `mmproj_path` and upload all weights to `device` (sets the
    /// device current on the calling thread). ~890 MiB device + the host
    /// copy retained in `self.host`.
    pub fn load(mmproj_path: &Path, device: Device) -> eyre::Result<Tower> {
        let host = MmprojHost::load(mmproj_path)?;
        Self::from_host(host, device)
    }

    /// Upload an already-loaded host copy.
    pub fn from_host(host: MmprojHost, device: Device) -> eyre::Result<Tower> {
        device.set_current().wrap_err("Tower::load: set_current")?;
        let arch = device.properties().wrap_err("Tower::load: properties")?.gcn_arch_name;
        let kernels = VitKernels::for_arch(&arch).wrap_err("Tower::load: kernels")?;
        let stream = Stream::new(device.id).wrap_err("Tower::load: stream")?;
        let mut tally = 0usize;
        let mut blocks = Vec::with_capacity(host.blocks.len());
        for (l, b) in host.blocks.iter().enumerate() {
            blocks.push(upload_block(device, l, b, &mut tally)?);
        }
        let dev = MmprojDev {
            patch_embd_w: upload(device, "v.patch_embd.weight", &pad_patch_embd(&host.patch_embd_w), &mut tally)?,
            patch_embd_b: upload(device, "v.patch_embd.bias", &host.patch_embd_b, &mut tally)?,
            blocks,
            post_ln: upload(device, "v.post_ln", &host.post_ln, &mut tally)?,
            mm1_w: upload(device, "mm.1.weight", &host.mm1_w, &mut tally)?,
            mm1_b: upload(device, "mm.1.bias", &host.mm1_b, &mut tally)?,
            mm2_w: upload(device, "mm.2.weight", &host.mm2_w, &mut tally)?,
            mm2_b: upload(device, "mm.2.bias", &host.mm2_b, &mut tally)?,
        };
        device.synchronize().wrap_err("Tower::load: synchronize")?;
        let sentinels = Sentinels {
            start: host.img_start.clone(),
            pad: host.img_pad.clone(),
            newline: host.image_newline.clone(),
            end: host.img_end.clone(),
        };
        tracing::info!(
            device = device.id,
            arch = %arch,
            gemm = ?kernels.gemm_path,
            dev_mib = tally as f64 / (1u64 << 20) as f64,
            "vision tower loaded (f16 weights, host copy retained)"
        );
        Ok(Tower {
            device,
            meta: host.meta.clone(),
            host: Some(host),
            dev,
            kernels,
            stream,
            ws: None,
            sentinels,
            dev_bytes: tally,
            last_encode_ms: 0.0,
            profile: std::env::var("VIT_PROFILE").map(|v| v == "1").unwrap_or(false),
            stage_ms: Vec::new(),
        })
    }

    /// Bytes of weights resident on the device (excludes the workspace).
    pub fn device_bytes(&self) -> usize {
        self.dev_bytes
    }

    /// Bytes of activation workspace currently allocated.
    pub fn workspace_bytes(&self) -> usize {
        self.ws.as_ref().map(|w| w.bytes()).unwrap_or(0)
    }

    /// Free the activation workspace (weights stay).
    pub fn free_workspace(&mut self) {
        self.ws = None;
    }

    /// Release the host copy (keeps sentinels).
    pub fn drop_host(&mut self) {
        self.host = None;
    }

    /// Sentinel row for a non-IMAGE block token type.
    pub fn sentinel(&self, ty: u8) -> Option<&[f32]> {
        match TokenType::from_u8(ty)? {
            TokenType::Start => Some(&self.sentinels.start),
            TokenType::Pad => Some(&self.sentinels.pad),
            TokenType::NewLine => Some(&self.sentinels.newline),
            TokenType::End => Some(&self.sentinels.end),
            TokenType::Image => None,
        }
    }

    /// Scatter aligner rows (`[n_llm_h*n_llm_w][4096]`, row-major over the
    /// LLM grid) into the block: `[types.len()][4096]` with row `i` =
    /// `aligner[perm[k]]` for the k-th IMAGE slot, sentinel otherwise
    /// (reference `merge_image_embeddings`; PAD covers both pad kinds).
    pub fn place_rows(&self, layout: &ImageLayout, aligner_rows: &[f32]) -> eyre::Result<Vec<f32>> {
        place_rows_with(layout, aligner_rows, |ty| self.sentinel(ty))
    }

    fn workspace(&mut self, n: usize, n_llm: usize) -> eyre::Result<&mut Workspace> {
        let ok = self.ws.as_ref().map(|w| w.n_cap >= n && w.llm_cap >= n_llm).unwrap_or(false);
        if !ok {
            // Capture the old caps BEFORE freeing, so a grow keeps the high-water mark.
            let cap_n = self.ws.as_ref().map(|w| w.n_cap).unwrap_or(0).max(n);
            let cap_l = self.ws.as_ref().map(|w| w.llm_cap).unwrap_or(0).max(n_llm);
            self.ws = None; // free first: the iGPU workspace comes out of host RAM
            self.ws = Some(Workspace::new(self.device.id, cap_n, cap_l).wrap_err("Tower: workspace alloc")?);
        }
        Ok(self.ws.as_mut().expect("just allocated"))
    }

    /// Run the ViT + aligner on `img` and return `[layout.types.len(), 4096]`
    /// f32 rows in block order.
    pub fn encode(&mut self, img: &PreprocessedImage, layout: &ImageLayout) -> eyre::Result<Vec<f32>> {
        let rows = self.encode_rows(img)?;
        self.place_rows(layout, &rows)
    }

    /// Residual stream `[n][1024]` f32 after `n_layers` ViT blocks (no
    /// post_ln) — the GPU twin of [`crate::reference::vit_trunk_x`], for
    /// the layer-bisect oracle. `n_layers = 0` returns the patch embedding.
    pub fn trunk_x(&mut self, img: &PreprocessedImage, n_layers: usize) -> eyre::Result<Vec<f32>> {
        let (n_h, n_w) = (img.n_vit_h as usize, img.n_vit_w as usize);
        let n = n_h * n_w;
        self.device.set_current()?;
        self.upload_inputs(img)?;
        let ws = self.ws.as_mut().expect("workspace");
        let kk = &self.kernels;
        let st = &self.stream;
        let dev = &self.dev;
        let nt = n as u32;
        kk.gemm(st, Some(&mut ws.x), None, &ws.patches16, &dev.patch_embd_w, Some(&dev.patch_embd_b), nt, PATCH_K_PAD as u32, VIT_DIM as u32, 0)?;
        Self::run_blocks(kk, st, ws, dev, nt, n_layers, None)?;
        st.synchronize()?;
        let mut out = vec![0f32; n * VIT_DIM];
        ws.x.slice_view(0, n * VIT_DIM).copy_to_host(&mut out)?;
        Ok(out)
    }

    /// ViT + aligner only: `[n_llm_h*n_llm_w][4096]` f32, row-major over the LLM grid.
    pub fn encode_rows(&mut self, img: &PreprocessedImage) -> eyre::Result<Vec<f32>> {
        let t0 = Instant::now();
        let (n_h, n_w) = (img.n_vit_h as usize, img.n_vit_w as usize);
        let n = n_h * n_w;
        let (lh, lw) = (n_h.div_ceil(3), n_w.div_ceil(3));
        let n_llm = lh * lw;
        self.device.set_current().wrap_err("Tower::encode: set_current")?;
        self.upload_inputs(img)?;

        let mut prof = self.profile.then(Prof::default);
        let mut prof = prof.as_mut();
        let ws = self.ws.as_mut().expect("workspace");
        let kk = &self.kernels;
        let st = &self.stream;
        let dev = &self.dev;
        let nt = n as u32;
        st.synchronize().wrap_err("Tower::encode: sync after upload")?;
        if let Some(p) = prof.as_deref_mut() {
            p.start();
        }

        kk.gemm(st, Some(&mut ws.x), None, &ws.patches16, &dev.patch_embd_w, Some(&dev.patch_embd_b), nt, PATCH_K_PAD as u32, VIT_DIM as u32, 0)
            .wrap_err("patch_embd")?;
        mark!(prof, st, "patch_embd");
        Self::run_blocks(kk, st, ws, dev, nt, dev.blocks.len(), prof.as_deref_mut())?;

        kk.rmsnorm_f16(st, &mut ws.h16, &ws.x, &dev.post_ln, nt, VIT_DIM as u32, VIT_RMS_EPS).wrap_err("post_ln")?;
        mark!(prof, st, "rmsnorm");
        kk.unfold(st, &mut ws.unf16, &ws.h16, n_h as u32, n_w as u32, lh as u32, lw as u32, VIT_DIM as u32).wrap_err("unfold")?;
        mark!(prof, st, "unfold");
        kk.gemm(st, None, Some(&mut ws.al16), &ws.unf16, &dev.mm1_w, Some(&dev.mm1_b), n_llm as u32, ALIGNER_IN as u32, TEXT_DIM as u32, FLAG_GELU)
            .wrap_err("mm.1")?;
        mark!(prof, st, "gemm_mm1");
        kk.gemm(st, Some(&mut ws.out32), None, &ws.al16, &dev.mm2_w, Some(&dev.mm2_b), n_llm as u32, TEXT_DIM as u32, TEXT_DIM as u32, 0)
            .wrap_err("mm.2")?;
        mark!(prof, st, "gemm_mm2");
        st.synchronize().wrap_err("Tower::encode: synchronize")?;
        let mut out = vec![0f32; n_llm * TEXT_DIM];
        ws.out32.slice_view(0, n_llm * TEXT_DIM).copy_to_host(&mut out)?;
        mark!(prof, st, "d2h");
        self.stage_ms = prof.map(|p| std::mem::take(&mut p.stages)).unwrap_or_default();
        self.last_encode_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(out)
    }

    /// Validate `img`, (re)size the workspace and upload the f16 patches +
    /// RoPE tables. Leaves `self.ws` populated.
    fn upload_inputs(&mut self, img: &PreprocessedImage) -> eyre::Result<()> {
        let (n_h, n_w) = (img.n_vit_h as usize, img.n_vit_w as usize);
        let n = n_h * n_w;
        if n == 0 || img.patches.len() != n * PATCH_ELEMS {
            return Err(eyre!("Tower::encode: patches len {} != {n}x{PATCH_ELEMS}", img.patches.len()));
        }
        let n_llm = n_h.div_ceil(3) * n_w.div_ceil(3);
        // Host-side prep: f16 patches padded to PATCH_K_PAD, RoPE tables.
        let mut p16 = vec![0u16; n * PATCH_K_PAD];
        for t in 0..n {
            let src = &img.patches[t * PATCH_ELEMS..(t + 1) * PATCH_ELEMS];
            let dst = &mut p16[t * PATCH_K_PAD..t * PATCH_K_PAD + PATCH_ELEMS];
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = f32_to_f16_bits(s);
            }
        }
        let (cos, sin) = vision_cos_sin(n_h as u32, n_w as u32);
        let ws = self.workspace(n, n_llm)?;
        ws.patches16.slice_view_mut(0, n * PATCH_K_PAD).copy_from_host(&p16)?;
        ws.cos.slice_view_mut(0, n * VIT_ROPE_DIM).copy_from_host(&cos)?;
        ws.sin.slice_view_mut(0, n * VIT_ROPE_DIM).copy_from_host(&sin)?;
        Ok(())
    }

    /// The `n_layers` ViT blocks, in place on `ws.x` (f32 residual stream).
    fn run_blocks(
        kk: &VitKernels,
        st: &Stream,
        ws: &mut Workspace,
        dev: &MmprojDev,
        nt: u32,
        n_layers: usize,
        mut prof: Option<&mut Prof>,
    ) -> eyre::Result<()> {
        let scale = 1.0 / (VIT_HEAD_DIM as f32).sqrt();
        for (l, blk) in dev.blocks.iter().take(n_layers).enumerate() {
            let ctx = |s: &str| format!("blk.{l}.{s}");
            kk.rmsnorm_f16(st, &mut ws.h16, &ws.x, &blk.ln1, nt, VIT_DIM as u32, VIT_RMS_EPS).wrap_err_with(|| ctx("ln1"))?;
            mark!(prof, st, "rmsnorm");
            kk.gemm(st, Some(&mut ws.qkv), None, &ws.h16, &blk.qkv_w, Some(&blk.qkv_b), nt, VIT_DIM as u32, 3 * VIT_DIM as u32, 0)
                .wrap_err_with(|| ctx("qkv"))?;
            mark!(prof, st, "gemm_qkv");
            kk.rope_split(st, &mut ws.q16, &mut ws.k16, &mut ws.v16, &ws.qkv, &ws.cos, &ws.sin, nt).wrap_err_with(|| ctx("rope"))?;
            mark!(prof, st, "rope_split");
            kk.attention(st, &mut ws.o16, &ws.q16, &ws.k16, &ws.v16, nt, scale).wrap_err_with(|| ctx("attn"))?;
            mark!(prof, st, "attention");
            kk.gemm(st, Some(&mut ws.x), None, &ws.o16, &blk.attn_out_w, Some(&blk.attn_out_b), nt, VIT_DIM as u32, VIT_DIM as u32, FLAG_ACCUM)
                .wrap_err_with(|| ctx("attn_out"))?;
            mark!(prof, st, "gemm_attn_out");
            kk.rmsnorm_f16(st, &mut ws.h16, &ws.x, &blk.ln2, nt, VIT_DIM as u32, VIT_RMS_EPS).wrap_err_with(|| ctx("ln2"))?;
            mark!(prof, st, "rmsnorm");
            kk.gemm(st, Some(&mut ws.gu), None, &ws.h16, &blk.gateup_w, None, nt, VIT_DIM as u32, 2 * VIT_FFN as u32, 0)
                .wrap_err_with(|| ctx("gateup"))?;
            mark!(prof, st, "gemm_gateup");
            kk.swiglu_f16(st, &mut ws.a16, &ws.gu, nt, VIT_FFN as u32).wrap_err_with(|| ctx("swiglu"))?;
            mark!(prof, st, "swiglu");
            kk.gemm(st, Some(&mut ws.x), None, &ws.a16, &blk.ffn_down_w, None, nt, VIT_FFN as u32, VIT_DIM as u32, FLAG_ACCUM)
                .wrap_err_with(|| ctx("down"))?;
            mark!(prof, st, "gemm_down");
        }
        Ok(())
    }
}

/// [`Tower::place_rows`] with an explicit sentinel lookup (testable without a device).
pub fn place_rows_with<'a>(
    layout: &ImageLayout,
    aligner_rows: &[f32],
    sentinel: impl Fn(u8) -> Option<&'a [f32]>,
) -> eyre::Result<Vec<f32>> {
    let n_rows = layout.n_llm_h as usize * layout.n_llm_w as usize;
    if aligner_rows.len() != n_rows * TEXT_DIM {
        return Err(eyre!("place_rows: aligner rows len {} != {}x{}", aligner_rows.len(), n_rows, TEXT_DIM));
    }
    let mut out = vec![0f32; layout.types.len() * TEXT_DIM];
    let mut k = 0usize;
    for (i, &ty) in layout.types.iter().enumerate() {
        let dst = &mut out[i * TEXT_DIM..(i + 1) * TEXT_DIM];
        if ty == TokenType::Image as u8 {
            let r = *layout.perm.get(k).ok_or_else(|| eyre!("place_rows: perm shorter than IMAGE slots"))? as usize;
            if r >= n_rows {
                return Err(eyre!("place_rows: perm[{k}] = {r} out of range {n_rows}"));
            }
            dst.copy_from_slice(&aligner_rows[r * TEXT_DIM..(r + 1) * TEXT_DIM]);
            k += 1;
        } else {
            let s = sentinel(ty).ok_or_else(|| eyre!("place_rows: no sentinel for type {ty}"))?;
            if s.len() != TEXT_DIM {
                return Err(eyre!("place_rows: sentinel len {}", s.len()));
            }
            dst.copy_from_slice(s);
        }
    }
    if k != layout.perm.len() {
        return Err(eyre!("place_rows: {} IMAGE slots but perm has {}", k, layout.perm.len()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::layout_for_grid;

    #[test]
    fn place_rows_scatters_by_perm() {
        // 5x4 ViT grid → 2x2 LLM grid (even h, row_len 3, pad_last = 1*3%2*2 = 2).
        let layout = layout_for_grid(5, 4, 0);
        assert_eq!((layout.n_llm_h, layout.n_llm_w), (2, 2));
        let n_rows = 4;
        let aligner: Vec<f32> = (0..n_rows * TEXT_DIM).map(|i| (i / TEXT_DIM) as f32 + 100.0).collect();
        let sent = |ty: u8| -> Option<&'static [f32]> {
            static S: [[f32; TEXT_DIM]; 5] = [[0.0; TEXT_DIM], [1.0; TEXT_DIM], [f32::NAN; TEXT_DIM], [3.0; TEXT_DIM], [4.0; TEXT_DIM]];
            if ty == 2 { None } else { Some(&S[ty as usize]) }
        };
        let out = place_rows_with(&layout, &aligner, sent).unwrap();
        assert_eq!(out.len(), layout.types.len() * TEXT_DIM);
        // Block: PAD PAD PAD START | (r0c0 r1c0 r0c1 r1c1 NL NL) | PAD PAD END
        let expect: Vec<f32> = vec![1.0, 1.0, 1.0, 0.0, 100.0, 102.0, 101.0, 103.0, 3.0, 3.0, 1.0, 1.0, 4.0];
        let got: Vec<f32> = (0..layout.types.len()).map(|i| out[i * TEXT_DIM]).collect();
        assert_eq!(got, expect);
        assert_eq!(layout.perm, vec![0, 2, 1, 3]);
    }
}
