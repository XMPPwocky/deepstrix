//! Qwen3-Embedding forward on the dGPU, for the hub's embed phase
//! (docs/v41/EMBED_PHASE_DESIGN.md §5).
//!
//! Layer-major over every token of a phase, weights STREAMED: reader threads
//! pread layer `l + 1` from the GGUF into a pinned host buffer while the dGPU
//! computes layer `l`; the copy stream moves it into one of two dGPU ring
//! buffers, where `qe_q8_0_repack_rows` turns the GGUF Q8_0 blocks into the
//! split layout the GEMM reads. All device memory comes from a [`LoanAlloc`]
//! (the hub: an in-place loan of immutable V4.1 weights; the standalone gate:
//! a plain allocation), carved in the order [`EmbedSizing::buffer_sizes`] lists.
//!
//! Projections: the production Q8_0 x f16 WMMA GEMM (`gemm_f16x`, gfx1201; the
//! caller passes the engine's resident module). Attention, one launch per input
//! segment of a sub-batch, from the `HGP_G = 4` build of `gqa_attention.hip`:
//! `prefill_flash_wmma_fa2` (one WG per query head; gate E2 covers Qwen's
//! kv_group 4), or with [`Qwen3EmbedKernels::set_packed_attention`] the
//! full-group packed kernel (4 query heads per WG; design §18).
//! Everything else: `kernels/qwen3_embed.hip`. The modules here are loaded per
//! phase and dropped after it.

use std::sync::mpsc;
use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_core::direct_io::DirectFiles;
use v4flash_core::qwen3_embed::{Qwen3EmbedConfig, Qwen3EmbedModel, LayerLayout};
use v4flash_core::MappedGguf;
use v4flash_hip::{launch_kernel, DeviceBuffer, Event, LaunchConfig, Module, PinnedBuffer, Stream};

use crate::dgpu_loan::LoanAlloc;
use crate::gqa_attention::GqaAttention;
use crate::q8_0::{F16xTile, Q8_0MatvecWmma};

const QWEN3_EMBED_GFX1201: &[u8] = include_bytes!(env!("KERNEL_QWEN3_EMBED_GFX1201"));
const QWEN3_EMBED_GFX1151: &[u8] = include_bytes!(env!("KERNEL_QWEN3_EMBED_GFX1151"));

/// Threads per block of the row kernels (mirrors `QE_BLOCK`).
const QE_BLOCK: u32 = 256;

/// GEMM activation row pitch for a K-wide input: `>= k`, a multiple of 8, and
/// off a power of two (`gemm_f16x`: power-of-two rows alias L2 sets).
pub fn act_pitch(k: usize) -> usize {
    let p = k.div_ceil(8) * 8;
    if p.is_power_of_two() { p + 8 } else { p }
}

/// Reader threads per layer read and for the token-embedding rows.
pub const READERS: usize = 4;

/// Max blocks per row of `qe_q8_0_repack_rows` (mirrors `QE_REPACK_MAX_BLOCKS`).
const REPACK_MAX_BLOCKS: usize = 480;

pub struct Qwen3EmbedKernels {
    module: Module,
    /// The `HGP_G = 4` attention module: `fa2` plus the full-group packed kernel
    /// at 4 query heads per WG (design §18).
    attn: GqaAttention,
    /// Attention through `prefill_flash_wmma_fa2_hg_packed` (when kv_group is a
    /// multiple of 4) instead of `prefill_flash_wmma_fa2`.
    packed_attn: bool,
}

impl Qwen3EmbedKernels {
    /// The dGPU (gfx1201) only: the WMMA GEMM and attention have no fast path
    /// elsewhere. Attention starts on `fa2`; see [`Self::set_packed_attention`].
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        if !arch.starts_with("gfx1201") {
            return Err(eyre!("qwen3 embed forward needs gfx1201 (the dGPU), got {arch}"));
        }
        let _ = QWEN3_EMBED_GFX1151; // built for every target; never loaded there
        Ok(Qwen3EmbedKernels {
            module: Module::load_data(QWEN3_EMBED_GFX1201)?,
            attn: GqaAttention::for_arch_group4(arch)?,
            packed_attn: false,
        })
    }

    /// Attention through the full-group packed kernel (4 query heads per WG,
    /// one K/V staging and softmax pass for all of them) instead of `fa2` (one WG
    /// per query head). A model whose kv_group is not a multiple of 4 stays on
    /// `fa2` either way.
    pub fn set_packed_attention(&mut self, on: bool) {
        self.packed_attn = on;
    }

    /// Whether `cfg`'s attention runs packed (the setting and the shape agree).
    pub fn attention_packed(&self, cfg: &Qwen3EmbedConfig) -> bool {
        self.packed_attn && (cfg.n_head / cfg.n_kv_head) as u32 % self.attn.packed_group() == 0
    }

    /// In place: `rows` GGUF Q8_0 rows of `blocks` blocks starting at `w[0]`
    /// -> the split layout `gemm_f16x` reads (`weights::repack_q8_0`).
    pub fn repack_q8_0_rows(&self, stream: &Stream, w: &mut DeviceBuffer<u8>, rows: usize, blocks: usize) -> eyre::Result<()> {
        if rows == 0 {
            return Ok(());
        }
        if blocks == 0 || blocks > REPACK_MAX_BLOCKS || w.byte_len() < rows * blocks * 34 || (w.raw() as usize) % 2 != 0 {
            return Err(eyre!("qe_q8_0_repack_rows: rows {rows} x blocks {blocks} do not fit the buffer / LDS / alignment"));
        }
        let f = self.module.get_function("qe_q8_0_repack_rows")?;
        let cfg = LaunchConfig { grid: (rows as u32, 1, 1), block: (QE_BLOCK, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [w.raw(), blocks as u32])
    }

    /// Repack every Q8_0 matrix of the layer in ring buffer `w`.
    fn repack_layer(&self, stream: &Stream, c: &Qwen3EmbedConfig, l: &LayerLayout, w: &DeviceBuffer<u8>) -> eyre::Result<()> {
        let (d, qw, ff) = (c.n_embd, c.q_width(), c.n_ff);
        let groups = [
            (l.q, c.qkv_rows(), d / 32),
            (l.o, d, qw / 32),
            (l.gate, 2 * ff, d / 32),
            (l.down, d, ff / 32),
        ];
        for (off, rows, blocks) in groups {
            let mut v = w.slice_view(off, rows * blocks * 34);
            self.repack_q8_0_rows(stream, &mut v, rows, blocks)?;
        }
        Ok(())
    }

    /// `out16[r, :n] = f16(rmsnorm(x[r, :]) * w)` for `rows` rows of `x` (pitch n).
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_f16(&self, stream: &Stream, out16: &mut DeviceBuffer<u16>, out_pitch: u32, x: &DeviceBuffer<f32>, w: &DeviceBuffer<f32>, n: u32, eps: f32, rows: u32) -> eyre::Result<()> {
        if rows == 0 {
            return Ok(());
        }
        if out_pitch < n || out16.len() < (rows * out_pitch) as usize || x.len() < (rows * n) as usize || w.len() < n as usize {
            return Err(eyre!("qe_rmsnorm_f16: buffers too small (rows {rows}, n {n}, pitch {out_pitch})"));
        }
        let f = self.module.get_function("qe_rmsnorm_f16")?;
        let cfg = LaunchConfig { grid: (rows, 1, 1), block: (QE_BLOCK, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [out16.raw(), out_pitch, x.raw(), w.raw(), n, eps])
    }

    /// Per (row, head) of the fused qkv GEMM output: q/k RMSNorm + NeoX RoPE,
    /// q -> `q16[r]`, k / v -> `kc` / `vc` at row `row0 + r`.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope(
        &self,
        stream: &Stream,
        cfg: &Qwen3EmbedConfig,
        qkv: &DeviceBuffer<f32>,
        q16: &mut DeviceBuffer<u16>,
        kc: &mut DeviceBuffer<u16>,
        vc: &mut DeviceBuffer<u16>,
        q_norm: &DeviceBuffer<f32>,
        k_norm: &DeviceBuffer<f32>,
        pos: &DeviceBuffer<u32>,
        row0: u32,
        rows: u32,
    ) -> eyre::Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let hd = cfg.head_dim as u32;
        if hd > QE_BLOCK || !hd.is_power_of_two() {
            return Err(eyre!("qe_qk_norm_rope: head_dim {hd} must be a power of two <= {QE_BLOCK}"));
        }
        let r = rows as usize;
        let kv_end = (row0 as usize + r) * cfg.kv_width();
        if qkv.len() < r * cfg.qkv_rows() || q16.len() < r * cfg.q_width() || kc.len() < kv_end || vc.len() < kv_end || pos.len() < r {
            return Err(eyre!("qe_qk_norm_rope: buffers too small (row0 {row0}, rows {rows})"));
        }
        let f = self.module.get_function("qe_qk_norm_rope")?;
        let heads = (cfg.n_head + 2 * cfg.n_kv_head) as u32;
        let lc = LaunchConfig { grid: (rows, heads, 1), block: (hd, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, lc, stream, [
            qkv.raw(), q16.raw(), kc.raw(), vc.raw(), q_norm.raw(), k_norm.raw(), pos.raw(),
            row0, cfg.n_head as u32, cfg.n_kv_head as u32, hd, cfg.rope_theta, cfg.eps
        ])
    }

    /// `out16[r, :n] = f16(x[r, :])`, x pitch n.
    pub fn cast_f16(&self, stream: &Stream, out16: &mut DeviceBuffer<u16>, out_pitch: u32, x: &DeviceBuffer<f32>, n: u32, rows: u32) -> eyre::Result<()> {
        if rows == 0 {
            return Ok(());
        }
        if out_pitch < n || out16.len() < (rows * out_pitch) as usize || x.len() < (rows * n) as usize {
            return Err(eyre!("qe_cast_f16: buffers too small"));
        }
        let f = self.module.get_function("qe_cast_f16")?;
        let cfg = LaunchConfig { grid: (n.div_ceil(QE_BLOCK), rows, 1), block: (QE_BLOCK, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [out16.raw(), out_pitch, x.raw(), n])
    }

    /// `out16[r, i] = f16(silu(gu[r, i]) * gu[r, n_ff + i])`.
    pub fn swiglu_f16(&self, stream: &Stream, out16: &mut DeviceBuffer<u16>, out_pitch: u32, gu: &DeviceBuffer<f32>, n_ff: u32, rows: u32) -> eyre::Result<()> {
        if rows == 0 {
            return Ok(());
        }
        if out_pitch < n_ff || out16.len() < (rows * out_pitch) as usize || gu.len() < (rows * 2 * n_ff) as usize {
            return Err(eyre!("qe_swiglu_f16: buffers too small"));
        }
        let f = self.module.get_function("qe_swiglu_f16")?;
        let cfg = LaunchConfig { grid: (n_ff.div_ceil(QE_BLOCK), rows, 1), block: (QE_BLOCK, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [out16.raw(), out_pitch, gu.raw(), n_ff])
    }

    /// `resid[i] += delta[i]` for `i < n`.
    pub fn add(&self, stream: &Stream, resid: &mut DeviceBuffer<f32>, delta: &DeviceBuffer<f32>, n: u32) -> eyre::Result<()> {
        if n == 0 {
            return Ok(());
        }
        if resid.len() < n as usize || delta.len() < n as usize {
            return Err(eyre!("qe_add: buffers too small"));
        }
        let f = self.module.get_function("qe_add")?;
        let cfg = LaunchConfig { grid: (n.div_ceil(QE_BLOCK).min(8192), 1, 1), block: (QE_BLOCK, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [resid.raw(), delta.raw(), n])
    }

    /// `out[c, :] = src[idx[c], :]` for `count` rows of `n`.
    pub fn gather_rows(&self, stream: &Stream, out: &mut DeviceBuffer<f32>, src: &DeviceBuffer<f32>, idx: &DeviceBuffer<u32>, n: u32, count: u32) -> eyre::Result<()> {
        if count == 0 {
            return Ok(());
        }
        if out.len() < (count * n) as usize || idx.len() < count as usize {
            return Err(eyre!("qe_gather_rows: buffers too small"));
        }
        let f = self.module.get_function("qe_gather_rows")?;
        let cfg = LaunchConfig { grid: (n.div_ceil(QE_BLOCK), count, 1), block: (QE_BLOCK, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [out.raw(), src.raw(), idx.raw(), n])
    }

}

/// The projection GEMM: `out[rows, m] = x16[rows, :k] · W[m, k]^T`, `W` in the
/// split Q8_0 layout (after `qe_q8_0_repack_rows`).
#[allow(clippy::too_many_arguments)]
fn gemm(g: &Q8_0MatvecWmma, stream: &Stream, out: &mut DeviceBuffer<f32>, w: &DeviceBuffer<u8>, x16: &DeviceBuffer<u16>, k: usize, m: usize, rows: usize, pitch: usize) -> eyre::Result<()> {
    g.gemm_f16x_tile(F16xTile::Base, stream, out, w, x16, k as u32, m as u32, 1, rows as u32, pitch as u32)
}

/// The two knobs that size one phase's device memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbedSizing {
    /// Max tokens in one phase (`T`).
    pub phase_tokens: usize,
    /// Rows per sub-batch (`R`).
    pub sub_rows: usize,
}

/// The loan's buffers, in carve order (largest first at the defaults).
const BUFFERS: [&str; 11] = ["resid", "ring0", "ring1", "gemm_out", "kc", "vc", "x16", "attn_out", "q16", "pos", "idx"];

impl EmbedSizing {
    fn x16_pitch(cfg: &Qwen3EmbedConfig) -> usize {
        act_pitch(cfg.n_embd).max(act_pitch(cfg.q_width())).max(act_pitch(cfg.n_ff))
    }

    fn gemm_out_width(cfg: &Qwen3EmbedConfig) -> usize {
        cfg.qkv_rows().max(2 * cfg.n_ff).max(cfg.n_embd)
    }

    /// Bytes of each buffer, in the order [`EmbedBuffers::carve`] takes them
    /// (the hub sizes its loan with this; the order IS the placement).
    pub fn buffer_sizes(&self, cfg: &Qwen3EmbedConfig, layout: &LayerLayout) -> Vec<(&'static str, usize)> {
        let (t, r) = (self.phase_tokens, self.sub_rows);
        let sizes = [
            t * cfg.n_embd * 4,
            layout.bytes,
            layout.bytes,
            r * Self::gemm_out_width(cfg) * 4,
            t * cfg.kv_width() * 2,
            t * cfg.kv_width() * 2,
            r * Self::x16_pitch(cfg) * 2,
            r * cfg.q_width() * 4,
            r * cfg.q_width() * 2,
            t * 4,
            t * 4,
        ];
        BUFFERS.iter().copied().zip(sizes).collect()
    }

    pub fn total_bytes(&self, cfg: &Qwen3EmbedConfig, layout: &LayerLayout) -> usize {
        self.buffer_sizes(cfg, layout).iter().map(|(_, b)| b.div_ceil(crate::dgpu_loan::LOAN_ALIGN) * crate::dgpu_loan::LOAN_ALIGN).sum()
    }
}

/// One phase's device buffers (views into the loan).
pub struct EmbedBuffers {
    sizing: EmbedSizing,
    resid: DeviceBuffer<f32>,
    ring: [DeviceBuffer<u8>; 2],
    gemm_out: DeviceBuffer<f32>,
    kc: DeviceBuffer<u16>,
    vc: DeviceBuffer<u16>,
    x16: DeviceBuffer<u16>,
    attn_out: DeviceBuffer<f32>,
    q16: DeviceBuffer<u16>,
    pos: DeviceBuffer<u32>,
    idx: DeviceBuffer<u32>,
}

impl EmbedBuffers {
    pub fn carve(alloc: &mut LoanAlloc, sizing: EmbedSizing, cfg: &Qwen3EmbedConfig, layout: &LayerLayout) -> eyre::Result<Self> {
        let s = sizing.buffer_sizes(cfg, layout);
        let n = |i: usize, elem: usize| s[i].1 / elem;
        Ok(EmbedBuffers {
            sizing,
            resid: alloc.take::<f32>(n(0, 4))?,
            ring: [alloc.take::<u8>(n(1, 1))?, alloc.take::<u8>(n(2, 1))?],
            gemm_out: alloc.take::<f32>(n(3, 4))?,
            kc: alloc.take::<u16>(n(4, 2))?,
            vc: alloc.take::<u16>(n(5, 2))?,
            x16: alloc.take::<u16>(n(6, 2))?,
            attn_out: alloc.take::<f32>(n(7, 4))?,
            q16: alloc.take::<u16>(n(8, 2))?,
            pos: alloc.take::<u32>(n(9, 4))?,
            idx: alloc.take::<u32>(n(10, 4))?,
        })
    }
}

/// Where one phase's time went.
#[derive(Clone, Copy, Debug, Default)]
pub struct EmbedTimings {
    pub tokens: usize,
    /// Token-embedding rows: preads + dequant + upload.
    pub rows_ms: f64,
    /// The reader thread's pread time, summed over layers.
    pub read_ms: f64,
    /// The engine thread waiting for the reader (the I/O-bound part).
    pub wait_read_ms: f64,
    pub total_ms: f64,
}

/// A pinned host buffer handed to the reader thread.
struct HostBuf(*mut u8, usize);
// SAFETY: the engine thread never touches a buffer while the reader fills it:
// it hands the buffer over only after its previous H2D completed, and
// reads it (H2D) only after the reader reported it filled.
unsafe impl Send for HostBuf {}

/// Run the forward over `inputs` (token ids, EOS included) and return each
/// input's LAST hidden row, before `output_norm` (the caller finishes it with
/// `Qwen3EmbedModel::finish` and the request's `dimensions`).
///
/// `g` = the GEMM module (the hub passes its engine's resident `q8_wmma`).
/// `file` serves the token-embedding rows; `direct` (the GGUF and its replicas
/// on other drives, O_DIRECT) streams the layers, each as one aligned span
/// read by `readers` threads. `host` = two pinned buffers of at least
/// [`host_bytes`]. `compute` and `copy` are streams on the dGPU.
/// `after_layer(l)` runs once layer `l` is queued (the hub pets its watchdog
/// there); an `Err` stops the forward (fault injection, cancellation). On
/// return every queued device operation has completed, also on error, and
/// the reader threads have exited.
#[allow(clippy::too_many_arguments)]
pub fn run(
    k: &Qwen3EmbedKernels,
    g: &Q8_0MatvecWmma,
    model: &Qwen3EmbedModel,
    file: &MappedGguf,
    direct: &DirectFiles,
    readers: usize,
    bufs: &mut EmbedBuffers,
    host: &mut [PinnedBuffer<u8>; 2],
    compute: &Stream,
    copy: &Stream,
    inputs: &[&[u32]],
    after_layer: &mut dyn FnMut(usize) -> eyre::Result<()>,
) -> eyre::Result<(Vec<Vec<f32>>, EmbedTimings)> {
    // The events live here, past the streams' synchronize below: on an error
    // path they may still be recorded / waited on when `run_inner` returns.
    let ev = Events {
        h2d_done: [Event::new_no_timing()?, Event::new_no_timing()?],
        comp_done: [Event::new_no_timing()?, Event::new_no_timing()?],
    };
    let r = run_inner(k, g, model, file, direct, readers, bufs, host, compute, copy, inputs, after_layer, &ev);
    // Nothing may still be reading the host buffers or writing the loan.
    let s1 = compute.synchronize();
    let s2 = copy.synchronize();
    drop(ev);
    let out = r?;
    s1?;
    s2?;
    Ok(out)
}

/// Per ring buffer: its H2D done (copy stream) / the compute that read it
/// done (compute stream).
struct Events {
    h2d_done: [Event; 2],
    comp_done: [Event; 2],
}

/// Bytes each of `run`'s two pinned host buffers must hold: the largest
/// layer's aligned O_DIRECT span.
pub fn host_bytes(model: &Qwen3EmbedModel) -> eyre::Result<usize> {
    model.max_layer_span_bytes()
}

#[allow(clippy::too_many_arguments)]
fn run_inner(
    k: &Qwen3EmbedKernels,
    g: &Q8_0MatvecWmma,
    model: &Qwen3EmbedModel,
    file: &MappedGguf,
    direct: &DirectFiles,
    readers: usize,
    bufs: &mut EmbedBuffers,
    host: &mut [PinnedBuffer<u8>; 2],
    compute: &Stream,
    copy: &Stream,
    inputs: &[&[u32]],
    after_layer: &mut dyn FnMut(usize) -> eyre::Result<()>,
    ev: &Events,
) -> eyre::Result<(Vec<Vec<f32>>, EmbedTimings)> {
    let Events { h2d_done, comp_done } = ev;
    let t0 = Instant::now();
    let c = &model.cfg;
    let layout = &model.layout;
    let d = c.n_embd;
    let mut tm = EmbedTimings::default();
    if inputs.is_empty() {
        return Ok((Vec::new(), tm));
    }
    let span_cap = host_bytes(model)?;
    if host.iter().any(|h| h.len() < span_cap) {
        return Err(eyre!("embed forward: pinned buffers must hold {span_cap} B"));
    }
    if direct.size() != file.gguf().file_size {
        return Err(eyre!("embed forward: the direct reader's file ({} B) is not the GGUF ({} B)", direct.size(), file.gguf().file_size));
    }
    // Pack the inputs.
    let mut starts = Vec::with_capacity(inputs.len());
    let mut ids: Vec<u32> = Vec::new();
    let mut pos: Vec<u32> = Vec::new();
    for inp in inputs {
        if inp.is_empty() {
            return Err(eyre!("embed forward: empty input"));
        }
        starts.push(ids.len());
        ids.extend_from_slice(inp);
        pos.extend(0..inp.len() as u32);
    }
    let t = ids.len();
    tm.tokens = t;
    if t > bufs.sizing.phase_tokens {
        return Err(eyre!("embed forward: {t} tokens > the phase's {}", bufs.sizing.phase_tokens));
    }
    let tr = Instant::now();
    let rows = model.token_rows(file, &ids, READERS)?;
    bufs.resid.slice_view_mut(0, t * d).copy_from_host(&rows)?;
    bufs.pos.slice_view_mut(0, t).copy_from_host(&pos)?;
    let last: Vec<u32> = starts.iter().zip(inputs).map(|(s, inp)| (s + inp.len() - 1) as u32).collect();
    bufs.idx.slice_view_mut(0, last.len()).copy_from_host(&last)?;
    tm.rows_ms = tr.elapsed().as_secs_f64() * 1e3;
    drop(rows);

    let n_layer = c.n_layer;
    let host_bufs: [HostBuf; 2] = [
        HostBuf(host[0].as_mut_slice().as_mut_ptr(), host[0].len()),
        HostBuf(host[1].as_mut_slice().as_mut_ptr(), host[1].len()),
    ];
    let [hs0, hs1] = host_bufs;
    let mut comp_recorded = [false, false];

    std::thread::scope(|scope| -> eyre::Result<()> {
        let (job_tx, job_rx) = mpsc::channel::<(usize, HostBuf)>();
        // (layer, where its span starts in the host buffer, read ms)
        let (done_tx, done_rx) = mpsc::channel::<eyre::Result<(usize, usize, f64)>>();
        scope.spawn(move || {
            for (l, hs) in job_rx {
                let t = Instant::now();
                // SAFETY: see `HostBuf`.
                let dst = unsafe { std::slice::from_raw_parts_mut(hs.0, hs.1) };
                // A reader panic becomes an error here, not a re-panic at the
                // end of the scope (which would unwind with the loan out).
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let (off, len) = model.layer_span(l)?;
                    direct.read_span(off, len, dst, readers)
                }))
                .unwrap_or_else(|_| Err(eyre!("embed reader panicked reading layer {l}")))
                .map(|head| (l, head, t.elapsed().as_secs_f64() * 1e3));
                let failed = r.is_err();
                if done_tx.send(r).is_err() || failed {
                    break;
                }
            }
        });
        let mut pending = [Some(hs0), Some(hs1)];
        for (l, p) in pending.iter_mut().enumerate().take(n_layer.min(2)) {
            job_tx.send((l, p.take().expect("host buffer"))).map_err(|_| eyre!("embed reader exited"))?;
        }
        for l in 0..n_layer {
            let s = l % 2;
            let tw = Instant::now();
            let (got, head, ms) = done_rx.recv().map_err(|_| eyre!("embed reader exited"))??;
            if got != l {
                return Err(eyre!("embed reader returned layer {got}, expected {l}"));
            }
            tm.wait_read_ms += tw.elapsed().as_secs_f64() * 1e3;
            tm.read_ms += ms;
            // Ring buffer `s` was last read by layer l - 2.
            if comp_recorded[s] {
                copy.wait_event(&comp_done[s])?;
            }
            // Each tensor from its place in the file span to its LayerLayout slot.
            let (off, _) = model.layer_span(l)?;
            for (tl, at) in layout.placements(&model.layers[l]) {
                let src = head + (tl.offset - off) as usize;
                let n = tl.bytes as usize;
                bufs.ring[s].slice_view_mut(at, n).copy_from_host_async(&host[s].as_slice()[src..src + n], copy)?;
            }
            h2d_done[s].record(copy)?;
            compute.wait_event(&h2d_done[s])?;
            let w = bufs.ring[s].slice_view(0, bufs.ring[s].len());
            // GGUF Q8_0 blocks -> the split layout the GEMM reads, in place.
            k.repack_layer(compute, c, layout, &w)?;
            layer(k, g, c, layout, &w, bufs, &starts, inputs, t, compute)?;
            comp_done[s].record(compute)?;
            comp_recorded[s] = true;
            if l + 2 < n_layer {
                // The host buffer is free once its H2D is done.
                h2d_done[s].synchronize()?;
                let hs = HostBuf(host[s].as_mut_slice().as_mut_ptr(), host[s].len());
                job_tx.send((l + 2, hs)).map_err(|_| eyre!("embed reader exited"))?;
            }
            after_layer(l)?;
        }
        drop(job_tx);
        Ok(())
    })?;

    // Last rows, in chunks that fit `gemm_out`.
    let n = inputs.len();
    let per = (bufs.gemm_out.len() / d).min(65_535);
    let mut out = Vec::with_capacity(n);
    let mut hostrows = vec![0f32; per.min(n) * d];
    let mut c0 = 0;
    while c0 < n {
        let cnt = per.min(n - c0);
        let idx = bufs.idx.slice_view(c0, cnt);
        k.gather_rows(compute, &mut bufs.gemm_out, &bufs.resid, &idx, d as u32, cnt as u32)?;
        compute.synchronize()?;
        bufs.gemm_out.slice_view(0, cnt * d).copy_to_host(&mut hostrows[..cnt * d])?;
        for i in 0..cnt {
            out.push(hostrows[i * d..(i + 1) * d].to_vec());
        }
        c0 += cnt;
    }
    tm.total_ms = t0.elapsed().as_secs_f64() * 1e3;
    Ok((out, tm))
}

/// One layer over all `t` rows, sub-batch by sub-batch in token order.
#[allow(clippy::too_many_arguments)]
fn layer(
    k: &Qwen3EmbedKernels,
    g: &Q8_0MatvecWmma,
    c: &Qwen3EmbedConfig,
    layout: &LayerLayout,
    w: &DeviceBuffer<u8>,
    bufs: &mut EmbedBuffers,
    starts: &[usize],
    inputs: &[&[u32]],
    t: usize,
    st: &Stream,
) -> eyre::Result<()> {
    let d = c.n_embd;
    let (qw, kvw, ff) = (c.q_width(), c.kv_width(), c.n_ff);
    let (ph, pa, pf) = (act_pitch(d), act_pitch(qw), act_pitch(ff));
    // SAFETY: the layout offsets are LAYOUT_ALIGN (256 B) aligned and in range.
    let f32v = |off: usize, n: usize| unsafe { w.view_as::<f32>(off, n) };
    let attn_norm = f32v(layout.attn_norm, d);
    let ffn_norm = f32v(layout.ffn_norm, d);
    let q_norm = f32v(layout.q_norm, c.head_dim);
    let k_norm = f32v(layout.k_norm, c.head_dim);
    let w_qkv = w.slice_view(layout.q, layout.q_bytes + 2 * layout.kv_bytes);
    let w_o = w.slice_view(layout.o, layout.o_bytes);
    let w_gu = w.slice_view(layout.gate, 2 * layout.ff_bytes);
    let w_down = w.slice_view(layout.down, layout.down_bytes);
    let scale = 1.0 / (c.head_dim as f32).sqrt();
    let sub = bufs.sizing.sub_rows;
    let packed = k.attention_packed(c);
    let mut seg = 0usize; // first input that may overlap the sub-batch
    let mut r0 = 0;
    while r0 < t {
        let r1 = (r0 + sub).min(t);
        let rows = r1 - r0;
        let mut resid = bufs.resid.slice_view_mut(r0 * d, rows * d);
        // Attention block.
        k.rmsnorm_f16(st, &mut bufs.x16, ph as u32, &resid, &attn_norm, d as u32, c.eps, rows as u32)?;
        gemm(g, st, &mut bufs.gemm_out, &w_qkv, &bufs.x16, d, c.qkv_rows(), rows, ph)?;
        let pos = bufs.pos.slice_view(r0, rows);
        k.qk_norm_rope(st, c, &bufs.gemm_out, &mut bufs.q16, &mut bufs.kc, &mut bufs.vc, &q_norm, &k_norm, &pos, r0 as u32, rows as u32)?;
        while seg < inputs.len() && starts[seg] + inputs[seg].len() <= r0 {
            seg += 1;
        }
        let mut s = seg;
        while s < inputs.len() && starts[s] < r1 {
            let (s0, len) = (starts[s], inputs[s].len());
            let (a, b) = (s0.max(r0), (s0 + len).min(r1));
            let q = bufs.q16.slice_view((a - r0) * qw, (b - a) * qw);
            let mut o = bufs.attn_out.slice_view_mut((a - r0) * qw, (b - a) * qw);
            let kc = bufs.kc.slice_view(s0 * kvw, len * kvw);
            let vc = bufs.vc.slice_view(s0 * kvw, len * kvw);
            let (batch, q_off) = ((b - a) as u32, (a - s0) as u32);
            let (nh, nkv, hd) = (c.n_head as u32, c.n_kv_head as u32, c.head_dim as u32);
            if packed {
                k.attn.prefill_flash_wmma_fa2_hg_packed(st, &mut o, &q, &kc, &vc, batch, nh, nkv, hd, q_off, scale, 0, len as u32)?;
            } else {
                k.attn.prefill_flash_wmma_fa2(st, &mut o, &q, &kc, &vc, batch, nh, nkv, hd, q_off, scale, 0, len as u32, true)?;
            }
            s += 1;
        }
        k.cast_f16(st, &mut bufs.x16, pa as u32, &bufs.attn_out, qw as u32, rows as u32)?;
        gemm(g, st, &mut bufs.gemm_out, &w_o, &bufs.x16, qw, d, rows, pa)?;
        k.add(st, &mut resid, &bufs.gemm_out, (rows * d) as u32)?;
        // FFN block.
        k.rmsnorm_f16(st, &mut bufs.x16, ph as u32, &resid, &ffn_norm, d as u32, c.eps, rows as u32)?;
        gemm(g, st, &mut bufs.gemm_out, &w_gu, &bufs.x16, d, 2 * ff, rows, ph)?;
        k.swiglu_f16(st, &mut bufs.x16, pf as u32, &bufs.gemm_out, ff as u32, rows as u32)?;
        gemm(g, st, &mut bufs.gemm_out, &w_down, &bufs.x16, ff, d, rows, pf)?;
        k.add(st, &mut resid, &bufs.gemm_out, (rows * d) as u32)?;
        r0 = r1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use v4flash_core::qwen3_embed::testing::tiny_config;

    /// `qe_q8_0_repack_rows`' word map, run on the host, against the
    /// production host repack (the GPU twin is `repack_matches_host`).
    #[test]
    fn repack_word_map_matches_host_repack() {
        for (rows, blocks) in [(3usize, 80usize), (2, 128), (2, 304), (1, REPACK_MAX_BLOCKS)] {
            let n = rows * blocks * 34;
            let src: Vec<u8> = (0..n).map(|i| (i as u32).wrapping_mul(2_654_435_761).rotate_left(11) as u8).collect();
            let want = crate::weights::repack_q8_0(&src, rows, blocks);
            let mut got = src.clone();
            for r in 0..rows {
                let row = &mut got[r * blocks * 34..(r + 1) * blocks * 34];
                let s: Vec<u16> = row.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                for i in 0..blocks * 17 {
                    let w = if i < blocks { s[i * 17] } else { let q = i - blocks; s[(q / 16) * 17 + 1 + q % 16] };
                    row[2 * i..2 * i + 2].copy_from_slice(&w.to_le_bytes());
                }
            }
            assert_eq!(got, want, "rows {rows} blocks {blocks}");
        }
        // The LDS stage holds the widest row: 480 blocks x 17 words x 2 B.
        assert!(REPACK_MAX_BLOCKS * 17 * 2 <= 64 * 1024);
    }

    #[test]
    fn pitches_avoid_powers_of_two() {
        assert_eq!(act_pitch(2560), 2560);
        assert_eq!(act_pitch(4096), 4104);
        assert_eq!(act_pitch(9728), 9728);
        assert_eq!(act_pitch(256), 264);
    }

    #[test]
    fn default_loan_is_about_575_mb() {
        let cfg = Qwen3EmbedConfig {
            n_layer: 36, n_embd: 2560, n_ff: 9728, n_head: 32, n_kv_head: 8, head_dim: 128,
            n_vocab: 151_665, rope_theta: 1e6, eps: 1e-6, n_ctx_train: 40960,
        };
        let layout = LayerLayout::new(&cfg);
        let s = EmbedSizing { phase_tokens: 16384, sub_rows: 1024 };
        let total = s.total_bytes(&cfg, &layout);
        assert!((560_000_000..600_000_000).contains(&total), "{total}");
        assert_eq!(s.buffer_sizes(&cfg, &layout).len(), BUFFERS.len());
    }

    #[test]
    fn tiny_shapes_meet_the_gemm_contract() {
        let c = tiny_config();
        for m in [c.qkv_rows(), c.n_embd, 2 * c.n_ff] {
            assert_eq!(m % 128, 0, "m {m}");
        }
        for kk in [c.n_embd, c.q_width(), c.n_ff] {
            assert_eq!(kk % 32, 0, "k {kk}");
        }
    }
}
