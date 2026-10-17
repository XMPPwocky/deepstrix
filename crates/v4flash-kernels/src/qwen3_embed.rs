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
//! segment of a sub-batch ([`Qwen3EmbedKernels::set_attention`]):
//! - [`AttnPath::Fa2`] (default): `prefill_flash_wmma_fa2` from the `HGP_G = 4`
//!   build of `gqa_attention.hip` (one WG per query head; gate E2);
//! - [`AttnPath::Packed`]: the full-group packed kernel (4 query heads per WG;
//!   design §18);
//! - [`AttnPath::G4`]: `qe_attn_g4` (`kernels/qwen3_attn_g4.hip`, design §19):
//!   keys on M, Q^T / P^T / O^T in VGPRs, 64 query tokens x 4 heads per WG, K/V in
//!   a fragment-native tile layout written by the producers ([`KvLayout`]), the
//!   output stored straight into `x16` as f16.
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
// The three cells of `qe_attn_g4` (one source, kernels/qwen3_attn_g4.hip): 16 waves
// in CU mode, 8 waves in CU mode, 16 waves in WGP mode. gfx1201 only (the gfx1151
// builds exist, every .hip is built for every target, and trap).
const QWEN3_ATTN_G4_GFX1201: &[u8] = include_bytes!(env!("KERNEL_QWEN3_ATTN_G4_GFX1201"));
const QWEN3_ATTN_G4N8_GFX1201: &[u8] = include_bytes!(env!("KERNEL_QWEN3_ATTN_G4N8_GFX1201"));
const QWEN3_ATTN_G4WGP_GFX1201: &[u8] = include_bytes!(env!("KERNEL_QWEN3_ATTN_G4WGP_GFX1201"));

/// Threads per block of the row kernels (mirrors `QE_BLOCK`).
const QE_BLOCK: u32 = 256;

/// Keys per tile of the G4 layout, and halfs per (kv head, tile) in each of kc / vc
/// (mirror `qwen3_embed.hip`'s `QE_G4_TILE_HALFS` and `qwen3_attn_g4.hip`'s `QA_TCH`).
pub const G4_TILE_KEYS: usize = 32;
pub const G4_TILE_HALFS: usize = 4096;
/// head_dim the G4 kernel and its producers are built for.
pub const G4_HEAD_DIM: usize = 128;

/// One compile-time cell of `qe_attn_g4` (design §19.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum G4Cell {
    /// `qe_attn_g4_n16`, `-mcumode`: 512 threads, 64 query tokens, 3 LDS stages.
    Cu16,
    /// `qe_attn_g4_n8`, `-mcumode`: 256 threads, 32 query tokens, 2 stages, 2 WGs/CU.
    Cu8,
    /// `qe_attn_g4_n16` built in WGP mode (2 WGs per WGP).
    Wgp16,
}

impl G4Cell {
    pub fn threads(self) -> u32 {
        match self {
            G4Cell::Cu16 | G4Cell::Wgp16 => 512,
            G4Cell::Cu8 => 256,
        }
    }
    /// Query tokens per WG (4 per wave).
    pub fn rows(self) -> u32 {
        self.threads() / 8
    }
    fn symbol(self) -> &'static str {
        match self {
            G4Cell::Cu16 | G4Cell::Wgp16 => "qe_attn_g4_n16",
            G4Cell::Cu8 => "qe_attn_g4_n8",
        }
    }
    fn image(self) -> &'static [u8] {
        match self {
            G4Cell::Cu16 => QWEN3_ATTN_G4_GFX1201,
            G4Cell::Cu8 => QWEN3_ATTN_G4N8_GFX1201,
            G4Cell::Wgp16 => QWEN3_ATTN_G4WGP_GFX1201,
        }
    }
}

/// The forward's attention kernel (`V41_EMBED_ATTN`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnPath {
    Fa2,
    Packed,
    G4(G4Cell),
}

impl AttnPath {
    /// The knob / log name.
    pub fn name(self) -> &'static str {
        match self {
            AttnPath::Fa2 => "fa2",
            AttnPath::Packed => "packed",
            AttnPath::G4(G4Cell::Cu16) => "g4",
            AttnPath::G4(G4Cell::Cu8) => "g4n8",
            AttnPath::G4(G4Cell::Wgp16) => "g4wgp",
        }
    }
    /// Inverse of [`Self::name`].
    pub fn from_name(s: &str) -> Option<Self> {
        [AttnPath::Fa2, AttnPath::Packed, AttnPath::G4(G4Cell::Cu16), AttnPath::G4(G4Cell::Cu8), AttnPath::G4(G4Cell::Wgp16)]
            .into_iter()
            .find(|p| p.name() == s)
    }
}

/// Where `qk_norm_rope` writes K (and whether it writes V).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvLayout {
    /// `kc` / `vc` row-major `[phase row][kv head][head_dim]` (fa2, packed).
    RowMajor,
    /// The G4 fragment-native tiles ([`g4_k_frag_index`]), `n_tiles` per kv head;
    /// V is written by [`Qwen3EmbedKernels::v_frag`].
    G4Tiles { n_tiles: u32 },
}

/// `attn_g4`'s output: f16 rows at `pitch` halfs (production: straight into `x16`,
/// `qe_f16`-rounded), or f32 `[batch][n_head][128]` (tests).
pub enum G4Out<'a> {
    F16 { buf: &'a mut DeviceBuffer<u16>, pitch: u32 },
    F32(&'a mut DeviceBuffer<f32>),
}

/// The kc index (in halfs) of `K[r][i]` (phase row `r`, dim `i`) of kv head `kh`
/// in the G4 layout; host twin of `qe_k_frag_index` (kernels/qwen3_embed.hip).
pub fn g4_k_frag_index(kh: usize, r: usize, i: usize, n_tiles: usize) -> usize {
    let chunk = ((i >> 4) * 2 + ((r >> 4) & 1)) * 32 + ((r & 15) | (((i >> 3) & 1) << 4));
    (kh * n_tiles + (r >> 5)) * G4_TILE_HALFS + chunk * 8 + (i & 7)
}

/// The vc index (in halfs) of `V[r][d]` of kv head `vh` in the G4 layout (chunk
/// `(kb*8 + dt)*32 + L`, half `e` = `V[32T + 16kb + 8(L>>4) + e][16dt + (L&15)]`);
/// host twin of `qe_v_frag`'s store map.
pub fn g4_v_frag_index(vh: usize, r: usize, d: usize, n_tiles: usize) -> usize {
    let (kb, hi, e) = ((r >> 4) & 1, (r >> 3) & 1, r & 7);
    let (dt, l) = (d >> 4, (d & 15) | (hi << 4));
    (vh * n_tiles + (r >> 5)) * G4_TILE_HALFS + ((kb * 8 + dt) * 32 + l) * 8 + e
}

/// Row-major K and V (`[rows][n_kv][128]` f16 bits) -> the G4 kc / vc layout with
/// `n_tiles` tiles per head; every slot no row maps to (rows past `rows`) holds
/// `fill`. The host twin the GPU tests pack with.
pub fn g4_pack_kv(k: &[u16], v: &[u16], rows: usize, n_kv: usize, n_tiles: usize, fill: u16) -> (Vec<u16>, Vec<u16>) {
    let hd = G4_HEAD_DIM;
    assert!(rows <= n_tiles * G4_TILE_KEYS, "g4_pack_kv: {rows} rows > {n_tiles} tiles");
    assert!(k.len() >= rows * n_kv * hd && v.len() >= rows * n_kv * hd, "g4_pack_kv: short input");
    let n = n_kv * n_tiles * G4_TILE_HALFS;
    let (mut kc, mut vc) = (vec![fill; n], vec![fill; n]);
    for r in 0..rows {
        for h in 0..n_kv {
            for i in 0..hd {
                let src = (r * n_kv + h) * hd + i;
                kc[g4_k_frag_index(h, r, i, n_tiles)] = k[src];
                vc[g4_v_frag_index(h, r, i, n_tiles)] = v[src];
            }
        }
    }
    (kc, vc)
}

/// Inverse of [`g4_pack_kv`] for rows `0..rows`: kc / vc tiles -> row-major K, V.
pub fn g4_unpack_kv(kc: &[u16], vc: &[u16], rows: usize, n_kv: usize, n_tiles: usize) -> (Vec<u16>, Vec<u16>) {
    let hd = G4_HEAD_DIM;
    let (mut k, mut v) = (vec![0u16; rows * n_kv * hd], vec![0u16; rows * n_kv * hd]);
    for r in 0..rows {
        for h in 0..n_kv {
            for i in 0..hd {
                let dst = (r * n_kv + h) * hd + i;
                k[dst] = kc[g4_k_frag_index(h, r, i, n_tiles)];
                v[dst] = vc[g4_v_frag_index(h, r, i, n_tiles)];
            }
        }
    }
    (k, v)
}

/// Host twin of `qe_v_frag` over one sub-batch (phase rows `row0 .. row0 + rows`):
/// `vrow(r, vh, d)` = the f16 bits of `V[row0 + r][vh][d]`. Every tile the
/// sub-batch touches is rewritten: keys below `row0` kept (the straddled first
/// tile's read-modify-write), keys at or past `row0 + rows` set to 0.
pub fn g4_v_frag_host(vc: &mut [u16], vrow: &dyn Fn(usize, usize, usize) -> u16, n_kv: usize, n_tiles: usize, row0: usize, rows: usize) {
    assert!(rows >= 1 && row0 + rows <= n_tiles * G4_TILE_KEYS);
    for tile in row0 / G4_TILE_KEYS..=(row0 + rows - 1) / G4_TILE_KEYS {
        for vh in 0..n_kv {
            for key in 0..G4_TILE_KEYS {
                let r = tile * G4_TILE_KEYS + key;
                if r < row0 {
                    continue;
                }
                for d in 0..G4_HEAD_DIM {
                    vc[g4_v_frag_index(vh, r, d, n_tiles)] = if r < row0 + rows { vrow(r - row0, vh, d) } else { 0 };
                }
            }
        }
    }
}

/// log2(kv_group / 4) when `n_head / n_kv_head` is 4 * 2^s (the G4 kernel's query
/// heads per WG are 4; consecutive WGs share a kv head at kv_group 8, 16, ...).
fn g4_kv_shift(n_head: usize, n_kv_head: usize) -> Option<u32> {
    if n_kv_head == 0 || n_head % (4 * n_kv_head) != 0 {
        return None;
    }
    let sub = n_head / (4 * n_kv_head);
    sub.is_power_of_two().then(|| sub.trailing_zeros())
}

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
    /// The attention kernel the forward runs ([`Self::set_attention`]).
    path: AttnPath,
    /// The loaded `qe_attn_g4` cell's module (only for [`AttnPath::G4`]).
    g4: Option<(G4Cell, Module)>,
}

impl Qwen3EmbedKernels {
    /// The dGPU (gfx1201) only: the WMMA GEMM and attention have no fast path
    /// elsewhere. Attention starts on `fa2`; see [`Self::set_attention`].
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        if !arch.starts_with("gfx1201") {
            return Err(eyre!("qwen3 embed forward needs gfx1201 (the dGPU), got {arch}"));
        }
        let _ = QWEN3_EMBED_GFX1151; // built for every target; never loaded there
        Ok(Qwen3EmbedKernels {
            module: Module::load_data(QWEN3_EMBED_GFX1201)?,
            attn: GqaAttention::for_arch_group4(arch)?,
            path: AttnPath::Fa2,
            g4: None,
        })
    }

    /// The forward's attention kernel. [`AttnPath::G4`] loads that cell's module
    /// (`KERNEL_QWEN3_ATTN_G4[N8|WGP]_GFX1201`) and switches the producers to the G4
    /// K/V layout for the whole phase.
    pub fn set_attention(&mut self, p: AttnPath) -> eyre::Result<()> {
        if let AttnPath::G4(cell) = p {
            if self.g4.as_ref().map(|(c, _)| *c) != Some(cell) {
                self.g4 = Some((cell, Module::load_data(cell.image())?));
            }
        }
        self.path = p;
        Ok(())
    }

    /// The path `cfg`'s forward takes: `packed` falls back to `fa2` when kv_group is
    /// not a multiple of 4 (as before); `G4` is an ERROR unless kv_group is 4 * 2^s
    /// and head_dim 128 (the kernel and its layout have no other shape).
    pub fn attention_path(&self, cfg: &Qwen3EmbedConfig) -> eyre::Result<AttnPath> {
        Ok(match self.path {
            AttnPath::Packed if (cfg.n_head / cfg.n_kv_head) as u32 % self.attn.packed_group() != 0 => AttnPath::Fa2,
            AttnPath::G4(cell) => {
                if g4_kv_shift(cfg.n_head, cfg.n_kv_head).is_none() || cfg.head_dim != G4_HEAD_DIM {
                    return Err(eyre!(
                        "qe_attn_g4 ({}) needs kv_group 4 * 2^s and head_dim {G4_HEAD_DIM}; model has n_head {} / n_kv_head {} / head_dim {}",
                        self.path.name(), cfg.n_head, cfg.n_kv_head, cfg.head_dim
                    ));
                }
                AttnPath::G4(cell)
            }
            p => p,
        })
    }

    /// `qe_attn_g4`: causal attention of `batch` query rows (phase rows `q_abs0 ..`,
    /// `q` = their q16 rows) over the keys `seg0 ..= query row` of the G4-layout
    /// `kc` / `vc` (whole buffers, `n_tiles` tiles per kv head). `c` = RN(scale *
    /// log2(e)). Every precondition is checked; a violation is an error naming it.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_g4(
        &self,
        st: &Stream,
        out: G4Out<'_>,
        q: &DeviceBuffer<u16>,
        kc: &DeviceBuffer<u16>,
        vc: &DeviceBuffer<u16>,
        n_head: u32,
        n_kv_head: u32,
        n_tiles: u32,
        batch: u32,
        q_abs0: u32,
        seg0: u32,
        c: f32,
    ) -> eyre::Result<()> {
        let (cell, module) = self.g4.as_ref().ok_or_else(|| eyre!("qe_attn_g4: no cell loaded (set_attention(AttnPath::G4(..)) first)"))?;
        let hd = G4_HEAD_DIM;
        let (nh, nkv, nt, b) = (n_head as usize, n_kv_head as usize, n_tiles as usize, batch as usize);
        let kv_shift = g4_kv_shift(nh, nkv).ok_or_else(|| eyre!("qe_attn_g4: n_head {n_head} is not 4 * 2^s * n_kv_head {n_kv_head}"))?;
        if batch == 0 {
            return Err(eyre!("qe_attn_g4: batch 0"));
        }
        if seg0 > q_abs0 {
            return Err(eyre!("qe_attn_g4: seg0 {seg0} > q_abs0 {q_abs0} (the segment starts after its first query)"));
        }
        if q_abs0 as usize + b > nt * G4_TILE_KEYS {
            return Err(eyre!("qe_attn_g4: query rows {q_abs0}..{} past the layout's {n_tiles} tiles", q_abs0 as usize + b));
        }
        if nt * G4_TILE_HALFS * 2 > u32::MAX as usize {
            return Err(eyre!("qe_attn_g4: {n_tiles} tiles per head overflow the kernel's 32-bit byte offsets"));
        }
        let kv_len = nkv * nt * G4_TILE_HALFS;
        if kc.len() < kv_len || vc.len() < kv_len {
            return Err(eyre!("qe_attn_g4: kc {} / vc {} halfs < n_kv_head * n_tiles * {G4_TILE_HALFS} = {kv_len}", kc.len(), vc.len()));
        }
        if q.len() < b * nh * hd {
            return Err(eyre!("qe_attn_g4: q {} halfs < batch * n_head * {hd}", q.len()));
        }
        if !(c.is_finite() && c > 0.0) {
            return Err(eyre!("qe_attn_g4: c = scale * log2(e) must be finite and > 0, got {c}"));
        }
        let rows = cell.rows();
        let grid_y = batch.div_ceil(rows);
        if grid_y > 65_535 {
            return Err(eyre!("qe_attn_g4: batch {batch} needs {grid_y} > 65535 q-tiles"));
        }
        let (out16, pitch, out32) = match out {
            G4Out::F16 { buf, pitch } => {
                if (pitch as usize) < nh * hd || pitch % 8 != 0 {
                    return Err(eyre!("qe_attn_g4: f16 out pitch {pitch} must be >= n_head * {hd} and a multiple of 8"));
                }
                if buf.len() < (b - 1) * pitch as usize + nh * hd {
                    return Err(eyre!("qe_attn_g4: f16 out {} halfs < (batch - 1) * pitch + n_head * {hd}", buf.len()));
                }
                (buf.raw(), pitch, std::ptr::null_mut())
            }
            G4Out::F32(buf) => {
                if buf.len() < b * nh * hd {
                    return Err(eyre!("qe_attn_g4: f32 out {} < batch * n_head * {hd}", buf.len()));
                }
                (std::ptr::null_mut(), 0, buf.raw())
            }
        };
        for (name, p) in [("out16", out16), ("out32", out32), ("q", q.raw()), ("kc", kc.raw()), ("vc", vc.raw())] {
            if (p as usize) % 16 != 0 {
                return Err(eyre!("qe_attn_g4: {name} is not 16-byte aligned"));
            }
        }
        let f = module.get_function(cell.symbol())?;
        let lc = LaunchConfig { grid: (n_head / 4, grid_y, 1), block: (cell.threads(), 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, lc, st, [out16, pitch, out32, q.raw(), kc.raw(), vc.raw(), n_head, kv_shift, n_tiles, batch, q_abs0, seg0, c])
    }

    /// `qe_v_frag`: the V columns of the fused qkv GEMM output for phase rows
    /// `row0 .. row0 + rows` -> the G4 vc tiles (`n_tiles` per kv head). The
    /// straddled first tile is read-modify-written (sub-batches need not be
    /// 32-aligned); keys past the sub-batch in its last tile are written 0.
    #[allow(clippy::too_many_arguments)]
    pub fn v_frag(
        &self,
        st: &Stream,
        cfg: &Qwen3EmbedConfig,
        qkv: &DeviceBuffer<f32>,
        vc: &mut DeviceBuffer<u16>,
        n_tiles: u32,
        row0: u32,
        rows: u32,
    ) -> eyre::Result<()> {
        if cfg.head_dim != G4_HEAD_DIM {
            return Err(eyre!("qe_v_frag: head_dim {} != {G4_HEAD_DIM}", cfg.head_dim));
        }
        if rows == 0 {
            return Err(eyre!("qe_v_frag: rows 0"));
        }
        if row0 as usize + rows as usize > n_tiles as usize * G4_TILE_KEYS {
            return Err(eyre!("qe_v_frag: rows {row0}..{} past {n_tiles} tiles", row0 + rows));
        }
        if vc.len() < cfg.n_kv_head * n_tiles as usize * G4_TILE_HALFS || qkv.len() < rows as usize * cfg.qkv_rows() {
            return Err(eyre!("qe_v_frag: buffers too small (vc {}, qkv {}, rows {rows})", vc.len(), qkv.len()));
        }
        if (vc.raw() as usize) % 16 != 0 || (qkv.raw() as usize) % 16 != 0 || cfg.qkv_rows() % 4 != 0 {
            return Err(eyre!("qe_v_frag: vc / qkv not 16-byte aligned (or the qkv row pitch is not a multiple of 4)"));
        }
        let tiles = (row0 + rows - 1) / G4_TILE_KEYS as u32 - row0 / G4_TILE_KEYS as u32 + 1;
        let f = self.module.get_function("qe_v_frag")?;
        let lc = LaunchConfig { grid: (tiles, cfg.n_kv_head as u32, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, lc, st, [vc.raw(), qkv.raw(), cfg.n_head as u32, cfg.n_kv_head as u32, n_tiles, row0, rows])
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
    /// q -> `q16[r]`, k / v -> `kc` / `vc` at phase row `row0 + r`. `kv`:
    /// [`KvLayout::RowMajor`] writes K and V row-major; [`KvLayout::G4Tiles`] writes
    /// K into the G4 tiles and does NOT write V (that is [`Self::v_frag`]).
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
        kv: KvLayout,
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
        if qkv.len() < r * cfg.qkv_rows() || q16.len() < r * cfg.q_width() || pos.len() < r {
            return Err(eyre!("qe_qk_norm_rope: buffers too small (row0 {row0}, rows {rows})"));
        }
        let (k_tiles, heads) = match kv {
            KvLayout::RowMajor => {
                let kv_end = (row0 as usize + r) * cfg.kv_width();
                if kc.len() < kv_end || vc.len() < kv_end {
                    return Err(eyre!("qe_qk_norm_rope: kc / vc too small for rows {row0}..{}", row0 as usize + r));
                }
                (0u32, (cfg.n_head + 2 * cfg.n_kv_head) as u32)
            }
            KvLayout::G4Tiles { n_tiles } => {
                if cfg.head_dim != G4_HEAD_DIM || n_tiles == 0 {
                    return Err(eyre!("qe_qk_norm_rope: the G4 layout needs head_dim {G4_HEAD_DIM} and n_tiles > 0"));
                }
                if row0 as usize + r > n_tiles as usize * G4_TILE_KEYS || kc.len() < cfg.n_kv_head * n_tiles as usize * G4_TILE_HALFS {
                    return Err(eyre!("qe_qk_norm_rope: rows {row0}..{} do not fit the G4 kc ({n_tiles} tiles)", row0 as usize + r));
                }
                // V heads are not launched: qe_v_frag writes vc.
                (n_tiles, (cfg.n_head + cfg.n_kv_head) as u32)
            }
        };
        let f = self.module.get_function("qe_qk_norm_rope")?;
        let lc = LaunchConfig { grid: (rows, heads, 1), block: (hd, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, lc, stream, [
            qkv.raw(), q16.raw(), kc.raw(), vc.raw(), q_norm.raw(), k_norm.raw(), pos.raw(),
            row0, cfg.n_head as u32, cfg.n_kv_head as u32, hd, cfg.rope_theta, cfg.eps, k_tiles
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
        // kc / vc: whole 32-key tiles in every mode (the G4 layout's unit; identical
        // bytes at T = 16384), so the knob can pick any attention path per phase.
        let t32 = t.div_ceil(G4_TILE_KEYS) * G4_TILE_KEYS;
        let sizes = [
            t * cfg.n_embd * 4,
            layout.bytes,
            layout.bytes,
            r * Self::gemm_out_width(cfg) * 4,
            t32 * cfg.kv_width() * 2,
            t32 * cfg.kv_width() * 2,
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

/// Gates only (design §19.11 G3, the first-affected-layer rule): with
/// [`EmbedBuffers::capture`] set, the forward copies layer 0's attention inputs and
/// output to the host -- q16 and the f16 attention output (`x16` after attention,
/// `[t][q_width]`) per sub-batch, and kc / vc whole (in the phase's layout) at the
/// end of layer 0. It synchronizes the stream per sub-batch: never in production.
#[derive(Debug, Default)]
pub struct Layer0Capture {
    pub q16: Vec<u16>,
    pub attn: Vec<u16>,
    pub kc: Vec<u16>,
    pub vc: Vec<u16>,
    /// The attention path layer 0 ran.
    pub path: Option<AttnPath>,
}

/// One phase's device buffers (views into the loan).
pub struct EmbedBuffers {
    /// See [`Layer0Capture`]; `None` (production) costs nothing.
    pub capture: Option<Layer0Capture>,
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
            capture: None,
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
            layer(k, g, c, layout, &w, bufs, &starts, inputs, t, compute, l)?;
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
    l: usize,
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
    // G4: the exp2 fold's constant, RN(scale * log2 e) in f32 (design §19.7).
    let c_log2 = scale * std::f32::consts::LOG2_E;
    let sub = bufs.sizing.sub_rows;
    let path = k.attention_path(c)?;
    let is_g4 = matches!(path, AttnPath::G4(_));
    // The G4 layout's tiles per kv head: from the SIZING (the buffers' shape), not t.
    let n_tiles = bufs.sizing.phase_tokens.div_ceil(G4_TILE_KEYS) as u32;
    let kv = if is_g4 { KvLayout::G4Tiles { n_tiles } } else { KvLayout::RowMajor };
    let (nh, nkv, hd) = (c.n_head as u32, c.n_kv_head as u32, c.head_dim as u32);
    let capture = l == 0 && bufs.capture.is_some();
    if capture {
        let cap = bufs.capture.as_mut().expect("capture");
        *cap = Layer0Capture { q16: vec![0; t * qw], attn: vec![0; t * qw], kc: Vec::new(), vc: Vec::new(), path: Some(path) };
    }
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
        k.qk_norm_rope(st, c, &bufs.gemm_out, &mut bufs.q16, &mut bufs.kc, &mut bufs.vc, &q_norm, &k_norm, &pos, kv, r0 as u32, rows as u32)?;
        if is_g4 {
            k.v_frag(st, c, &bufs.gemm_out, &mut bufs.vc, n_tiles, r0 as u32, rows as u32)?;
        }
        while seg < inputs.len() && starts[seg] + inputs[seg].len() <= r0 {
            seg += 1;
        }
        let mut s = seg;
        while s < inputs.len() && starts[s] < r1 {
            let (s0, len) = (starts[s], inputs[s].len());
            let (a, b) = (s0.max(r0), (s0 + len).min(r1));
            let q = bufs.q16.slice_view((a - r0) * qw, (b - a) * qw);
            let (batch, q_off) = ((b - a) as u32, (a - s0) as u32);
            match path {
                AttnPath::G4(_) => {
                    // Straight into x16 (free: the QKV GEMM consumed it), qe_f16-rounded,
                    // at the O-projection's pitch; whole kc / vc (absolute phase rows).
                    let mut o = bufs.x16.slice_view_mut((a - r0) * pa, (b - a - 1) * pa + qw);
                    let out = G4Out::F16 { buf: &mut o, pitch: pa as u32 };
                    k.attn_g4(st, out, &q, &bufs.kc, &bufs.vc, nh, nkv, n_tiles, batch, a as u32, s0 as u32, c_log2)?;
                }
                AttnPath::Packed | AttnPath::Fa2 => {
                    let mut o = bufs.attn_out.slice_view_mut((a - r0) * qw, (b - a) * qw);
                    let kc = bufs.kc.slice_view(s0 * kvw, len * kvw);
                    let vc = bufs.vc.slice_view(s0 * kvw, len * kvw);
                    if path == AttnPath::Packed {
                        k.attn.prefill_flash_wmma_fa2_hg_packed(st, &mut o, &q, &kc, &vc, batch, nh, nkv, hd, q_off, scale, 0, len as u32)?;
                    } else {
                        k.attn.prefill_flash_wmma_fa2(st, &mut o, &q, &kc, &vc, batch, nh, nkv, hd, q_off, scale, 0, len as u32, true)?;
                    }
                }
            }
            s += 1;
        }
        if !is_g4 {
            k.cast_f16(st, &mut bufs.x16, pa as u32, &bufs.attn_out, qw as u32, rows as u32)?;
        }
        if capture {
            st.synchronize()?;
            let cap = bufs.capture.as_mut().expect("capture");
            bufs.q16.slice_view(0, rows * qw).copy_to_host(&mut cap.q16[r0 * qw..r1 * qw])?;
            let mut x = vec![0u16; (rows - 1) * pa + qw];
            bufs.x16.slice_view(0, x.len()).copy_to_host(&mut x)?;
            for r in 0..rows {
                cap.attn[(r0 + r) * qw..(r0 + r + 1) * qw].copy_from_slice(&x[r * pa..r * pa + qw]);
            }
        }
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
    if capture {
        st.synchronize()?;
        let cap = bufs.capture.as_mut().expect("capture");
        cap.kc = vec![0; bufs.kc.len()];
        cap.vc = vec![0; bufs.vc.len()];
        bufs.kc.copy_to_host(&mut cap.kc)?;
        bufs.vc.copy_to_host(&mut cap.vc)?;
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

    fn qwen4b() -> Qwen3EmbedConfig {
        Qwen3EmbedConfig {
            n_layer: 36, n_embd: 2560, n_ff: 9728, n_head: 32, n_kv_head: 8, head_dim: 128,
            n_vocab: 151_665, rope_theta: 1e6, eps: 1e-6, n_ctx_train: 40960,
        }
    }

    /// H5: kc / vc are whole 32-key tiles: unchanged at T = 16384, T = 100 -> 128 rows.
    #[test]
    fn g4_sizing_rounds_kv_to_tiles() {
        let cfg = qwen4b();
        let layout = LayerLayout::new(&cfg);
        let kv = |t: usize| {
            let s = EmbedSizing { phase_tokens: t, sub_rows: 64 }.buffer_sizes(&cfg, &layout);
            assert_eq!((s[4].0, s[5].0), ("kc", "vc"));
            assert_eq!(s[4].1, s[5].1);
            s[4].1
        };
        assert_eq!(kv(16384), 16384 * cfg.kv_width() * 2);
        assert_eq!(kv(100), 128 * cfg.kv_width() * 2);
        assert_eq!(kv(128), 128 * cfg.kv_width() * 2);
        assert_eq!(kv(129), 160 * cfg.kv_width() * 2);
        // n_tiles from the sizing covers every row the G4 index can produce.
        let n_tiles = 100usize.div_ceil(G4_TILE_KEYS);
        assert!(g4_k_frag_index(cfg.n_kv_head - 1, 99, 127, n_tiles) < kv(100) / 2);
        assert!(g4_v_frag_index(cfg.n_kv_head - 1, 99, 127, n_tiles) < kv(100) / 2);
    }

    /// H1: the K and V tile maps are bijections of (head, row, dim) onto the buffer.
    #[test]
    fn g4_layout_is_a_bijection() {
        let (n_kv, n_tiles) = (2usize, 3usize);
        let n = n_kv * n_tiles * G4_TILE_HALFS;
        for (name, f) in [("k", g4_k_frag_index as fn(usize, usize, usize, usize) -> usize), ("v", g4_v_frag_index)] {
            let mut seen = vec![false; n];
            for h in 0..n_kv {
                for r in 0..n_tiles * G4_TILE_KEYS {
                    for d in 0..G4_HEAD_DIM {
                        let i = f(h, r, d, n_tiles);
                        assert!(i < n, "{name}: ({h}, {r}, {d}) -> {i} out of range");
                        assert!(!seen[i], "{name}: ({h}, {r}, {d}) -> {i} twice");
                        seen[i] = true;
                    }
                }
            }
            assert!(seen.iter().all(|&s| s), "{name}: not onto");
        }
        // Each 16-B chunk is one fragment row: 8 consecutive halfs = 8 dims of one key (K)
        // / 8 keys of one dim (V), as the lane maps need.
        for r in 0..64 {
            for d in (0..128).step_by(8) {
                let k0 = g4_k_frag_index(1, r, d, n_tiles);
                assert_eq!(k0 % 8, 0);
                for e in 0..8 {
                    assert_eq!(g4_k_frag_index(1, r, d + e, n_tiles), k0 + e);
                }
            }
        }
        for r in (0..64).step_by(8) {
            for d in 0..128 {
                let v0 = g4_v_frag_index(1, r, d, n_tiles);
                for e in 0..8 {
                    assert_eq!(g4_v_frag_index(1, r + e, d, n_tiles), v0 + e);
                }
            }
        }
    }

    fn hash(i: u64) -> u64 {
        let mut x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5DEE_CE66_D;
        x ^= x >> 29;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^ (x >> 32)
    }

    /// The gfx12 wave32 WMMA lane maps (verified in-repo): A elem e of lane L =
    /// (m = L&15, k = 8(L>>4)+e); B = (k = 8(L>>4)+e, n = L&15); C = (m = 8(L>>4)+e,
    /// n = L&15). One 16x16x16 product on per-lane fragments, exactly (f64).
    fn wmma(a: &[[f64; 8]; 32], b: &[[f64; 8]; 32], c: &mut [[f64; 8]; 32]) {
        let mut am = [[0f64; 16]; 16];
        let mut bm = [[0f64; 16]; 16];
        for l in 0..32 {
            for e in 0..8 {
                am[l & 15][8 * (l >> 4) + e] = a[l][e];
                bm[8 * (l >> 4) + e][l & 15] = b[l][e];
            }
        }
        for l in 0..32 {
            for e in 0..8 {
                let (m, n) = (8 * (l >> 4) + e, l & 15);
                c[l][e] += (0..16).map(|k| am[m][k] * bm[k][n]).sum::<f64>();
            }
        }
    }

    /// H2: the kc / vc chunk maps, read as WMMA operands through the lane maps,
    /// reproduce S^T = K Q^T and O^T = V^T P^T with P^T read IN PLACE from the C
    /// layout of S^T -- exactly (values k/8, so f64 sums are exact). Negative control:
    /// flipping the hi bit of either operand map fails.
    #[test]
    fn g4_lane_maps_reproduce_tile_products() {
        let val = |i: u64| (hash(i) % 33) as f64 / 8.0 - 2.0; // k/8 in [-2, 2]: exact in f16 and f64
        let n_tiles = 1;
        let kr: Vec<Vec<f64>> = (0..32).map(|key| (0..128).map(|d| val(key * 128 + d)).collect()).collect();
        let vr: Vec<Vec<f64>> = (0..32).map(|key| (0..128).map(|d| val(10_000 + key * 128 + d)).collect()).collect();
        let qr: Vec<Vec<f64>> = (0..16).map(|n| (0..128).map(|d| val(20_000 + n * 128 + d)).collect()).collect();
        let bits = |x: f64| v4flash_core::kquants::f32_to_f16_bits(x as f32);
        let kflat: Vec<u16> = kr.iter().flatten().map(|&x| bits(x)).collect();
        let vflat: Vec<u16> = vr.iter().flatten().map(|&x| bits(x)).collect();
        let (kc, vc) = g4_pack_kv(&kflat, &vflat, 32, 1, n_tiles, 0);
        let h16 = |b: u16| f64::from(v4flash_core::kquants::f16_to_f32(b));
        for flip in [None, Some("k"), Some("v")] {
            // QK: s_kb += A(kc chunk (ks*2+kb)*32 + L) . B(Q^T: lane L holds q[n][16ks + 8hi + e]).
            let mut s = [[[0f64; 8]; 32]; 2];
            for kb in 0..2 {
                for ks in 0..8 {
                    let mut a = [[0f64; 8]; 32];
                    let mut b = [[0f64; 8]; 32];
                    for l in 0..32 {
                        let la = if flip == Some("k") { l ^ 16 } else { l };
                        for e in 0..8 {
                            a[l][e] = h16(kc[((ks * 2 + kb) * 32 + la) * 8 + e]);
                            b[l][e] = qr[l & 15][16 * ks + 8 * (l >> 4) + e];
                        }
                    }
                    wmma(&a, &b, &mut s[kb]);
                }
            }
            let mut qk_ok = true;
            for kb in 0..2 {
                for l in 0..32 {
                    for e in 0..8 {
                        let (key, n) = (16 * kb + 8 * (l >> 4) + e, l & 15);
                        let want: f64 = (0..128).map(|d| kr[key][d] * qr[n][d]).sum();
                        qk_ok &= s[kb][l][e] == want;
                    }
                }
            }
            // PV: P = the S^T C layout itself (scaled down to stay exact), O^T += A(vc chunk
            // (kb*8 + dt)*32 + L) . B(P^T in place).
            let mut o = [[[0f64; 8]; 32]; 8];
            for kb in 0..2 {
                let mut b = [[0f64; 8]; 32];
                for l in 0..32 {
                    for e in 0..8 {
                        b[l][e] = s[kb][l][e] / 64.0;
                    }
                }
                for (dt, od) in o.iter_mut().enumerate() {
                    let mut a = [[0f64; 8]; 32];
                    for l in 0..32 {
                        let la = if flip == Some("v") { l ^ 16 } else { l };
                        for e in 0..8 {
                            a[l][e] = h16(vc[((kb * 8 + dt) * 32 + la) * 8 + e]);
                        }
                    }
                    wmma(&a, &b, od);
                }
            }
            let p = |key: usize, n: usize| s[key >> 4][n | (((key >> 3) & 1) << 4)][key & 7] / 64.0;
            let mut pv_ok = true;
            for (dt, od) in o.iter().enumerate() {
                for l in 0..32 {
                    for e in 0..8 {
                        let (d, n) = (16 * dt + 8 * (l >> 4) + e, l & 15);
                        let want: f64 = (0..32).map(|key| vr[key][d] * p(key, n)).sum();
                        pv_ok &= od[l][e] == want;
                    }
                }
            }
            match flip {
                None => assert!(qk_ok && pv_ok, "lane maps: qk {qk_ok} pv {pv_ok}"),
                Some("k") => assert!(!qk_ok, "negative control: a flipped K map still matched"),
                _ => assert!(qk_ok && !pv_ok, "negative control: a flipped V map still matched"),
            }
        }
    }

    /// H4: `qe_v_frag`'s host twin over sub-batches of 8 / 48 / 1024 rows (unaligned
    /// starts: the straddled-tile read-modify-write) equals one pass over the phase and
    /// the plain pack (pad rows 0).
    #[test]
    fn g4_v_frag_rmw_matches_one_shot() {
        let (t, n_kv) = (1000usize, 2usize);
        let n_tiles = t.div_ceil(G4_TILE_KEYS);
        let v: Vec<u16> = (0..t * n_kv * 128).map(|i| (hash(i as u64) & 0x7bff) as u16).collect();
        let n = n_kv * n_tiles * G4_TILE_HALFS;
        let run = |sub: usize| {
            let mut vc = vec![0xAAAAu16; n]; // garbage: everything must be (re)written
            let mut r0 = 0;
            while r0 < t {
                let rows = sub.min(t - r0);
                let vrow = |r: usize, vh: usize, d: usize| v[((r0 + r) * n_kv + vh) * 128 + d];
                g4_v_frag_host(&mut vc, &vrow, n_kv, n_tiles, r0, rows);
                r0 += rows;
            }
            vc
        };
        let one = run(t);
        let (_, packed) = g4_pack_kv(&v, &v, t, n_kv, n_tiles, 0);
        assert_eq!(one, packed, "one-shot vs pack");
        for sub in [8usize, 48, 1024] {
            assert_eq!(run(sub), one, "sub-batches of {sub}");
        }
    }

    /// The f16 value of bits (NaN stays NaN).
    fn h(b: u16) -> f32 {
        v4flash_core::kquants::f16_to_f32(b)
    }

    struct ModelOpts {
        zero_fill: bool,
        mask: bool,
    }

    /// H3: a host emulation of `qe_attn_g4`'s per-WG / wave / lane loop: tile range,
    /// the wave's last tile, the edge zero-fill, the unsigned mask, the exact skip with
    /// the corr select, the exp2 fold (`mul_add` + `exp2`), RNE f16 P, two l partials,
    /// the L^16 exchanges. `q` f32 `[batch][nh][128]`; returns f32 `[batch][nh][128]`.
    #[allow(clippy::too_many_arguments)]
    fn g4_model(q: &[f32], kc: &[u16], vc: &[u16], nh: usize, nkv: usize, n_tiles: usize, batch: usize, q_abs0: usize, seg0: usize, c: f32, br: usize, opt: &ModelOpts) -> Vec<f32> {
        let shift = g4_kv_shift(nh, nkv).expect("kv_group 4 * 2^s");
        let mut out = vec![f32::NAN; batch * nh * 128];
        for hq in 0..nh / 4 {
            let kvh = hq >> shift;
            for qb in 0..batch.div_ceil(br) {
                let wg_qmax = q_abs0 + (qb * br + br).min(batch) - 1;
                let (t0, t1) = (seg0 / 32, wg_qmax / 32);
                for w in 0..br / 4 {
                    let rw0 = qb * br + w * 4;
                    if rw0 >= batch {
                        continue; // inactive wave: stages tiles, computes nothing
                    }
                    let wv_qmin = q_abs0 + rw0;
                    let wv_t1 = (q_abs0 + (rw0 + 3).min(batch - 1)) / 32;
                    let lane_r = |l: usize| (rw0 + ((l & 15) >> 2)).min(batch - 1);
                    let lane_h = |l: usize| hq * 4 + (l & 3);
                    let mut o = vec![[0f32; 64]; 32]; // lane L: d = 16dt + 8hi + e at [dt * 8 + e]
                    let (mut m, mut la, mut lb) = ([f32::NEG_INFINITY; 32], [0f32; 32], [0f32; 32]);
                    for t in t0..=t1.min(wv_t1) {
                        let inside = |key: usize| key >= seg0 && key <= wg_qmax;
                        let kval = |key: usize, d: usize| {
                            if opt.zero_fill && !inside(key) { 0.0 } else { h(kc[g4_k_frag_index(kvh, key, d, n_tiles)]) }
                        };
                        let vval = |key: usize, d: usize| {
                            if opt.zero_fill && !inside(key) { 0.0 } else { h(vc[g4_v_frag_index(kvh, key, d, n_tiles)]) }
                        };
                        // S^T: lane L elem e of half kb = key 32t + 16kb + 8hi + e, column L&15.
                        let mut s = [[[0f32; 8]; 2]; 32];
                        for (l, sl) in s.iter_mut().enumerate() {
                            let qrow = &q[(lane_r(l) * nh + lane_h(l)) * 128..][..128];
                            for kb in 0..2 {
                                for e in 0..8 {
                                    let key = 32 * t + 16 * kb + 8 * (l >> 4) + e;
                                    sl[kb][e] = (0..128).map(|d| f64::from(kval(key, d)) * f64::from(qrow[d])).sum::<f64>() as f32;
                                }
                            }
                        }
                        let kbase = 32 * t;
                        if opt.mask && (kbase < seg0 || kbase + 31 > wv_qmin) {
                            for (l, sl) in s.iter_mut().enumerate() {
                                let span = (q_abs0 + lane_r(l)).wrapping_sub(seg0) as u32;
                                for kb in 0..2 {
                                    for e in 0..8 {
                                        let key = kbase + 16 * kb + 8 * (l >> 4) + e;
                                        if (key as u32).wrapping_sub(seg0 as u32) > span {
                                            sl[kb][e] = f32::NEG_INFINITY;
                                        }
                                    }
                                }
                            }
                        }
                        let tl: Vec<f32> = s.iter().map(|sl| sl.iter().flatten().copied().fold(f32::NEG_INFINITY, f32::max)).collect();
                        let tmc: Vec<f32> = (0..32).map(|l| tl[l].max(tl[l ^ 16]) * c).collect();
                        if (0..32).any(|l| tmc[l] > m[l]) {
                            for l in 0..32 {
                                let mn = m[l].max(tmc[l]);
                                let corr = if mn == m[l] { 1.0 } else { (m[l] - mn).exp2() };
                                o[l].iter_mut().for_each(|x| *x *= corr);
                                la[l] *= corr;
                                lb[l] *= corr;
                                m[l] = mn;
                            }
                        }
                        // P: f32 p (l from it), f16 RNE P16 for PV.
                        let mut p16 = [[[0f32; 8]; 2]; 32];
                        for l in 0..32 {
                            for kb in 0..2 {
                                for e in 0..8 {
                                    let p = s[l][kb][e].mul_add(c, -m[l]).exp2();
                                    if kb == 0 { la[l] += p } else { lb[l] += p }
                                    p16[l][kb][e] = h(v4flash_core::kquants::f32_to_f16_bits(p));
                                }
                            }
                        }
                        // PV: O^T[d][n] += sum over the kb half's 16 keys, one WMMA per (kb, dt).
                        for kb in 0..2 {
                            for l in 0..32 {
                                for dt in 0..8 {
                                    for e in 0..8 {
                                        let d = 16 * dt + 8 * (l >> 4) + e;
                                        let n = l & 15;
                                        let acc: f64 = (0..16)
                                            .map(|k| {
                                                let key = 32 * t + 16 * kb + k;
                                                let pl = n | ((k >> 3) << 4);
                                                f64::from(vval(key, d)) * f64::from(p16[pl][kb][k & 7])
                                            })
                                            .sum();
                                        o[l][dt * 8 + e] = (f64::from(o[l][dt * 8 + e]) + acc) as f32;
                                    }
                                }
                            }
                        }
                    }
                    for l in 0..32 {
                        let lt = (la[l] + lb[l]) + (la[l ^ 16] + lb[l ^ 16]);
                        assert!(lt > 0.0 && lt.is_finite() || !opt.zero_fill || !opt.mask, "lane {l}: l = {lt} (the kernel traps)");
                        let r_raw = rw0 + ((l & 15) >> 2);
                        if r_raw < batch {
                            let inv = 1.0 / lt;
                            for dt in 0..8 {
                                for e in 0..8 {
                                    out[(r_raw * nh + lane_h(l)) * 128 + 16 * dt + 8 * (l >> 4) + e] = o[l][dt * 8 + e] * inv;
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// f64 causal attention: query row i (phase row q_abs0 + i) over keys seg0 ..= q_abs0 + i
    /// of row-major `k` / `v` (`[row][nkv][128]` f32).
    #[allow(clippy::too_many_arguments)]
    fn ref_attn(q: &[f32], k: &[f32], v: &[f32], nh: usize, nkv: usize, batch: usize, q_abs0: usize, seg0: usize, scale: f64) -> Vec<f64> {
        let g = nh / nkv;
        let mut out = vec![0f64; batch * nh * 128];
        for i in 0..batch {
            for hh in 0..nh {
                let kh = hh / g;
                let qr = &q[(i * nh + hh) * 128..][..128];
                let keys = seg0..=q_abs0 + i;
                let sc: Vec<f64> = keys
                    .clone()
                    .map(|j| (0..128).map(|d| f64::from(qr[d]) * f64::from(k[(j * nkv + kh) * 128 + d])).sum::<f64>() * scale)
                    .collect();
                let mx = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
                let den: f64 = w.iter().sum();
                let o = &mut out[(i * nh + hh) * 128..][..128];
                for (j, wj) in keys.zip(&w) {
                    for d in 0..128 {
                        o[d] += wj / den * f64::from(v[(j * nkv + kh) * 128 + d]);
                    }
                }
            }
        }
        out
    }

    /// H3: the kernel model against the f64 reference on the gate's geometry cases, at
    /// kv_group 4 and 8 and both WG heights (NW16: 64 rows, NW8: 32); NaN in every K/V
    /// row past the launch's last query, finite garbage before seg0 (another segment).
    /// Negative controls: without the edge zero-fill the output is NaN; without the mask
    /// it fails the bar.
    #[test]
    fn g4_kernel_model_matches_reference() {
        let scale = 1.0 / 128f32.sqrt();
        let c = scale * std::f32::consts::LOG2_E;
        // (seg0, q_abs0, batch, q amplitude): partial tiles, unaligned segment starts,
        // q_offset > 0, one row, one full sub-batch; amplitude 4 = peaked softmax (rescales).
        let mut cases = vec![(0usize, 0usize, 1usize, 1.0f32), (0, 0, 7, 1.0), (37, 37, 77, 4.0), (37, 100, 130, 1.0), (0, 1024, 64, 4.0)];
        if !cfg!(debug_assertions) {
            cases.push((5, 5, 1024, 1.0));
        }
        let mut worst = 0f64;
        for &(nh, nkv) in &[(8usize, 2usize), (8, 1)] {
            for &(seg0, q_abs0, batch, amp) in &cases {
                let end = q_abs0 + batch;
                let n_tiles = end.div_ceil(32) + 1; // a spare tile: the layout is sized, not fitted
                let rows = n_tiles * 32;
                let seed = (seg0 * 7919 + q_abs0 * 131 + batch + nkv * 1_000_003) as u64;
                let f16v = |i: u64, a: f32| h(v4flash_core::kquants::f32_to_f16_bits(((hash(i) >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * a));
                // K/V rows: before seg0 finite garbage (another segment), [seg0, end) data, past end NaN.
                let kv_val = |r: usize, i: u64| -> f32 {
                    if r >= end { f32::NAN } else if r < seg0 { 1000.0 } else { f16v(i, 1.0) }
                };
                let n = rows * nkv * 128;
                let kf: Vec<f32> = (0..n).map(|i| kv_val(i / (nkv * 128), seed ^ (i as u64) << 1)).collect();
                let vf: Vec<f32> = (0..n).map(|i| kv_val(i / (nkv * 128), seed ^ ((i as u64) << 1 | 1) ^ 0xABCD)).collect();
                let bits = |x: &f32| v4flash_core::kquants::f32_to_f16_bits(*x);
                let (kb, vb): (Vec<u16>, Vec<u16>) = (kf.iter().map(bits).collect(), vf.iter().map(bits).collect());
                let (kc, vc) = g4_pack_kv(&kb, &vb, rows, nkv, n_tiles, 0x7E00);
                let q: Vec<f32> = (0..batch * nh * 128).map(|i| f16v(seed ^ 0x5151 ^ (i as u64) << 2, amp)).collect();
                let want = ref_attn(&q, &kf, &vf, nh, nkv, batch, q_abs0, seg0, f64::from(scale));
                for br in [64usize, 32] {
                    let got = g4_model(&q, &kc, &vc, nh, nkv, n_tiles, batch, q_abs0, seg0, c, br, &ModelOpts { zero_fill: true, mask: true });
                    assert!(got.iter().all(|x| x.is_finite()), "case {seg0}/{q_abs0}/{batch} kv {nkv} br {br}: non-finite output");
                    let err = got.iter().zip(&want).map(|(g, w)| (f64::from(*g) - w).abs()).fold(0f64, f64::max);
                    assert!(err < 2e-3, "case {seg0}/{q_abs0}/{batch} kv {nkv} br {br}: max abs {err:.3e}");
                    worst = worst.max(err);
                }
            }
        }
        eprintln!("g4 kernel model: worst max abs {worst:.3e} vs f64");
        // Negative controls on a case whose last tile holds NaN rows and whose first
        // tile holds garbage before seg0.
        let (seg0, q_abs0, batch, nh, nkv) = (37usize, 37usize, 77usize, 8usize, 2usize);
        let n_tiles = (q_abs0 + batch).div_ceil(32);
        let rows = n_tiles * 32;
        let row = |r: usize, i: usize| if r >= q_abs0 + batch { f32::NAN } else if r < seg0 { 1000.0 } else { ((i * 37 % 101) as f32 / 101.0) - 0.5 };
        let kf: Vec<f32> = (0..rows * nkv * 128).map(|i| row(i / (nkv * 128), i)).collect();
        let vf: Vec<f32> = (0..rows * nkv * 128).map(|i| row(i / (nkv * 128), i + 7)).collect();
        let bits = |x: &f32| v4flash_core::kquants::f32_to_f16_bits(*x);
        let (kb, vb): (Vec<u16>, Vec<u16>) = (kf.iter().map(bits).collect(), vf.iter().map(bits).collect());
        let kf: Vec<f32> = kb.iter().map(|&b| h(b)).collect();
        let vf: Vec<f32> = vb.iter().map(|&b| h(b)).collect();
        let (kc, vc) = g4_pack_kv(&kb, &vb, rows, nkv, n_tiles, 0x7E00);
        let q: Vec<f32> = (0..batch * nh * 128).map(|i| h(bits(&(((i * 53 % 97) as f32 / 97.0) - 0.5)))).collect();
        let want = ref_attn(&q, &kf, &vf, nh, nkv, batch, q_abs0, seg0, f64::from(scale));
        let ok = g4_model(&q, &kc, &vc, nh, nkv, n_tiles, batch, q_abs0, seg0, c, 64, &ModelOpts { zero_fill: true, mask: true });
        let err = |g: &[f32]| g.iter().zip(&want).map(|(g, w)| (f64::from(*g) - w).abs()).fold(0f64, f64::max);
        assert!(err(&ok) < 2e-3, "negative-control baseline: {:.3e}", err(&ok));
        let no_fill = g4_model(&q, &kc, &vc, nh, nkv, n_tiles, batch, q_abs0, seg0, c, 64, &ModelOpts { zero_fill: false, mask: true });
        assert!(no_fill.iter().any(|x| x.is_nan()), "negative control: without the edge zero-fill the NaN rows must reach the output");
        let no_mask = g4_model(&q, &kc, &vc, nh, nkv, n_tiles, batch, q_abs0, seg0, c, 64, &ModelOpts { zero_fill: true, mask: false });
        let e = err(&no_mask);
        assert!(!(e < 2e-3), "negative control: without the mask the model still met the bar ({e:.3e})");
    }

    /// The G4 shape gate: kv_group 4 * 2^s only.
    #[test]
    fn g4_kv_shift_accepts_only_group_4_times_a_power_of_two() {
        assert_eq!(g4_kv_shift(32, 8), Some(0));
        assert_eq!(g4_kv_shift(32, 4), Some(1));
        assert_eq!(g4_kv_shift(64, 4), Some(2));
        assert_eq!(g4_kv_shift(8, 2), Some(0));
        assert_eq!(g4_kv_shift(24, 4), None); // group 6
        assert_eq!(g4_kv_shift(48, 4), None); // group 12 = 4 * 3
        assert_eq!(g4_kv_shift(16, 8), None); // group 2
        assert_eq!(g4_kv_shift(4, 0), None);
        for p in [AttnPath::Fa2, AttnPath::Packed, AttnPath::G4(G4Cell::Cu16), AttnPath::G4(G4Cell::Cu8), AttnPath::G4(G4Cell::Wgp16)] {
            assert_eq!(AttnPath::from_name(p.name()), Some(p));
        }
        assert_eq!((G4Cell::Cu16.rows(), G4Cell::Cu8.rows(), G4Cell::Wgp16.rows()), (64, 32, 64));
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
