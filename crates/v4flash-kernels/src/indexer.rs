//! V4 Flash CSA indexer — produces the `comp_allowed` boolean mask that
//! `attention_mixed` (M6) consumes for ratio==4 layers.
//!
//! Composition (per token in a ratio==4 layer):
//!   1. F16 matvec(indexer.attn_q_b × qr_norm) → indexer_q[64, 128]
//!   2. RoPE forward on indexer_q
//!   3. F16 matvec(indexer.proj × attn_norm) → head_weights[64]
//!   4. Scale head_weights by 1/sqrt(head_dim * n_head)
//!   5. Per-comp-row score via `IndexerScore` kernel
//!   6. Top-K = DS4_N_INDEXER_TOP_K = 512 greedy selection → bool mask
//!
//! Early return: if `n_comp <= top_k`, ds4 returns all-permit without
//! computing q/weights/scores. Our pipeline mirrors that.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const INDEXER_SCORE_GFX1201: &[u8] = include_bytes!(env!("KERNEL_INDEXER_SCORE_GFX1201"));
const INDEXER_SCORE_GFX1151: &[u8] = include_bytes!(env!("KERNEL_INDEXER_SCORE_GFX1151"));

const INDEXER_SCORE_WMMA_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_INDEXER_SCORE_WMMA_GFX1201"));
const INDEXER_SCORE_WMMA_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_INDEXER_SCORE_WMMA_GFX1151"));

const INDEXER_TOPK_GFX1201: &[u8] = include_bytes!(env!("KERNEL_INDEXER_TOPK_GFX1201"));
const INDEXER_TOPK_GFX1151: &[u8] = include_bytes!(env!("KERNEL_INDEXER_TOPK_GFX1151"));

const INDEXER_GATHER_GFX1201: &[u8] = include_bytes!(env!("KERNEL_INDEXER_GATHER_GFX1201"));
const INDEXER_GATHER_GFX1151: &[u8] = include_bytes!(env!("KERNEL_INDEXER_GATHER_GFX1151"));

const INDEXER_TOPK_BITONIC_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_INDEXER_TOPK_BITONIC_GFX1201"));
const INDEXER_TOPK_BITONIC_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_INDEXER_TOPK_BITONIC_GFX1151"));

const VEC_SCALE_INPLACE_GFX1201: &[u8] = include_bytes!(env!("KERNEL_VEC_SCALE_INPLACE_GFX1201"));
const VEC_SCALE_INPLACE_GFX1151: &[u8] = include_bytes!(env!("KERNEL_VEC_SCALE_INPLACE_GFX1151"));

const INDEXER_BITPACK_GFX1201: &[u8] = include_bytes!(env!("KERNEL_INDEXER_BITPACK_GFX1201"));
const INDEXER_BITPACK_GFX1151: &[u8] = include_bytes!(env!("KERNEL_INDEXER_BITPACK_GFX1151"));

const INDEXER_QAT_GFX1201: &[u8] = include_bytes!(env!("KERNEL_INDEXER_QAT_GFX1201"));
const INDEXER_QAT_GFX1151: &[u8] = include_bytes!(env!("KERNEL_INDEXER_QAT_GFX1151"));

pub const INDEXER_TOP_K: u32 = 512;
/// Index heads for THIS model. Was hard-coded 64 (V4-Flash) while
/// `config::N_INDEXER_HEAD` is cfg-gated 64/32 — a second source of truth that made
/// `tests/indexer_score.rs` drive the 32-head V4.1 kernel with 64-head data.
pub const INDEXER_N_HEAD: u32 = crate::config::N_INDEXER_HEAD;
pub const INDEXER_HEAD_DIM: u32 = 128;

/// Hadamard128 + E2M1 FP4 QAT round trip on 128-wide indexer rows,
/// in-place. Mirrors ds4's `dsv4_indexer_qat_rows_inplace_cpu`
/// (5bc1e6d, "Flash graph correctness"): the official V4 graph rotates
/// indexer Q rows and ratio-4 indexer compressor KV rows with a
/// normalised 128-wide Hadamard transform, then quantize-dequantizes
/// through E2M1 FP4 (per-32-block power-of-two scale) — after RoPE,
/// before top-k scoring / comp-cache append.
pub struct IndexerQat {
    module: Module,
}

impl IndexerQat {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            INDEXER_QAT_GFX1201
        } else if arch.starts_with("gfx1151") {
            INDEXER_QAT_GFX1151
        } else {
            return Err(eyre!("unsupported arch for indexer_qat: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// In-place QAT on `n_rows` contiguous rows of `INDEXER_HEAD_DIM`
    /// (=128) floats each. One workgroup of 128 threads per row.
    pub fn launch(
        &self,
        stream: &Stream,
        x: &mut DeviceBuffer<f32>,
        n_rows: u32,
    ) -> eyre::Result<()> {
        if n_rows == 0 {
            return Ok(());
        }
        let function = self.module.get_function("indexer_qat")?;
        let cfg = LaunchConfig {
            grid: (n_rows, 1, 1),
            block: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [x.raw(), n_rows])
    }

    /// V4.1: FP4 round trip ONLY — no Hadamard128, no 1/sqrt(128).
    ///
    /// `fp4_act_quant(x, 32, True)` from the reference `inference/kernel.py:184`.
    /// V4-Flash rotates before quantizing; V4.1 does not (there is no "hadamard"
    /// anywhere in its reference). Using `launch` here would give a plausible but
    /// WRONG selection; skipping quantization ENTIRELY is also wrong — it leaves Q
    /// in f32 while the index keys are FP4.
    pub fn launch_fp4(
        &self,
        stream: &Stream,
        x: &mut DeviceBuffer<f32>,
        n_rows: u32,
    ) -> eyre::Result<()> {
        if n_rows == 0 {
            return Ok(());
        }
        let function = self.module.get_function("indexer_fp4")?;
        let cfg = LaunchConfig {
            grid: (n_rows, 1, 1),
            block: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [x.raw(), n_rows])
    }
}

/// Per-comp-row scoring kernel.
pub struct IndexerScore {
    module: Module,
}

impl IndexerScore {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            INDEXER_SCORE_GFX1201
        } else if arch.starts_with("gfx1151") {
            INDEXER_SCORE_GFX1151
        } else {
            return Err(eyre!("unsupported arch for indexer_score: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// `scores[c] = sum_h max(0, dot(q[h], index_comp_kv[c])) * head_weights[h]`
    /// for `c in 0..n_comp`. `index_comp_kv` is f16-stored (matches the
    /// indexer compressor's output buffer format and ds4's
    /// `index_comp_post_fp8` dump tag).
    pub fn launch(
        &self,
        stream: &Stream,
        scores: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        head_weights: &DeviceBuffer<f32>,
        index_comp_kv: &DeviceBuffer<u16>,
        n_comp: u32,
        n_head: u32,
        head_dim: u32,
    ) -> eyre::Result<()> {
        if n_comp == 0 {
            return Err(eyre!("indexer_score: n_comp must be > 0"));
        }
        let function = self.module.get_function("indexer_score")?;
        let cfg = LaunchConfig {
            grid: (n_comp, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            scores.raw(), q.raw(), head_weights.raw(), index_comp_kv.raw(),
            n_comp, n_head, head_dim
        ])
    }

    /// [`IndexerScore::launch`] over a packed-E2M1 key cache
    /// (`index_kv_e2m1`, 80-B rows, head_dim must be 128). Same math,
    /// bit-identical scores.
    pub fn launch_e2m1(
        &self,
        stream: &Stream,
        scores: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        head_weights: &DeviceBuffer<f32>,
        index_comp_kv: &DeviceBuffer<u8>,
        n_comp: u32,
        n_head: u32,
        head_dim: u32,
    ) -> eyre::Result<()> {
        if n_comp == 0 {
            return Err(eyre!("indexer_score_e2m1: n_comp must be > 0"));
        }
        if head_dim as usize != crate::index_kv_e2m1::E2M1_KEY_DIM {
            return Err(eyre!("indexer_score_e2m1: head_dim must be 128, got {head_dim}"));
        }
        if index_comp_kv.len() < (n_comp as usize) * crate::index_kv_e2m1::E2M1_KEY_ROW_BYTES {
            return Err(eyre!("indexer_score_e2m1: packed keys too small for n_comp={n_comp}"));
        }
        let function = self.module.get_function("indexer_score_e2m1")?;
        let cfg = LaunchConfig {
            grid: (n_comp, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            scores.raw(), q.raw(), head_weights.raw(), index_comp_kv.raw(),
            n_comp, n_head, head_dim
        ])
    }
}

/// WMMA-based variant of [`IndexerScore`]. Same math, identical I/O
/// contract; rewrites the per-row scalar scoring as a Q × K^T GEMM fused
/// with the per-row ReLU + head_weights + sum-across-heads tail so the
/// 64-head intermediate never materialises in DRAM. Hardcoded to
/// N_INDEXER_HEAD=64 and N_INDEXER_HEAD_DIM=128 (the V4-Flash shape).
/// Requires gfx12 (RDNA4 WMMA); falls back to a no-op on other arches.
pub struct IndexerScoreWmma {
    module: Module,
}

impl IndexerScoreWmma {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            INDEXER_SCORE_WMMA_GFX1201
        } else if arch.starts_with("gfx1151") {
            INDEXER_SCORE_WMMA_GFX1151
        } else {
            return Err(eyre!("unsupported arch for indexer_score_wmma: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    pub fn launch(
        &self,
        stream: &Stream,
        scores: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        head_weights: &DeviceBuffer<f32>,
        index_comp_kv: &DeviceBuffer<u16>,
        n_comp: u32,
    ) -> eyre::Result<()> {
        if n_comp == 0 {
            return Err(eyre!("indexer_score_wmma: n_comp must be > 0"));
        }
        let function = self.module.get_function("indexer_score_wmma")?;
        // One WG per 16-comp-row n-tile.
        let n_tiles = (n_comp + 15) / 16;
        let cfg = LaunchConfig {
            grid: (n_tiles, 1, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            scores.raw(), q.raw(), head_weights.raw(), index_comp_kv.raw(), n_comp
        ])
    }

    /// Batched variant: one launch handles `batch` tokens. Per-token n_comp
    /// comes from `n_idx_per[bi]`. `scores`/`q`/`head_weights` are strided
    /// by `bi` (scores stride = `n_idx_stride`). Scores past valid range
    /// are stamped with -INF (so the bitonic topk that follows picks only
    /// valid entries).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched(
        &self,
        stream: &Stream,
        scores: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        head_weights: &DeviceBuffer<f32>,
        index_comp_kv: &DeviceBuffer<u16>,
        n_idx_per: &DeviceBuffer<u32>,
        n_idx_max: u32,
        n_idx_stride: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_idx_max == 0 {
            return Ok(());
        }
        let function = self.module.get_function("indexer_score_wmma_batched")?;
        // Must match ISW_NT_PER_WG in kernels/indexer_score_wmma.hip.
        const NT_PER_WG: u32 = 8;
        let n_tiles_max = (n_idx_max + 15) / 16;
        let n_chunks_x = (n_tiles_max + NT_PER_WG - 1) / NT_PER_WG;
        let cfg = LaunchConfig {
            grid: (n_chunks_x, batch, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            scores.raw(), q.raw(), head_weights.raw(), index_comp_kv.raw(),
            n_idx_per.raw(), n_idx_stride
        ])
    }

    /// **Multi-wave NON-batched variant (M58, decode path)** — same fix as
    /// `launch_batched_mw`: the 1-wave kernel re-stages 16 KB of Q per
    /// 16-col WG (~25 MB/layer of Q re-reads at decode depth 96K).
    pub fn launch_mw(
        &self,
        stream: &Stream,
        scores: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        head_weights: &DeviceBuffer<f32>,
        index_comp_kv: &DeviceBuffer<u16>,
        n_comp: u32,
    ) -> eyre::Result<()> {
        if n_comp == 0 {
            return Err(eyre!("indexer_score_wmma_mw: n_comp must be > 0"));
        }
        let function = self.module.get_function("indexer_score_wmma_mw")?;
        const COLS_PER_WG: u32 = 8 * 8 * 16; // 1024
        let cfg = LaunchConfig {
            grid: ((n_comp + COLS_PER_WG - 1) / COLS_PER_WG, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            scores.raw(), q.raw(), head_weights.raw(), index_comp_kv.raw(), n_comp
        ])
    }

    /// [`Self::launch_mw`] over a packed-E2M1 key cache (80-B rows).
    /// Bit-identical scores (the B-fragments expand to the same f16).
    pub fn launch_mw_e2m1(
        &self,
        stream: &Stream,
        scores: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        head_weights: &DeviceBuffer<f32>,
        index_comp_kv: &DeviceBuffer<u8>,
        n_comp: u32,
    ) -> eyre::Result<()> {
        if n_comp == 0 {
            return Err(eyre!("indexer_score_wmma_mw_e2m1: n_comp must be > 0"));
        }
        if index_comp_kv.len() < (n_comp as usize) * crate::index_kv_e2m1::E2M1_KEY_ROW_BYTES {
            return Err(eyre!("indexer_score_wmma_mw_e2m1: packed keys too small for n_comp={n_comp}"));
        }
        let function = self.module.get_function("indexer_score_wmma_mw_e2m1")?;
        const COLS_PER_WG: u32 = 8 * 8 * 16; // 1024
        let cfg = LaunchConfig {
            grid: ((n_comp + COLS_PER_WG - 1) / COLS_PER_WG, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            scores.raw(), q.raw(), head_weights.raw(), index_comp_kv.raw(), n_comp
        ])
    }

    /// **Multi-wave batched variant (M52)** — 8 waves/WG share one Q staging
    /// (the 1-wave kernel re-staged Q per 128 cols: ~6.3 GB redundant reads
    /// and a 4× staging-to-WMMA instruction ratio at 96K ctx); B-fragments
    /// load straight from global (K is small + MALL-resident), giving the WG
    /// a single barrier. Same args/semantics as `launch_batched`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched_mw(
        &self,
        stream: &Stream,
        scores: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        head_weights: &DeviceBuffer<f32>,
        index_comp_kv: &DeviceBuffer<u16>,
        n_idx_per: &DeviceBuffer<u32>,
        n_idx_max: u32,
        n_idx_stride: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_idx_max == 0 {
            return Ok(());
        }
        let function = self.module.get_function("indexer_score_wmma_batched_mw")?;
        // Must match ISWMW_WAVES × ISW_NT_PER_WG × ISW_N_TILE in the kernel.
        const COLS_PER_WG: u32 = 8 * 8 * 16; // 1024
        let n_chunks_x = (n_idx_max + COLS_PER_WG - 1) / COLS_PER_WG;
        let cfg = LaunchConfig {
            grid: (n_chunks_x, batch, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            scores.raw(), q.raw(), head_weights.raw(), index_comp_kv.raw(),
            n_idx_per.raw(), n_idx_stride
        ])
    }

    /// [`Self::launch_batched_mw`] over a packed-E2M1 key cache.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched_mw_e2m1(
        &self,
        stream: &Stream,
        scores: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        head_weights: &DeviceBuffer<f32>,
        index_comp_kv: &DeviceBuffer<u8>,
        n_idx_per: &DeviceBuffer<u32>,
        n_idx_max: u32,
        n_idx_stride: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_idx_max == 0 {
            return Ok(());
        }
        if index_comp_kv.len() < (n_idx_max as usize) * crate::index_kv_e2m1::E2M1_KEY_ROW_BYTES {
            return Err(eyre!("indexer_score_wmma_batched_mw_e2m1: packed keys too small for n_idx_max={n_idx_max}"));
        }
        let function = self.module.get_function("indexer_score_wmma_batched_mw_e2m1")?;
        const COLS_PER_WG: u32 = 8 * 8 * 16; // 1024
        let n_chunks_x = (n_idx_max + COLS_PER_WG - 1) / COLS_PER_WG;
        let cfg = LaunchConfig {
            grid: (n_chunks_x, batch, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            scores.raw(), q.raw(), head_weights.raw(), index_comp_kv.raw(),
            n_idx_per.raw(), n_idx_stride
        ])
    }

    /// GEMM-shaped batched indexer score (2026-09-08): 8 tokens per WG share
    /// each staged K tile; Q is `q16` = f16 [B, 64*128] (cast once by the
    /// caller). Bit-exact with `launch_batched` / `_mw`. `n_splits` WGs per
    /// 8-token group split the row range; pass 0 for the built-in choice
    /// (~4 WGs per CU).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched_gemm(
        &self,
        stream: &Stream,
        scores: &mut DeviceBuffer<f32>,
        q16: &DeviceBuffer<u16>,
        head_weights: &DeviceBuffer<f32>,
        index_comp_kv: &DeviceBuffer<u16>,
        n_idx_per: &DeviceBuffer<u32>,
        n_idx_max: u32,
        n_idx_stride: u32,
        batch: u32,
        n_splits: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_idx_max == 0 {
            return Ok(());
        }
        // Head count is a MODEL dimension (64 V4-Flash / 32 V4.1), not a constant:
        // the kernel derives it from `-DDEEPSTRIX_V41`. Hard-coding 64 here rejected
        // every V4.1 call with "q16 too small" and was the blocker on the prefill
        // indexer.
        let q16_need = (batch as usize)
            * (crate::config::N_INDEXER_HEAD as usize)
            * (crate::config::N_INDEXER_HEAD_DIM as usize);
        if q16.len() < q16_need {
            return Err(eyre!("indexer gemm: q16 too small for batch={batch}"));
        }
        const TILE: u32 = 64; // ISG_TILE_ROWS
        let groups = batch.div_ceil(8);
        let n_tiles = n_idx_stride.div_ceil(TILE).max(1);
        let n_splits = if n_splits == 0 {
            // ~256 WGs total (4/CU on the 64-CU 9070 XT), but never more splits than tiles
            (256u32.div_ceil(groups)).clamp(1, n_tiles)
        } else {
            n_splits.clamp(1, n_tiles)
        };
        let span = n_tiles.div_ceil(n_splits) * TILE;
        let n_splits = n_idx_stride.div_ceil(span);
        let function = self.module.get_function("indexer_score_wmma_gemm")?;
        let cfg = LaunchConfig {
            grid: (n_splits, groups, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            scores.raw(), q16.raw(), head_weights.raw(), index_comp_kv.raw(),
            n_idx_per.raw(), n_idx_stride, n_idx_max, batch, span
        ])
    }

    /// [`Self::launch_batched_gemm`] over a packed-E2M1 key cache (80-B
    /// rows; the 64-row K tiles expand to f16 at LDS publish). Same split
    /// arithmetic, bit-identical scores.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched_gemm_e2m1(
        &self,
        stream: &Stream,
        scores: &mut DeviceBuffer<f32>,
        q16: &DeviceBuffer<u16>,
        head_weights: &DeviceBuffer<f32>,
        index_comp_kv: &DeviceBuffer<u8>,
        n_idx_per: &DeviceBuffer<u32>,
        n_idx_max: u32,
        n_idx_stride: u32,
        batch: u32,
        n_splits: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_idx_max == 0 {
            return Ok(());
        }
        // Head count is a MODEL dimension (64 V4-Flash / 32 V4.1), not a constant:
        // the kernel derives it from `-DDEEPSTRIX_V41`. Hard-coding 64 here rejected
        // every V4.1 call with "q16 too small" and was the blocker on the prefill
        // indexer.
        let q16_need = (batch as usize)
            * (crate::config::N_INDEXER_HEAD as usize)
            * (crate::config::N_INDEXER_HEAD_DIM as usize);
        if q16.len() < q16_need {
            return Err(eyre!("indexer gemm e2m1: q16 too small for batch={batch}"));
        }
        if index_comp_kv.len() < (n_idx_max as usize) * crate::index_kv_e2m1::E2M1_KEY_ROW_BYTES {
            return Err(eyre!("indexer gemm e2m1: packed keys too small for n_idx_max={n_idx_max}"));
        }
        const TILE: u32 = 64; // ISG_TILE_ROWS
        let groups = batch.div_ceil(8);
        let n_tiles = n_idx_stride.div_ceil(TILE).max(1);
        let n_splits = if n_splits == 0 {
            (256u32.div_ceil(groups)).clamp(1, n_tiles)
        } else {
            n_splits.clamp(1, n_tiles)
        };
        let span = n_tiles.div_ceil(n_splits) * TILE;
        let n_splits = n_idx_stride.div_ceil(span);
        let function = self.module.get_function("indexer_score_wmma_gemm_e2m1")?;
        let cfg = LaunchConfig {
            grid: (n_splits, groups, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            scores.raw(), q16.raw(), head_weights.raw(), index_comp_kv.raw(),
            n_idx_per.raw(), n_idx_stride, n_idx_max, batch, span
        ])
    }
}

/// Greedy top-K selection over indexer scores. Mirrors ds4's iterative
/// max-find (ds4.c:7022-7032): for each of K iterations, find the
/// strictly-largest score not yet selected; ties break to the FIRST
/// (smallest) index. The strict-`>` semantics is load-bearing — any
/// reduction primitive used inside the kernel must preserve it.
///
/// Outputs:
///  - `selected[k]`     = chosen comp-row index for k = 0..min(top_k, n_comp);
///                        positions [min(top_k,n_comp), top_k) are sentinel `-1`.
///  - `allowed_bits[w]` = packed bitmap; bit (c & 31) of word (c >> 5) is 1
///                        iff comp row c is in the selected set.
///
/// **Early-permit (n_comp ≤ INDEXER_TOP_K) is the CALLER'S responsibility.**
/// ds4 short-circuits in that regime and returns an all-allowed mask without
/// running any of: indexer Q matvec, RoPE, head_weights matvec, IndexerScore,
/// or IndexerTopk. To stay bit-exact and zero-overhead at short ctx, the
/// caller MUST skip this kernel when n_comp ≤ top_k. The kernel handles
/// the case defensively (degenerates to selecting all rows) but it would
/// be unnecessary cost.
pub struct IndexerTopk {
    module: Module,
}

impl IndexerTopk {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            INDEXER_TOPK_GFX1201
        } else if arch.starts_with("gfx1151") {
            INDEXER_TOPK_GFX1151
        } else {
            return Err(eyre!("unsupported arch for indexer_topk: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// `selected` is `[top_k]` i32. `allowed_bits` is `[ceil(n_comp/32)]` u32.
    /// `scores` is `[n_comp]` f32.
    pub fn launch(
        &self,
        stream: &Stream,
        selected: &mut DeviceBuffer<i32>,
        allowed_bits: &mut DeviceBuffer<u32>,
        scores: &DeviceBuffer<f32>,
        n_comp: u32,
        top_k: u32,
    ) -> eyre::Result<()> {
        if n_comp == 0 {
            return Err(eyre!("indexer_topk: n_comp must be > 0"));
        }
        if top_k == 0 {
            return Err(eyre!("indexer_topk: top_k must be > 0"));
        }
        let needed_bits_words = ((n_comp + 31) / 32) as usize;
        if allowed_bits.len() < needed_bits_words {
            return Err(eyre!(
                "indexer_topk: allowed_bits has {} words, need {}",
                allowed_bits.len(),
                needed_bits_words
            ));
        }
        if selected.len() < top_k as usize {
            return Err(eyre!(
                "indexer_topk: selected has {} slots, need {}",
                selected.len(),
                top_k
            ));
        }
        if scores.len() < n_comp as usize {
            return Err(eyre!(
                "indexer_topk: scores has {} elems, need {}",
                scores.len(),
                n_comp
            ));
        }
        let function = self.module.get_function("indexer_topk")?;
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            selected.raw(), allowed_bits.raw(), scores.raw(), n_comp, top_k
        ])
    }
}

/// Bitonic-sort variant of [`IndexerTopk`]. Ported from ds4's
/// `indexer_topk_chunk_pow2_kernel` + `indexer_topk_merge_pow2_kernel`
/// design. Same bit-exact tie-break:
///
///   better(a, b) := a.score > b.score
///                || (a.score == b.score && a.idx < b.idx)
///
/// Dispatch:
///   - `n_comp ≤ 4096`            → single `indexer_topk_bitonic_4096` WG
///                                  (no scratch).
///   - `4096 < n_comp ≤ 32768`    → `indexer_topk_chunk_4096` (one WG per
///                                  4096-element chunk, ≤ 8 chunks) then one
///                                  `indexer_topk_merge_4096` WG that
///                                  bitonic-sorts the (n_chunks × top_k)
///                                  candidates → selected[] + bitmap.
///   - `32768 < n_comp ≤ 262144`  → two-level tree: chunk →
///                                  `indexer_topk_regroup_4096` (folds groups
///                                  of 8 chunks' top_k down to top_k each) →
///                                  merge. Covers ATTN_MIXED_MAX_KEYS=49408.
///
/// Self-zeroes `allowed_bits` so the API matches [`IndexerTopk`].
/// Bitonic-merge sort width. A single workgroup sorts this many candidates.
pub const TOPK_SORT_N: u32 = 4096;

/// Candidate counts at each level of the bitonic merge ladder.
///
/// `[0]` is the L0 chunk output (`ceil(n_comp / SORT_N) * top_k`); each later
/// level folds the previous by `SORT_N / top_k` (8 at top_k=512) until one
/// merge workgroup can finish. The SUM is the per-row scratch the launcher
/// needs, so the launchers and the two scratch sizers all derive from here --
/// they used to compute it three times and a single regroup pass was baked in.
///
/// Why this is iterative: one regroup pass caps `n_comp` at 262,144, which
/// `--ctx 307200` exceeds. The launcher then failed MID-LAYER instead of at
/// admission, which wedged the GPU for every later request. Two passes reach
/// 2,097,152. The kernels needed no change -- `indexer_topk_regroup_4096`
/// already bounds its input by `in_stride` and drops padding via `idx < n_comp`.
pub fn topk_merge_levels(n_comp: u32, top_k: u32) -> Vec<u32> {
    if n_comp <= TOPK_SORT_N || top_k == 0 {
        return Vec::new();
    }
    let group_chunks = (TOPK_SORT_N / top_k).max(1);
    let group_span = group_chunks * top_k;
    let mut levels = vec![n_comp.div_ceil(TOPK_SORT_N) * top_k];
    while *levels.last().unwrap() > TOPK_SORT_N {
        let n = *levels.last().unwrap();
        let next = n.div_ceil(group_span) * top_k;
        // `next < n` because group_span > top_k; guard anyway so a pathological
        // top_k cannot spin here.
        if next >= n {
            break;
        }
        levels.push(next);
    }
    levels
}

/// Byte offsets (in u32s) of each ladder level inside one row's scratch.
pub fn topk_merge_offsets(levels: &[u32]) -> Vec<usize> {
    let mut offs = Vec::with_capacity(levels.len());
    let mut o = 0usize;
    for &n in levels {
        offs.push(o);
        o += n as usize;
    }
    offs
}

pub struct IndexerTopkBitonic {
    module: Module,
}

impl IndexerTopkBitonic {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            INDEXER_TOPK_BITONIC_GFX1201
        } else if arch.starts_with("gfx1151") {
            INDEXER_TOPK_BITONIC_GFX1151
        } else {
            return Err(eyre!("unsupported arch for indexer_topk_bitonic: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// `selected[top_k]` i32 (sorted descending by score, sentinel -1
    /// past the valid range). `allowed_bits[ceil(n_comp/32)]` u32
    /// (self-zeroed by the kernel before bits are atomically OR'd in).
    /// `scratch` must be sized for at least `n_chunks × top_k` u32
    /// candidates; for ATTN_MIXED_MAX_KEYS=24576 and top_k=512, that's
    /// 6 × 512 = 3072 u32 = 12 KB.
    pub fn launch(
        &self,
        stream: &Stream,
        selected: &mut DeviceBuffer<i32>,
        allowed_bits: &mut DeviceBuffer<u32>,
        scratch: &mut DeviceBuffer<u32>,
        scores: &DeviceBuffer<f32>,
        n_comp: u32,
        top_k: u32,
    ) -> eyre::Result<()> {
        if n_comp == 0 {
            return Err(eyre!("indexer_topk_bitonic: n_comp must be > 0"));
        }
        if top_k == 0 {
            return Err(eyre!("indexer_topk_bitonic: top_k must be > 0"));
        }
        const SORT_N: u32 = 4096;
        const BLOCK: u32 = 1024;
        let needed_bits_words = ((n_comp + 31) / 32) as usize;
        if allowed_bits.len() < needed_bits_words {
            return Err(eyre!(
                "indexer_topk_bitonic: allowed_bits has {} words, need {}",
                allowed_bits.len(),
                needed_bits_words
            ));
        }
        if selected.len() < top_k as usize {
            return Err(eyre!(
                "indexer_topk_bitonic: selected has {} slots, need {}",
                selected.len(),
                top_k
            ));
        }

        if n_comp <= SORT_N {
            let function = self.module.get_function("indexer_topk_bitonic_4096")?;
            let cfg = LaunchConfig {
                grid: (1, 1, 1),
                block: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            return launch_kernel!(function, cfg, stream, [
                selected.raw(), allowed_bits.raw(), scores.raw(), n_comp, top_k
            ]);
        }

        // Chunked path.
        let n_chunks = (n_comp + SORT_N - 1) / SORT_N;
        let n_candidates = n_chunks * top_k;

        let chunk_fn = self.module.get_function("indexer_topk_chunk_4096")?;
        let chunk_cfg = LaunchConfig {
            grid: (n_chunks, 1, 1),
            block: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let merge_fn = self.module.get_function("indexer_topk_merge_4096")?;
        let merge_cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        if n_candidates <= SORT_N {
            // Single-level merge: chunk -> merge.
            if scratch.len() < n_candidates as usize {
                return Err(eyre!(
                    "indexer_topk_bitonic: scratch has {} u32, need {}",
                    scratch.len(),
                    n_candidates
                ));
            }
            launch_kernel!(chunk_fn, chunk_cfg, stream, [
                scratch.raw(), scores.raw(), n_comp, top_k
            ])?;
            return launch_kernel!(merge_fn, merge_cfg, stream, [
                selected.raw(), allowed_bits.raw(), scratch.raw(), scores.raw(),
                n_comp, top_k, n_candidates
            ]);
        }

        // N-level tree merge: chunk -> regroup* -> merge. See
        // [`topk_merge_levels`] for why this iterates instead of doing one pass.
        let group_chunks = SORT_N / top_k; // 8 for SORT_N=4096, top_k=512
        let group_span = group_chunks * top_k;
        let levels = crate::indexer::topk_merge_levels(n_comp, top_k);
        let offs = crate::indexer::topk_merge_offsets(&levels);
        let scratch_need: usize = levels.iter().map(|&n| n as usize).sum();
        if scratch.len() < scratch_need {
            return Err(eyre!(
                "indexer_topk_bitonic: scratch has {} u32, need {} (ladder {levels:?})",
                scratch.len(),
                scratch_need
            ));
        }
        // Non-owning views; `.raw()` copies the pointer out so no borrow of
        // `scratch` outlives the expression.
        let ptrs: Vec<_> = levels
            .iter()
            .zip(&offs)
            .map(|(&n, &o)| scratch.slice_view(o, n as usize).raw())
            .collect();

        launch_kernel!(chunk_fn, chunk_cfg, stream, [
            ptrs[0], scores.raw(), n_comp, top_k
        ])?;

        let regroup_fn = self.module.get_function("indexer_topk_regroup_4096")?;
        for i in 0..levels.len() - 1 {
            let n_in = levels[i];
            let regroup_cfg = LaunchConfig {
                grid: (n_in.div_ceil(group_span), 1, 1),
                block: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            launch_kernel!(regroup_fn, regroup_cfg, stream, [
                ptrs[i + 1], ptrs[i], scores.raw(),
                n_comp, n_in, top_k, group_span
            ])?;
        }

        let last = *levels.last().unwrap();
        launch_kernel!(merge_fn, merge_cfg, stream, [
            selected.raw(), allowed_bits.raw(), ptrs[levels.len() - 1], scores.raw(),
            n_comp, top_k, last
        ])
    }

    /// Batched variant: one launch covers `batch` tokens. Each token's
    /// `n_comp = n_idx_per[bi]` (may vary within batch). Strides:
    ///   - `selected`     stride `top_k`
    ///   - `allowed_bits` stride `n_words_per_b`
    ///   - `scores`       stride `n_idx_stride`
    ///   - `scratch`      stride `n_chunks_max * top_k` (chunked path only)
    /// Self-zeros the per-token `allowed_bits` slice before atomic-OR'ing
    /// in selected bits. `allowed_bits = None` passes a null pointer and
    /// the kernels skip the bitmap entirely (batched prefill has no
    /// bitmap consumer — attention reads the gathered top-K rows).
    /// Pre-condition: scores past `n_idx_per[bi]` must be `-INF` (the
    /// batched IndexerScoreWmma stamps this).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched(
        &self,
        stream: &Stream,
        selected: &mut DeviceBuffer<i32>,
        allowed_bits: Option<&mut DeviceBuffer<u32>>,
        scratch: &mut DeviceBuffer<u32>,
        scores: &DeviceBuffer<f32>,
        n_idx_per: &DeviceBuffer<u32>,
        n_idx_max: u32,
        n_idx_stride: u32,
        n_words_per_b: u32,
        top_k: u32,
        batch: u32,
        done: Option<&mut DeviceBuffer<u32>>,
    ) -> eyre::Result<()> {
        if batch == 0 || top_k == 0 {
            return Ok(());
        }
        const SORT_N: u32 = 4096;
        const BLOCK: u32 = 1024;
        let allowed_ptr = allowed_bits
            .map(|b| b.raw())
            .unwrap_or(std::ptr::null_mut());
        // 2026-09-08: exact threshold select first (O(n) per token, same
        // selection and order as the chain); tokens it cannot bracket are
        // left with done=0 and fall through to the chain below, whose
        // kernels early-out on done=1. Only for the no-bitmap (batched
        // prefill) path; `done` = [batch] u32.
        let done_ptr: *const u32 = match done {
            Some(d) if allowed_ptr.is_null() && top_k <= SORT_N / 2 => {
                if d.len() < batch as usize {
                    return Err(eyre!("indexer_topk: done buffer {} < batch {batch}", d.len()));
                }
                let f = self.module.get_function("indexer_topk_select_batched")?;
                let cfg = LaunchConfig { grid: (batch, 1, 1), block: (BLOCK, 1, 1), shared_mem_bytes: 0 };
                launch_kernel!(f, cfg, stream, [
                    selected.raw(), d.raw(), scores.raw(), n_idx_per.raw(), n_idx_stride, top_k
                ])?;
                if n_idx_max <= SORT_N {
                    return Ok(()); // the select kernel's small path is exact and complete
                }
                d.raw() as *const u32
            }
            _ => std::ptr::null(),
        };

        if n_idx_max <= SORT_N {
            let function = self.module.get_function("indexer_topk_bitonic_4096_batched")?;
            let cfg = LaunchConfig {
                grid: (1, batch, 1),
                block: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            return launch_kernel!(function, cfg, stream, [
                selected.raw(), allowed_ptr, scores.raw(), n_idx_per.raw(),
                n_idx_stride, n_words_per_b, top_k
            ]);
        }

        let n_chunks = (n_idx_max + SORT_N - 1) / SORT_N;
        let n_candidates = n_chunks * top_k;
        let candidates_stride = n_candidates;

        let chunk_fn = self.module.get_function("indexer_topk_chunk_4096_batched")?;
        let chunk_cfg = LaunchConfig {
            grid: (n_chunks, batch, 1),
            block: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let merge_fn = self.module.get_function("indexer_topk_merge_4096_batched")?;
        let merge_cfg = LaunchConfig {
            grid: (1, batch, 1),
            block: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        if n_candidates <= SORT_N {
            // Single-level merge: chunk -> merge.
            let scratch_need = (batch as usize) * (candidates_stride as usize);
            if scratch.len() < scratch_need {
                return Err(eyre!(
                    "indexer_topk_bitonic_batched: scratch has {} u32, need {} (B={batch}, stride={candidates_stride})",
                    scratch.len(),
                    scratch_need,
                ));
            }
            launch_kernel!(chunk_fn, chunk_cfg, stream, [
                scratch.raw(), scores.raw(), n_idx_per.raw(),
                n_idx_stride, candidates_stride, top_k, done_ptr
            ])?;
            return launch_kernel!(merge_fn, merge_cfg, stream, [
                selected.raw(), allowed_ptr, scratch.raw(), scores.raw(), n_idx_per.raw(),
                n_idx_stride, candidates_stride, n_words_per_b, top_k, n_candidates, done_ptr
            ]);
        }

        // N-level tree merge: chunk -> regroup* -> merge. See
        // [`topk_merge_levels`]. THIS is the launcher `--ctx 307200` broke: one
        // regroup pass caps n_comp at 262,144, and exceeding it errored
        // mid-layer, which left the GPU faulted for every later request.
        let group_chunks = SORT_N / top_k;
        let group_span = group_chunks * top_k;
        let levels = crate::indexer::topk_merge_levels(n_idx_max, top_k);
        let offs = crate::indexer::topk_merge_offsets(&levels);
        // Scratch layout: each level is [B, levels[i]], levels laid end to end.
        let per_row: usize = levels.iter().map(|&n| n as usize).sum();
        let need = (batch as usize) * per_row;
        if scratch.len() < need {
            return Err(eyre!(
                "indexer_topk_bitonic_batched: scratch has {} u32, need {} \
                 (B={batch}, ladder {levels:?} = {per_row} per token)",
                scratch.len(),
                need
            ));
        }
        let ptrs: Vec<_> = levels
            .iter()
            .zip(&offs)
            .map(|(&n, &o)| {
                scratch
                    .slice_view((batch as usize) * o, (batch as usize) * n as usize)
                    .raw()
            })
            .collect();

        launch_kernel!(chunk_fn, chunk_cfg, stream, [
            ptrs[0], scores.raw(), n_idx_per.raw(),
            n_idx_stride, candidates_stride, top_k, done_ptr
        ])?;

        let regroup_fn = self.module.get_function("indexer_topk_regroup_4096_batched")?;
        for i in 0..levels.len() - 1 {
            let n_in = levels[i];
            let regroup_cfg = LaunchConfig {
                grid: (n_in.div_ceil(group_span), batch, 1),
                block: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            launch_kernel!(regroup_fn, regroup_cfg, stream, [
                ptrs[i + 1], ptrs[i], scores.raw(), n_idx_per.raw(),
                n_idx_stride, n_in, levels[i + 1], top_k, group_span, done_ptr
            ])?;
        }

        let last = *levels.last().unwrap();
        launch_kernel!(merge_fn, merge_cfg, stream, [
            selected.raw(), allowed_ptr, ptrs[levels.len() - 1], scores.raw(), n_idx_per.raw(),
            n_idx_stride, last, n_words_per_b, top_k, last, done_ptr
        ])
    }
}

/// Gather selected rows of `comp_kv` into a contiguous `active_comp_kv`
/// buffer that the existing attention kernels can consume as a smaller
/// dense `comp_kv` (with `n_comp = top_k`). Pairs with `IndexerTopk`'s
/// `selected[]` output.
pub struct IndexerGather {
    module: Module,
}

impl IndexerGather {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            INDEXER_GATHER_GFX1201
        } else if arch.starts_with("gfx1151") {
            INDEXER_GATHER_GFX1151
        } else {
            return Err(eyre!("unsupported arch for indexer_gather: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// `active_comp_kv[i, d] = comp_kv[selected[i], d]` for i in 0..top_k,
    /// d in 0..head_dim. Sentinel `selected[i] == -1` rows are skipped
    /// (slot left unwritten — caller's responsibility to either zero-init
    /// or know it won't be read).
    pub fn launch(
        &self,
        stream: &Stream,
        active_comp_kv: &mut DeviceBuffer<u16>,
        comp_kv: &DeviceBuffer<u16>,
        selected: &DeviceBuffer<i32>,
        top_k: u32,
        head_dim: u32,
    ) -> eyre::Result<()> {
        if top_k == 0 || head_dim == 0 {
            return Ok(());
        }
        if active_comp_kv.len() < (top_k as usize) * (head_dim as usize) {
            return Err(eyre!(
                "indexer_gather: active_comp_kv has {} elems, need top_k*head_dim={}",
                active_comp_kv.len(),
                (top_k as usize) * (head_dim as usize)
            ));
        }
        if selected.len() < top_k as usize {
            return Err(eyre!(
                "indexer_gather: selected has {} slots, need {}",
                selected.len(),
                top_k
            ));
        }
        let function = self.module.get_function("indexer_gather")?;
        const BLOCK: u32 = 256;
        let dim_blocks = (head_dim + BLOCK - 1) / BLOCK;
        let cfg = LaunchConfig {
            grid: (top_k, dim_blocks, 1),
            block: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            active_comp_kv.raw(), comp_kv.raw(), selected.raw(), top_k, head_dim
        ])
    }

    /// Batched variant: one launch covers `batch` tokens. Per-batch
    /// selected indices at stride `top_k`; per-batch destination strides
    /// `top_k * head_dim`. `comp_kv` is shared across the batch (the layer's
    /// single main compressor). Sentinel slots (selected[bi, i] == -1) are
    /// skipped; downstream attention must respect per-token n_comp_per ≤ top_k.
    pub fn launch_batched(
        &self,
        stream: &Stream,
        active_comp_kv_b: &mut DeviceBuffer<u16>,
        comp_kv: &DeviceBuffer<u16>,
        selected_b: &DeviceBuffer<i32>,
        top_k: u32,
        head_dim: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || top_k == 0 || head_dim == 0 {
            return Ok(());
        }
        let need = (batch as usize) * (top_k as usize) * (head_dim as usize);
        if active_comp_kv_b.len() < need {
            return Err(eyre!(
                "indexer_gather_batched: active_comp_kv_b has {} f16, need {} (B={batch})",
                active_comp_kv_b.len(),
                need
            ));
        }
        if selected_b.len() < (batch as usize) * (top_k as usize) {
            return Err(eyre!(
                "indexer_gather_batched: selected_b too small (have {}, need {})",
                selected_b.len(),
                (batch as usize) * (top_k as usize)
            ));
        }
        let function = self.module.get_function("indexer_gather_batched")?;
        const BLOCK: u32 = 256;
        let dim_blocks = (head_dim + BLOCK - 1) / BLOCK;
        let cfg = LaunchConfig {
            grid: (top_k, batch, dim_blocks),
            block: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            active_comp_kv_b.raw(), comp_kv.raw(), selected_b.raw(), top_k, head_dim
        ])
    }
}

/// In-place scalar multiply on an f32 device buffer.
/// `x[i] *= scalar` for i in 0..n. Used by the indexer pipeline to apply
/// `1/sqrt(head_dim*n_head)` to head_weights[64] without round-tripping
/// through the host.
pub struct VecScaleInplace {
    module: Module,
}

impl VecScaleInplace {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            VEC_SCALE_INPLACE_GFX1201
        } else if arch.starts_with("gfx1151") {
            VEC_SCALE_INPLACE_GFX1151
        } else {
            return Err(eyre!("unsupported arch for vec_scale_inplace: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    pub fn launch(
        &self,
        stream: &Stream,
        x: &mut DeviceBuffer<f32>,
        scalar: f32,
        n: u32,
    ) -> eyre::Result<()> {
        if n == 0 {
            return Ok(());
        }
        if x.len() < n as usize {
            return Err(eyre!(
                "vec_scale_inplace: x has {} elems, need n={}",
                x.len(),
                n
            ));
        }
        let function = self.module.get_function("vec_scale_inplace")?;
        const BLOCK: u32 = 256;
        let grid = (n + BLOCK - 1) / BLOCK;
        let cfg = LaunchConfig {
            grid: (grid, 1, 1),
            block: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [x.raw(), scalar, n])
    }
}

/// Bitpack helpers for the prefill CSA mask. Three kernels:
///  - `bitpack_zero`     — clear the first `n_words` words of a slice.
///  - `bitpack_set`      — OR in the bits named by IndexerTopk's
///                          `selected[k]` list (sentinel `-1` skipped).
///  - `bitpack_all_ones` — set bits [0..n_comp) (early-permit branch
///                          where every comp row is allowed).
pub struct IndexerBitpack {
    module: Module,
}

impl IndexerBitpack {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            INDEXER_BITPACK_GFX1201
        } else if arch.starts_with("gfx1151") {
            INDEXER_BITPACK_GFX1151
        } else {
            return Err(eyre!("unsupported arch for indexer_bitpack: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    pub fn launch_zero(
        &self,
        stream: &Stream,
        bits: &mut DeviceBuffer<u32>,
        n_words: u32,
    ) -> eyre::Result<()> {
        if n_words == 0 {
            return Ok(());
        }
        let function = self.module.get_function("indexer_bitpack_zero")?;
        const BLOCK: u32 = 256;
        let grid = (n_words + BLOCK - 1) / BLOCK;
        let cfg = LaunchConfig {
            grid: (grid, 1, 1),
            block: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [bits.raw(), n_words])
    }

    pub fn launch_set(
        &self,
        stream: &Stream,
        bits: &mut DeviceBuffer<u32>,
        selected: &DeviceBuffer<i32>,
        k: u32,
    ) -> eyre::Result<()> {
        if k == 0 {
            return Ok(());
        }
        let function = self.module.get_function("indexer_bitpack_set")?;
        const BLOCK: u32 = 256;
        let grid = (k + BLOCK - 1) / BLOCK;
        let cfg = LaunchConfig {
            grid: (grid, 1, 1),
            block: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [bits.raw(), selected.raw(), k])
    }

    pub fn launch_all_ones(
        &self,
        stream: &Stream,
        bits: &mut DeviceBuffer<u32>,
        n_comp: u32,
    ) -> eyre::Result<()> {
        if n_comp == 0 {
            return Ok(());
        }
        let function = self.module.get_function("indexer_bitpack_all_ones")?;
        const BLOCK: u32 = 256;
        let n_words = (n_comp + 31) / 32;
        let grid = (n_words + BLOCK - 1) / BLOCK;
        let cfg = LaunchConfig {
            grid: (grid, 1, 1),
            block: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [bits.raw(), n_comp])
    }
}

#[cfg(test)]
mod topk_ladder_tests {
    use super::*;
    use crate::config::INDEXER_TOP_K;

    /// REGRESSION for the 2026-09-18 outage: the launcher hardcoded a single
    /// regroup pass, which caps `n_comp` at 262,144. `--ctx 307200` exceeded it
    /// and the launcher returned an error MID-LAYER -- leaving the GPU faulted,
    /// so every later request died with `hipErrorIllegalAddress` until restart.
    #[test]
    fn merge_ladder_covers_the_shipped_cap() {
        let k = INDEXER_TOP_K;

        // The old two-level scheme's exact ceiling.
        let at_cap = topk_merge_levels(262_144, k);
        assert_eq!(at_cap.len(), 2, "262144 should need 2 levels, got {at_cap:?}");

        // One past it needs a third -- this is what used to hard-error.
        let over = topk_merge_levels(307_200, k);
        assert!(over.len() >= 3, "307200 needs a 3rd level, got {over:?}");

        // Whatever the engine's cap is, the ladder must terminate inside one
        // merge workgroup and strictly shrink at every step.
        let shipped = topk_merge_levels(crate::attention::ATTN_MIXED_MAX_KEYS, k);
        assert!(
            *shipped.last().unwrap() <= TOPK_SORT_N,
            "ladder does not reach a single merge WG: {shipped:?}"
        );
        for w in shipped.windows(2) {
            assert!(w[1] < w[0], "ladder must shrink: {shipped:?}");
        }
    }
}
