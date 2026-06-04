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

pub const INDEXER_TOP_K: u32 = 512;
pub const INDEXER_N_HEAD: u32 = 64;
pub const INDEXER_HEAD_DIM: u32 = 128;

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
///   - `n_comp ≤ 4096`         → single `indexer_topk_bitonic_4096` WG
///                               (no scratch).
///   - `4096 < n_comp ≤ 24576` → `indexer_topk_chunk_4096` (one WG per
///                               4096-element chunk, up to 6 chunks for
///                               ATTN_MIXED_MAX_KEYS=24576) followed by
///                               one `indexer_topk_merge_4096` WG that
///                               bitonic-sorts the (n_chunks × top_k)
///                               candidates and writes the final
///                               selected[] + bitmap.
///
/// Self-zeroes `allowed_bits` so the API matches [`IndexerTopk`].
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
        if n_candidates > SORT_N {
            return Err(eyre!(
                "indexer_topk_bitonic: n_candidates={n_candidates} exceeds merge cap {SORT_N} \
                 (n_chunks={n_chunks}, top_k={top_k}). Multi-level tree merge not yet implemented."
            ));
        }
        if scratch.len() < n_candidates as usize {
            return Err(eyre!(
                "indexer_topk_bitonic: scratch has {} u32, need {}",
                scratch.len(),
                n_candidates
            ));
        }

        let chunk_fn = self.module.get_function("indexer_topk_chunk_4096")?;
        let chunk_cfg = LaunchConfig {
            grid: (n_chunks, 1, 1),
            block: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(chunk_fn, chunk_cfg, stream, [
            scratch.raw(), scores.raw(), n_comp, top_k
        ])?;

        let merge_fn = self.module.get_function("indexer_topk_merge_4096")?;
        let merge_cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(merge_fn, merge_cfg, stream, [
            selected.raw(), allowed_bits.raw(), scratch.raw(), scores.raw(),
            n_comp, top_k, n_candidates
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
