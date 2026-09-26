//! ARCH_SPEC §1.5 — hierarchical candidate pool (level one of the two-level top-k).
//!
//! The candidate source layer (20) scores every reachable compressed position,
//! then keeps the `CANDIDATE_TOPK_BLOCKS` best BLOCKS of `CANDIDATE_BLOCK_SIZE`
//! consecutive positions. Index sources above it (24/28/32/36) score with their
//! own weights but only inside that mask, so their scan is bounded by
//! `2048 * 8 = 16384` positions instead of the whole store.
//!
//! Reference: `inference/model.py::select_candidate_blocks`, shipped with the
//! weights. Transcribed here as the oracle the kernels are tested against.
//!
//! NOTE the no-op regime: `topk(min(topk_blocks, num_blocks))` means that while
//! `n_comp <= topk_blocks * block_size` (16384) every block is kept and the mask
//! is all-true. Below that context the level-two layers are conformant WITHOUT
//! this pass; above it they are not.

use crate::config::{CANDIDATE_BLOCK_SIZE, CANDIDATE_TOPK_BLOCKS};
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const CANDIDATE_BLOCKS_GFX1201: &[u8] = include_bytes!(env!("KERNEL_CANDIDATE_BLOCKS_GFX1201"));
const CANDIDATE_BLOCKS_GFX1151: &[u8] = include_bytes!(env!("KERNEL_CANDIDATE_BLOCKS_GFX1151"));

/// CPU reference. `logits` holds one query's scores over `n_comp` reachable
/// compressed positions (unreachable ones are already `-inf`). Returns a
/// per-POSITION keep mask of length `n_comp`.
///
/// Mirrors the reference exactly:
///   * block score = max over its positions, `-inf` padding the last block;
///   * the block holding the newest position is pinned to `+inf` — it is only
///     partly filled and would otherwise lose to an older full block;
///   * top-k blocks, then any pick that came back `-inf` (fewer reachable
///     blocks than `topk_blocks`) is dropped.
pub fn select_candidate_blocks_cpu(logits: &[f32], n_comp: usize) -> Vec<bool> {
    let bs = CANDIDATE_BLOCK_SIZE as usize;
    let k = CANDIDATE_TOPK_BLOCKS as usize;
    let n_blocks = n_comp.div_ceil(bs);
    let mut block_score = vec![f32::NEG_INFINITY; n_blocks];
    for (p, &v) in logits.iter().take(n_comp).enumerate() {
        let b = p / bs;
        if v > block_score[b] {
            block_score[b] = v;
        }
    }
    // Pin the block holding the newest position.
    if n_comp > 0 {
        block_score[(n_comp - 1) / bs] = f32::INFINITY;
    }
    // Top-min(k, n_blocks) by score; ties are broken arbitrarily by the
    // reference too (torch.topk), so any valid set is acceptable.
    let mut order: Vec<usize> = (0..n_blocks).collect();
    let take = k.min(n_blocks);
    order.sort_by(|&a, &b| block_score[b].total_cmp(&block_score[a]));
    let mut keep_block = vec![false; n_blocks];
    for &b in order.iter().take(take) {
        // "leftover picks came back -inf: drop them"
        if block_score[b] > f32::NEG_INFINITY {
            keep_block[b] = true;
        }
    }
    (0..n_comp).map(|p| keep_block[p / bs]).collect()
}

/// Is the mask a no-op at this size? (Every block fits in the top-k.)
pub fn candidates_are_vacuous(n_comp: usize) -> bool {
    n_comp.div_ceil(CANDIDATE_BLOCK_SIZE as usize) <= CANDIDATE_TOPK_BLOCKS as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vacuous_below_the_threshold() {
        // 2048 blocks x 8 = 16384 positions: everything is kept.
        assert!(candidates_are_vacuous(16384));
        assert!(!candidates_are_vacuous(16385));
        let n = 4096;
        let logits: Vec<f32> = (0..n).map(|i| (i % 97) as f32).collect();
        assert!(select_candidate_blocks_cpu(&logits, n).iter().all(|&b| b));
    }

    #[test]
    fn newest_block_is_pinned_even_when_it_scores_worst() {
        let bs = CANDIDATE_BLOCK_SIZE as usize;
        let n = (CANDIDATE_TOPK_BLOCKS as usize + 4) * bs;
        // Descending, so the LAST block is the worst-scoring and would be cut.
        let logits: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
        let keep = select_candidate_blocks_cpu(&logits, n);
        assert!(keep[n - 1], "newest position must survive");
        assert_eq!(keep.iter().filter(|&&b| b).count(), CANDIDATE_TOPK_BLOCKS as usize * bs);
    }

    #[test]
    fn unreachable_blocks_are_dropped_not_padded_in() {
        let bs = CANDIDATE_BLOCK_SIZE as usize;
        let n = (CANDIDATE_TOPK_BLOCKS as usize + 8) * bs;
        // Only the first 3 blocks are reachable; everything else -inf.
        let mut logits = vec![f32::NEG_INFINITY; n];
        for p in 0..3 * bs {
            logits[p] = p as f32;
        }
        // n_comp says the newest position is inside block 2.
        let keep = select_candidate_blocks_cpu(&logits, 3 * bs);
        assert_eq!(keep.len(), 3 * bs);
        assert!(keep.iter().all(|&b| b), "3 blocks < topk: all kept");
    }
}

/// GPU side of §1.5: block max (+pin) -> top-k threshold -> mask apply.
///
/// The reachable-count buffer is taken as a RAW device pointer: the kernel reads
/// it as `unsigned int` (counts are non-negative) while callers hold it as
/// `DeviceBuffer<i32>` on the decode path and `<u32>` in the oracle.
pub struct CandidateBlocks {
    module: Module,
}

/// Largest batch that runs `candidate_threshold_ilp` (see [`cand_thresh_ilp_for`]).
pub const CAND_THRESH_ILP_MAX_B: u32 = 8;

/// `V41_CAND_THRESH_ILP` (default ON; `0` = `candidate_threshold` at every
/// batch): the level-one threshold at `batch <= 8` (decode lanes at L20) runs
/// `candidate_threshold_ilp`, block 1024 with 8 independent block-score loads
/// per thread per radix pass; same passes / histogram / scan, so the threshold
/// is BIT-IDENTICAL (tests/indexer_sweep_bitexact.rs; the sweep review: 12654 +
/// 67836 row checks vs production and a CPU nth_element). 2026-09-26 sweep
/// (E_indexer/candidate_threshold_ilp, dGPU): 235K 87.4 -> 43.8 us, 131K 57.3
/// -> 32.4, 65K 41.9 -> 30.0. The batch gate is the review's: at prefill shapes
/// (b = 512/1024) the 1024-wide kernel is 2.0-2.7x SLOWER (15 waves idle through
/// the serial bin scan), so those keep the 256-thread production kernel.
pub fn cand_thresh_ilp_for(batch: u32) -> bool {
    static D: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_CAND_THRESH_ILP").as_deref() != Ok("0"));
    *D && batch <= CAND_THRESH_ILP_MAX_B
}

impl CandidateBlocks {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            CANDIDATE_BLOCKS_GFX1201
        } else if arch.starts_with("gfx1151") {
            CANDIDATE_BLOCKS_GFX1151
        } else {
            return Err(eyre!("unsupported arch for candidate_blocks kernel: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)? })
    }

    /// Number of blocks a row of `n_comp` positions needs.
    pub fn n_blocks(n_comp: u32) -> u32 {
        n_comp.div_ceil(CANDIDATE_BLOCK_SIZE)
    }

    /// LEVEL ONE, at the candidate source (layer 20): score each block and pick
    /// the threshold, leaving `block_score` / `threshold` for the layers above to
    /// consume. The source does NOT mask its own scores — the reference publishes
    /// `shared_attn.candidates` and then takes its own full top-k.
    pub fn launch_build(
        &self,
        stream: &Stream,
        scores: &DeviceBuffer<f32>,
        block_score: &mut DeviceBuffer<f32>,
        threshold: &mut DeviceBuffer<u32>,
        n_per: v4flash_hip::sys::hipDeviceptr_t,
        stride: u32,
        nb_stride: u32,
        n_comp_max: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_comp_max == 0 {
            return Ok(());
        }
        const T: u32 = 256;
        let nb_max = Self::n_blocks(n_comp_max);
        let f = self.module.get_function("candidate_block_max")?;
        let cfg = LaunchConfig { grid: (nb_max.div_ceil(T), batch, 1), block: (T, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [block_score.raw(), scores.raw(), n_per, stride, nb_stride, CANDIDATE_BLOCK_SIZE])?;
        let (f, cfg) = self.threshold_kernel(batch)?;
        launch_kernel!(f, cfg, stream, [threshold.raw(), block_score.raw(), n_per, nb_stride, CANDIDATE_BLOCK_SIZE, CANDIDATE_TOPK_BLOCKS])
    }

    /// The threshold kernel and its launch for `batch` rows: the ILP twin at
    /// block 1024 under `V41_CAND_THRESH_ILP` for `batch <= 8`, else
    /// `candidate_threshold` at block 256 (its own size, not the shared `T`).
    fn threshold_kernel(&self, batch: u32) -> eyre::Result<(v4flash_hip::Function<'_>, LaunchConfig)> {
        let (sym, block) = if cand_thresh_ilp_for(batch) {
            ("candidate_threshold_ilp", 1024)
        } else {
            ("candidate_threshold", 256)
        };
        let cfg = LaunchConfig { grid: (batch, 1, 1), block: (block, 1, 1), shared_mem_bytes: 0 };
        Ok((self.module.get_function(sym)?, cfg))
    }

    /// The loaded module (tests: explicit-symbol launches).
    pub fn module(&self) -> &Module { &self.module }

    /// LEVEL TWO, at an index source ABOVE the candidate source (24/28/32/36):
    /// `-inf` every position outside the published candidate blocks, so the
    /// top-k that follows selects only from within them. Mirrors
    /// `index_score.masked_fill(~shared_attn.candidates, -inf)`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_mask(
        &self,
        stream: &Stream,
        scores: &mut DeviceBuffer<f32>,
        block_score: &DeviceBuffer<f32>,
        threshold: &DeviceBuffer<u32>,
        n_per: v4flash_hip::sys::hipDeviceptr_t,
        stride: u32,
        nb_stride: u32,
        n_comp_max: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_comp_max == 0 {
            return Ok(());
        }
        const T: u32 = 256;
        let f = self.module.get_function("candidate_mask_apply")?;
        let cfg = LaunchConfig { grid: (n_comp_max.div_ceil(T), batch, 1), block: (T, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [scores.raw(), block_score.raw(), threshold.raw(), n_per, stride, nb_stride, CANDIDATE_BLOCK_SIZE])
    }

    /// Build the mask AND apply it, in place, to `scores`.
    ///
    /// `scores` is the batched indexer layout (row `b` at `b * stride`), `n_per`
    /// the reachable count per row. `block_score` and `threshold` are scratch,
    /// sized `batch * nb_stride` and `batch`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_apply(
        &self,
        stream: &Stream,
        scores: &mut DeviceBuffer<f32>,
        block_score: &mut DeviceBuffer<f32>,
        threshold: &mut DeviceBuffer<u32>,
        n_per: v4flash_hip::sys::hipDeviceptr_t,
        stride: u32,
        nb_stride: u32,
        n_comp_max: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_comp_max == 0 {
            return Ok(());
        }
        let cb = CANDIDATE_BLOCK_SIZE;
        let nb_max = Self::n_blocks(n_comp_max);
        const T: u32 = 256;

        let f = self.module.get_function("candidate_block_max")?;
        let cfg = LaunchConfig { grid: (nb_max.div_ceil(T), batch, 1), block: (T, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [block_score.raw(), scores.raw(), n_per, stride, nb_stride, cb])?;

        let (f, cfg) = self.threshold_kernel(batch)?;
        launch_kernel!(f, cfg, stream, [threshold.raw(), block_score.raw(), n_per, nb_stride, cb, CANDIDATE_TOPK_BLOCKS])?;

        let f = self.module.get_function("candidate_mask_apply")?;
        let cfg = LaunchConfig { grid: (n_comp_max.div_ceil(T), batch, 1), block: (T, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, stream, [scores.raw(), block_score.raw(), threshold.raw(), n_per, stride, nb_stride, cb])
    }
}
