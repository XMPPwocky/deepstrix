//! SWA attention compute — mirrors ds4's `layer_attention_rows_one`
//! (ds4.c:4955). Sink-aware causal softmax + weighted sum over the raw KV
//! cache. Used by V4 Flash layers L=0, L=1 (the dense / `ratio==0` layers).
//!
//! For L≥2, ds4 dispatches to `layer_attention_mixed_one` which extends
//! the softmax with compressed-KV rows + indexer masking — that's M6/M7.
//! The M5 SWA kernel is the building block both variants share.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const ATTENTION_SWA_GFX1201: &[u8] = include_bytes!(env!("KERNEL_ATTENTION_SWA_GFX1201"));
const ATTENTION_SWA_GFX1151: &[u8] = include_bytes!(env!("KERNEL_ATTENTION_SWA_GFX1151"));

const ATTENTION_MIXED_GFX1201: &[u8] = include_bytes!(env!("KERNEL_ATTENTION_MIXED_GFX1201"));
const ATTENTION_MIXED_GFX1151: &[u8] = include_bytes!(env!("KERNEL_ATTENTION_MIXED_GFX1151"));

/// Compile-time max for the kernel's shared-memory `scores`/`weights`
/// arrays. Matches the SWA window size (`SWA_WINDOW = 128`); the
/// forward orchestrator caps `n_raw` at this value via memmove-eviction
/// so this is never exceeded.
pub const ATTN_SWA_MAX_KV: u32 = 128;

/// Compile-time max for `attention_mixed`'s `scores`/`weights` arrays;
/// must cover `n_raw + n_comp`. Raw is permanently capped at SWA_WINDOW
/// (128) by the forward orchestrator's sliding eviction; comp grows
/// unbounded until cap_comp (set per allocation in `ModelState::alloc`).
/// 2304 covers up to 128 raw + 2176 comp ≈ 8832 tokens at ratio=4, sized
/// for chat sessions up to CHAT_KV_MAX=8192. Costs ~16 KiB extra LDS/WG
/// vs the original 256-cap; one WG per head so occupancy is fine.
/// Hard cap on `n_raw + n_comp` for the split decode kernels
/// (`attention_mixed_score`, `attention_mixed_softmax_wsum`). Scratch
/// lives in `DgpuScratch.attn_scores` (global memory), so this isn't
/// LDS-bound. 17664 covers 128 raw + 17536 comp ≈ 70K context at ratio=4
/// (headroom above 64K so a full chunk prefilled on top of a 64K prefix
/// still fits). Must match `#define ATTN_MIXED_MAX_KEYS` in `kernels/attention_mixed.hip`.
pub const ATTN_MIXED_MAX_KEYS: u32 = 17664;

/// LDS capacity of the monolithic `attention_mixed_batched` prefill kernel's
/// `scores`/`weights` arrays. Must match `#define ATTN_PREFILL_LDS_MAX` in
/// `kernels/attention_mixed.hip`. When a chunk's max `n_raw + n_comp` exceeds
/// this, prefill dispatches to the batched split kernels (global scratch,
/// no LDS cap) instead — both for correctness (the LDS arrays overflow past
/// ~9K tokens) and perf (the monolithic kernel's single-thread softmax +
/// sequential per-row reduction is the long-context bottleneck).
pub const ATTN_PREFILL_LDS_MAX: u32 = 2304;

/// Head-group size for `attention_mixed_score_batched_htiled`. Must match
/// `#define SCORE_HEAD_TILE` in `kernels/attention_mixed.hip`. One WG covers
/// this many heads, loading the shared KV row once and reusing it across them.
pub const SCORE_HEAD_TILE: u32 = 8;

/// Head-group size for `attention_mixed_softmax_wsum_batched_htiled`. Must
/// match `#define SMWSUM_HEAD_TILE` in `kernels/attention_mixed.hip`.
pub const SMWSUM_HEAD_TILE: u32 = 16;

pub struct AttentionSwa {
    module: Module,
}

impl AttentionSwa {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            ATTENTION_SWA_GFX1201
        } else if arch.starts_with("gfx1151") {
            ATTENTION_SWA_GFX1151
        } else {
            return Err(eyre!("unsupported arch for attention_swa kernel: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Launch the SWA attention kernel.
    ///
    /// - `out`: `[n_head * head_dim]`
    /// - `q`:   `[n_head * head_dim]` (post-RoPE)
    /// - `kv`:  `[n_kv * head_dim]`   (f16-precision values in f32 cells —
    ///                                 ds4's cache stores `f16_to_f32(f32_to_f16(x))`)
    /// - `sinks`: `[n_head]`
    /// - `n_kv ≤ ATTN_SWA_MAX_KV`
    pub fn launch(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        kv: &DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        n_head: u32,
        head_dim: u32,
        n_kv: u32,
    ) -> eyre::Result<()> {
        if n_kv > ATTN_SWA_MAX_KV {
            return Err(eyre!(
                "attention_swa: n_kv={n_kv} exceeds kernel cap {ATTN_SWA_MAX_KV}"
            ));
        }
        if n_kv == 0 {
            return Err(eyre!("attention_swa: n_kv must be > 0"));
        }
        let needed_out = (n_head as usize) * (head_dim as usize);
        if out.len() < needed_out || q.len() < needed_out {
            return Err(eyre!(
                "attention_swa: out/q have {}/{} elems, need {}",
                out.len(),
                q.len(),
                needed_out
            ));
        }
        if kv.len() < (n_kv as usize) * (head_dim as usize) {
            return Err(eyre!(
                "attention_swa: kv has {} elems, need n_kv*head_dim={}",
                kv.len(),
                (n_kv as usize) * (head_dim as usize)
            ));
        }
        if sinks.len() < n_head as usize {
            return Err(eyre!(
                "attention_swa: sinks has {} elems, need n_head={}",
                sinks.len(),
                n_head
            ));
        }

        let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
        let function = self.module.get_function("attention_swa")?;
        let cfg = LaunchConfig {
            grid: (n_head, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), q.raw(), kv.raw(), sinks.raw(), n_head, head_dim, n_kv, kq_scale
        ])
    }

    /// M50 Phase 4: per-token causal SWA attention. Grid (n_head, B, 1).
    /// `n_raw_per[B]` gives each token's causal prefix length over the
    /// SHARED `kv` cache. Per-token q[B, n_head, head_dim], out[B, ...].
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        kv: &DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        n_raw_per: &DeviceBuffer<i32>,
        n_raw_offset_per: &DeviceBuffer<i32>,
        n_head: u32,
        head_dim: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
        let function = self.module.get_function("attention_swa_batched")?;
        let cfg = LaunchConfig {
            grid: (n_head, batch, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), q.raw(), kv.raw(), sinks.raw(),
            n_raw_per.raw(), n_raw_offset_per.raw(),
            n_head, head_dim, kq_scale
        ])
    }
}

/// Mixed-attention compute — mirrors ds4's `layer_attention_mixed_one_decode_scratch`
/// (ds4.c:6738). Extends [`AttentionSwa`] with compressed-KV rows and an
/// optional per-comp-row allow mask. Used by V4 Flash layers L≥2 (ratio>0).
///
/// When `n_comp == 0` and `mask` is `None`, this reduces to the SWA case
/// bit-for-bit — useful for covering all attention paths with one kernel.
pub struct AttentionMixed {
    module: Module,
}

impl AttentionMixed {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            ATTENTION_MIXED_GFX1201
        } else if arch.starts_with("gfx1151") {
            ATTENTION_MIXED_GFX1151
        } else {
            return Err(eyre!("unsupported arch for attention_mixed kernel: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// M50 Phase 4: per-token causal mixed attention. Grid (n_head, B, 1).
    /// `n_raw_per[B]` and `n_comp_per[B]` give each token's per-cache
    /// causal prefixes over the SHARED raw_kv and comp_kv buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<f32>,
        comp_kv: Option<&DeviceBuffer<f32>>,
        sinks: &DeviceBuffer<f32>,
        n_raw_per: &DeviceBuffer<i32>,
        n_comp_per: &DeviceBuffer<i32>,
        n_head: u32,
        head_dim: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
        let function = self.module.get_function("attention_mixed_batched")?;
        let comp_ptr: v4flash_hip::sys::hipDeviceptr_t = match comp_kv {
            Some(c) => c.raw(),
            None => std::ptr::null_mut(),
        };
        let cfg = LaunchConfig {
            grid: (n_head, batch, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), q.raw(), raw_kv.raw(), comp_ptr, sinks.raw(),
            n_raw_per.raw(), n_comp_per.raw(), n_head, head_dim, kq_scale
        ])
    }

    /// Split-kernel decode (perf diagnosis): phase 1 (dot-product scores).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_score(
        &self,
        stream: &Stream,
        scores: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<f32>,
        comp_kv: Option<&DeviceBuffer<f32>>,
        n_head: u32,
        head_dim: u32,
        n_raw: u32,
        n_comp: u32,
    ) -> eyre::Result<()> {
        if n_raw + n_comp > ATTN_MIXED_MAX_KEYS {
            return Err(eyre!(
                "attention_mixed_score: n_raw+n_comp={} exceeds cap {ATTN_MIXED_MAX_KEYS}",
                n_raw + n_comp
            ));
        }
        let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
        let function = self.module.get_function("attention_mixed_score")?;
        let comp_kv_ptr = comp_kv
            .map(|b| b.raw())
            .unwrap_or(std::ptr::null_mut());
        // WG handles ROWS_PER_WG rows: grid (n_head, ceil(n_total/ROWS)).
        // ROWS_PER_WG must match the #define in kernels/attention_mixed.hip.
        const ROWS_PER_WG: u32 = 1;
        let n_total = n_raw + n_comp;
        let grid_y = (n_total + ROWS_PER_WG - 1) / ROWS_PER_WG;
        let cfg = LaunchConfig {
            grid: (n_head, grid_y, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            scores.raw(), q.raw(), raw_kv.raw(), comp_kv_ptr,
            n_head, head_dim, n_raw, n_comp, ATTN_MIXED_MAX_KEYS, kq_scale
        ])
    }

    /// Merged softmax + weighted sum (phases 2-4). Reads scores from
    /// global, does softmax in place via wave 0 + warp-shuffle reductions,
    /// then all 256 threads do the weighted sum reading weights from the
    /// same global buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_softmax_wsum(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        scores: &mut DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<f32>,
        comp_kv: Option<&DeviceBuffer<f32>>,
        n_head: u32,
        head_dim: u32,
        n_raw: u32,
        n_comp: u32,
    ) -> eyre::Result<()> {
        let function = self.module.get_function("attention_mixed_softmax_wsum")?;
        let comp_kv_ptr = comp_kv
            .map(|b| b.raw())
            .unwrap_or(std::ptr::null_mut());
        // block=512 = 16 waves/WG → 4 waves/SIMD on a 64-CU dGPU. More
        // waves give more concurrent in-flight loads to hide L2 latency.
        // Softmax uses only wave 0; other waves idle during phase B.
        let cfg = LaunchConfig { grid: (n_head, 1, 1), block: (512, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [
            out.raw(), scores.raw(), sinks.raw(), raw_kv.raw(), comp_kv_ptr,
            n_head, head_dim, n_raw, n_comp, ATTN_MIXED_MAX_KEYS
        ])
    }

    /// Batched split-kernel PREFILL — phase 1 (scores), head-tiled. Each
    /// token `b` uses its own causal prefix `n_raw_per[b]`/`n_comp_per[b]`
    /// over the shared raw_kv/comp_kv with per-token starting slot
    /// `n_raw_offset_per[b]` into the (oversized) raw cache. `scores_g` is
    /// `[batch, n_head, max_keys]` global. `n_total_max` = max over the
    /// chunk of `n_raw+n_comp` (grid.x extent). Each WG covers
    /// `SCORE_HEAD_TILE` heads, loading the shared KV row once and reusing
    /// it across them. Grid (n_total_max, ceil(n_head/SCORE_HEAD_TILE),
    /// batch), block 32.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_score_batched_htiled(
        &self,
        stream: &Stream,
        scores_g: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<f32>,
        comp_kv: Option<&DeviceBuffer<f32>>,
        n_raw_per: &DeviceBuffer<i32>,
        n_raw_offset_per: &DeviceBuffer<i32>,
        n_comp_per: &DeviceBuffer<i32>,
        n_head: u32,
        head_dim: u32,
        n_total_max: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_total_max == 0 {
            return Ok(());
        }
        if n_total_max > ATTN_MIXED_MAX_KEYS {
            return Err(eyre!(
                "attention_mixed_score_batched_htiled: n_total_max={n_total_max} exceeds cap {ATTN_MIXED_MAX_KEYS}"
            ));
        }
        let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
        let function = self
            .module
            .get_function("attention_mixed_score_batched_htiled")?;
        let comp_kv_ptr = comp_kv.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        let n_head_groups = n_head.div_ceil(SCORE_HEAD_TILE);
        let cfg = LaunchConfig {
            grid: (n_total_max, n_head_groups, batch),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            scores_g.raw(), q.raw(), raw_kv.raw(), comp_kv_ptr,
            n_raw_per.raw(), n_raw_offset_per.raw(), n_comp_per.raw(),
            n_head, head_dim, ATTN_MIXED_MAX_KEYS, kq_scale
        ])
    }

    /// Batched split-kernel PREFILL — phase 2 (merged softmax + weighted
    /// sum), head-tiled. Each WG covers `SMWSUM_HEAD_TILE` heads — softmax
    /// runs one wave per head, and the wsum loads each shared V-latent
    /// element once and reuses it across the head group. Reads/overwrites
    /// `scores_g` from phase 1, writes `out` `[batch, n_head, head_dim]`.
    /// Grid (ceil(n_head/SMWSUM_HEAD_TILE), batch), block 512.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_softmax_wsum_batched_htiled(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        scores_g: &mut DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<f32>,
        comp_kv: Option<&DeviceBuffer<f32>>,
        n_raw_per: &DeviceBuffer<i32>,
        n_raw_offset_per: &DeviceBuffer<i32>,
        n_comp_per: &DeviceBuffer<i32>,
        n_head: u32,
        head_dim: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let function = self
            .module
            .get_function("attention_mixed_softmax_wsum_batched_htiled")?;
        let comp_kv_ptr = comp_kv.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        let n_head_groups = n_head.div_ceil(SMWSUM_HEAD_TILE);
        let cfg = LaunchConfig {
            grid: (n_head_groups, batch, 1),
            block: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), scores_g.raw(), sinks.raw(), raw_kv.raw(), comp_kv_ptr,
            n_raw_per.raw(), n_raw_offset_per.raw(), n_comp_per.raw(),
            n_head, head_dim, ATTN_MIXED_MAX_KEYS
        ])
    }

    /// WMMA variant of [`Self::launch_score_batched_htiled`]. Computes the
    /// score GEMM O[heads,keys]=Q·Kᵀ as a RDNA4 16x16x16 f16 WMMA (f32->f16 at
    /// fragment-load, no f16 KV cache). Requires head_dim==512 and n_head a
    /// multiple of 16. Grid (ceil(n_total_max/256), 1, batch), block 512.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_score_batched_htiled_wmma(
        &self,
        stream: &Stream,
        scores_g: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<f32>,
        comp_kv: Option<&DeviceBuffer<f32>>,
        n_raw_per: &DeviceBuffer<i32>,
        n_raw_offset_per: &DeviceBuffer<i32>,
        n_comp_per: &DeviceBuffer<i32>,
        n_head: u32,
        head_dim: u32,
        n_total_max: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_total_max == 0 {
            return Ok(());
        }
        if n_total_max > ATTN_MIXED_MAX_KEYS {
            return Err(eyre!(
                "attention_mixed_score_batched_htiled_wmma: n_total_max={n_total_max} exceeds cap {ATTN_MIXED_MAX_KEYS}"
            ));
        }
        let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
        let function = self
            .module
            .get_function("attention_mixed_score_batched_htiled_wmma")?;
        let comp_kv_ptr = comp_kv.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        let key_blocks = n_total_max.div_ceil(256);
        let cfg = LaunchConfig {
            grid: (key_blocks, 1, batch),
            block: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            scores_g.raw(), q.raw(), raw_kv.raw(), comp_kv_ptr,
            n_raw_per.raw(), n_raw_offset_per.raw(), n_comp_per.raw(),
            n_head, head_dim, ATTN_MIXED_MAX_KEYS, kq_scale
        ])
    }

    /// WMMA Phase-B variant of [`Self::launch_softmax_wsum_batched_htiled`].
    /// Phase A (softmax) is identical; Phase B is a RDNA4 16x16x16 f16 WMMA
    /// GEMM (f32->f16 converted at fragment-load, no f16 KV cache). Requires
    /// head_dim==512 and SMWSUM_HEAD_TILE==16. Same grid/block/output.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_softmax_wsum_batched_htiled_wmma(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        scores_g: &mut DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<f32>,
        comp_kv: Option<&DeviceBuffer<f32>>,
        n_raw_per: &DeviceBuffer<i32>,
        n_raw_offset_per: &DeviceBuffer<i32>,
        n_comp_per: &DeviceBuffer<i32>,
        n_head: u32,
        head_dim: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let function = self
            .module
            .get_function("attention_mixed_softmax_wsum_batched_htiled_wmma")?;
        let comp_kv_ptr = comp_kv.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        let n_head_groups = n_head.div_ceil(SMWSUM_HEAD_TILE);
        let cfg = LaunchConfig {
            grid: (n_head_groups, batch, 1),
            block: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), scores_g.raw(), sinks.raw(), raw_kv.raw(), comp_kv_ptr,
            n_raw_per.raw(), n_raw_offset_per.raw(), n_comp_per.raw(),
            n_head, head_dim, ATTN_MIXED_MAX_KEYS
        ])
    }

    /// f16-scores experimental variant. Same WMMA math as the f32-scores
    /// kernels, but the scores buffer is reinterpreted as `_Float16*` —
    /// halves the largest DRAM read in attn at long context. Score writes
    /// the result as f16; the matching smwsum reads f16 (Phase A softmax
    /// computes in f32, stores back as f16). The Rust-side buffer stays
    /// `DeviceBuffer<f32>` (oversized — only half is used) so no scratch
    /// re-allocation is needed.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_score_batched_htiled_wmma_f16s(
        &self,
        stream: &Stream,
        scores_g: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<f32>,
        comp_kv: Option<&DeviceBuffer<f32>>,
        n_raw_per: &DeviceBuffer<i32>,
        n_raw_offset_per: &DeviceBuffer<i32>,
        n_comp_per: &DeviceBuffer<i32>,
        n_head: u32,
        head_dim: u32,
        n_total_max: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_total_max == 0 {
            return Ok(());
        }
        if n_total_max > ATTN_MIXED_MAX_KEYS {
            return Err(eyre!(
                "attention_mixed_score_batched_htiled_wmma_f16s: n_total_max={n_total_max} exceeds cap {ATTN_MIXED_MAX_KEYS}"
            ));
        }
        let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
        let function = self
            .module
            .get_function("attention_mixed_score_batched_htiled_wmma_f16s")?;
        let comp_kv_ptr = comp_kv.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        let key_blocks = n_total_max.div_ceil(256);
        let cfg = LaunchConfig {
            grid: (key_blocks, 1, batch),
            block: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            scores_g.raw(), q.raw(), raw_kv.raw(), comp_kv_ptr,
            n_raw_per.raw(), n_raw_offset_per.raw(), n_comp_per.raw(),
            n_head, head_dim, ATTN_MIXED_MAX_KEYS, kq_scale
        ])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_softmax_wsum_batched_htiled_wmma_f16s(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        scores_g: &mut DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<f32>,
        comp_kv: Option<&DeviceBuffer<f32>>,
        n_raw_per: &DeviceBuffer<i32>,
        n_raw_offset_per: &DeviceBuffer<i32>,
        n_comp_per: &DeviceBuffer<i32>,
        n_head: u32,
        head_dim: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let function = self
            .module
            .get_function("attention_mixed_softmax_wsum_batched_htiled_wmma_f16s")?;
        let comp_kv_ptr = comp_kv.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        let n_head_groups = n_head.div_ceil(SMWSUM_HEAD_TILE);
        let cfg = LaunchConfig {
            grid: (n_head_groups, batch, 1),
            block: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), scores_g.raw(), sinks.raw(), raw_kv.raw(), comp_kv_ptr,
            n_raw_per.raw(), n_raw_offset_per.raw(), n_comp_per.raw(),
            n_head, head_dim, ATTN_MIXED_MAX_KEYS
        ])
    }

    /// LDS-staged V variant. Same WMMA math + same softmax as the f32-
    /// scores baseline, but each K-tile of 16 keys cooperatively stages V
    /// into 16 KB of LDS (f16) once per tile. WMMA B-fragment loads then
    /// read from LDS instead of DRAM. Designed to eliminate the
    /// `s_wait_loadcnt`-on-V-loads stall (82.8% of stall cycles in the
    /// non-LDS WMMA variant per rocprofv3 ATT).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_softmax_wsum_batched_htiled_wmma_ldsv(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        scores_g: &mut DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<f32>,
        comp_kv: Option<&DeviceBuffer<f32>>,
        n_raw_per: &DeviceBuffer<i32>,
        n_raw_offset_per: &DeviceBuffer<i32>,
        n_comp_per: &DeviceBuffer<i32>,
        n_head: u32,
        head_dim: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let function = self
            .module
            .get_function("attention_mixed_softmax_wsum_batched_htiled_wmma_ldsv")?;
        let comp_kv_ptr = comp_kv.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        let n_head_groups = n_head.div_ceil(SMWSUM_HEAD_TILE);
        let cfg = LaunchConfig {
            grid: (n_head_groups, batch, 1),
            block: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), scores_g.raw(), sinks.raw(), raw_kv.raw(), comp_kv_ptr,
            n_raw_per.raw(), n_raw_offset_per.raw(), n_comp_per.raw(),
            n_head, head_dim, ATTN_MIXED_MAX_KEYS
        ])
    }
}
