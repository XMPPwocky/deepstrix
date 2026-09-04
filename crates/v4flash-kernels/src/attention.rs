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

/// `attention_swa_batched`'s raw-key cap (`#define ATTN_SWA_BATCHED_MAX_KV`).
/// Text rows stay at `SWA_WINDOW` (128); Vision-Exp image rows widen the raw
/// window to at most `SWA_WINDOW + vision_max_n_token = 512` keys
/// (`het::image_spans::IMAGE_RAW_WINDOW_MAX`). The kernel's scores/weights
/// arrays are DYNAMIC LDS sized from the caller's `max_n_kv`, so a text-only
/// chunk allocates exactly the 1 KiB the old static `[128]` arrays did and
/// its occupancy is unchanged — only image chunks pay 4 KiB.
pub const ATTN_SWA_BATCHED_MAX_KV: u32 = 512;

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
/// LDS-bound. 49408 covers 128 raw + 49152 comp + 128 chunk-headroom ≈ 192K
/// context at ratio=4 (worst-case layer ratio across [[compress-ratios]] is 4;
/// min layers have 128, but they barely grow). Headroom above 192K so a full
/// chunk (B_MAX tokens, +128 comp rows) prefilled on top of a 192K prefix
/// still fits. Requires DGPU_HOT_EXPERTS≤6 at 192K to leave ~1 GiB dGPU KV
/// margin (K=8 overflows the ~17.1 GiB budget past ~128K). Must match
/// `#define ATTN_MIXED_MAX_KEYS` in `kernels/attention_mixed.hip`.
pub const ATTN_MIXED_MAX_KEYS: u32 = 49408;

/// Stride (in keys) of the `attn_scores` scratch buffer per (batch, head).
/// Smaller than ATTN_MIXED_MAX_KEYS because the production attention path
/// runs after the CSA indexer has gathered the top-K=512 most-relevant
/// comp_kv rows into a dense buffer. So n_total = n_raw + n_keys (≤512)
/// ≈ 640 at any depth on a ratio==4 layer. Set to 2048 to cover:
///   - ratio==4 with CSA: ≤ 640 keys (post-gather)
///   - ratio==4 without CSA (n_index_comp ≤ INDEXER_TOP_K): n_total ≤ 640
///   - ratio==128 (the ungathered worst case): n_raw + n_kv/128.
///
/// THE BINDING CASE IS ratio==128 WITH VISION. A text row's raw window is
/// SWA_WINDOW = 128 keys; a Vision-Exp image row's is widened to at most
/// `het::image_spans::IMAGE_RAW_WINDOW_MAX` = SWA_WINDOW + 384 = 512
/// (a row inside an image block sees the whole block plus its trailing
/// text window). So the budget is
///
///     n_kv_max/128 + IMAGE_RAW_WINDOW_MAX <= ATTN_SCORES_STRIDE
///
/// which at the SHIPPED 192K context is 1536 + 512 = 2048 — exact, zero
/// spare. (The true peak is 508, so 4 keys of real margin.) Raising the
/// context past 192K, or widening the image window, therefore REQUIRES
/// raising this constant; `check_vision_ctx_fits` enforces that at
/// startup so it is a launch-time refusal rather than a mid-prefill 500
/// after part of the chunk's KV was already appended.
///
/// The old comment here said "headroom ok at our actual max ctx of 96K
/// (1024)" and predated both the 192K default and vision.
///
/// At B=512: 64 * 2048 * 2 (f16) = 256 KB/B = 128 MB scratch (was 1.5 GiB).
/// At B=1024: 256 MB. Plenty of room for bigger batches.
pub const ATTN_SCORES_STRIDE: u32 = 2048;

/// Refuse to start when a vision tower is loaded and `n_kv_max` would let
/// an image row's `n_raw + n_comp` overrun [`ATTN_SCORES_STRIDE`].
///
/// Text-only servers are unaffected: without image rows the raw window is
/// SWA_WINDOW and the bound is ~1.5x looser.
pub fn check_vision_ctx_fits(n_kv_max: u32) -> eyre::Result<()> {
    let max_ratio = crate::config::COMPRESS_RATIOS
        .iter()
        .copied()
        .max()
        .unwrap_or(1)
        .max(1);
    let n_comp_max = n_kv_max.div_ceil(max_ratio);
    let need = n_comp_max + crate::het::image_spans::IMAGE_RAW_WINDOW_MAX;
    if need > ATTN_SCORES_STRIDE {
        return Err(eyre!(
            "vision + ctx {n_kv_max} needs {need} attention score slots per (row, head)              (n_comp {n_comp_max} at ratio {max_ratio} + image raw window {}) but              ATTN_SCORES_STRIDE is {ATTN_SCORES_STRIDE}. Lower --ctx to {} or raise              ATTN_SCORES_STRIDE (and the attn_scores scratch with it).",
            crate::het::image_spans::IMAGE_RAW_WINDOW_MAX,
            (ATTN_SCORES_STRIDE - crate::het::image_spans::IMAGE_RAW_WINDOW_MAX) * max_ratio,
        ));
    }
    Ok(())
}

/// Head-group size for the head-tiled WMMA smwsum kernels. Must match
/// `#define SMWSUM_HEAD_TILE` in `kernels/attention_mixed.hip`.
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
        kv: &DeviceBuffer<u16>,
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
        kv: &DeviceBuffer<u16>,
        sinks: &DeviceBuffer<f32>,
        n_raw_per: &DeviceBuffer<i32>,
        n_raw_offset_per: &DeviceBuffer<i32>,
        n_head: u32,
        head_dim: u32,
        batch: u32,
        // Dynamic-LDS stride: must be >= every `n_raw_per[b]` of this
        // launch. Text-only chunks pass `SWA_WINDOW`; image chunks pass the
        // widest window in the chunk (<= `ATTN_SWA_BATCHED_MAX_KV`).
        max_n_kv: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if max_n_kv == 0 || max_n_kv > ATTN_SWA_BATCHED_MAX_KV {
            return Err(eyre!(
                "attention_swa_batched: max_n_kv {max_n_kv} must be in [1, {ATTN_SWA_BATCHED_MAX_KV}]"
            ));
        }
        let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
        let function = self.module.get_function("attention_swa_batched")?;
        let cfg = LaunchConfig {
            grid: (n_head, batch, 1),
            block: (256, 1, 1),
            // scores[max_n_kv] + weights[max_n_kv], f32.
            shared_mem_bytes: 2 * max_n_kv * 4,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), q.raw(), kv.raw(), sinks.raw(),
            n_raw_per.raw(), n_raw_offset_per.raw(),
            n_head, head_dim, max_n_kv, kq_scale
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


    /// Split-kernel decode (perf diagnosis): phase 1 (dot-product scores).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_score(
        &self,
        stream: &Stream,
        scores: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<u16>,
        comp_kv: Option<&DeviceBuffer<u16>>,
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
        raw_kv: &DeviceBuffer<u16>,
        comp_kv: Option<&DeviceBuffer<u16>>,
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

    /// B=1 scalar-arg variant of `launch_score_batched_htiled_wmma` —
    /// f32 scores, drop-in compatible with the existing f32-reading
    /// `launch_softmax_wsum`. Decode-path scoring.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_score_b1_htiled_wmma(
        &self,
        stream: &Stream,
        scores_g: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<u16>,
        comp_kv: Option<&DeviceBuffer<u16>>,
        n_raw: u32,
        raw_off: u32,
        n_comp: u32,
        n_head: u32,
        head_dim: u32,
        n_total_max: u32,
    ) -> eyre::Result<()> {
        if n_total_max == 0 {
            return Ok(());
        }
        if n_total_max > ATTN_MIXED_MAX_KEYS {
            return Err(eyre!(
                "launch_score_b1_htiled_wmma: n_total_max={n_total_max} exceeds cap {ATTN_MIXED_MAX_KEYS}"
            ));
        }
        let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
        let function = self
            .module
            .get_function("attention_mixed_score_b1_htiled_wmma")?;
        let cfg = LaunchConfig {
            grid: ((n_total_max + 255) / 256, 1, 1),
            block: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        let comp_kv_ptr = comp_kv.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        launch_kernel!(function, cfg, stream, [
            scores_g.raw(), q.raw(), raw_kv.raw(), comp_kv_ptr,
            n_raw, raw_off, n_comp, n_head, head_dim, ATTN_MIXED_MAX_KEYS, kq_scale
        ])
    }

    /// B=1 scalar-arg variant of `launch_score_batched_htiled_wmma_f16s`.
    /// Writes f16 scores; takes scalar n_raw/raw_off/n_comp instead of
    /// per-batch device buffers — used in the decode path so we don't pay
    /// 86 copy_from_host calls per token to stamp the buffer-indexed
    /// variant's counters. Grid (ceil(n_total_max/256), 1, 1), block 512.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_score_b1_htiled_wmma_f16s(
        &self,
        stream: &Stream,
        scores_g: &mut DeviceBuffer<f32>,    // type-aliased to f16 — buffer sized for f32 holds 2× f16
        q: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<u16>,
        comp_kv: Option<&DeviceBuffer<u16>>,
        n_raw: u32,
        raw_off: u32,
        n_comp: u32,
        n_head: u32,
        head_dim: u32,
        n_total_max: u32,
    ) -> eyre::Result<()> {
        if n_total_max == 0 {
            return Ok(());
        }
        if n_total_max > ATTN_MIXED_MAX_KEYS {
            return Err(eyre!(
                "launch_score_b1_htiled_wmma_f16s: n_total_max={n_total_max} exceeds cap {ATTN_MIXED_MAX_KEYS}"
            ));
        }
        let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
        let function = self
            .module
            .get_function("attention_mixed_score_b1_htiled_wmma_f16s")?;
        // SCORE_WMMA_KEYS_PER_BLK = 256 (16 warps × 16 keys/warp).
        let cfg = LaunchConfig {
            grid: ((n_total_max + 255) / 256, 1, 1),
            block: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        let comp_kv_ptr = comp_kv.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        launch_kernel!(function, cfg, stream, [
            scores_g.raw(), q.raw(), raw_kv.raw(), comp_kv_ptr,
            n_raw, raw_off, n_comp, n_head, head_dim, ATTN_MIXED_MAX_KEYS, kq_scale
        ])
    }

    /// Decode-attention K-split smwsum pipeline, pass 1: softmax_only.
    /// Per-head softmax across all keys, writes weights in place to scores
    /// buffer, writes per-head inv = 1/sum to `inv_per_head` for pass 3.
    /// Grid (n_head, 1, 1), block 32 (wave 0 only).
    pub fn launch_softmax_only(
        &self,
        stream: &Stream,
        scores: &mut DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        inv_per_head: &mut DeviceBuffer<f32>,
        n_head: u32,
        n_raw: u32,
        n_comp: u32,
    ) -> eyre::Result<()> {
        let function = self.module.get_function("attention_mixed_softmax_only")?;
        let cfg = LaunchConfig { grid: (n_head, 1, 1), block: (32, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [
            scores.raw(), sinks.raw(), inv_per_head.raw(),
            n_head, n_raw, n_comp, ATTN_MIXED_MAX_KEYS
        ])
    }

    /// Decode K-split smwsum pass 2: head-tiled (16 heads/WG) WMMA wsum,
    /// K-split across WGs. Writes per-chunk partials [k_split, n_head,
    /// head_dim] to `partials`. Requires head_dim == 512, n_head % 16 == 0.
    /// Caller must run `launch_softmax_only` first to populate weights in
    /// `scores` and inv values.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_wsum_b1_htiled_ksplit_ldsv(
        &self,
        stream: &Stream,
        partials: &mut DeviceBuffer<f32>,
        scores: &DeviceBuffer<f32>,         // post-softmax weights, unscaled
        raw_kv: &DeviceBuffer<u16>,
        comp_kv: Option<&DeviceBuffer<u16>>,
        n_head: u32,
        head_dim: u32,
        n_raw: u32,
        n_comp: u32,
        k_split: u32,
    ) -> eyre::Result<()> {
        if head_dim != 512 || n_head % 16 != 0 {
            return Err(eyre!(
                "launch_wsum_b1_htiled_ksplit_ldsv: requires head_dim=512 and n_head%16==0 (got {head_dim}, {n_head})"
            ));
        }
        let function = self
            .module
            .get_function("attention_mixed_wsum_b1_htiled_ksplit_ldsv")?;
        let h_tiles = n_head / 16;
        let cfg = LaunchConfig {
            grid: (h_tiles, k_split, 1),
            block: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        let comp_kv_ptr = comp_kv.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        launch_kernel!(function, cfg, stream, [
            partials.raw(), scores.raw(), raw_kv.raw(), comp_kv_ptr,
            n_raw, n_comp, n_head, head_dim, ATTN_MIXED_MAX_KEYS, k_split
        ])
    }

    /// Decode K-split smwsum pass 3: reduce k_split partials per (h, d)
    /// and apply inv[h]. Writes final out [n_head, head_dim].
    pub fn launch_reduce_partials_apply_inv(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        partials: &DeviceBuffer<f32>,
        inv_per_head: &DeviceBuffer<f32>,
        n_head: u32,
        head_dim: u32,
        k_split: u32,
    ) -> eyre::Result<()> {
        let function = self
            .module
            .get_function("attention_mixed_reduce_partials_apply_inv")?;
        let total = (n_head as usize) * (head_dim as usize);
        let block: u32 = 256;
        let grid: u32 = ((total + (block as usize) - 1) / (block as usize)) as u32;
        let cfg = LaunchConfig { grid: (grid, 1, 1), block: (block, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [
            out.raw(), partials.raw(), inv_per_head.raw(),
            n_head, head_dim, k_split
        ])
    }

    /// LDS-V variant of `launch_softmax_wsum` — same semantics, V tile is
    /// cooperatively staged to LDS once per K-tile then per-d reads come
    /// from LDS. Targets long-ctx decode where the per-K-tile DRAM reads
    /// of V dominate (~540 µs per dispatch at ratio=4 n_comp=16384).
    /// Requires head_dim == 512 (the MLA latent width).
    pub fn launch_softmax_wsum_ldsv(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        scores: &mut DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<u16>,
        comp_kv: Option<&DeviceBuffer<u16>>,
        n_head: u32,
        head_dim: u32,
        n_raw: u32,
        n_comp: u32,
    ) -> eyre::Result<()> {
        if head_dim != 512 {
            return Err(eyre!(
                "launch_softmax_wsum_ldsv requires head_dim==512, got {head_dim}"
            ));
        }
        let function = self.module.get_function("attention_mixed_softmax_wsum_ldsv")?;
        let comp_kv_ptr = comp_kv.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        let cfg = LaunchConfig { grid: (n_head, 1, 1), block: (512, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [
            out.raw(), scores.raw(), sinks.raw(), raw_kv.raw(), comp_kv_ptr,
            n_head, head_dim, n_raw, n_comp, ATTN_MIXED_MAX_KEYS
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
        raw_kv: &DeviceBuffer<u16>,
        comp_kv: Option<&DeviceBuffer<u16>>,
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
        raw_kv: &DeviceBuffer<u16>,
        comp_kv: Option<&DeviceBuffer<u16>>,
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
    /// Mask-aware variant of [`Self::launch_score_batched_htiled_wmma_f16s`].
    /// `comp_allowed_bits` is bitpacked `[B, max_keys_words]` u32 where
    /// `max_keys_words = ceil(ATTN_MIXED_MAX_KEYS / 32)`. When `None`, the
    /// kernel skips the bit test (bit-exact identical output to the
    /// pre-mask version). When `Some(_)`, masked comp rows are stamped as
    /// f16 -INFINITY in the score buffer — softmax converts to zero
    /// weight downstream.
    ///
    /// `comp_kv_batch_stride` (rows): `0` = legacy shared comp_kv (all batches
    /// read the same row 0..n_comp). `>0` = per-batch comp_kv (batch b reads
    /// rows starting at `b * comp_kv_batch_stride`). Pairs with the CSA
    /// gather path where `comp_kv` is `active_comp_kv[B, top_k, head_dim]`
    /// and `comp_kv_batch_stride = top_k` — score kernel then reads only the
    /// gathered top-K dense rows per batch instead of doing per-row mask
    /// tests on the full sparse set.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_score_batched_htiled_wmma_f16s(
        &self,
        stream: &Stream,
        scores_g: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<u16>,
        comp_kv: Option<&DeviceBuffer<u16>>,
        n_raw_per: &DeviceBuffer<i32>,
        n_raw_offset_per: &DeviceBuffer<i32>,
        n_comp_per: &DeviceBuffer<i32>,
        comp_allowed_bits: Option<&DeviceBuffer<u32>>,
        n_head: u32,
        head_dim: u32,
        n_total_max: u32,
        batch: u32,
        comp_kv_batch_stride: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_total_max == 0 {
            return Ok(());
        }
        // BUFFER bound: scratch is sized at ATTN_SCORES_STRIDE per (b,head).
        // The user-facing cap stays ATTN_MIXED_MAX_KEYS but the production
        // CSA gather path never approaches it (post-gather n_total ≤ ~640).
        if n_total_max > ATTN_SCORES_STRIDE {
            return Err(eyre!(
                "attention_mixed_score_batched_htiled_wmma_f16s: n_total_max={n_total_max} exceeds scratch stride {ATTN_SCORES_STRIDE}"
            ));
        }
        let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
        let function = self
            .module
            .get_function("attention_mixed_score_batched_htiled_wmma_f16s")?;
        let comp_kv_ptr = comp_kv.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        let mask_ptr = comp_allowed_bits
            .map(|b| b.raw())
            .unwrap_or(std::ptr::null_mut());
        // Mask word stride is tied to ATTN_MIXED_MAX_KEYS regardless of
        // caller: decode passes `DgpuScratch::indexer_allowed_bits`
        // (sized ceil(ATTN_MIXED_MAX_KEYS/32) words); batched prefill
        // passes no mask (null).
        let max_keys_words: u32 = (ATTN_MIXED_MAX_KEYS + 31) / 32;
        let key_blocks = n_total_max.div_ceil(256);
        let cfg = LaunchConfig {
            grid: (key_blocks, 1, batch),
            block: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            scores_g.raw(), q.raw(), raw_kv.raw(), comp_kv_ptr,
            n_raw_per.raw(), n_raw_offset_per.raw(), n_comp_per.raw(),
            mask_ptr, max_keys_words,
            n_head, head_dim, ATTN_SCORES_STRIDE, kq_scale,
            comp_kv_batch_stride
        ])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_softmax_wsum_batched_htiled_wmma_f16s(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        scores_g: &mut DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<u16>,
        comp_kv: Option<&DeviceBuffer<u16>>,
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
        raw_kv: &DeviceBuffer<u16>,
        comp_kv: Option<&DeviceBuffer<u16>>,
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

    /// Combined LDS-V + f16-scores smwsum. Pairs with
    /// `launch_score_batched_htiled_wmma_f16s` (writes f16 scores). Cuts
    /// the Phase A score-DRAM stall (~28% of stall pre-fix) by halving
    /// scores buffer bytes.
    ///
    /// `comp_kv_batch_stride` (rows): same semantics as the score launcher.
    /// `0` = legacy shared comp_kv; `>0` = per-batch comp_kv at offset
    /// `b * comp_kv_batch_stride * head_dim`. Pairs with CSA gather.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        // f32 buffer reinterpreted as f16 by the kernel (kernel only writes
        // n_head × max_keys × 2 bytes; the f32 buffer is 2× oversized).
        scores_g: &mut DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<u16>,
        comp_kv: Option<&DeviceBuffer<u16>>,
        n_raw_per: &DeviceBuffer<i32>,
        n_raw_offset_per: &DeviceBuffer<i32>,
        n_comp_per: &DeviceBuffer<i32>,
        n_head: u32,
        head_dim: u32,
        batch: u32,
        comp_kv_batch_stride: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let function = self
            .module
            .get_function("attention_mixed_softmax_wsum_batched_htiled_wmma_ldsv_f16s")?;
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
            n_head, head_dim, ATTN_SCORES_STRIDE, comp_kv_batch_stride
        ])
    }

    /// Double-buffered LDS-V variant of the WMMA smwsum. Two ping-pong
    /// 16 KB LDS V tiles let stage_v(tile N+1) issue DRAM loads while
    /// WMMA(tile N) runs, halving the per-iter barrier count.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_softmax_wsum_batched_htiled_wmma_ldsv_db(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        scores_g: &mut DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<u16>,
        comp_kv: Option<&DeviceBuffer<u16>>,
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
            .get_function("attention_mixed_softmax_wsum_batched_htiled_wmma_ldsv_db")?;
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

    /// Register-V double-buffered LDS-V smwsum variant. V lives entirely
    /// in VGPRs (per-warp B-fragment slice loaded directly from DRAM);
    /// only s_inv (64 B) stays in LDS. Wave-occupancy-limited at 100% vs
    /// 75% for the LDS-V variants, and zero per-tile barriers.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_softmax_wsum_batched_htiled_wmma_regv_db(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        scores_g: &mut DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<u16>,
        comp_kv: Option<&DeviceBuffer<u16>>,
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
            .get_function("attention_mixed_softmax_wsum_batched_htiled_wmma_regv_db")?;
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

    /// Fused FlashAttention-style WMMA kernel. Replaces (score → smwsum)
    /// with one streaming kernel that keeps Q in LDS, stages V to LDS per
    /// K-tile, and runs an online softmax in f32 registers — never writes
    /// scores to DRAM. Designed to take ~6-8 ms at depth 32k vs the
    /// current ~22 ms split chain.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_wmma(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>,
        sinks: &DeviceBuffer<f32>,
        raw_kv: &DeviceBuffer<u16>,
        comp_kv: Option<&DeviceBuffer<u16>>,
        n_raw_per: &DeviceBuffer<i32>,
        n_raw_offset_per: &DeviceBuffer<i32>,
        n_comp_per: &DeviceBuffer<i32>,
        n_head: u32,
        head_dim: u32,
        n_total_max: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
        let function = self.module.get_function("attention_mixed_fused_wmma")?;
        let comp_kv_ptr = comp_kv.map(|b| b.raw()).unwrap_or(std::ptr::null_mut());
        let n_head_groups = n_head.div_ceil(SMWSUM_HEAD_TILE);
        let cfg = LaunchConfig {
            grid: (n_head_groups, batch, 1),
            block: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), q.raw(),
            raw_kv.raw(), comp_kv_ptr, sinks.raw(),
            n_raw_per.raw(), n_raw_offset_per.raw(), n_comp_per.raw(),
            n_head, head_dim, n_total_max, kq_scale
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped configuration (192K ctx + a vision tower) must start,
    /// and it must sit exactly at the edge — this is the test that fails
    /// first if someone raises the context without raising the stride.
    #[test]
    fn vision_ctx_budget_is_exact_at_192k() {
        let ctx_192k = 192 * 1024;
        check_vision_ctx_fits(ctx_192k).expect("192K + vision must fit");
        // One comp row more (ratio 128 => 128 tokens) does not.
        assert!(check_vision_ctx_fits(ctx_192k + 128).is_err());
        // Text-only servers were never near the bound.
        check_vision_ctx_fits(96 * 1024).unwrap();
    }

    #[test]
    fn vision_ctx_error_names_a_workable_ctx() {
        let e = check_vision_ctx_fits(512 * 1024).unwrap_err().to_string();
        assert!(e.contains("ATTN_SCORES_STRIDE"), "{e}");
        // The suggested ctx must itself pass.
        check_vision_ctx_fits(
            (ATTN_SCORES_STRIDE - crate::het::image_spans::IMAGE_RAW_WINDOW_MAX) * 128,
        )
        .unwrap();
    }
}
