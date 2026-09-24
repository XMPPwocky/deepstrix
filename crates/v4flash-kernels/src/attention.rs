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

/// Largest `--ctx` V4.1 can serve. The INDEXER scores its whole compressed
/// store densely to PRODUCE the top-k, and layers 20-39 are ratio 1, so the
/// widest store is one row per token: this cap is a context limit 1:1.
///
/// RAISED 2026-09-22, 307_200 -> 368_640 (360K). A client hit
/// `context_length_exceeded: prompt length 320795 >= 307200` and could not
/// recover: auto-compaction sends the WHOLE conversation plus a summarisation
/// wrapper, so once a transcript passes the window the request that would
/// shrink it no longer fits. Headroom above the transcript is what breaks that
/// deadlock. Cost, measured on the live box (1,027 MB dGPU free of 17,095):
/// ~45 MB -- `attn_scores` and `verify_scores` are each N_HEAD(64) x this x f32,
/// so +15.7 MB apiece, the per-lane batch scores ~+12 MB, and the indexer-side
/// buffers +0.3 MB.
///
/// NOTE the asymmetry, worth fixing separately: only the INDEXER-side buffers
/// genuinely scale with context (~1.4 MB total). `attn_scores` is 79 MB because
/// it is sized by this constant, yet with the indexer on attention only ever
/// scores `raw_window + INDEXER_TOP_K` = 640 keys -- 0.29 MB. Splitting this
/// into an indexer ceiling and an attention-scores ceiling would free ~150 MB
/// and make future `--ctx` raises cost ~1.4 MB instead of ~45. The trap is that
/// `scored_keys_are_gathered` reads `V41_INDEX_K` at RUNTIME, which is the
/// env-dependence that caused the truncation bug documented below; any split
/// must resolve the mode at startup and hard-error if it changes.
pub const V41_MAX_CTX: u32 = 368_640;

/// RAISED 2026-09-18 from `131_072 + SWA_WINDOW`, closing a SILENT TRUNCATION.
///
/// [`scored_keys_are_gathered`] reads `V41_INDEX_K` at RUNTIME, so with the
/// indexer on [`attn_max_scored_keys`] collapsed to `SWA_WINDOW +
/// INDEXER_TOP_K` = 640 and the server admitted `--ctx 307200` — while every
/// comp-indexed scratch buffer was still sized for 131_200. `n_index_comp` was
/// then `.min()`ed to the cap, so past 131_200 tokens the indexer scored only
/// the FIRST 131_200 compressed positions and every later token was invisible
/// to all 38 compressed layers, reachable only through the 128-token raw
/// window. No test caught it because the env var is unset under `cargo test`,
/// which is exactly the regime where the old bound is correct.
///
/// Two things prevent a recurrence: the admission check now derives from
/// [`indexer_max_scored_keys`], which has NO gathered shortcut and no env
/// dependence, and the truncating `.min()`s are hard errors.
pub const ATTN_MIXED_MAX_KEYS: u32 =
    V41_MAX_CTX + crate::het::image_spans::IMAGE_RAW_WINDOW_MAX;

/// FLOOR stride (in keys) of the `attn_scores` scratch buffer per
/// (batch, head). The stride actually used is per-launch — see
/// [`attn_scores_stride`] — and is never below this value, so V4-Flash's
/// production layout (3072) is unchanged.
///
/// Why 3072 was enough for V4-Flash and is NOT a model-independent number:
/// the production V4-Flash attention path runs after the CSA indexer has
/// gathered the top-K=512 most relevant comp_kv rows into a dense buffer, so
/// a ratio-4 layer scores n_raw + <=512 keys at any depth. Only the
/// *ungathered* layers (ratio 128) grow with context, at n_kv/128. With
/// vision the raw window widens to `het::image_spans::IMAGE_RAW_WINDOW_MAX`
/// = 512, giving the historical budget
///
///     n_kv_max/128 + IMAGE_RAW_WINDOW_MAX <= ATTN_SCORES_STRIDE
///
/// -> 3072 covers 320K with vision and ~377K text-only.
///
/// V4.1 breaks every term of that: no indexer is ported (ENGINE_PORT M5), so
/// NO layer is gathered, and `COMPRESS_RATIOS` are 2 (layers 2-19) and 1
/// (layers 20-39). The worst-case ungathered contribution is therefore
/// n_kv/1, not n_kv/128 — 3072 capped the CED decoder replay at 2944 prompt
/// tokens and the ratio-2 encoder at 5888. [`attn_max_scored_keys`] is the
/// model-independent derivation; `BatchDgpuShared::alloc_rows_ctx` sizes the
/// scratch from it.
///
/// Cost: 64 heads x stride x 2 B (f16 scores) per scratch row.
/// At rows=512: 64 * 3072 * 2 = 384 KiB/row = 201 MiB.
pub const ATTN_SCORES_STRIDE: u32 = 3072;

/// Does the CSA indexer gather layers of this compress ratio down to a dense
/// `INDEXER_TOP_K` buffer before attention scores them?
///
/// Only V4-Flash's ratio-4 layers. This is the V4-FLASH predicate and stays that
/// way; for "is this layer's score count bounded by the top-k", which is what
/// sizing wants and which IS true for V4.1 under S2, use
/// [`scored_keys_are_gathered`]. (Corrected 2026-09-18: the note here used to say
/// V4.1's indexer was unported and every V4.1 layer scored densely. S1+S2 landed;
/// it does not.)
pub fn indexer_gathers(_ratio: u32) -> bool {
    // V4-Flash's ratio-4 compressor indexer only; V4.1 has no ratio-4 layer.
    false
}

/// `V41_INDEX_K=1` — mirror of `het::forward_layer::index_k_enabled`, needed here
/// because SIZING has to agree with the runtime gate.
fn v41_index_k_on() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        matches!(std::env::var("V41_INDEX_K").as_deref(), Ok("1") | Ok("on"))
    });
    *B
}

/// SIZING predicate: is every compressed layer's score count bounded by
/// `INDEXER_TOP_K` rather than by its whole store?
///
/// True for V4.1 with the indexer on. S2 is implemented: index sources gather to
/// top-k and reuse layers take that same selection (`s2_reuse` in
/// `het::forward_layer` / `het::forward_prefill`), matching the reference
/// `inference/model.py`, whose `_compress_topk_idxs` returns
/// `shared_attn.topk_idxs` for every non-source layer and which has NO dense
/// attention path at all.
///
/// Dense still runs below the threshold, where the reference itself takes
/// `topk = min(index_topk, end_pos // ratio)` — so at most `INDEXER_TOP_K` keys
/// are scored either way, and this bound holds for both paths.
///
/// Keep in lockstep with the `use_sparse` / `s2_reuse` gates.
pub fn scored_keys_are_gathered(ratio: u32) -> bool {
    ratio > 0 && v41_index_k_on()
}

/// Can the CSA indexer fire on ANY layer of this model? False for V4.1
/// (unported indexer), which makes every per-token indexer scratch buffer
/// dead weight — see `het::batch_scratch::indexer_scratch_keys`.
/// Can the indexer fire in THIS PROCESS? `indexer_ever_fires()` is a static property
/// of the model (V4-Flash ratio-4 only); this additionally returns true when V4.1's
/// ported indexer is switched on with `V41_INDEX_K=1`, which is what decides whether the
/// per-token indexer SCRATCH must be allocated.
///
/// STALE, corrected 2026-09-18: S2 landed, so the reuse layers do NOT score their
/// whole store — they take their index source's selection (`s2_reuse`). Sizing now
/// goes through [`scored_keys_are_gathered`], which is true for every compressed
/// V4.1 layer when `V41_INDEX_K=1`, so the scores-scratch and the `--ctx` cap are
/// bounded by `raw_window + INDEXER_TOP_K` rather than `n_kv_max / ratio`. This
/// predicate stays separate because it answers a different question: whether the
/// per-token indexer SCRATCH must exist at all.
pub fn indexer_scratch_needed() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        indexer_ever_fires() || matches!(std::env::var("V41_INDEX_K").as_deref(), Ok("1") | Ok("on"))
    });
    *B
}

pub fn indexer_ever_fires() -> bool {
    crate::config::COMPRESS_RATIOS.iter().any(|&r| indexer_gathers(r))
}

/// Worst-case `n_raw + n_comp` any single (row, head) can be asked to score
/// at `n_kv_max` context, over every layer of THIS model.
///
/// This is the one derivation the three context caps
/// (`ATTN_MIXED_MAX_KEYS`, the `attn_scores` scratch, and the server's
/// `--ctx` admission check) must all come from. A gathered layer contributes
/// at most `INDEXER_TOP_K`; an ungathered one contributes its full store,
/// `ceil(n_kv_max / ratio)`. Dense layers (ratio 0) have no compressed store.
///
/// `raw_window` is `SWA_WINDOW` for a text-only engine and
/// `het::image_spans::IMAGE_RAW_WINDOW_MAX` when a vision tower is loaded.
///
/// Sanity: V4-Flash at 320K text-only -> 128 + max(512, 320K/128 = 2560)
/// = 2688, under the historical 3072. V4.1 at 100K -> 128 + max(50000,
/// 100000) = 100128, i.e. the store itself.
pub fn attn_max_scored_keys(n_kv_max: u32, raw_window: u32) -> u32 {
    let mut worst = 0u32;
    for &ratio in crate::config::COMPRESS_RATIOS.iter() {
        if ratio == 0 {
            continue;
        }
        let n_comp = n_kv_max.div_ceil(ratio);
        let scored = if scored_keys_are_gathered(ratio) {
            n_comp.min(crate::config::INDEXER_TOP_K)
        } else {
            n_comp
        };
        worst = worst.max(scored);
    }
    raw_window.saturating_add(worst)
}

/// Largest context [`attn_max_scored_keys`] keeps within `keys`, i.e. the
/// inverse of the function above. Used for the "lower --ctx to N" half of
/// every admission error.
pub fn attn_max_ctx_for_keys(keys: u32, raw_window: u32) -> u32 {
    let budget = keys.saturating_sub(raw_window);
    let mut min_ratio = u32::MAX;
    for &ratio in crate::config::COMPRESS_RATIOS.iter() {
        if ratio == 0 || scored_keys_are_gathered(ratio) {
            continue;
        }
        min_ratio = min_ratio.min(ratio);
    }
    if min_ratio == u32::MAX {
        // Every compressed layer is gathered: context is unbounded by this
        // budget as long as the gather itself fits.
        return u32::MAX;
    }
    budget.saturating_mul(min_ratio)
}

/// Worst-case compressed positions the INDEXER scores DENSELY at `n_kv_max`,
/// over every layer of this model.
///
/// Distinct from [`attn_max_scored_keys`], and the distinction is the whole
/// point: that function bounds what ATTENTION scores AFTER the gather, so a
/// gathered layer contributes only `INDEXER_TOP_K`. The indexer has to rank the
/// WHOLE store to produce that top-k, so its bound does not shrink when a layer
/// is gathered — and unlike the attention bound it must not depend on a runtime
/// flag, because the buffers it sizes are allocated once.
///
/// This is the quantity `indexer_scores`, `indexer_allowed_bits`,
/// `indexer_topk_scratch` and `candidate_block_score` are all strided by.
pub fn indexer_max_scored_keys(n_kv_max: u32, raw_window: u32) -> u32 {
    let mut worst = 0u32;
    for &ratio in crate::config::COMPRESS_RATIOS.iter() {
        if ratio == 0 {
            continue;
        }
        worst = worst.max(n_kv_max.div_ceil(ratio));
    }
    raw_window.saturating_add(worst)
}

/// Inverse of [`indexer_max_scored_keys`]: the largest context whose dense
/// indexer scoring fits in `keys`.
pub fn indexer_max_ctx_for_keys(keys: u32, raw_window: u32) -> u32 {
    let budget = keys.saturating_sub(raw_window);
    let min_ratio = crate::config::COMPRESS_RATIOS
        .iter()
        .copied()
        .filter(|&r| r > 0)
        .min()
        .unwrap_or(1);
    budget.saturating_mul(min_ratio)
}

/// `DEEPSTRIX_ATTN_LEGACY_STRIDE=1` restores the pre-2026-09-13 behaviour:
/// a fixed [`ATTN_SCORES_STRIDE`] and a hard error past it, plus legacy
/// scratch sizing. Rollback and before/after demonstration only.
pub fn attn_legacy_stride() -> bool {
    std::env::var("DEEPSTRIX_ATTN_LEGACY_STRIDE").map(|v| v != "0").unwrap_or(false)
}

/// Stride (keys per (row, head)) for one batched score + softmax-wsum pair.
///
/// The two kernels MUST be given the same value. `capacity_keys` is how many
/// score slots the caller's `attn_scores` buffer holds in total (f16 slots
/// for the `_f16s` pair, f32 slots otherwise).
///
/// Returns [`ATTN_SCORES_STRIDE`] whenever that is both sufficient and
/// affordable, so V4-Flash and short-context V4.1 keep today's exact layout;
/// otherwise the smallest stride that holds `n_total_max`.
pub fn attn_scores_stride(
    capacity_keys: usize,
    batch: u32,
    n_head: u32,
    n_total_max: u32,
) -> eyre::Result<u32> {
    let rows = (batch as usize) * (n_head as usize);
    if rows == 0 {
        return Ok(ATTN_SCORES_STRIDE);
    }
    // ROLLBACK / demonstration knob: pin the stride to the old compile-time
    // constant and refuse anything past it, exactly as the code did before
    // 2026-09-13. `DEEPSTRIX_ATTN_LEGACY_STRIDE=1` is the switch that turns
    // "a 3000-token V4.1 prompt" back into a hard error.
    if attn_legacy_stride() {
        if n_total_max > ATTN_SCORES_STRIDE {
            return Err(eyre!(
                "attention scores: n_total_max={n_total_max} exceeds scratch stride \
                 {ATTN_SCORES_STRIDE} (DEEPSTRIX_ATTN_LEGACY_STRIDE=1)"
            ));
        }
        return Ok(ATTN_SCORES_STRIDE);
    }
    let fits = capacity_keys / rows;
    if (n_total_max as usize) > fits {
        return Err(eyre!(
            "attention scores scratch holds {fits} keys per (row, head) at batch {batch} \
             (capacity {capacity_keys} keys) but this layer needs {n_total_max}. The scratch is \
             sized by `BatchDgpuShared::alloc_rows_ctx(rows, n_kv_max)`, which charges the CED \
             decoder layers only the bounded replay's row count — a NON-CED prefill \
             (V41_CED=0, or per-token logits) runs those layers over full chunks and needs 8x \
             more. Lower --ctx, use CED, or size the shared set for the batch you run."
        ));
    }
    if n_total_max <= ATTN_SCORES_STRIDE && (ATTN_SCORES_STRIDE as usize) <= fits {
        return Ok(ATTN_SCORES_STRIDE);
    }
    Ok(n_total_max.max(1))
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
        scores_stride: u32,
    ) -> eyre::Result<()> {
        if batch == 0 || n_total_max == 0 {
            return Ok(());
        }
        if n_total_max > scores_stride {
            return Err(eyre!(
                "attention_mixed_score_batched_htiled_wmma: n_total_max={n_total_max} exceeds scores stride {scores_stride}"
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
            n_head, head_dim, scores_stride, kq_scale
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
        scores_stride: u32,
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
            n_head, head_dim, scores_stride
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
    /// Single-sequence form of [`Self::launch_score_batched_htiled_wmma_f16s_rows`] (per-row base = none).
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
        scores_stride: u32,
    ) -> eyre::Result<()> {
        self.launch_score_batched_htiled_wmma_f16s_rows(stream, scores_g, q, raw_kv, comp_kv, n_raw_per, n_raw_offset_per, n_comp_per, comp_allowed_bits, n_head, head_dim, n_total_max, batch, comp_kv_batch_stride, scores_stride, None)
    }

    pub fn launch_score_batched_htiled_wmma_f16s_rows(
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
        scores_stride: u32,
        comp_base_per: Option<&DeviceBuffer<i32>>,
    ) -> eyre::Result<()> {
        let comp_base_per_ptr = comp_base_per.map(|b| b.raw() as *const i32).unwrap_or(std::ptr::null());
        if batch == 0 || n_total_max == 0 {
            return Ok(());
        }
        // BUFFER bound: the caller derives `scores_stride` from the actual
        // capacity of `scores_g` at this batch (`attn_scores_stride`), so this
        // is a real bounds check, not a model-shaped guess.
        if n_total_max > scores_stride {
            return Err(eyre!(
                "attention_mixed_score_batched_htiled_wmma_f16s: n_total_max={n_total_max} exceeds scores stride {scores_stride}"
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
            n_head, head_dim, scores_stride, kq_scale,
            comp_kv_batch_stride, comp_base_per_ptr
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
        scores_stride: u32,
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
            n_head, head_dim, scores_stride
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
        scores_stride: u32,
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
            n_head, head_dim, scores_stride
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
    /// Single-sequence form of [`Self::launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s_rows`] (per-row base = none).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s(
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
        comp_kv_batch_stride: u32,
        scores_stride: u32,
    ) -> eyre::Result<()> {
        self.launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s_rows(stream, out, scores_g, sinks, raw_kv, comp_kv, n_raw_per, n_raw_offset_per, n_comp_per, n_head, head_dim, batch, comp_kv_batch_stride, scores_stride, None)
    }

    pub fn launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s_rows(
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
        scores_stride: u32,
        comp_base_per: Option<&DeviceBuffer<i32>>,
    ) -> eyre::Result<()> {
        let comp_base_per_ptr = comp_base_per.map(|b| b.raw() as *const i32).unwrap_or(std::ptr::null());
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
            n_head, head_dim, scores_stride, comp_kv_batch_stride, comp_base_per_ptr
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
        scores_stride: u32,
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
            n_head, head_dim, scores_stride
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
        scores_stride: u32,
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
            n_head, head_dim, scores_stride
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

    /// The three context caps must all come from `attn_max_scored_keys`.
    /// This is the regression test for the 2026-09-13 finding: the caps were
    /// derived assuming compression ratio >= 4, which V4.1 does not have.
    #[test]
    fn max_scored_keys_uses_the_minimum_ungathered_ratio() {
        let w = crate::config::SWA_WINDOW;
        let min_ungathered = crate::config::COMPRESS_RATIOS
            .iter()
            .copied()
            .filter(|&r| r > 0 && !indexer_gathers(r))
            .min()
            .expect("every model has at least one ungathered compressed layer");
        // 64K of context: the worst layer scores the whole of its own store.
        let keys = attn_max_scored_keys(65536, w);
        let expect = w + (65536u32.div_ceil(min_ungathered))
            .max(if crate::config::COMPRESS_RATIOS.iter().any(|&r| indexer_gathers(r)) {
                crate::config::INDEXER_TOP_K
            } else {
                0
            });
        assert_eq!(keys, expect, "min ungathered ratio {min_ungathered}");
        // Round-trips through the inverse.
        let ctx = attn_max_ctx_for_keys(keys, w);
        assert!(ctx >= 65536, "inverse lost context: {ctx}");
        assert!(attn_max_scored_keys(ctx, w) <= keys);
    }

    /// REGRESSION for the 2026-09-18 silent truncation.
    ///
    /// `attn_max_scored_keys` reaches `scored_keys_are_gathered`, which reads
    /// `V41_INDEX_K` at RUNTIME — so it answers one thing under `cargo test`
    /// (var unset) and another in production (`=1`), and the production answer
    /// admitted a `--ctx` 2.3x wider than the buffers. That is why every
    /// existing cap test passed while production truncated: they all went
    /// through the env-dependent function. This one does not.
    #[test]
    fn indexer_bound_does_not_depend_on_the_env() {
        let w = crate::config::SWA_WINDOW;
        let min_ratio = crate::config::COMPRESS_RATIOS
            .iter()
            .copied()
            .filter(|&r| r > 0)
            .min()
            .expect("model has a compressed layer");
        // No gathered shortcut: the indexer ranks the WHOLE store.
        assert_eq!(
            indexer_max_scored_keys(65536, w),
            w + 65536u32.div_ceil(min_ratio)
        );
        // Whatever the cap is, the inverse round-trips inside it.
        let ctx = indexer_max_ctx_for_keys(ATTN_MIXED_MAX_KEYS, w);
        assert!(indexer_max_scored_keys(ctx, w) <= ATTN_MIXED_MAX_KEYS);
    }

    /// The cap must cover the context we actually ship, computed through the
    /// env-independent bound.
    #[test]
    fn cap_covers_the_shipped_ctx() {
        // The WIDEST raw window, not the text-only one: `engine_worker` passes
        // `IMAGE_RAW_WINDOW_MAX` whenever `--mmproj` is set, which production
        // does. Sizing this test off `SWA_WINDOW` made it pass against a cap
        // that was 384 keys short, and the server refused to start.
        let w = crate::het::image_spans::IMAGE_RAW_WINDOW_MAX;
        let need = indexer_max_scored_keys(V41_MAX_CTX, w);
        assert!(
            need <= ATTN_MIXED_MAX_KEYS,
            "--ctx {V41_MAX_CTX} needs {need} indexer keys, cap is {ATTN_MIXED_MAX_KEYS}"
        );
    }

    /// The decode cap must cover the context the server is allowed to accept.
    #[test]
    fn decode_cap_admits_a_real_context() {
        let w = crate::config::SWA_WINDOW;
        let ctx = attn_max_ctx_for_keys(ATTN_MIXED_MAX_KEYS, w);
        assert!(ctx >= 8192, "decode cap only reaches {ctx} tokens");
        assert!(attn_max_scored_keys(ctx, w) <= ATTN_MIXED_MAX_KEYS);
        {
            // V4.1: ratio-1 layers make this a 1:1 context limit. `ctx` here is
            // the TEXT-only inverse while the cap carries the wider vision raw
            // window, so it sits a little above `V41_MAX_CTX` — the invariant
            // that matters is that it covers what we ship, not equality.
            assert!(
                ctx >= V41_MAX_CTX,
                "text-only inverse {ctx} below the shipped --ctx {V41_MAX_CTX}"
            );
        }
    }

    /// `attn_scores_stride` never hands the two kernels a stride that
    /// overruns the buffer, and keeps the legacy layout when it fits.
    #[test]
    fn scores_stride_is_the_floor_when_it_fits_and_errors_when_it_cannot() {
        let cap = 512 * 64 * (ATTN_SCORES_STRIDE as usize);
        assert_eq!(attn_scores_stride(cap, 512, 64, 640).unwrap(), ATTN_SCORES_STRIDE);
        // Same buffer, a quarter of the rows: deeper contexts become legal.
        assert_eq!(attn_scores_stride(cap, 128, 64, 9000).unwrap(), 9000);
        // ... but not without bound.
        assert!(attn_scores_stride(cap, 512, 64, 4000).is_err());
    }
}
