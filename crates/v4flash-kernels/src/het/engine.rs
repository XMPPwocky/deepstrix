//! [`DeviceEngine`] — per-device bundle of HIP kernel modules + streams.
//! [`HeterogeneousEngine`] — pair of engines plus [`ExecMode`] policy.
//!
//! Two execution modes:
//! * `HetSingleStream` — every kernel followed by `compute.synchronize()`,
//!   peer copies serial. Correctness oracle, slower than even single-
//!   device because of cross-device sync overhead.
//! * `HetParallel` — separate `compute` + `xfer` streams per device,
//!   cross-device handoffs gated by HIP events. The production mode.

use color_eyre::eyre;
use v4flash_hip::{Device, Event, Stream};

use super::graph_cache::GraphCache;

use crate::config::N_LAYER;

use crate::attention::{AttentionMixed, AttentionSwa};
use crate::compressor::{
    CompressorPool, CompressorStateShuffleR4, CompressorStateSnapshot, CompressorStateWrite,
    F16Roundtrip, Fp8E4m3fnQuantize,
};
use crate::f16::F16Matvec;
use crate::ffn::{Swiglu, SwigluClampWeighted, VecAddInplace};
use crate::head::{HcPost, HcSigmoidBias, HcSinkhorn, HcWeightedSum};
use crate::comp_kv_append::CompKvAppend;
use crate::iq2_xxs::Iq2XxsPairMatvec;
use crate::kv_cache_append::KvCacheAppend;
use crate::q2_k::Q2KAccumulateMatvec;
use crate::q8_0::{Q8_0GroupedMatvec, Q8_0Matvec, Q8_0MatvecWmma};
use crate::q8_k::Q8KQuantize;
use crate::rms_norm::{RmsNorm, RmsNormNoWeight};
use crate::rope::RopeTail;
use crate::router_topk::RouterTopk;

use super::perfetto::DeviceTimingExporter;
use super::trace::EventPool;

/// Per-device EventPool capacity. The forward path emits per-kernel
/// sub-spans (one event-pair per kernel launch) inside every multi-kernel
/// stage; this can hit ~100 pairs/layer × 43 layers ≈ 8600 events in
/// decode and ~120 pairs/layer × 43 ≈ 10000 events in prefill. Sized
/// generously so the pool never trips when perfetto is attached. The
/// pool is reset per token/chunk; allocation is per-session.
pub const EVENT_POOL_CAPACITY: usize = 16384;

/// Execution policy for [`HeterogeneousEngine::forward_token`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecMode {
    /// Serial het execution — every kernel followed by `.synchronize()`,
    /// peer copies serialized. Correctness oracle for the parallel mode.
    /// Expected to be *slower* than single-device because of cross-device
    /// sync overhead.
    HetSingleStream,
    /// Event-driven overlap. Compressor / shared / routed-MoE run
    /// concurrently across the two devices, gated by HIP events.
    HetParallel,
}

/// Per-step sampling policy for `HeterogeneousEngine::sample_next`.
#[derive(Debug, Clone, Copy)]
pub enum SampleMode {
    /// Deterministic — picks argmax. Lowest-index tie-break.
    Argmax,
    /// Multinomial sample from softmax(logits / temperature).
    /// V4-Flash's recommended setting is `temperature = 1.0,
    /// min_p_rel = 0.0` (no pruning).
    Multinomial {
        temperature: f32,
        /// Min-p threshold relative to the most-likely token (e.g. 0.05).
        /// Set to 0.0 to disable pruning. Tokens whose unnormalised
        /// probability `exp((x*inv_T) - gmax)` falls below this threshold
        /// are skipped during the cumulative walk.
        min_p_rel: f32,
    },
}

/// All kernel modules + streams for one HIP device. Both devices carry
/// the full kernel set (cheap — kernels are HSACO blobs, not weights).
/// Memory split happens in [`super::weights`].
pub struct DeviceEngine {
    pub device: Device,
    /// Compute stream — all kernel launches go here for single-token paths.
    pub compute: Stream,
    /// Transfer stream — peer copies originating on this device go here.
    /// Per the load-bearing peer-copy rule, `hipMemcpyPeerAsync` MUST be
    /// queued on the **source** device's stream.
    pub xfer: Stream,

    pub rms_w: RmsNorm,
    pub rms_nw: RmsNormNoWeight,
    pub q8: Q8_0Matvec,
    /// int8-WMMA Q8_0 GEMM (gfx12 dGPU only). Same math as `q8.matvec_batched`,
    /// via the matrix cores. Used for the prefill qb up-projection.
    pub q8_wmma: Q8_0MatvecWmma,
    pub q8_grouped: Q8_0GroupedMatvec,
    pub f16: F16Matvec,
    pub rope: RopeTail,
    pub attn_swa: AttentionSwa,
    pub attn_mixed: AttentionMixed,
    pub swiglu: Swiglu,
    pub swiglu_cw: SwigluClampWeighted,
    pub vec_add: VecAddInplace,
    pub q8k: Q8KQuantize,
    pub iq2: Iq2XxsPairMatvec,
    pub q2k: Q2KAccumulateMatvec,
    pub hc_sigmoid: HcSigmoidBias,
    pub hc_weighted: HcWeightedSum,
    pub hc_sinkhorn: HcSinkhorn,
    pub mhc_pre_fused: crate::MhcPreFused,
    pub rms_nw_mw: crate::RmsNormNoWeightMultiWG,
    /// On-device sampler. Used only on dGPU (logits live there) but
    /// instantiated on both arches so the engine struct stays symmetric.
    pub sampler: crate::Sampler,
    pub hc_post: HcPost,
    pub compressor_pool: CompressorPool,
    pub compressor_state_write: CompressorStateWrite,
    pub compressor_shuffle: CompressorStateShuffleR4,
    pub compressor_state_snapshot: CompressorStateSnapshot,
    pub fp8: Fp8E4m3fnQuantize,
    pub f16rt: F16Roundtrip,
    pub kv_append: KvCacheAppend,
    pub comp_kv_append: CompKvAppend,
    pub router_topk: RouterTopk,
    /// CSA indexer kernels (used only on the dGPU's ratio==4 layers, but
    /// instantiated unconditionally so the engine struct stays symmetric).
    pub indexer_score: crate::IndexerScore,
    /// WMMA-fused IndexerScore variant. Available on gfx12 dGPU only;
    /// `None` on iGPU. Callers should prefer it when present (28× faster
    /// at production decode shape).
    pub indexer_score_wmma: Option<crate::IndexerScoreWmma>,
    pub indexer_topk: crate::IndexerTopk,
    /// Bitonic-sort IndexerTopk variant (ported from ds4). 72× faster
    /// at n_comp=16384 than the greedy fallback. Always available.
    pub indexer_topk_bitonic: crate::IndexerTopkBitonic,
    pub indexer_gather: crate::IndexerGather,
    pub indexer_bitpack: crate::IndexerBitpack,
    pub vec_scale: crate::VecScaleInplace,
    /// By-expert MoE pre-pass — inverts d_selected into per-expert
    /// (token, slot) lists. Used by the prefill iGPU MoE path.
    pub moe_group_builder: crate::moe_group_builder::MoeGroupBuilder,

    /// Per-device event pool for kernel-scope timing. Use
    /// `events.stage(name, &compute)` to wrap a kernel-group.
    pub events: EventPool,
}

impl DeviceEngine {
    /// Build a DeviceEngine for `device`. The device must already be
    /// reachable; this function will `set_current` it during construction.
    pub fn for_arch(device: Device, arch: &str) -> eyre::Result<Self> {
        device.set_current()?;
        let compute = Stream::new(device.id)?;
        let xfer = Stream::new(device.id)?;
        let is_igpu = device.properties()?.integrated;
        let label: &'static str = if is_igpu { "igpu" } else { "dgpu" };
        let events = EventPool::new(label, EVENT_POOL_CAPACITY)?;
        Ok(Self {
            device,
            compute,
            xfer,
            rms_w: RmsNorm::for_arch(arch)?,
            rms_nw: RmsNormNoWeight::for_arch(arch)?,
            q8: Q8_0Matvec::for_arch(arch)?,
            q8_wmma: Q8_0MatvecWmma::for_arch(arch)?,
            q8_grouped: Q8_0GroupedMatvec::for_arch(arch)?,
            f16: F16Matvec::for_arch(arch)?,
            rope: RopeTail::for_arch(arch)?,
            attn_swa: AttentionSwa::for_arch(arch)?,
            attn_mixed: AttentionMixed::for_arch(arch)?,
            swiglu: Swiglu::for_arch(arch)?,
            swiglu_cw: SwigluClampWeighted::for_arch(arch)?,
            vec_add: VecAddInplace::for_arch(arch)?,
            q8k: Q8KQuantize::for_arch(arch)?,
            iq2: Iq2XxsPairMatvec::for_arch(arch)?,
            q2k: Q2KAccumulateMatvec::for_arch(arch)?,
            hc_sigmoid: HcSigmoidBias::for_arch(arch)?,
            hc_weighted: HcWeightedSum::for_arch(arch)?,
            hc_sinkhorn: HcSinkhorn::for_arch(arch)?,
            mhc_pre_fused: crate::MhcPreFused::for_arch(arch)?,
            rms_nw_mw: crate::RmsNormNoWeightMultiWG::for_arch(arch)?,
            sampler: crate::Sampler::for_arch(arch)?,
            hc_post: HcPost::for_arch(arch)?,
            compressor_pool: CompressorPool::for_arch(arch)?,
            compressor_state_write: CompressorStateWrite::for_arch(arch)?,
            compressor_shuffle: CompressorStateShuffleR4::for_arch(arch)?,
            compressor_state_snapshot: CompressorStateSnapshot::for_arch(arch)?,
            fp8: Fp8E4m3fnQuantize::for_arch(arch)?,
            f16rt: F16Roundtrip::for_arch(arch)?,
            kv_append: KvCacheAppend::for_arch(arch)?,
            comp_kv_append: CompKvAppend::for_arch(arch)?,
            router_topk: RouterTopk::for_arch(arch)?,
            indexer_score: crate::IndexerScore::for_arch(arch)?,
            indexer_score_wmma: if arch.starts_with("gfx1200") || arch.starts_with("gfx1201") {
                Some(crate::IndexerScoreWmma::for_arch(arch)?)
            } else {
                None
            },
            indexer_topk: crate::IndexerTopk::for_arch(arch)?,
            indexer_topk_bitonic: crate::IndexerTopkBitonic::for_arch(arch)?,
            indexer_gather: crate::IndexerGather::for_arch(arch)?,
            indexer_bitpack: crate::IndexerBitpack::for_arch(arch)?,
            vec_scale: crate::VecScaleInplace::for_arch(arch)?,
            moe_group_builder: crate::moe_group_builder::MoeGroupBuilder::for_arch(arch)?,
            events,
        })
    }
}

/// Per-layer cross-device sync events for the [`ExecMode::HetParallel`]
/// pipeline. Pre-allocated up front, reused per token.
///
/// FFN handoff:
/// * `ain_ready` (dgpu.compute) — `ffn_input_norm` finished computing.
/// * `ain_pushed` (dgpu.xfer)   — `ffn_input_norm` finished copying to iGPU.
/// * `moe_done`  (igpu.compute) — routed MoE finished writing `ffn_moe`.
/// * `moe_arrived` (igpu.xfer)  — `ffn_moe` finished copying to dGPU.
///
/// Router handoff:
/// * `selected_ready` (dgpu.compute) — selected/d_ew written and ready for
///   peer-push to iGPU.
/// * `selected_pushed` (dgpu.xfer) — selected/d_ew pushed to iGPU;
///   iGPU MoE waits on this.
pub struct LayerSyncEvents {
    pub ain_ready: Event,
    pub ain_pushed: Event,
    pub moe_done: Event,
    pub moe_arrived: Event,
    pub selected_ready: Event,
    pub selected_pushed: Event,
}

pub struct HetSyncEvents {
    pub layers: Vec<LayerSyncEvents>,
}

impl HetSyncEvents {
    /// Allocate per-layer no-timing events. Caller must have `dgpu`
    /// current when creating dGPU events and `igpu` current when
    /// creating iGPU events; this function handles the switching.
    pub fn alloc(dgpu: Device, igpu: Device) -> eyre::Result<Self> {
        let mut layers = Vec::with_capacity(N_LAYER as usize);
        for _ in 0..N_LAYER {
            dgpu.set_current()?;
            let ain_ready = Event::new_no_timing()?;
            let ain_pushed = Event::new_no_timing()?;
            let selected_ready = Event::new_no_timing()?;
            let selected_pushed = Event::new_no_timing()?;
            igpu.set_current()?;
            let moe_done = Event::new_no_timing()?;
            let moe_arrived = Event::new_no_timing()?;
            layers.push(LayerSyncEvents {
                ain_ready,
                ain_pushed,
                moe_done,
                moe_arrived,
                selected_ready,
                selected_pushed,
            });
        }
        Ok(Self { layers })
    }
}

/// Pair of [`DeviceEngine`]s plus an [`ExecMode`].
pub struct HeterogeneousEngine {
    pub dgpu: DeviceEngine,
    pub igpu: DeviceEngine,
    pub mode: ExecMode,
    /// Pre-allocated per-layer sync events for `HetParallel`. Unused in
    /// `HetSingleStream` but cheap to keep allocated.
    pub sync_events: HetSyncEvents,
    /// Second per-layer event set, used by the two-lane pipelined prefill
    /// in `forward_prefill_pipelined` — lane A uses `sync_events`, lane B
    /// uses this. Allocated identically.
    pub sync_events_t1: HetSyncEvents,
    /// Optional per-token device-time perfetto exporter. Drains the
    /// EventPools at the end of each `forward_token` into per-stream
    /// perfetto tracks. Enable by calling
    /// [`HeterogeneousEngine::attach_perfetto`].
    pub perfetto: Option<std::sync::Mutex<DeviceTimingExporter>>,

    /// Captured HIP-graph cache, keyed by `(stage_name, layer)`. Each
    /// per-layer forward stage that is purely device-resident with
    /// layer-constant scalar params is captured once and replayed per
    /// token. Stages currently captured on the dGPU side: `mhc_pre_attn`,
    /// `mhc_pre_ffn`, `shared_expert`, `q_chain_pre_rope`,
    /// `output_proj_post_rope`, `ffn_combine`, `combined_ffn_pre_attn`
    /// (the cross-layer fusion of ffn_combine_N + mhc_pre_attn_{N+1};
    /// occupies slot `(name, layer)` for the transition out of `layer`).
    pub dgpu_graphs: GraphCache,
    /// Same as `dgpu_graphs` for the iGPU routed-MoE sub-pipeline
    /// (`routed_moe`).
    pub igpu_graphs: GraphCache,

    /// Thread-local cache of the currently-bound HIP device, so
    /// `set_current_cached()` can skip the driver call when the device
    /// hasn't actually changed. AtomicI32 (not Cell) keeps the engine
    /// `Sync`. `-1` = unknown.
    pub current_device: std::sync::atomic::AtomicI32,
    /// Diagnostic: last `forward_token`'s host-enqueue time (µs), before
    /// the final `dgpu.compute.synchronize()`. Bench reads this to split
    /// per-token wall into host vs device-wait.
    pub last_host_us: std::sync::atomic::AtomicU64,
    /// Diagnostic: time the host spent inside the final `synchronize()`.
    pub last_sync_us: std::sync::atomic::AtomicU64,
}

impl HeterogeneousEngine {
    /// Run a full token forward across both devices. Reads the layer-0
    /// residual stream from `input_hc_host` (size `HC_DIM`), runs all
    /// 43 layers, then the head. On return, `dgpu_scratch.logits` holds
    /// the final logits (size `N_VOCAB`).
    ///
    /// Each call resets both EventPools, runs the layers, then harvests
    /// per-kernel timings and emits a per-token INFO summary (see
    /// [`super::trace::TokenTiming`]). The summary's busy/idle breakdown
    /// uses the EventPool data directly so it survives changes to the
    /// internal sync points.
    pub fn forward_token(
        &self,
        dgpu_scratch: &mut super::DgpuScratch,
        igpu_scratch: &mut super::IgpuScratch,
        state: &mut super::HetModelState,
        weights: &super::HetModelWeights,
        input_hc_host: &[f32],
        pos: u32,
        token_id: i32,
    ) -> color_eyre::eyre::Result<()> {
        use crate::config::{HC_DIM, N_LAYER};
        use tracing::debug_span;

        if input_hc_host.len() != HC_DIM as usize {
            return Err(color_eyre::eyre::eyre!(
                "input_hc_host len {} != HC_DIM {}",
                input_hc_host.len(),
                HC_DIM
            ));
        }
        let _token_span = debug_span!("het.token", pos, token_id).entered();

        // Reset event pools for this token.
        self.dgpu.events.reset();
        self.igpu.events.reset();

        self.set_current_cached(self.dgpu.device)?;
        dgpu_scratch.residual.copy_from_host(input_hc_host)?;
        // Dump the layer-0 input (== embedded token vector) if
        // DEEPSTRIX_DUMP_RESIDUAL_DIR is set. Index 00 in the file
        // naming. Per-layer post-output dumps land at indices 01..43.
        maybe_dump_residual(0, &dgpu_scratch.residual)?;

        let token_start = std::time::Instant::now();
        let dump_subtensor_layers: Vec<usize> = subtensor_dump_spec()
            .as_ref()
            .map(|(ls, _)| ls.clone())
            .unwrap_or_default();

        // M54: pre-issue the token's ENTIRE iGPU MoE lane (43 × wait →
        // graph → push, all event-gated) before the dGPU layer loop. The
        // decode pftrace showed ~125 µs/layer of host-submission lag on the
        // iGPU stream when its commands were interleaved with the ~25 dGPU
        // submissions per layer — lag that lands on the dGPU's MoE wait.
        // DECODE_PREISSUE=1 opts in (default off until gated).
        static PREISSUE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::env::var("DECODE_PREISSUE").map(|v| v == "1").unwrap_or(false)
        });
        let preissue = *PREISSUE
            && matches!(self.mode, ExecMode::HetParallel)
            && dump_subtensor_layers.is_empty();
        if preissue {
            let _span = debug_span!("igpu.preissue_lane").entered();
            for layer in 0..N_LAYER as usize {
                self.issue_igpu_moe(
                    dgpu_scratch,
                    igpu_scratch,
                    &weights.igpu_layers[layer],
                )?;
            }
            self.set_current_cached(self.dgpu.device)?;
        }

        for layer in 0..N_LAYER as usize {
            let next_dlw = if layer + 1 < N_LAYER as usize {
                Some(&weights.dgpu_layers[layer + 1])
            } else {
                None
            };
            // When sub-tensor dumping is active at this layer, fall
            // back to the standalone-graphs path. The combined
            // cross-layer graph (ffn_combine fused with next layer's
            // mhc_pre_attn) writes the NEXT layer's attn_cur /
            // attn_input_norm into the same scratch fields, clobbering
            // the current layer's values before we can read them.
            // Standalone runs each layer's mhc_pre_attn separately so
            // the post-layer-N buffers are stable.
            let force_standalone = dump_subtensor_layers.contains(&layer);
            if force_standalone {
                self.forward_layer_standalone_graphs(
                    dgpu_scratch,
                    igpu_scratch,
                    &mut state.layers[layer],
                    &weights.dgpu_layers[layer],
                    &weights.igpu_layers[layer],
                    pos,
                    token_id,
                )?;
                self.dgpu.compute.synchronize()?;
                // Dump every scratch field that ds4 also emits. Names
                // match ds4's dump tags so per-tag diff is mechanical.
                maybe_dump_subtensor_f32(layer, "attn_cur", &dgpu_scratch.attn_cur)?;
                maybe_dump_subtensor_f32(layer, "attn_input_norm", &dgpu_scratch.attn_input_norm)?;
                maybe_dump_subtensor_f32(layer, "q_a_out", &dgpu_scratch.qr)?;
                maybe_dump_subtensor_f32(layer, "q_a_normed", &dgpu_scratch.qr_normed)?;
                maybe_dump_subtensor_f32(layer, "q_post_rope", &dgpu_scratch.q_normed)?;
                maybe_dump_subtensor_f32(layer, "kv_post_rope", &dgpu_scratch.kv_normed)?;
                maybe_dump_subtensor_f32(layer, "attn_heads", &dgpu_scratch.heads)?;
                maybe_dump_subtensor_f32(layer, "attn_out", &dgpu_scratch.attn_out)?;
                maybe_dump_subtensor_f32(layer, "after_attn_hc", &dgpu_scratch.after_attn_hc)?;
                maybe_dump_subtensor_f32(layer, "ffn_cur", &dgpu_scratch.ffn_cur)?;
                maybe_dump_subtensor_f32(layer, "ffn_input_norm", &dgpu_scratch.ffn_input_norm)?;
                maybe_dump_subtensor_f32(layer, "ffn_shared", &dgpu_scratch.ffn_shared)?;
                maybe_dump_subtensor_f32(layer, "ffn_moe", &dgpu_scratch.ffn_moe_recv)?;
                // Dump the cached K/V state that attention reads —
                // both the committed compressed rows (`comp_kv`,
                // ratio>0 layers) and the raw SWA window (`kv_cache`).
                // f16 stored on device, dumped as f32 to match ds4's
                // f32 dump format.
                let head_dim = crate::config::N_HEAD_DIM as usize;
                if let Some(comp) = &state.layers[layer].compressor {
                    let n_comp_elems = (comp.n_comp as usize) * head_dim;
                    maybe_dump_subtensor_f16_as_f32(
                        layer, "attn_comp_kv", &comp.comp_kv, n_comp_elems
                    )?;
                }
                let n_raw_elems = (state.layers[layer].n_raw as usize) * head_dim;
                maybe_dump_subtensor_f16_as_f32(
                    layer, "raw_kv", &state.layers[layer].kv_cache, n_raw_elems
                )?;
                // Router + expert-selection diagnostic: lets us
                // compare against ds4's `expert_selected` /
                // `expert_weight_out` / router_logits to see whether
                // the top-K experts we pick match ds4's. If they
                // don't, the MoE divergence is a router-precision
                // issue, not a per-expert kernel issue.
                maybe_dump_subtensor_f32(
                    layer, "router_logits", &dgpu_scratch.router_logits
                )?;
                maybe_dump_subtensor_i32(
                    layer, "d_selected", &dgpu_scratch.d_selected,
                    crate::config::N_EXPERT_USED,
                )?;
                maybe_dump_subtensor_f32(
                    layer, "d_ew", &dgpu_scratch.d_ew
                )?;
            } else if preissue {
                self.forward_layer_preissued_moe(
                    dgpu_scratch,
                    igpu_scratch,
                    &mut state.layers[layer],
                    &weights.dgpu_layers[layer],
                    next_dlw,
                    &weights.igpu_layers[layer],
                    pos,
                    token_id,
                )?;
            } else {
                self.forward_layer(
                    dgpu_scratch,
                    igpu_scratch,
                    &mut state.layers[layer],
                    &weights.dgpu_layers[layer],
                    next_dlw,
                    &weights.igpu_layers[layer],
                    pos,
                    token_id,
                )?;
            }
            std::mem::swap(&mut dgpu_scratch.residual, &mut dgpu_scratch.residual_next);
            // Diagnostic-only: substitute the per-layer output residual
            // with a host-supplied vector (typically ds4's
            // `layer_input_residual` for layer+1, loaded from a file).
            // Used to bisect cross-impl divergence by layer — when set,
            // the next layer reads the substituted residual instead of
            // ours, isolating whether our layer-`layer` compute is
            // upstream of the diverging logit.
            //
            // Format of env var:
            //   DEEPSTRIX_SUBSTITUTE_RESIDUAL=<after_layer>:<path>
            // where <after_layer> is the layer index whose OUTPUT to
            // overwrite (i.e. the substitution happens AFTER that
            // layer's forward + swap, so layer <after_layer+1> reads
            // the injected value). <path> is a binary file containing
            // exactly HC_DIM little-endian f32 values.
            //
            // No-op when the env var is unset or doesn't match this
            // layer. Read once per token via OnceLock; flipping the
            // env mid-run does nothing.
            maybe_substitute_residual(layer, &mut dgpu_scratch.residual)?;
            // Companion DUMP hook. Env: DEEPSTRIX_DUMP_RESIDUAL_DIR=/path
            // — when set, copy `dgpu_scratch.residual` (= layer's
            // output = layer+1's input) to host and write to
            // <dir>/layer_<NN+1>_residual.bin. Naming matches ds4's
            // convention: file index = INPUT-LAYER-NUMBER, i.e. our
            // layer-K-output is dumped as the file for layer K+1's
            // input. We also dump layer 0's INPUT separately at the
            // top of forward_token. Together this gives us a
            // ds4-comparable set of 43 files (indices 00..42).
            maybe_dump_residual(layer + 1, &dgpu_scratch.residual)?;
        }
        self.forward_head(dgpu_scratch, &weights.global)?;
        // N_LAYER (43) is odd, so 43 in-loop swaps leave residual /
        // residual_next inverted from token start. Without an extra swap
        // here, every token's layer 0 would read from a different
        // physical DeviceBuffer than the previous token's layer 0,
        // making it impossible to capture mhc_pre_attn / mhc_post_ffn
        // into HIP graphs (the captured pointer would be wrong on
        // alternating tokens). The extra swap restores the initial state
        // so layer N always operates on the same physical buffers across
        // every token.
        std::mem::swap(&mut dgpu_scratch.residual, &mut dgpu_scratch.residual_next);
        self.set_current_cached(self.dgpu.device)?;
        // Diagnostic split of per-token wall:
        //   host_us = time in this loop before the final sync. If big,
        //             OS preempted between launches.
        //   sync_us = time the host waited for the device. If big, the
        //             device was slow (clock/thermal).
        let host_us = token_start.elapsed().as_micros() as u64;
        self.dgpu.compute.synchronize()?;
        let token_elapsed_us = token_start.elapsed().as_micros() as u64;
        let sync_us = token_elapsed_us.saturating_sub(host_us);
        use std::sync::atomic::Ordering;
        self.last_host_us.store(host_us, Ordering::Relaxed);
        self.last_sync_us.store(sync_us, Ordering::Relaxed);

        // Emit device-time perfetto tracks (if enabled) before the
        // summary harvest — this only reads events that have already
        // completed (we sync on the last event inside for_each_pair).
        if let Some(exp_lock) = &self.perfetto {
            let mut exp = exp_lock.lock().unwrap();
            self.dgpu.events.for_each_pair(|name, s, e| {
                let track = if name.contains(".xfer") || name.contains(".peer_push") {
                    &exp.dgpu_xfer
                } else {
                    &exp.dgpu_compute
                };
                exp.emit_slice(track, name, s, e)
            })?;
            self.igpu.events.for_each_pair(|name, s, e| {
                let track = if name.contains(".xfer") || name.contains(".peer_push") {
                    &exp.igpu_xfer
                } else {
                    &exp.igpu_compute
                };
                exp.emit_slice(track, name, s, e)
            })?;
            // Re-anchor for the next token to bound dGPU/iGPU clock drift.
            // All streams are already synced at this point (compute via
            // `self.dgpu.compute.synchronize()` above, igpu via the various
            // readbacks inside forward_layer), so Anchor::new's per-stream
            // synchronize is essentially free.
            exp.re_anchor(
                self.dgpu.device,
                &self.dgpu.compute,
                &self.dgpu.xfer,
                self.igpu.device,
                &self.igpu.compute,
                &self.igpu.xfer,
            )?;
            // re_anchor calls device.set_current() internally for each
            // of the 4 tracks (last is igpu.xfer → igpu), bypassing
            // set_current_cached and leaving the cache stale. Invalidate
            // so the next forward_token's first set_current_cached is
            // forced through.
            self.current_device.store(-1, std::sync::atomic::Ordering::Relaxed);
        }

        // Harvest per-kernel timings.
        let dgpu_timings = self.dgpu.events.harvest()?;
        let igpu_timings = self.igpu.events.harvest()?;

        let dgpu_busy_us: u64 = (dgpu_timings.iter().map(|t| t.ms as f64).sum::<f64>() * 1000.0) as u64;
        let igpu_busy_us: u64 = (igpu_timings.iter().map(|t| t.ms as f64).sum::<f64>() * 1000.0) as u64;

        let dgpu_idle_us = token_elapsed_us.saturating_sub(dgpu_busy_us);
        let igpu_idle_us = token_elapsed_us.saturating_sub(igpu_busy_us);

        // peer copies: ffn_input_norm (N_EMBD f32) + ffn_moe (N_EMBD f32) per layer.
        let peer_bytes = (N_LAYER as u64) * 2 * (crate::config::N_EMBD as u64) * 4;

        let summary = super::trace::TokenTiming {
            token_pos: pos,
            total_us: token_elapsed_us,
            dgpu_busy_us,
            igpu_busy_us,
            dgpu_idle_us,
            igpu_idle_us,
            peer_bytes,
        };
        summary.emit();

        // DEBUG rollup: per-stage totals.
        if tracing::enabled!(tracing::Level::DEBUG) {
            let dgpu_roll = super::trace::rollup_by_name(&dgpu_timings);
            let igpu_roll = super::trace::rollup_by_name(&igpu_timings);
            for (name, total_ms, calls) in dgpu_roll {
                tracing::debug!(
                    device = "dgpu",
                    stage = name,
                    total_us = (total_ms * 1000.0) as u64,
                    calls,
                    "het.stage"
                );
            }
            for (name, total_ms, calls) in igpu_roll {
                tracing::debug!(
                    device = "igpu",
                    stage = name,
                    total_us = (total_ms * 1000.0) as u64,
                    calls,
                    "het.stage"
                );
            }
        }
        Ok(())
    }

    /// Build a het engine over (dgpu, igpu). Enables peer access both
    /// directions — if the runtime refuses, that's surfaced as an error
    /// here rather than at the first `hipMemcpyPeerAsync` call.
    pub fn new(
        dgpu_device: Device,
        dgpu_arch: &str,
        igpu_device: Device,
        igpu_arch: &str,
        mode: ExecMode,
    ) -> eyre::Result<Self> {
        let dgpu = DeviceEngine::for_arch(dgpu_device, dgpu_arch)?;
        let igpu = DeviceEngine::for_arch(igpu_device, igpu_arch)?;

        // Enable peer access both directions. Re-entrant-safe per HIP
        // docs but check the can_access_peer flag first — if peer access
        // isn't supported the user needs to know up front.
        dgpu_device.set_current()?;
        if !dgpu_device.can_access_peer(igpu_device)? {
            return Err(color_eyre::eyre::eyre!(
                "dGPU {} cannot access iGPU {} as a peer",
                dgpu_device.id,
                igpu_device.id
            ));
        }
        let _ = dgpu_device.enable_peer_access(igpu_device);

        igpu_device.set_current()?;
        if !igpu_device.can_access_peer(dgpu_device)? {
            return Err(color_eyre::eyre::eyre!(
                "iGPU {} cannot access dGPU {} as a peer",
                igpu_device.id,
                dgpu_device.id
            ));
        }
        let _ = igpu_device.enable_peer_access(dgpu_device);

        let sync_events = HetSyncEvents::alloc(dgpu_device, igpu_device)?;
        let sync_events_t1 = HetSyncEvents::alloc(dgpu_device, igpu_device)?;
        dgpu_device.set_current()?;

        Ok(Self {
            dgpu,
            igpu,
            mode,
            sync_events,
            sync_events_t1,
            perfetto: None,
            dgpu_graphs: GraphCache::new(),
            igpu_graphs: GraphCache::new(),
            current_device: std::sync::atomic::AtomicI32::new(-1),
            last_host_us: std::sync::atomic::AtomicU64::new(0),
            last_sync_us: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Drain both devices to idle before teardown. Call this once after
    /// all forward work is done and before `HeterogeneousEngine`,
    /// `HetModelState`, and the scratch buffers go out of scope. It
    /// guarantees every stream (compute, xfer, the pipeline lane streams,
    /// and any in-flight cross-device event signal packets) has fully
    /// completed, so the implicit `SyncAllStreams` that each buffer's
    /// `hipFree` performs during Drop finds quiescent queues. Without
    /// this drain, a stream destroyed before its signal packet executes
    /// can orphan a peer's `hipStreamWaitEvent`, making teardown
    /// busy-spin forever (the intermittent ROCm teardown hang).
    pub fn shutdown(&self) -> eyre::Result<()> {
        self.dgpu.device.synchronize()?;
        self.igpu.device.synchronize()?;
        Ok(())
    }

    /// Sample the next token from `dgpu_scratch.logits` on-device.
    ///
    /// Three modes:
    ///   - argmax: deterministic, ignores temperature / u01.
    ///   - multinomial: full softmax sample with optional min-p pruning.
    ///   - argmax (T == 0.0): falls through to argmax mode automatically.
    ///
    /// `u01` is the host-supplied uniform sample in [0, 1) for this token.
    /// Pass `0.0` in argmax mode (ignored). The returned token id is read
    /// back via a 4-byte D→H copy after a stream sync — total overhead
    /// per token is well under 100 µs.
    ///
    /// Caller must hold the dGPU as current device (or this method will
    /// re-set it via `set_current_cached`).
    pub fn sample_next(
        &self,
        dgpu_scratch: &mut super::DgpuScratch,
        mode: SampleMode,
        u01: f32,
    ) -> eyre::Result<i32> {
        self.set_current_cached(self.dgpu.device)?;
        let n = crate::config::N_VOCAB;
        match mode {
            SampleMode::Argmax => {
                self.dgpu.sampler.launch_argmax(
                    &self.dgpu.compute,
                    &mut dgpu_scratch.sampler_next_token_id,
                    &dgpu_scratch.logits,
                    n,
                )?;
            }
            SampleMode::Multinomial { temperature, min_p_rel } => {
                if temperature <= 0.0 {
                    self.dgpu.sampler.launch_argmax(
                        &self.dgpu.compute,
                        &mut dgpu_scratch.sampler_next_token_id,
                        &dgpu_scratch.logits,
                        n,
                    )?;
                } else {
                    dgpu_scratch.sampler_u01.copy_from_host(&[u01])?;
                    self.dgpu.sampler.launch_multinomial(
                        &self.dgpu.compute,
                        &mut dgpu_scratch.sampler_next_token_id,
                        &dgpu_scratch.logits,
                        &mut dgpu_scratch.sampler_partials_max,
                        &mut dgpu_scratch.sampler_partials_z,
                        &dgpu_scratch.sampler_u01,
                        n,
                        temperature,
                        min_p_rel,
                    )?;
                }
            }
        }
        self.dgpu.compute.synchronize()?;
        let mut id = [0i32; 1];
        dgpu_scratch.sampler_next_token_id.copy_to_host(&mut id)?;
        Ok(id[0])
    }

    /// Invalidate the set_current_cached cache. Call this any time
    /// external code may have changed the actual current HIP device
    /// behind the cache's back — the next set_current_cached call will
    /// then forcibly re-set.
    pub fn invalidate_device_cache(&self) {
        self.current_device
            .store(-1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Set the HIP device only if the cached value differs from the
    /// request. Skips redundant `hipSetDevice` driver calls in the
    /// forward_layer loop (which toggles dGPU ↔ iGPU multiple times
    /// per layer; each unconditional `set_current` was a few µs of
    /// host time).
    pub fn set_current_cached(&self, dev: Device) -> eyre::Result<()> {
        use std::sync::atomic::Ordering;
        if self.current_device.load(Ordering::Relaxed) != dev.id {
            dev.set_current()?;
            self.current_device.store(dev.id, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Drain the current device event pools into perfetto slices and
    /// re-anchor. forward_prefill / forward_token do this implicitly per
    /// chunk / token; tests that drive forward_layer_batch_v2 directly in
    /// a loop must call this between iterations to keep slice timestamps
    /// from drifting against the anchor reference.
    pub fn flush_perfetto(&self) -> eyre::Result<()> {
        if let Some(exp_lock) = &self.perfetto {
            let mut exp = exp_lock.lock().unwrap();
            self.dgpu.events.for_each_pair(|name, s, e| {
                let track = if name.contains(".xfer") || name.contains(".peer_push") {
                    &exp.dgpu_xfer
                } else {
                    &exp.dgpu_compute
                };
                exp.emit_slice(track, name, s, e)
            })?;
            self.igpu.events.for_each_pair(|name, s, e| {
                let track = if name.contains(".xfer") || name.contains(".peer_push") {
                    &exp.igpu_xfer
                } else {
                    &exp.igpu_compute
                };
                exp.emit_slice(track, name, s, e)
            })?;
            exp.re_anchor(
                self.dgpu.device,
                &self.dgpu.compute,
                &self.dgpu.xfer,
                self.igpu.device,
                &self.igpu.compute,
                &self.igpu.xfer,
            )?;
            self.current_device
                .store(-1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    /// Open a perfetto device-time trace file. Subsequent
    /// `forward_token` calls will append per-stream slices for every
    /// kernel-stage event-pair. Call after `new()` and before forward
    /// passes. Pair with a host-side `PerfettoLayer` writing the same
    /// file for the full picture.
    pub fn attach_perfetto(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> eyre::Result<()> {
        let exporter = DeviceTimingExporter::open(
            path,
            self.dgpu.device,
            &self.dgpu.compute,
            &self.dgpu.xfer,
            self.igpu.device,
            &self.igpu.compute,
            &self.igpu.xfer,
        )?;
        self.perfetto = Some(std::sync::Mutex::new(exporter));
        self.dgpu.events.set_enabled(true);
        self.igpu.events.set_enabled(true);
        self.current_device.store(-1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let s: u32 = (sign as u32) << 31;
    let f32_bits: u32 = match exp {
        0 if mant == 0 => s,
        0 => {
            let m = (mant as f32) / 1024.0;
            let v = m * (1.0 / (1u64 << 14) as f32);
            return if sign == 1 { -v } else { v };
        }
        0x1f => s | 0x7f800000 | ((mant as u32) << 13),
        _ => s | ((exp as u32 + 112) << 23) | ((mant as u32) << 13),
    };
    f32::from_bits(f32_bits)
}

/// Parsed `DEEPSTRIX_SUBSTITUTE_RESIDUAL` setting. See the call site
/// in `forward_token` for the full rationale.
struct SubstituteResidualSpec {
    after_layer: usize,
    bytes: Vec<u8>,
}

fn substitute_residual_spec() -> &'static Option<SubstituteResidualSpec> {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Option<SubstituteResidualSpec>> = OnceLock::new();
    CACHED.get_or_init(|| {
        let raw = std::env::var("DEEPSTRIX_SUBSTITUTE_RESIDUAL").ok()?;
        let mut parts = raw.splitn(2, ':');
        let layer_str = parts.next()?;
        let path = parts.next()?;
        let after_layer: usize = layer_str.parse().ok()?;
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "DEEPSTRIX_SUBSTITUTE_RESIDUAL: failed to read {path}: {e}"
                );
                return None;
            }
        };
        let expected = (crate::config::HC_DIM as usize) * std::mem::size_of::<f32>();
        if bytes.len() != expected {
            eprintln!(
                "DEEPSTRIX_SUBSTITUTE_RESIDUAL: {path} is {} bytes, expected {expected} (HC_DIM={} f32)",
                bytes.len(),
                crate::config::HC_DIM
            );
            return None;
        }
        eprintln!(
            "DEEPSTRIX_SUBSTITUTE_RESIDUAL: armed — after layer {after_layer}, overwrite residual from {path}"
        );
        Some(SubstituteResidualSpec { after_layer, bytes })
    })
}

fn maybe_substitute_residual(
    layer: usize,
    residual: &mut v4flash_hip::DeviceBuffer<f32>,
) -> eyre::Result<()> {
    let Some(spec) = substitute_residual_spec().as_ref() else {
        return Ok(());
    };
    if spec.after_layer != layer {
        return Ok(());
    }
    // Reinterpret the cached bytes as &[f32] for copy_from_host.
    // Safe: spec.bytes.len() was checked == HC_DIM * sizeof(f32) at
    // env-var parse time, and we trust the on-disk byte order matches
    // the device's f32 layout (LE on both Linux x86_64 and the GPUs
    // we target).
    let n = crate::config::HC_DIM as usize;
    let floats: &[f32] = unsafe {
        std::slice::from_raw_parts(spec.bytes.as_ptr() as *const f32, n)
    };
    residual.copy_from_host(floats)?;
    eprintln!("substituted residual after layer {layer} from ds4 dump");
    Ok(())
}

fn dump_residual_dir() -> &'static Option<String> {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Option<String>> = OnceLock::new();
    CACHED.get_or_init(|| {
        let dir = std::env::var("DEEPSTRIX_DUMP_RESIDUAL_DIR").ok()?;
        if dir.is_empty() { return None; }
        if let Err(e) = std::fs::create_dir_all(&dir) {
            eprintln!("DEEPSTRIX_DUMP_RESIDUAL_DIR: mkdir {dir} failed: {e}");
            return None;
        }
        eprintln!("DEEPSTRIX_DUMP_RESIDUAL_DIR: dumping per-layer residuals to {dir}");
        Some(dir)
    })
}

fn maybe_dump_residual(
    layer_index_for_name: usize,
    residual: &v4flash_hip::DeviceBuffer<f32>,
) -> eyre::Result<()> {
    let Some(dir) = dump_residual_dir().as_ref() else {
        return Ok(());
    };
    let n = crate::config::HC_DIM as usize;
    let mut host = vec![0.0f32; n];
    residual.copy_to_host(&mut host)?;
    let path = format!("{dir}/layer_{:02}_residual.bin", layer_index_for_name);
    // Reinterpret as bytes for fs::write. f32→u8 LE is the host
    // representation; matches ds4's on-disk f32 layout.
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(host.as_ptr() as *const u8, n * std::mem::size_of::<f32>())
    };
    std::fs::write(&path, bytes)
        .map_err(|e| eyre::eyre!("write {path}: {e}"))?;
    eprintln!("  dumped layer_{:02}_residual.bin", layer_index_for_name);
    Ok(())
}

/// Diagnostic: when env `DEEPSTRIX_DUMP_SUBTENSOR_LAYER=N` and
/// `DEEPSTRIX_DUMP_SUBTENSOR_DIR=/path` are set, forward_layer can
/// call this after each major stage to capture per-sub-tensor f32
/// values for the layer `N`. Output filename:
/// `<dir>/layer_<NN>_<tag>.bin`. Used to bisect WITHIN a layer to find
/// which specific kernel first diverges from ds4-CPU's reference,
/// matching ds4's `ds4_dump_emit_1d` tagging.
fn subtensor_dump_spec() -> &'static Option<(Vec<usize>, String)> {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Option<(Vec<usize>, String)>> = OnceLock::new();
    CACHED.get_or_init(|| {
        // Accept either DEEPSTRIX_DUMP_SUBTENSOR_LAYERS (comma list)
        // or the legacy DEEPSTRIX_DUMP_SUBTENSOR_LAYER (single int).
        let layers_s = std::env::var("DEEPSTRIX_DUMP_SUBTENSOR_LAYERS")
            .ok()
            .or_else(|| std::env::var("DEEPSTRIX_DUMP_SUBTENSOR_LAYER").ok())?;
        let dir = std::env::var("DEEPSTRIX_DUMP_SUBTENSOR_DIR").ok()?;
        let layers: Vec<usize> = layers_s
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        if layers.is_empty() {
            return None;
        }
        if let Err(e) = std::fs::create_dir_all(&dir) {
            eprintln!("DEEPSTRIX_DUMP_SUBTENSOR_DIR: mkdir {dir} failed: {e}");
            return None;
        }
        eprintln!(
            "DEEPSTRIX_DUMP_SUBTENSOR_LAYERS={layers:?} dir={dir}: sub-tensor dump armed"
        );
        Some((layers, dir))
    })
}

pub(super) fn maybe_dump_subtensor_f32(
    layer: usize,
    tag: &str,
    buf: &v4flash_hip::DeviceBuffer<f32>,
) -> eyre::Result<()> {
    let Some((target_layers, dir)) = subtensor_dump_spec().as_ref() else {
        return Ok(());
    };
    if !target_layers.contains(&layer) {
        return Ok(());
    }
    let n = buf.len();
    let mut host = vec![0.0f32; n];
    buf.copy_to_host(&mut host)?;
    let path = format!("{dir}/layer_{:02}_{tag}.bin", layer);
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(host.as_ptr() as *const u8, n * std::mem::size_of::<f32>())
    };
    std::fs::write(&path, bytes)
        .map_err(|e| eyre::eyre!("write {path}: {e}"))?;
    eprintln!("  dumped layer_{:02}_{tag}.bin ({} f32)", layer, n);
    Ok(())
}

pub(super) fn maybe_dump_subtensor_i32(
    layer: usize,
    tag: &str,
    buf: &v4flash_hip::DeviceBuffer<i32>,
    n_elem: usize,
) -> eyre::Result<()> {
    let Some((target_layers, dir)) = subtensor_dump_spec().as_ref() else {
        return Ok(());
    };
    if !target_layers.contains(&layer) || n_elem == 0 {
        return Ok(());
    }
    let take = n_elem.min(buf.len());
    let mut full = vec![0i32; buf.len()];
    buf.copy_to_host(&mut full)?;
    let host: Vec<i32> = full[..take].to_vec();
    let path = format!("{dir}/layer_{:02}_{tag}.bin", layer);
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * std::mem::size_of::<i32>())
    };
    std::fs::write(&path, bytes)
        .map_err(|e| eyre::eyre!("write {path}: {e}"))?;
    eprintln!("  dumped layer_{:02}_{tag}.bin ({} i32)", layer, host.len());
    Ok(())
}

/// Dump f16-stored values (u16 buffer) as f32 to match ds4's f32-only
/// dump callback. Only the first `n_elem` u16 values are read +
/// converted; subsequent buffer bytes are unused (some buffers like
/// kv_cache and comp_kv are over-allocated to a max capacity but only
/// the first n_used rows are valid).
pub(super) fn maybe_dump_subtensor_f16_as_f32(
    layer: usize,
    tag: &str,
    buf: &v4flash_hip::DeviceBuffer<u16>,
    n_elem: usize,
) -> eyre::Result<()> {
    let Some((target_layers, dir)) = subtensor_dump_spec().as_ref() else {
        return Ok(());
    };
    if !target_layers.contains(&layer) {
        return Ok(());
    }
    if n_elem == 0 {
        return Ok(());
    }
    // Read the full buffer into u16s, slice to n_elem, convert each
    // u16 → f16 → f32, then dump as little-endian f32 bytes.
    let mut raw = vec![0u16; buf.len()];
    buf.copy_to_host(&mut raw)?;
    let take = n_elem.min(raw.len());
    let f32s: Vec<f32> = raw[..take]
        .iter()
        .map(|&u| f16_bits_to_f32(u))
        .collect();
    let path = format!("{dir}/layer_{:02}_{tag}.bin", layer);
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            f32s.as_ptr() as *const u8,
            f32s.len() * std::mem::size_of::<f32>(),
        )
    };
    std::fs::write(&path, bytes)
        .map_err(|e| eyre::eyre!("write {path}: {e}"))?;
    eprintln!(
        "  dumped layer_{:02}_{tag}.bin ({} f32 from f16 source)",
        layer, take
    );
    Ok(())
}
