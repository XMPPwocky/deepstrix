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
use v4flash_hip::{Device, DeviceBuffer, Event, Stream};

use super::graph_cache::GraphCache;
use super::state::CompKvStore;

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
    /// DeepSeek's agentic recipe for this model is `temperature = 1.0,
    /// top_p = 0.95, min_p_rel = 0.0`.
    Multinomial {
        temperature: f32,
        /// Min-p threshold relative to the most-likely token (e.g. 0.05).
        /// Set to 0.0 to disable pruning. Tokens whose unnormalised
        /// probability `exp((x*inv_T) - gmax)` falls below this threshold
        /// are skipped during the cumulative walk.
        min_p_rel: f32,
        /// Nucleus cutoff in (0, 1]. `1.0` disables truncation and takes
        /// the exact pre-top-p kernel chain (bit-identical results).
        /// Composes with `min_p_rel` the way llama.cpp does: top-p is taken
        /// over the full distribution, min-p prunes what is left, and the
        /// mass is renormalised over the intersection.
        top_p: f32,
    },
}

/// All kernel modules + streams for one HIP device. Both devices carry
/// the full kernel set (cheap — kernels are HSACO blobs, not weights).
/// Memory split happens in [`super::weights`].
pub struct DeviceEngine {
    pub device: Device,
    /// RDNA3-class (gfx11xx) target: gates the gfx11-layout WMMA MoE kernels.
    pub is_gfx11: bool,
    /// Compute stream — all kernel launches go here for single-token paths.
    pub compute: Stream,
    /// Transfer stream — peer copies originating on this device go here.
    /// Per the load-bearing peer-copy rule, `hipMemcpyPeerAsync` MUST be
    /// queued on the **source** device's stream.
    pub xfer: Stream,
    /// Side stream for V4.1 single-pass mHC coefficient work (`hc_mixes`).
    ///
    /// ARCH_SPEC §1.1 collapses the 4 copies with the PREVIOUS sub-block's
    /// `pre`, so a sub-block's own `hc_mixes` (RMS + [24,20480] fp32 matvec +
    /// 20-iteration sinkhorn) feeds only `hc_post` and the NEXT sub-block —
    /// it is off this sub-block's critical path by construction. That shift is
    /// the whole point of the design; running the mixes here lets them overlap
    /// the ~413 us of q/kv/attention/output_proj that follows the collapse.
    pub hc: Stream,

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
    // unsloth UD mix kernels (per-layer dispatch; None-cost when unused).
    pub iq3: crate::iq3_xxs::Iq3XxsMatvec,
    pub mxfp4: crate::mxfp4::Mxfp4Matvec,
    pub iq2s: crate::iq2_s::Iq2SPairMatvec,
    /// UD-Q2_K_XL gate/up: IQ2_XS on 42 layers, IQ3_XXS (paired form) on blk.26.
    pub iq2xs: crate::iq2_xs::Iq2XsPairMatvec,
    pub iq3pair: crate::iq3_xxs_pair::Iq3XxsPairMatvec,
    /// UD-IQ3_XXS gate/up: IQ2_S on 42 layers, IQ3_S (paired form) on blk.26.
    pub iq3s: crate::iq3_s::Iq3SPairMatvec,
    /// V4.1-Flash native experts: MXFP4 gate/up (paired form).
    pub mxfp4pair: crate::mxfp4_pair::Mxfp4PairMatvec,
    /// V4.1 Engram gate + residual add (layers 1, 14).
    pub engram_gate: crate::engram_gate::EngramGateAdd,
    /// V4.1 compressed-KV fake quant (E2M1 × E4M3 per 16).
    pub fp4kv: crate::fp4_kv::Fp4KvQuant,
    pub q4d: crate::q4_k_dense::Q4_KDenseMatvec,
    pub q5d: crate::q5_k_dense::Q5_KDenseMatvec,
    pub q6d: crate::q6_k_dense::Q6_KDenseMatvec,
    pub dense_gemm: crate::dense_gemm::DenseGemmDp4a,
    pub hc_sigmoid: HcSigmoidBias,
    pub hc_weighted: HcWeightedSum,
    pub hc_sinkhorn: HcSinkhorn,
    pub mhc_pre_fused: crate::MhcPreFused,
    /// Arena mHC pre-mix in 2 launches per sub-block (`V41_MHC_ARENA_FUSED`).
    pub mhc_arena: crate::MhcArena,
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
    /// Packed-FP8 compressed-KV producer / gather / expand (ratio-4 main
    /// compressors, see `CompKvStore`).
    pub comp_kv_fp8: crate::CompKvFp8,
    /// Packed-E2M1 indexer-key producer / expand (ratio-4 indexer
    /// compressors, see `CompKvStore::E2m1`).
    pub index_kv_e2m1: crate::IndexKvE2m1,
    pub router_topk: RouterTopk,
    /// CSA indexer kernels (used only on the dGPU's ratio==4 layers, but
    /// instantiated unconditionally so the engine struct stays symmetric).
    /// Hadamard128 + FP4 QAT round trip on indexer Q / indexer compressor
    /// KV rows (ds4 5bc1e6d graph-correctness fix).
    pub indexer_qat: crate::IndexerQat,
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
    /// ARCH_SPEC 1.5 level one: layer 20 publishes the candidate blocks that
    /// index sources 24/28/32/36 mask against. See `candidate_blocks`.
    pub candidate_blocks: crate::candidate_blocks::CandidateBlocks,
    pub indexer_bitpack: crate::IndexerBitpack,
    pub vec_scale: crate::VecScaleInplace,
    /// By-expert MoE pre-pass — inverts d_selected into per-expert
    /// (token, slot) lists. Used by the prefill iGPU MoE path.
    pub moe_group_builder: crate::moe_group_builder::MoeGroupBuilder,
    /// M62: expert-selection histogram accumulator. Launched on the dGPU
    /// only (router lives there) but instantiated on both arches so the
    /// engine struct stays symmetric.
    pub expert_sel_count: crate::ExpertSelCount,

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
        let hc = Stream::new(device.id)?;
        let is_igpu = device.properties()?.integrated;
        let label: &'static str = if is_igpu { "igpu" } else { "dgpu" };
        let events = EventPool::new(label, EVENT_POOL_CAPACITY)?;
        Ok(Self {
            device,
            is_gfx11: arch.starts_with("gfx11"),
            compute,
            xfer,
            hc,
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
            iq3: crate::iq3_xxs::Iq3XxsMatvec::for_arch(arch)?,
            mxfp4: crate::mxfp4::Mxfp4Matvec::for_arch(arch)?,
            iq2s: crate::iq2_s::Iq2SPairMatvec::for_arch(arch)?,
            iq2xs: crate::iq2_xs::Iq2XsPairMatvec::for_arch(arch)?,
            iq3pair: crate::iq3_xxs_pair::Iq3XxsPairMatvec::for_arch(arch)?,
            iq3s: crate::iq3_s::Iq3SPairMatvec::for_arch(arch)?,
            mxfp4pair: crate::mxfp4_pair::Mxfp4PairMatvec::for_arch(arch)?,
            engram_gate: crate::engram_gate::EngramGateAdd::for_arch(arch)?,
            fp4kv: crate::fp4_kv::Fp4KvQuant::for_arch(arch)?,
            q4d: crate::q4_k_dense::Q4_KDenseMatvec::for_arch(arch)?,
            q5d: crate::q5_k_dense::Q5_KDenseMatvec::for_arch(arch)?,
            q6d: crate::q6_k_dense::Q6_KDenseMatvec::for_arch(arch)?,
            dense_gemm: crate::dense_gemm::DenseGemmDp4a::for_arch(arch)?,
            hc_sigmoid: HcSigmoidBias::for_arch(arch)?,
            hc_weighted: HcWeightedSum::for_arch(arch)?,
            hc_sinkhorn: HcSinkhorn::for_arch(arch)?,
            mhc_pre_fused: crate::MhcPreFused::for_arch(arch)?,
            mhc_arena: crate::MhcArena::for_arch(arch)?,
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
            comp_kv_fp8: crate::CompKvFp8::for_arch(arch)?,
            index_kv_e2m1: crate::IndexKvE2m1::for_arch(arch)?,
            router_topk: RouterTopk::for_arch(arch)?,
            indexer_qat: crate::IndexerQat::for_arch(arch)?,
            indexer_score: crate::IndexerScore::for_arch(arch)?,
            indexer_score_wmma: if arch.starts_with("gfx1200") || arch.starts_with("gfx1201") {
                Some(crate::IndexerScoreWmma::for_arch(arch)?)
            } else {
                None
            },
            indexer_topk: crate::IndexerTopk::for_arch(arch)?,
            indexer_topk_bitonic: crate::IndexerTopkBitonic::for_arch(arch)?,
            indexer_gather: crate::IndexerGather::for_arch(arch)?,
            candidate_blocks: crate::candidate_blocks::CandidateBlocks::for_arch(arch)?,
            indexer_bitpack: crate::IndexerBitpack::for_arch(arch)?,
            vec_scale: crate::VecScaleInplace::for_arch(arch)?,
            moe_group_builder: crate::moe_group_builder::MoeGroupBuilder::for_arch(arch)?,
            expert_sel_count: crate::ExpertSelCount::for_arch(arch)?,
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
    /// mHC split (V4.1): `hc_src_*` = `residual` final on dgpu.compute, so the
    /// side stream may read it; `hc_collapse_*` = `hc_weighted` has consumed the
    /// OLD carry, so the side stream may overwrite it with this sub-block's pre;
    /// `hc_mixes_*` = `split` written, so `hc_post` may consume it.
    pub hc_src_attn: Event,
    pub hc_collapse_attn: Event,
    pub hc_mixes_attn: Event,
    pub hc_src_ffn: Event,
    pub hc_collapse_ffn: Event,
    pub hc_mixes_ffn: Event,
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
            let hc_src_attn = Event::new_no_timing()?;
            let hc_collapse_attn = Event::new_no_timing()?;
            let hc_mixes_attn = Event::new_no_timing()?;
            let hc_src_ffn = Event::new_no_timing()?;
            let hc_collapse_ffn = Event::new_no_timing()?;
            let hc_mixes_ffn = Event::new_no_timing()?;
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
                hc_src_attn,
                hc_collapse_attn,
                hc_mixes_attn,
                hc_src_ffn,
                hc_collapse_ffn,
                hc_mixes_ffn,
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
    /// Third per-layer event set: lane C of the N-lane arena decode step
    /// (`forward_step_arena_lanes`, 2026-09-21).
    pub sync_events_t2: HetSyncEvents,
    /// M54: per-layer MoE-ready signal words (pinned host memory). The
    /// dGPU xfer stream writes `signal[layer] = token_seq` right after the
    /// selected push; the pre-issued iGPU lane waits GTE on it. Value
    /// waits compare at execution time, so the whole iGPU lane can be
    /// enqueued at token start (event waits would snapshot-no-op).
    pub moe_signal: v4flash_hip::PinnedBuffer<u32>,
    /// Monotonic per-token sequence for `moe_signal` (starts at 1 on the
    /// first token; u32 wrap is ~4B tokens, ignored).
    pub token_seq: std::sync::atomic::AtomicU32,
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
    /// Diagnostic: `trace::epoch_ns()` at which the last `forward_token_impl`
    /// returned; the next call reports the elapsed gap as `gap_us`. 0 = none yet.
    pub last_token_end_ns: std::sync::atomic::AtomicU64,
    /// S2 shared selection (V4.1 CSA2 §1.4). The 8 `index_source_layer_ids` run the
    /// indexer; every layer between two sources REUSES the most recent source's
    /// selection (`shared_attn.topk_idxs` in the reference). Because reuse layers also
    /// share the compressed store, the GATHERED buffer is reusable as-is — no rescore,
    /// no regather, just point attention at it.
    ///
    /// `src` is the store group `kv_source_of(layer).unwrap_or(layer)` the cached gather
    /// belongs to, or -1 for "none this token"; `rows` is its valid row count. Guarding
    /// on the store group is what stops a selection leaking across a kv-source boundary
    /// (e.g. layer 8 starts a new store, so layer 2's selection must not carry into it).
    pub last_idx_gather_src: std::sync::atomic::AtomicI32,
    pub last_idx_gather_rows: std::sync::atomic::AtomicU32,

    /// M62: persistent expert-selection histogram banks on the dGPU,
    /// `u32[N_LAYER × N_EXPERT]` each (88 KB). Accumulated by
    /// `record_sel_stats` on de.compute after router top-k (no hot-path
    /// readbacks); harvested + zeroed by `harvest_sel_stats` at the
    /// server's snapshot-save points. Mutex keeps the engine `Sync`
    /// (uncontended in practice — one engine worker thread).
    pub sel_stats_prefill: std::sync::Mutex<DeviceBuffer<u32>>,
    pub sel_stats_decode: std::sync::Mutex<DeviceBuffer<u32>>,
    /// Tokens accumulated into each bank since the last harvest.
    pub sel_tokens_prefill: std::sync::atomic::AtomicU64,
    pub sel_tokens_decode: std::sync::atomic::AtomicU64,
    /// Optional second box serving a slice of the routed experts
    /// (`docs/v41/REMOTE_EXPERTS.md`). Connected when `V41_REMOTE_ADDR` is set.
    ///
    /// The point is residency, not compute: with the daemon owning
    /// `L0-L19:192-383`, this box only has to hold the other half of each
    /// encoder layer, which brings prefill's working set inside the pool so it
    /// stops re-paging every layer for every chunk.
    ///
    /// `Mutex` because the two prefill lanes share one connection and the
    /// protocol is a single FIFO request stream.
    pub remote: Option<std::sync::Mutex<super::remote_experts::RemoteExpertClient>>,
}

/// Connect to the remote expert daemon named by `V41_REMOTE_ADDR` (e.g.
/// `10.99.0.2:7431`). Returns `None` when unset. A connection FAILURE is a hard
/// error rather than a silent fallback: running with the remote half of the
/// experts quietly missing would either page them locally (destroying the point)
/// or compute without them (silently wrong), and both are worse than refusing to
/// start.
fn connect_remote_experts() -> Option<std::sync::Mutex<super::remote_experts::RemoteExpertClient>> {
    let addr = std::env::var("V41_REMOTE_ADDR").ok()?;
    let mut opts = super::remote_experts::SocketOptions::default();
    // Link A/B knobs (profile audit 2026-09-21: the hub's socket sees ~24k
    // out-of-order segments and box 2 sends ~3.8k spurious retransmits per
    // 8-row burst; suspects are the hub's SO_BUSY_POLL receive path and the
    // delayed ACKs that inflate box 2's srtt to ~6 ms).
    if let Some(v) = std::env::var("V41_REMOTE_BUSY_POLL_US").ok().and_then(|v| v.parse().ok()) {
        opts.busy_poll_us = v;
    }
    if std::env::var("V41_REMOTE_QUICKACK").as_deref() == Ok("1") {
        opts.quickack = true;
    }
    match super::remote_experts::RemoteExpertClient::connect(&addr, &opts) {
        Ok(c) => {
            let info = c.info();
            let owned: Vec<u32> = (0..info.n_layer).filter(|&l| info.owned_count(l) > 0).collect();
            let per_layer = owned.first().map(|&l| info.owned_count(l)).unwrap_or(0);
            eprintln!(
                "remote experts: {addr} — {} resident experts, {} layers {:?}, {} per layer, \
                 max_batch {}, decode_max_b {}",
                info.n_resident, owned.len(),
                if owned.len() > 6 { format!("{}..{}", owned[0], owned[owned.len()-1]) }
                else { format!("{owned:?}") },
                per_layer, info.max_batch, info.decode_max_b,
            );
            Some(std::sync::Mutex::new(c))
        }
        Err(e) => panic!("V41_REMOTE_ADDR={addr} set but connect failed: {e:#}"),
    }
}

impl HeterogeneousEngine {
    /// Lanes that have their own sync events (`sync_events_lane`).
    pub const MAX_LANES: usize = 3;
    /// Per-layer sync events for lane `lane` of an N-lane step (0..MAX_LANES).
    /// A lane past the last set used to alias lane 2's events silently; the
    /// drivers now reject `n > MAX_LANES`, so reaching the panic is a bug.
    pub fn sync_events_lane(&self, lane: usize) -> &HetSyncEvents {
        match lane {
            0 => &self.sync_events,
            1 => &self.sync_events_t1,
            2 => &self.sync_events_t2,
            _ => panic!("sync_events_lane({lane}): only {} lanes have sync events", Self::MAX_LANES),
        }
    }
    /// See `RemoteExpertClient::drain_in_flight`. Returns the number drained.
    /// Redial box 2 if the link is known broken. Called at a request BOUNDARY,
    /// where nothing is in flight. Returns Ok(false) if no reconnect was needed.
    ///
    /// Without this a single box-2 error is a permanent outage: the writer thread
    /// exits on the broken pipe and every later submit fails, so the process
    /// serves 500s until someone restarts it by hand.
    pub fn remote_reconnect_if_dead(&self) -> eyre::Result<bool> {
        let Some(m) = self.remote.as_ref() else { return Ok(false) };
        let Ok(mut c) = m.lock() else { return Ok(false) };
        if !c.is_dead() {
            return Ok(false);
        }
        c.ensure_connected()?;
        Ok(true)
    }

    /// Per-phase `SO_BUSY_POLL` on the box-2 link (docs/v41/LINK_IDLE_LATENCY.md,
    /// 2026-09-19). Decode's 20 KB response is ONE TCP segment: a reader that is
    /// still spinning when it lands drains all its 4 KB frames in one poll
    /// (~50 us of link) where a sleeping reader pays an interrupt+wake per frame
    /// (~190 us). The reader's spin starts at the previous handoff, so the
    /// window must cover a whole per-layer period (~2.2 ms): 3000 us. But a
    /// response LARGER than the 65,520 B MTU (>= 2 segments: any B >= 4, every
    /// prefill/verify chunk) is HELD by the busy-poll loop until the window
    /// expires -- link ~= window - 1.9 ms -- so those phases must run at 500.
    /// Measured with `deepstrix-expert-bench`, table in the doc.
    ///
    /// The kernel refuses a window above `net.core.busy_read` for an
    /// unprivileged process (`scripts/link_latency_step.sh 4s on` raises it);
    /// refusal is logged once and the socket keeps its previous window.
    /// `V41_DECODE_BUSY_POLL_US` / `V41_BATCH_BUSY_POLL_US` override the
    /// defaults (3000 / 50); `V41_DECODE_BUSY_POLL_US=0` disables the switch.
    ///
    /// 2026-09-21: the multi-segment hold is TWO effects. The sender's TSO/GSO
    /// deferral (~1 ms per reply over one segment at ANY window) is removed by
    /// `ethtool -K thunderbolt0 tso off gso off` on the sending box
    /// (scripts/apply_host_tuning.sh); what remains scales with the RECEIVER's
    /// window (2.1 ms at 3000, 0.36 ms at 20 for an 82 KB reply), so the batch
    /// phase now defaults to 50 us: measured link 364/404/503/781 us at 4/8/16/32
    /// rows f32 with TSO off + window 20, against 1075-1272 before.
    pub fn remote_set_phase_busy_poll(&self, decode: bool) {
        static DECODE_US: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
            std::env::var("V41_DECODE_BUSY_POLL_US").ok().and_then(|v| v.parse().ok()).unwrap_or(3000)
        });
        static BATCH_US: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
            std::env::var("V41_BATCH_BUSY_POLL_US").ok().and_then(|v| v.parse().ok()).unwrap_or(50)
        });
        let Some(m) = self.remote.as_ref() else { return };
        let Ok(mut c) = m.lock() else { return };
        // The daemon adapts its own reader window to this (REQ_FLAG_DECODE),
        // whether or not the hub's switch is enabled.
        c.set_decode_phase(decode);
        if *DECODE_US == 0 {
            return;
        }
        c.set_busy_poll_us(if decode { *DECODE_US } else { *BATCH_US });
    }

    pub fn remote_drain_in_flight(&self) -> usize {
        self.remote
            .as_ref()
            .and_then(|m| m.lock().ok())
            .map(|mut c| c.drain_in_flight())
            .unwrap_or(0)
    }
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
        self.remote_set_phase_busy_poll(true);
        self.forward_token_impl(
            dgpu_scratch, igpu_scratch, state, weights, input_hc_host, pos, token_id, None, None,
            None,
        )
    }

    /// M7 paged-expert decode: identical to [`Self::forward_token`] except each
    /// layer's routed MoE pages the router's actual picks out of `pager` instead
    /// of reading iGPU-resident experts. Correctness-first — this takes the
    /// standalone (non-fused) per-layer path and syncs once per layer to read back
    /// `d_selected`, so it is slow by construction.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_token_paged(
        &self,
        dgpu_scratch: &mut super::DgpuScratch,
        igpu_scratch: &mut super::IgpuScratch,
        state: &mut super::HetModelState,
        weights: &super::HetModelWeights,
        input_hc_host: &[f32],
        pos: u32,
        token_id: i32,
        pager: &mut super::expert_pager::ExpertPager,
        engram_rows: Option<&[Vec<f32>]>,
    ) -> color_eyre::eyre::Result<()> {
        self.remote_set_phase_busy_poll(true);
        self.forward_token_impl(
            dgpu_scratch,
            igpu_scratch,
            state,
            weights,
            input_hc_host,
            pos,
            token_id,
            Some(pager),
            engram_rows,
            None,
        )
    }

    /// Publish the per-token device scalars a single layer consumes: the rope
    /// position and the monotonic KV append slot.
    ///
    /// `forward_token_impl` writes these ONCE before its layer loop, because in
    /// token-major order every layer's counters evolve in lockstep. A
    /// LAYER-MAJOR driver (the speculative verify: all B rows at layer L, then
    /// layer L+1) breaks that invariant — after layer 0 has run all B rows its
    /// `n_raw` is +B while layer 1's is still +0 — so it must publish per
    /// (row, layer) with the slot taken from THAT layer's state. Without this
    /// the rope ropes at a stale position and attention is quietly wrong.
    pub fn publish_pos_slot(
        &self,
        dgpu_scratch: &mut super::DgpuScratch,
        pos: u32,
        slot: u32,
    ) -> color_eyre::eyre::Result<()> {
        self.set_current_cached(self.dgpu.device)?;
        let pos_ptr = dgpu_scratch.pos_dev.raw() as *mut u32;
        let slot_ptr = dgpu_scratch.kv_slot_dev.raw() as *mut u32;
        // SAFETY: scratch outlives the layer; the writes are stream-ordered on
        // `de.compute` ahead of every consumer, exactly as in
        // `forward_token_impl`.
        unsafe {
            self.dgpu.compute.write_value32(pos_ptr, pos)?;
            self.dgpu.compute.write_value32(slot_ptr, slot)?;
        }
        Ok(())
    }

    /// Compact every layer's raw KV window down to slots `[0, n_raw)` and reset
    /// `raw_off = 0`, so a subsequent PREFILL-path verify (which addresses the
    /// window from slot 0) attends to exactly the same keys DECODE does.
    ///
    /// Decode keeps a MONOTONIC ring `[raw_off, raw_off + n_raw)` and only
    /// compacts once per B_MAX tokens; the speculative verify runs the prefill
    /// path, which assumes `[0, n_raw)`. While `raw_off == 0` they coincide, but
    /// once the window has slid the verify reads the wrong slots (KL(decode||
    /// verify) jumps from ~0.0008 to ~4 nats). Calling this before the verify's
    /// `mark_kv` keeps them in agreement. Same overlap-safe two-hop copy the
    /// decode wrap uses; a no-op on layers already at `raw_off == 0`.
    pub fn normalize_raw_windows(
        &self,
        dgpu_scratch: &mut super::DgpuScratch,
        state: &mut super::HetModelState,
    ) -> color_eyre::eyre::Result<()> {
        use crate::config::N_HEAD_DIM;
        self.set_current_cached(self.dgpu.device)?;
        let head_dim = N_HEAD_DIM as usize;
        for ls in state.layers.iter_mut() {
            if ls.raw_off == 0 || ls.n_raw == 0 {
                ls.raw_off = 0;
                continue;
            }
            let win_len = (ls.n_raw as usize) * head_dim;
            let src_off = (ls.raw_off as usize) * head_dim;
            {
                let mut sc = dgpu_scratch.kv_wrap_scratch.slice_view_mut(0, win_len);
                let src = ls.kv_cache.slice_view(src_off, win_len);
                sc.copy_from_buffer_async(&src, &self.dgpu.compute)?;
            }
            {
                let sc = dgpu_scratch.kv_wrap_scratch.slice_view(0, win_len);
                let mut dst = ls.kv_cache.slice_view_mut(0, win_len);
                dst.copy_from_buffer_async(&sc, &self.dgpu.compute)?;
            }
            ls.raw_off = 0;
        }
        self.dgpu.compute.synchronize()?;
        Ok(())
    }

    /// One DSpark draft step: entry, three drafter layers, exit.
    ///
    /// Spans both devices. The drafter's 7.93 GB of layers only fit on the iGPU;
    /// its head is TIED to the main model's `output`, which is dGPU-resident.
    /// So the layer stack runs on the iGPU and the exit on the dGPU, joined by a
    /// 410 KB host round trip — at decode rates that is far under the noise
    /// floor, and it sidesteps the peer-copy stream rule entirely.
    ///
    /// `pos` is the reference's `start_pos`: the position of the token whose
    /// residual `main_hidden` holds (from `MtpCapture::read` on the decode
    /// path, or the batched verify's per-row capture on the accept path). `token_row` is `embed_lookup` of the token
    /// sampled FROM that forward, which sits at `pos + 1`. The returned drafts
    /// are therefore predictions for positions `pos + 2 ..= pos + 1 + MTP_BLOCK`.
    #[allow(clippy::too_many_arguments)]
    /// Ring-write-only drafter advance (see `MtpState::advance_ring`): populate
    /// the drafter's KV ring at `pos` from `main_hidden` without drafting.
    pub fn dspark_advance_ring(
        &self,
        mtp_state: &mut super::mtp::MtpState,
        w: &super::weights::MtpWeights,
        pos: u32,
        main_hidden: &[f32],
        token_row: &[f32],
        noise_row: &[f32],
    ) -> color_eyre::eyre::Result<()> {
        self.set_current_cached(self.igpu.device)?;
        mtp_state.advance_ring(
            &self.igpu, &self.igpu.compute, w, &super::mtp::mtp_rope(), pos, main_hidden,
            token_row, noise_row,
        )?;
        self.igpu.compute.synchronize()?;
        Ok(())
    }

    /// Cheap ring-only advance for an intermediate accepted position. See
    /// `MtpState::ring_write_only`.
    pub fn dspark_ring_write_only(
        &self,
        mtp_state: &mut super::mtp::MtpState,
        w: &super::weights::MtpWeights,
        pos: u32,
        main_hidden: &[f32],
    ) -> color_eyre::eyre::Result<()> {
        self.set_current_cached(self.igpu.device)?;
        mtp_state.inject_main_hidden(main_hidden)?;
        mtp_state.ring_write_only(
            &self.igpu, &self.igpu.compute, w, &super::mtp::mtp_rope(), pos,
        )?;
        self.igpu.compute.synchronize()?;
        Ok(())
    }

    pub fn dspark_draft(
        &self,
        mtp_state: &mut super::mtp::MtpState,
        exit: &mut super::mtp::MtpExit,
        main_hidden: &[f32],
        w: &super::weights::MtpWeights,
        xw: &super::weights::MtpExitWeights,
        weights: &super::HetModelWeights,
        markov_embd: &[u8],
        markov_dtype: v4flash_core::gguf::GgufType,
        pos: u32,
        token_row: &[f32],
        noise_row: &[f32],
        first_token: i32,
    ) -> color_eyre::eyre::Result<([i32; super::mtp::MTP_BLOCK], [i32; super::mtp::MTP_BLOCK])> {
        // `V41_DSPARK_DRAFT_TIMING=1`: split the draft into its three parts. The
        // drafter is only 3 layers but costs ~26 ms/step against the 40-layer
        // main model's 68.7 ms/token, and it straddles BOTH GPUs: the layers run
        // on the iGPU, the exit on the dGPU (the head is tied to the main model's
        // `output`, which lives there), with a blocking iGPU stream drain and a
        // ~410 KB host round trip between them. Without this split there is no way
        // to tell layer cost from handoff cost.
        let dt = std::env::var("V41_DSPARK_DRAFT_TIMING").as_deref() == Ok("1");
        let t0 = std::time::Instant::now();
        self.set_current_cached(self.igpu.device)?;
        mtp_state.inject_main_hidden(main_hidden)?;
        mtp_state.forward(
            &self.igpu,
            &self.igpu.compute,
            w,
            &super::mtp::mtp_rope(),
            pos,
            token_row,
            noise_row,
        )?;
        // SLACK PROBE: stall the drafter's stream by a known amount, so the
        // step can be regressed against it. See `slack_probe_spin`. Sweeping
        // this answers whether the drafter is on the critical path at all --
        // which a device-time counter cannot, and which we got wrong once:
        // B-packing attn_q_b removed 8.8 ms/step of drafter device time for
        // zero end-to-end gain.
        if let Some(ticks) = super::mtp::slack_probe_ticks("draft") {
            self.igpu.q8.slack_probe_spin(&self.igpu.compute, ticks)?;
        }
        let t_enq = std::time::Instant::now();
        self.igpu.compute.synchronize()?;
        // The drafter's whole forward is now complete on the device, so every
        // mode-3 event pair has resolved and can be charged to its counter.
        super::mtp::drain_event_spans();
        let t_sync = std::time::Instant::now();
        let mut h_host = vec![0.0f32; mtp_state.h.len()];
        mtp_state.h.copy_to_host(&mut h_host)?;
        let mut pre_host = vec![0.0f32; mtp_state.pre_carry().len()];
        mtp_state.pre_carry().copy_to_host(&mut pre_host)?;
        let t_copy = std::time::Instant::now();

        self.set_current_cached(self.dgpu.device)?;
        if dt {
            let ms = |a: std::time::Instant, b: std::time::Instant| {
                format!("{:.2}", (b - a).as_secs_f64() * 1e3)
            };
            let (hcmix, rms, attn, moe, post) = super::mtp::take_layer_host_us();
            tracing::info!(
                igpu_enqueue_ms = ms(t0, t_enq),
                igpu_sync_ms = ms(t_enq, t_sync),
                h2d_copy_ms = ms(t_sync, t_copy),
                bytes = (mtp_state.h.len() + mtp_state.pre_carry().len()) * 4,
                // Host (enqueue) time INSIDE the 3 layers, summed, us.
                // `V41_DSPARK_LAYER_TIMING=1` or these are all zero.
                l_hcmix_us = hcmix,
                l_rms_us = rms,
                l_attn_us = attn,
                l_moe_us = moe,
                l_hcpost_us = post,
                l_attn_split_us = {
                    let (q, kv, qa, o) = super::mtp::take_attn_split_us();
                    format!("qloop={q} kv={kv} qa={qa} outproj={o} kvcopy={}", super::mtp::take_kvcopy_us())
                },
                l_kernel_split_us = {
                    let k = super::mtp::take_kernel_split_us();
                    format!(
                        "oquant={} owa={} owb={} | mrouter={} mtopk={} mq8k={} mgateup={} mdown={}",
                        k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]
                    )
                },
                "dspark.draft.split"
            );
        }
        let r = exit.forward(
            &self.dgpu,
            &self.dgpu.compute,
            &h_host,
            &pre_host,
            xw,
            &weights.global.output,
            markov_embd,
            markov_dtype,
            first_token,
        );
        if dt {
            tracing::info!(
                exit_ms = format!("{:.2}", t_copy.elapsed().as_secs_f64() * 1e3),
                total_ms = format!("{:.2}", t0.elapsed().as_secs_f64() * 1e3),
                "dspark.draft.exit"
            );
        }
        // The drafter's own stages live in the same pools the per-token export
        // already drained, so drain again or they are lost at the next reset.
        let _ = self.export_pending_perfetto();
        r
    }

    /// Emit any perfetto pairs recorded since the last export.
    ///
    /// `forward_token_impl` exports at its END, so anything the caller runs after
    /// it -- notably `dspark_draft` from the accept loop -- was recorded into the
    /// pools and then discarded by the next token's `reset()`. The drafter is
    /// ~26 ms/step across BOTH GPUs and was invisible on every trace because of
    /// this. Idempotent: the pool's watermark means a pair is emitted once.
    pub fn export_pending_perfetto(&self) -> eyre::Result<()> {
        let Some(exp_lock) = &self.perfetto else { return Ok(()) };
        let Ok(mut exp) = exp_lock.lock() else { return Ok(()) };
        self.dgpu.events.for_each_pair_new(|name, s, e| {
            let track = if name.contains(".xfer") || name.contains(".peer_push") {
                &exp.dgpu_xfer
            } else {
                &exp.dgpu_compute
            };
            exp.emit_slice(track, name, s, e)
        })?;
        self.igpu.events.for_each_pair_new(|name, s, e| {
            let track = if name.contains(".xfer") || name.contains(".peer_push") {
                &exp.igpu_xfer
            } else {
                &exp.igpu_compute
            };
            exp.emit_slice(track, name, s, e)
        })?;
        Ok(())
    }

    /// `forward_token_paged` that also captures the residuals the DSpark drafter
    /// eats (entering layers 37/38/39). Separate entry point so the hot path
    /// keeps its signature; `mtp.begin()` is the caller's to call.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_token_paged_mtp(
        &self,
        dgpu_scratch: &mut super::DgpuScratch,
        igpu_scratch: &mut super::IgpuScratch,
        state: &mut super::HetModelState,
        weights: &super::HetModelWeights,
        input_hc_host: &[f32],
        pos: u32,
        token_id: i32,
        pager: &mut super::expert_pager::ExpertPager,
        engram_rows: Option<&[Vec<f32>]>,
        mtp: &mut super::mtp::MtpCapture,
    ) -> color_eyre::eyre::Result<()> {
        self.remote_set_phase_busy_poll(true);
        self.forward_token_impl(
            dgpu_scratch,
            igpu_scratch,
            state,
            weights,
            input_hc_host,
            pos,
            token_id,
            Some(pager),
            engram_rows,
            Some(mtp),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_token_impl(
        &self,
        dgpu_scratch: &mut super::DgpuScratch,
        igpu_scratch: &mut super::IgpuScratch,
        state: &mut super::HetModelState,
        weights: &super::HetModelWeights,
        input_hc_host: &[f32],
        pos: u32,
        token_id: i32,
        mut pager: Option<&mut super::expert_pager::ExpertPager>,
        engram_rows: Option<&[Vec<f32>]>,
        mut mtp: Option<&mut super::mtp::MtpCapture>,
    ) -> color_eyre::eyre::Result<()> {
        use crate::config::{HC_DIM, N_EXPERT, N_LAYER};
        use tracing::debug_span;

        // A previous request that errored between a compressor lend and its
        // hand-back leaves the store on the reuse layer. Repair before we read it.
        state.restore_compressor_lending();

        if input_hc_host.len() != HC_DIM as usize {
            return Err(color_eyre::eyre::eyre!(
                "input_hc_host len {} != HC_DIM {}",
                input_hc_host.len(),
                HC_DIM
            ));
        }
        // Caller-side gap since the previous token's return (see
        // `TokenTiming::gap_us`), taken BEFORE anything else in this call.
        let fn_entry = std::time::Instant::now();
        let gap_us = {
            let prev = self.last_token_end_ns.load(std::sync::atomic::Ordering::Relaxed);
            if prev == 0 { 0 } else { super::trace::epoch_ns().saturating_sub(prev) / 1000 }
        };
        let _token_span = debug_span!("het.token", pos, token_id).entered();

        // Reset event pools for this token.
        self.dgpu.events.reset();
        self.igpu.events.reset();
        // A cached selection never crosses a token boundary.
        self.last_idx_gather_src
            .store(-1, std::sync::atomic::Ordering::Relaxed);
        super::trace::phase::reset();
        let pager_c0 = pager.as_deref().map(|p| p.counters()).unwrap_or_default();

        self.set_current_cached(self.dgpu.device)?;
        dgpu_scratch.residual.copy_from_host(input_hc_host)?;
        // Dump the layer-0 input (== embedded token vector) if
        // DEEPSTRIX_DUMP_RESIDUAL_DIR is set. Index 00 in the file
        // naming. Per-layer post-output dumps land at indices 01..43.
        maybe_dump_residual(0, &dgpu_scratch.residual)?;

        // M59: publish the per-token device scalars consumed by the merged
        // qkv_chain graphs (rope pos + monotonic KV append slot; the slot is
        // identical across layers — counters evolve in lockstep). One pair
        // of 4-byte stream writes per token, ordered before layer 0 on the
        // dGPU compute stream.
        {
            let slot = state.layers[0].raw_off + state.layers[0].n_raw;
            let pos_ptr = dgpu_scratch.pos_dev.raw() as *mut u32;
            let slot_ptr = dgpu_scratch.kv_slot_dev.raw() as *mut u32;
            // SAFETY: scratch buffers outlive the token; writes are stream-
            // ordered on de.compute ahead of all consumers.
            unsafe {
                self.dgpu.compute.write_value32(pos_ptr, pos)?;
                self.dgpu.compute.write_value32(slot_ptr, slot)?;
            }
        }
        let token_start = std::time::Instant::now();
        let pre_us = token_start.duration_since(fn_entry).as_micros() as u64;
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
            && dump_subtensor_layers.is_empty()
            // M56: pre-issue's iGPU lane doesn't carry the het-split
            // branch; the two are not composed yet.
            && weights.dgpu_layers[0].hot_experts.is_none();
        // Advance the token sequence for the moe_signal protocol (the
        // write side in forward_layer reads this with `load`). Done
        // unconditionally so the signal words stay in lockstep with
        // tokens whether or not pre-issue is active.
        self.token_seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
            // Per-layer residual dump, decode side, WITHOUT the `force_standalone`
            // path -- arming `DEEPSTRIX_DUMP_SUBTENSOR_LAYERS` routes decode into
            // `forward_layer_standalone_graphs`, which crashes with
            // HIP 700 (hipErrorIllegalAddress), so decode has never actually been
            // able to dump. See KNOWN_BUGS.
            //
            // Reading the residual ENTERING the layer is safe whatever the graph
            // fusion does, because nothing for this layer has run yet. Tag matches
            // the prefill side (`pf_pre_residual_p<POS>`) so the two are directly
            // diffable to find the first layer where verify departs from decode.
            if subtensor_dump_armed(layer) {
                self.dgpu.compute.synchronize()?;
                maybe_dump_subtensor_f32(
                    layer,
                    &format!("dec_pre_residual_p{pos}"),
                    &dgpu_scratch.residual,
                )?;
            }
            // DSpark: the drafter eats the residual ENTERING layers 37/38/39,
            // so this must run before the layer does. A no-op for every other
            // layer.
            if let Some(c) = mtp.as_deref_mut() {
                c.on_layer(&self.dgpu, &self.dgpu.compute, layer as i32, &dgpu_scratch.residual)?;
            }
            let next_dlw = if layer + 1 < N_LAYER as usize {
                Some(&weights.dgpu_layers[layer + 1])
            } else {
                None
            };
            // V4.1 reuse layers: lend the KV source's store for this layer's forward.
            let kv_src = crate::config::kv_source_of(layer);
            if let Some(src) = kv_src {
                let st = state.layers[src].compressor.take();
                state.layers[layer].compressor = st;
            }
            // When sub-tensor dumping is active at this layer, fall
            // back to the standalone-graphs path. The combined
            // cross-layer graph (ffn_combine fused with next layer's
            // mhc_pre_attn) writes the NEXT layer's attn_cur /
            // attn_input_norm into the same scratch fields, clobbering
            // the current layer's values before we can read them.
            // Standalone runs each layer's mhc_pre_attn separately so
            // the post-layer-N buffers are stable.
            // V4.1 Engram (layers 1 and 14): the n-gram rows for THIS token were
            // gathered host-side by the caller (SSD-backed 189 GiB tables, never
            // resident). Stage them before the layer reads them, in the same
            // ENGRAM_LAYERS order the caller built them in.
            if weights.dgpu_layers[layer].engram.is_some() {
                let rows = engram_rows.and_then(|rs| {
                    crate::config::ENGRAM_LAYERS
                        .iter()
                        .position(|&l| l as usize == layer)
                        .and_then(|i| rs.get(i))
                });
                match rows {
                    Some(r) => {
                        let t = std::time::Instant::now();
                        self.stage_engram_rows(dgpu_scratch, r)?;
                        super::trace::phase::add(
                            &super::trace::phase::ENGRAM_STAGE_NS,
                            t.elapsed().as_nanos() as u64,
                        );
                    }
                    None => {
                        return Err(color_eyre::eyre::eyre!(
                            "layer {layer} needs Engram rows but the caller staged none"
                        ))
                    }
                }
            }
            // NOTE: this deliberately no longer includes the dump-layer list.
            // Routing decode through `forward_layer_standalone_graphs` to dump
            // crashes it (HIP 700); the residual dump above does not need it.
            let force_standalone = false && dump_subtensor_layers.contains(&layer);
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
                    match &comp.comp_kv {
                        CompKvStore::F16(buf) => {
                            maybe_dump_subtensor_f16_as_f32(
                                layer, "attn_comp_kv", buf, n_comp_elems
                            )?;
                        }
                        CompKvStore::Fp8 { rows, .. } => {
                            // Expand through the production kernel so the
                            // dump is exactly what attention would gather.
                            let mut tmp: DeviceBuffer<u16> =
                                DeviceBuffer::new(self.dgpu.device.id, n_comp_elems.max(head_dim))?;
                            self.dgpu.comp_kv_fp8.launch_expand(
                                &self.dgpu.compute, &mut tmp, rows, comp.n_comp,
                            )?;
                            self.dgpu.compute.synchronize()?;
                            maybe_dump_subtensor_f16_as_f32(
                                layer, "attn_comp_kv", &tmp, n_comp_elems
                            )?;
                        }
                        CompKvStore::E2m1(_) => {
                            return Err(eyre::eyre!("L{layer}: main compressor store cannot be E2M1"));
                        }
                    }
                }
                let n_raw_elems = (state.layers[layer].n_raw as usize) * head_dim;
                // M55: live window starts at raw_off (monotonic append).
                let raw_win = state.layers[layer].kv_cache.slice_view(
                    (state.layers[layer].raw_off as usize) * head_dim,
                    n_raw_elems,
                );
                maybe_dump_subtensor_f16_as_f32(layer, "raw_kv", &raw_win, n_raw_elems)?;
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
            } else if let Some(pg) = pager.as_deref_mut() {
                // M7: page this layer's routed experts from the pager's pool.
                self.forward_layer_standalone_graphs_paged(
                    dgpu_scratch,
                    igpu_scratch,
                    &mut state.layers[layer],
                    &weights.dgpu_layers[layer],
                    &weights.igpu_layers[layer],
                    pos,
                    token_id,
                    pg,
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
            // DEEPSTRIX_EXPERT_STATS=1: per-layer expert-selection histogram
            // (diagnostic; syncs every layer — slow). Prints cumulative
            // top-16 hit-rates every 32 tokens. Used to size M55 static
            // hot-expert placement.
            if let Some(src) = kv_src {
                let st = state.layers[layer].compressor.take();
                state.layers[src].compressor = st;
            }
            {
                static STATS: std::sync::LazyLock<
                    Option<std::sync::Mutex<(Vec<[u64; N_EXPERT as usize]>, u64)>>,
                > = std::sync::LazyLock::new(|| {
                    std::env::var_os("DEEPSTRIX_EXPERT_STATS").map(|_| {
                        std::sync::Mutex::new((vec![[0u64; N_EXPERT as usize]; N_LAYER as usize], 0u64))
                    })
                });
                // Optional hot-set hit-rate histogram: resident sets from the
                // same placement file + K the het-split loader uses.
                static HOTSETS: std::sync::LazyLock<Option<Vec<[bool; N_EXPERT as usize]>>> =
                    std::sync::LazyLock::new(|| {
                        let k: usize = super::weights::dgpu_hot_experts();
                        if k == 0 {
                            return None;
                        }
                        let path = super::weights::hot_expert_file_path();
                        super::weights::parse_hot_expert_file(&path, k).ok().map(|lists| {
                            lists
                                .iter()
                                .map(|ids| {
                                    let mut m = [false; N_EXPERT as usize];
                                    for &e in ids {
                                        m[e as usize] = true;
                                    }
                                    m
                                })
                                .collect()
                        })
                    });
                // (per-token hit counter, 21-bin hit-rate histogram in 5% steps)
                static HITHIST: std::sync::LazyLock<std::sync::Mutex<(u32, [u32; 21])>> =
                    std::sync::LazyLock::new(|| std::sync::Mutex::new((0, [0; 21])));
                // DEEPSTRIX_EXPERT_TRACE=<path>: per-token, per-layer expert ids
                // (u16 LE, N_EXPERT_USED per layer, N_LAYER layers per token,
                // in decode order). Feeds the cold-expert prefetch study — we
                // need the *sequence*, which the histogram above throws away.
                static TRACE: std::sync::LazyLock<
                    Option<std::sync::Mutex<(Vec<u16>, u64)>>,
                > = std::sync::LazyLock::new(|| {
                    std::env::var_os("DEEPSTRIX_EXPERT_TRACE")
                        .map(|_| std::sync::Mutex::new((Vec::new(), 0u64)))
                });
                if STATS.is_some() || TRACE.is_some() {
                    self.dgpu.compute.synchronize()?;
                    let mut sel = vec![0i32; crate::config::N_EXPERT_USED];
                    dgpu_scratch
                        .d_selected
                        .slice_view(0, crate::config::N_EXPERT_USED)
                        .copy_to_host(&mut sel)?;
                    if let Some(t) = &*TRACE {
                        let mut g = t.lock().unwrap();
                        for &e in &sel {
                            g.0.push(e.clamp(0, u16::MAX as i32) as u16);
                        }
                        if layer == (N_LAYER as usize) - 1 {
                            g.1 += 1;
                            if g.1 % 64 == 0 {
                                if let Ok(path) = std::env::var("DEEPSTRIX_EXPERT_TRACE") {
                                    let bytes: Vec<u8> = g
                                        .0
                                        .iter()
                                        .flat_map(|v| v.to_le_bytes())
                                        .collect();
                                    let _ = std::fs::write(&path, &bytes);
                                    eprintln!(
                                        "EXPERT_TRACE: {} tokens -> {} ({} bytes)",
                                        g.1,
                                        path,
                                        bytes.len()
                                    );
                                }
                            }
                        }
                    }
                  if let Some(m) = &*STATS {
                    let mut g = m.lock().unwrap();
                    for &e in &sel {
                        if (0..N_EXPERT as i32).contains(&e) {
                            g.0[layer][e as usize] += 1;
                        }
                    }
                    if let Some(hs) = &*HOTSETS {
                        let mut h = HITHIST.lock().unwrap();
                        for &e in &sel {
                            if (0..N_EXPERT as i32).contains(&e) && hs[layer][e as usize] {
                                h.0 += 1;
                            }
                        }
                        if layer == (N_LAYER as usize) - 1 {
                            let total = (N_LAYER as usize)
                                * crate::config::N_EXPERT_USED;
                            let rate = h.0 as f64 / total as f64;
                            let bin = ((rate * 20.0).round() as usize).min(20);
                            h.1[bin] += 1;
                            h.0 = 0;
                            let tokens: u32 = h.1.iter().sum();
                            if tokens % 32 == 0 {
                                let hist: Vec<String> = h
                                    .1
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, c)| **c > 0)
                                    .map(|(b, c)| format!("{}%:{}", b * 5, c))
                                    .collect();
                                eprintln!(
                                    "HOT_HIT_HIST after {tokens} tokens (per-token hit-rate bins): {}",
                                    hist.join(" ")
                                );
                            }
                        }
                    }
                    if layer == (N_LAYER as usize) - 1 {
                        g.1 += 1;
                        if g.1 % 32 == 0 {
                            // Per-layer top-K hit rates, summarized.
                            let mut hit16 = 0f64;
                            let mut hit32 = 0f64;
                            let mut tot = 0f64;
                            for l in 0..N_LAYER as usize {
                                let mut c: Vec<u64> = g.0[l].to_vec();
                                let sum: u64 = c.iter().sum();
                                c.sort_unstable_by(|a, b| b.cmp(a));
                                let t16: u64 = c.iter().take(16).sum();
                                let t32: u64 = c.iter().take(32).sum();
                                hit16 += t16 as f64;
                                hit32 += t32 as f64;
                                tot += sum as f64;
                            }
                            eprintln!(
                                "EXPERT_STATS after {} tokens: top16 hit {:.1}%  top32 hit {:.1}%",
                                g.1,
                                100.0 * hit16 / tot,
                                100.0 * hit32 / tot
                            );
                            // Dump per-layer descending-frequency expert ids
                            // (placement input). One line per layer.
                            if let Ok(path) = std::env::var("DEEPSTRIX_EXPERT_STATS") {
                                if path != "1" {
                                    // id:count pairs, descending frequency —
                                    // counts let the loader do GLOBAL greedy
                                    // slot allocation across layers.
                                    let mut out = String::new();
                                    for l in 0..N_LAYER as usize {
                                        let mut idx: Vec<usize> = (0..N_EXPERT as usize).collect();
                                        idx.sort_unstable_by_key(|&e| {
                                            std::cmp::Reverse(g.0[l][e])
                                        });
                                        let row: Vec<String> = idx
                                            .iter()
                                            .take(64)
                                            .map(|&e| format!("{}:{}", e, g.0[l][e]))
                                            .collect();
                                        out.push_str(&row.join(","));
                                        out.push('\n');
                                    }
                                    let _ = std::fs::write(&path, out);
                                }
                            }
                        }
                    }
                  }
                }
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
        // DECODE logits dump, companion to prefill's `V41_PREFILL_LOGITS_DUMP`.
        // Added 2026-09-14 because validating a DECODE-path change against PREFILL
        // logits proves nothing — which I did twice before noticing. Appends
        // N_VOCAB little-endian f32 per token.
        if let Ok(path) = std::env::var("V41_DECODE_LOGITS_DUMP") {
            use std::io::Write;
            self.dgpu.compute.synchronize()?;
            let mut host = vec![0f32; crate::config::N_VOCAB as usize];
            dgpu_scratch.logits.slice_view(0, crate::config::N_VOCAB as usize).copy_to_host(&mut host)?;
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
            let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
            f.write_all(&bytes)?;
        }
        // When N_LAYER is odd (43 in V4-Flash), the in-loop swaps leave residual /
        // residual_next inverted from token start. Without an extra swap
        // here, every token's layer 0 would read from a different
        // physical DeviceBuffer than the previous token's layer 0,
        // making it impossible to capture mhc_pre_attn / mhc_post_ffn
        // into HIP graphs (the captured pointer would be wrong on
        // alternating tokens). The extra swap restores the initial state
        // so layer N always operates on the same physical buffers across
        // every token.
        if N_LAYER % 2 == 1 {
            std::mem::swap(&mut dgpu_scratch.residual, &mut dgpu_scratch.residual_next);
        }
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
            // `_new`: emit only what has not been emitted yet. Anything recorded
            // AFTER this point in the step -- the DSpark drafter, which runs from
            // the accept loop -- is picked up by `export_pending_perfetto()`.
            self.dgpu.events.for_each_pair_new(|name, s, e| {
                let track = if name.contains(".xfer") || name.contains(".peer_push") {
                    &exp.dgpu_xfer
                } else {
                    &exp.dgpu_compute
                };
                exp.emit_slice(track, name, s, e)
            })?;
            self.igpu.events.for_each_pair_new(|name, s, e| {
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

        // INFLATED, TWICE OVER (2026-09-22 audit B10) -- these are not "busy":
        //  (1) they sum the `.wait` scopes, which are pure STALLS. In decode
        //      `dgpu.ffn_combine.wait` (forward_layer.rs, right before
        //      `wait_event(&sev.moe_arrived)`) covers the whole MoE + box-2 leg.
        //      `trace.rs`'s own note states the convention: "the `.wait` stages
        //      and the wall DO NOT [survive]. Read the busy times, not the wait."
        //  (2) with `set_kernel_stages(true)` they sum parent stages AND their
        //      nested `k.*` children.
        // The multistream rollup already fixed exactly this (f25b715: "Parent
        // stages only"; it filters `dgpu.` / `igpu.` prefixes); this twin was
        // missed. Any "the dGPU is N% busy" figure taken from `het.token.summary`
        // is therefore too high. Fix by filtering the same way here.
        let dgpu_busy_us: u64 = (dgpu_timings.iter().map(|t| t.ms as f64).sum::<f64>() * 1000.0) as u64;
        let igpu_busy_us: u64 = (igpu_timings.iter().map(|t| t.ms as f64).sum::<f64>() * 1000.0) as u64;

        let dgpu_idle_us = token_elapsed_us.saturating_sub(dgpu_busy_us);
        let igpu_idle_us = token_elapsed_us.saturating_sub(igpu_busy_us);

        // peer copies: ffn_input_norm (N_EMBD f32) + ffn_moe (N_EMBD f32) per layer.
        let peer_bytes = (N_LAYER as u64) * 2 * (crate::config::N_EMBD as u64) * 4;

        let pager_d = pager
            .as_deref()
            .map(|p| p.counters() - pager_c0)
            .unwrap_or_default();
        // Everything between the bracket's closing sync and this line
        // (perfetto export, event harvest) — outside `total_us` by construction.
        let post_us = (token_start.elapsed().as_micros() as u64).saturating_sub(token_elapsed_us);
        let summary = super::trace::TokenTiming {
            token_pos: pos,
            total_us: token_elapsed_us,
            dgpu_busy_us,
            igpu_busy_us,
            dgpu_idle_us,
            igpu_idle_us,
            peer_bytes,
            host_us,
            sync_us,
            sel_sync_us: super::trace::phase::get(&super::trace::phase::SEL_SYNC_NS) / 1000,
            pager_ensure_us: super::trace::phase::get(&super::trace::phase::ENSURE_NS) / 1000,
            pager_read_us: pager_d.decode_read_ns / 1000,
            pager_h2d_us: pager_d.decode_h2d_ns / 1000,
            pager_misses: pager_d.decode_misses,
            remote_rtt_us: super::trace::phase::get(&super::trace::phase::REMOTE_RTT_NS) / 1000,
            remote_wait_us: super::trace::phase::get(&super::trace::phase::REMOTE_WAIT_NS) / 1000,
            remote_srv_us: super::trace::phase::get(&super::trace::phase::REMOTE_SRV_NS) / 1000,
            engram_stage_us: super::trace::phase::get(&super::trace::phase::ENGRAM_STAGE_NS) / 1000,
            pre_us,
            post_us,
            gap_us,
            // Drained (read-and-clear): these were added by the caller between
            // the previous forward's return and this one's entry.
            engram_us: super::trace::phase::take(&super::trace::phase::CALLER_ENGRAM_NS) / 1000,
            embed_us: super::trace::phase::take(&super::trace::phase::CALLER_EMBED_NS) / 1000,
            sample_us: super::trace::phase::take(&super::trace::phase::CALLER_SAMPLE_NS) / 1000,
            stream_us: super::trace::phase::take(&super::trace::phase::CALLER_STREAM_NS) / 1000,
        };
        summary.emit();

        // Per-stage rollup. At INFO under DEEPSTRIX_TOKEN_PROFILE (the M8 decode
        // breakdown), at DEBUG otherwise.
        if super::trace::token_profile() {
            let dgpu_roll = super::trace::rollup_by_name(&dgpu_timings);
            let igpu_roll = super::trace::rollup_by_name(&igpu_timings);
            for (name, total_ms, calls) in dgpu_roll {
                tracing::info!(
                    token_pos = pos, device = "dgpu", stage = name,
                    total_us = (total_ms * 1000.0) as u64, calls, "het.stage"
                );
            }
            for (name, total_ms, calls) in igpu_roll {
                tracing::info!(
                    token_pos = pos, device = "igpu", stage = name,
                    total_us = (total_ms * 1000.0) as u64, calls, "het.stage"
                );
            }
        } else if tracing::enabled!(tracing::Level::DEBUG) {
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
        // Stamp the return so the next call can report the caller-side gap.
        self.last_token_end_ns
            .store(super::trace::epoch_ns(), std::sync::atomic::Ordering::Relaxed);
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
        let sync_events_t2 = HetSyncEvents::alloc(dgpu_device, igpu_device)?;
        dgpu_device.set_current()?;

        Ok(Self {
            dgpu,
            igpu,
            mode,
            sync_events,
            sync_events_t1,
            sync_events_t2,
            moe_signal: v4flash_hip::PinnedBuffer::new(N_LAYER as usize)?,
            token_seq: std::sync::atomic::AtomicU32::new(0),
            perfetto: None,
            dgpu_graphs: GraphCache::new(),
            igpu_graphs: GraphCache::new(),
            current_device: std::sync::atomic::AtomicI32::new(-1),
            last_host_us: std::sync::atomic::AtomicU64::new(0),
            last_sync_us: std::sync::atomic::AtomicU64::new(0),
            last_token_end_ns: std::sync::atomic::AtomicU64::new(0),
            last_idx_gather_src: std::sync::atomic::AtomicI32::new(-1),
            last_idx_gather_rows: std::sync::atomic::AtomicU32::new(0),
            sel_stats_prefill: std::sync::Mutex::new({
                let mut b: DeviceBuffer<u32> = DeviceBuffer::new(
                    dgpu_device.id,
                    (N_LAYER as usize) * (crate::config::N_EXPERT as usize),
                )?;
                b.fill_zero()?;
                b
            }),
            sel_stats_decode: std::sync::Mutex::new({
                let mut b: DeviceBuffer<u32> = DeviceBuffer::new(
                    dgpu_device.id,
                    (N_LAYER as usize) * (crate::config::N_EXPERT as usize),
                )?;
                b.fill_zero()?;
                b
            }),
            sel_tokens_prefill: std::sync::atomic::AtomicU64::new(0),
            sel_tokens_decode: std::sync::atomic::AtomicU64::new(0),
            remote: connect_remote_experts(),
        })
    }

    /// M62: stats collection default-on; `DEEPSTRIX_SEL_STATS=0` opts out.
    pub fn sel_stats_enabled() -> bool {
        static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::env::var("DEEPSTRIX_SEL_STATS").map(|v| v != "0").unwrap_or(true)
        });
        *ON
    }

    /// M62: accumulate this layer's router picks (`tokens × N_EXPERT_USED`
    /// flat in `d_selected`) into the prefill or decode bank. Queued on
    /// de.compute — caller must already be in a context where d_selected
    /// has been written on that stream (right after router top-k).
    pub fn record_sel_stats(
        &self,
        is_prefill: bool,
        d_selected: &DeviceBuffer<i32>,
        layer: u32,
        tokens: u32,
    ) -> eyre::Result<()> {
        if !Self::sel_stats_enabled() || tokens == 0 {
            return Ok(());
        }
        let bank = if is_prefill { &self.sel_stats_prefill } else { &self.sel_stats_decode };
        let mut bank = bank.lock().expect("sel_stats lock");
        self.dgpu.expert_sel_count.launch(
            &self.dgpu.compute,
            &mut bank,
            d_selected,
            layer,
            crate::config::N_EXPERT,
            tokens * (crate::config::N_EXPERT_USED as u32),
        )?;
        if layer == 0 {
            let ctr = if is_prefill { &self.sel_tokens_prefill } else { &self.sel_tokens_decode };
            ctr.fetch_add(tokens as u64, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    /// M62: read back and zero both banks. Returns
    /// `((prefill_counts, prefill_tokens), (decode_counts, decode_tokens))`
    /// with counts laid out `[N_LAYER × N_EXPERT]`. Synchronizes de.compute
    /// — call only at harvest points (turn end / shutdown), never per token.
    #[allow(clippy::type_complexity)]
    pub fn harvest_sel_stats(&self) -> eyre::Result<((Vec<u32>, u64), (Vec<u32>, u64))> {
        self.set_current_cached(self.dgpu.device)?;
        self.dgpu.compute.synchronize()?;
        let take = |m: &std::sync::Mutex<DeviceBuffer<u32>>,
                    t: &std::sync::atomic::AtomicU64|
         -> eyre::Result<(Vec<u32>, u64)> {
            let mut b = m.lock().expect("sel_stats lock");
            let mut host = vec![0u32; b.len()];
            b.copy_to_host(&mut host)?;
            b.fill_zero()?;
            Ok((host, t.swap(0, std::sync::atomic::Ordering::Relaxed)))
        };
        Ok((
            take(&self.sel_stats_prefill, &self.sel_tokens_prefill)?,
            take(&self.sel_stats_decode, &self.sel_tokens_decode)?,
        ))
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
            SampleMode::Multinomial { temperature, min_p_rel, top_p } => {
                if temperature <= 0.0 {
                    self.dgpu.sampler.launch_argmax(
                        &self.dgpu.compute,
                        &mut dgpu_scratch.sampler_next_token_id,
                        &dgpu_scratch.logits,
                        n,
                    )?;
                } else {
                    dgpu_scratch.sampler_u01.copy_from_host(&[u01])?;
                    self.dgpu.sampler.launch_multinomial_topp(
                        &self.dgpu.compute,
                        &mut dgpu_scratch.sampler_next_token_id,
                        &dgpu_scratch.logits,
                        &mut dgpu_scratch.sampler_partials_max,
                        &mut dgpu_scratch.sampler_partials_z,
                        &mut dgpu_scratch.sampler_topp_mass,
                        &mut dgpu_scratch.sampler_topp_bracket,
                        &mut dgpu_scratch.sampler_topp_thr,
                        &dgpu_scratch.sampler_u01,
                        n,
                        temperature,
                        min_p_rel,
                        top_p,
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
        self.dgpu.events.set_kernel_stages(true);
        self.igpu.events.set_kernel_stages(true);
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

/// Is the sub-tensor dump armed for `layer`? Lets the batched prefill path skip
/// its device sync entirely when dumping is off.
pub(super) fn subtensor_dump_armed(layer: usize) -> bool {
    matches!(subtensor_dump_spec(), Some((layers, _)) if layers.contains(&layer))
}

/// `maybe_dump_subtensor_f32` for a VIEW (the batched path dumps row 0 of a
/// `[B, ...]` scratch buffer, not a whole buffer).
pub(super) fn maybe_dump_subtensor_f32_view(
    layer: usize,
    tag: &str,
    buf: &v4flash_hip::DeviceBuffer<f32>,
) -> eyre::Result<()> {
    maybe_dump_subtensor_f32(layer, tag, buf)
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
