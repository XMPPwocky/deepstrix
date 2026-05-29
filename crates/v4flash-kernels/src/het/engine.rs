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
    CompressorPool, CompressorStateShuffleR4, CompressorStateWrite, F16Roundtrip, Fp8E4m3fnQuantize,
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
    /// Q4_K matvec kernels (par-batched + pair-swiglu-batched). Loaded
    /// for the retired MTP draft layer (whose GGUF stored routed experts
    /// as Q4_K). No live caller in the current forward path — retained
    /// because `kernels/q4_k_matvec_par.hip` exists and is exercised by
    /// `tests/q4_k_matvec.rs`. Candidate for removal from DeviceEngine.
    pub q4k: crate::q4_k::Q4KMatvec,
    /// Broadcast/tile kernel (expand N_EMBD vector to N_HC × N_EMBD).
    /// Loaded for retired MTP HC-combine; same dead-engine-field status
    /// as `q4k` above.
    pub broadcast: crate::broadcast::BroadcastToHc,
    pub hc_sigmoid: HcSigmoidBias,
    pub hc_weighted: HcWeightedSum,
    pub hc_sinkhorn: HcSinkhorn,
    pub hc_post: HcPost,
    pub compressor_pool: CompressorPool,
    pub compressor_state_write: CompressorStateWrite,
    pub compressor_shuffle: CompressorStateShuffleR4,
    pub fp8: Fp8E4m3fnQuantize,
    pub f16rt: F16Roundtrip,
    pub kv_append: KvCacheAppend,
    pub comp_kv_append: CompKvAppend,
    pub router_topk: RouterTopk,
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
            q4k: crate::q4_k::Q4KMatvec::for_arch(arch)?,
            broadcast: crate::broadcast::BroadcastToHc::for_arch(arch)?,
            hc_sigmoid: HcSigmoidBias::for_arch(arch)?,
            hc_weighted: HcWeightedSum::for_arch(arch)?,
            hc_sinkhorn: HcSinkhorn::for_arch(arch)?,
            hc_post: HcPost::for_arch(arch)?,
            compressor_pool: CompressorPool::for_arch(arch)?,
            compressor_state_write: CompressorStateWrite::for_arch(arch)?,
            compressor_shuffle: CompressorStateShuffleR4::for_arch(arch)?,
            fp8: Fp8E4m3fnQuantize::for_arch(arch)?,
            f16rt: F16Roundtrip::for_arch(arch)?,
            kv_append: KvCacheAppend::for_arch(arch)?,
            comp_kv_append: CompKvAppend::for_arch(arch)?,
            router_topk: RouterTopk::for_arch(arch)?,
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
///
/// The `attn_in_*` and `comp_row_*` fields are pre-allocated for a
/// retired compressor-handoff path (the compressor now runs entirely on
/// dGPU). They are unread in the current code; candidates for removal.
pub struct LayerSyncEvents {
    pub ain_ready: Event,
    pub ain_pushed: Event,
    pub moe_done: Event,
    pub moe_arrived: Event,
    pub attn_in_ready: Event,
    pub attn_in_pushed: Event,
    pub comp_row_ready: Event,
    pub comp_row_arrived: Event,
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
            let attn_in_ready = Event::new_no_timing()?;
            let attn_in_pushed = Event::new_no_timing()?;
            let selected_ready = Event::new_no_timing()?;
            let selected_pushed = Event::new_no_timing()?;
            igpu.set_current()?;
            let moe_done = Event::new_no_timing()?;
            let moe_arrived = Event::new_no_timing()?;
            let comp_row_ready = Event::new_no_timing()?;
            let comp_row_arrived = Event::new_no_timing()?;
            layers.push(LayerSyncEvents {
                ain_ready,
                ain_pushed,
                moe_done,
                moe_arrived,
                attn_in_ready,
                attn_in_pushed,
                comp_row_ready,
                comp_row_arrived,
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

        let token_start = std::time::Instant::now();
        for layer in 0..N_LAYER as usize {
            let next_dlw = if layer + 1 < N_LAYER as usize {
                Some(&weights.dgpu_layers[layer + 1])
            } else {
                None
            };
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
            std::mem::swap(&mut dgpu_scratch.residual, &mut dgpu_scratch.residual_next);
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
