//! [`DeviceEngine`] — per-device bundle of HIP kernel modules + streams.
//! [`HeterogeneousEngine`] — pair of engines plus [`ExecMode`] policy.
//!
//! M13.1 implements `ExecMode::HetSingleStream` only — every kernel is
//! followed by `compute.synchronize()` and peer copies are serial. The
//! engine layout is already structured for M13.4's parallel mode (separate
//! `compute` + `xfer` streams per device), but the wait-events haven't
//! been wired yet.

use color_eyre::eyre;
use v4flash_hip::{Device, Event, GraphExec, Stream};

use crate::forward::N_LAYER;

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
use crate::q8_0::{Q8_0GroupedMatvec, Q8_0Matvec};
use crate::q8_k::Q8KQuantize;
use crate::rms_norm::{RmsNorm, RmsNormNoWeight};
use crate::rope::RopeTail;
use crate::router_topk::RouterTopk;

use super::perfetto::DeviceTimingExporter;
use super::trace::EventPool;

/// Per-device EventPool capacity. With wait-vs-work split bracketing
/// plus per-kernel timing inside the heavy stages (mhc_pre_attn,
/// mhc_pre_ffn, output_proj, q_chain, routed_moe — see forward_layer.rs)
/// we approach ~70 pairs/layer × 43 layers ≈ 6000 events per device.
pub const EVENT_POOL_CAPACITY: usize = 8192;

/// Execution policy for [`HeterogeneousEngine::forward_token`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecMode {
    /// Serial het execution — every kernel followed by `.synchronize()`,
    /// peer copies serialized. Correctness oracle for the parallel mode.
    /// Expected to be *slower* than single-device because of cross-device
    /// sync overhead.
    HetSingleStream,
    /// Event-driven overlap (M13.4+). Compressor / shared / routed-MoE run
    /// concurrently across the two devices, gated by HIP events.
    HetParallel,
}

/// All kernel modules + streams for one HIP device. We carry the full
/// kernel set on both devices in M13.1 (cheap — kernels are HSACO blobs,
/// not weights). Memory split happens in [`super::weights`].
pub struct DeviceEngine {
    pub device: Device,
    /// Compute stream — all kernel launches go here.
    pub compute: Stream,
    /// Transfer stream — peer copies originating on this device go here.
    /// Per the load-bearing peer-copy rule, `hipMemcpyPeerAsync` MUST be
    /// queued on the **source** device's stream.
    pub xfer: Stream,

    pub rms_w: RmsNorm,
    pub rms_nw: RmsNormNoWeight,
    pub q8: Q8_0Matvec,
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
    pub hc_post: HcPost,
    pub compressor_pool: CompressorPool,
    pub compressor_state_write: CompressorStateWrite,
    pub compressor_shuffle: CompressorStateShuffleR4,
    pub fp8: Fp8E4m3fnQuantize,
    pub f16rt: F16Roundtrip,
    pub kv_append: KvCacheAppend,
    pub comp_kv_append: CompKvAppend,
    pub router_topk: RouterTopk,

    /// Per-device event pool for kernel-scope timing (M13.2). Use
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
        let label: &'static str = if device.properties()?.integrated {
            "igpu"
        } else {
            "dgpu"
        };
        let events = EventPool::new(label, EVENT_POOL_CAPACITY)?;
        Ok(Self {
            device,
            compute,
            xfer,
            rms_w: RmsNorm::for_arch(arch)?,
            rms_nw: RmsNormNoWeight::for_arch(arch)?,
            q8: Q8_0Matvec::for_arch(arch)?,
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
            hc_post: HcPost::for_arch(arch)?,
            compressor_pool: CompressorPool::for_arch(arch)?,
            compressor_state_write: CompressorStateWrite::for_arch(arch)?,
            compressor_shuffle: CompressorStateShuffleR4::for_arch(arch)?,
            fp8: Fp8E4m3fnQuantize::for_arch(arch)?,
            f16rt: F16Roundtrip::for_arch(arch)?,
            kv_append: KvCacheAppend::for_arch(arch)?,
            comp_kv_append: CompKvAppend::for_arch(arch)?,
            router_topk: RouterTopk::for_arch(arch)?,
            events,
        })
    }
}

/// Per-layer cross-device sync events for the [`ExecMode::HetParallel`]
/// pipeline. Pre-allocated up front, reused per token.
///
/// FFN handoff (M13.4):
/// * `ain_ready` (dgpu.compute) — `ffn_input_norm` finished computing.
/// * `ain_pushed` (dgpu.xfer)   — `ffn_input_norm` finished copying to iGPU.
/// * `moe_done`  (igpu.compute) — routed MoE finished writing `ffn_moe`.
/// * `moe_arrived` (igpu.xfer)  — `ffn_moe` finished copying to dGPU.
///
/// Compressor handoff (M13.5):
/// * `attn_in_ready`   (dgpu.compute) — `attn_input_norm` finished computing.
/// * `attn_in_pushed`  (dgpu.xfer)    — `attn_input_norm` finished copying to iGPU.
/// * `comp_row_ready`  (igpu.compute) — boundary `comp_row` finished computing.
/// * `comp_row_arrived`(igpu.xfer)    — `comp_row` finished copying to dGPU.
pub struct LayerSyncEvents {
    pub ain_ready: Event,
    pub ain_pushed: Event,
    pub moe_done: Event,
    pub moe_arrived: Event,
    pub attn_in_ready: Event,
    pub attn_in_pushed: Event,
    pub comp_row_ready: Event,
    pub comp_row_arrived: Event,
    /// M16: router runs on dGPU; this fires (dGPU compute) once
    /// selected/d_ew are written and ready for peer-push to iGPU.
    pub selected_ready: Event,
    /// M16: fires (dGPU xfer) once selected/d_ew have been pushed to
    /// the iGPU's d_selected/d_ew. iGPU MoE waits on this.
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
    /// Optional per-token device-time perfetto exporter. Drains the
    /// EventPools at the end of each `forward_token` into per-stream
    /// perfetto tracks. Enable by calling
    /// [`HeterogeneousEngine::attach_perfetto`].
    pub perfetto: Option<std::sync::Mutex<DeviceTimingExporter>>,

    /// M15: per-layer captured HIP graphs for the iGPU routed-MoE
    /// sub-pipeline (q8k_xq → iq2_fused_swiglu_batch → q8k_mid_batch →
    /// q2k_down_batch). All 4 launches have device-resident inputs and
    /// layer-constant scalar params, so the graph captures once on the
    /// first call to each layer and replays for every subsequent token.
    /// Each layer slot is `None` until the first forward_layer call
    /// initializes it.
    pub igpu_moe_graphs: Vec<std::sync::Mutex<Option<GraphExec>>>,
    /// M15: per-layer captured graphs for the dGPU mHC pre-attn block
    /// (5 kernels: rms_nw → f16_matvec → hc_sinkhorn → hc_weighted →
    /// rms_w_weighted). All inputs are device-resident; scalar params
    /// are layer-constant.
    pub dgpu_mhc_pre_attn_graphs: Vec<std::sync::Mutex<Option<GraphExec>>>,
    /// M15: per-layer captured graphs for the dGPU mHC pre-ffn block
    /// (5 kernels, same shape as pre-attn but using the ffn-side
    /// projection / scale / base / norm weights).
    pub dgpu_mhc_pre_ffn_graphs: Vec<std::sync::Mutex<Option<GraphExec>>>,
    /// M15: per-layer captured graphs for the dGPU shared-expert chain
    /// (6 kernels: q8_quantize → q8 gate matvec → q8 up matvec → swiglu
    /// → q8_quantize_mid → q8 down matvec). All inputs are
    /// device-resident; scalar params are layer-constant.
    pub dgpu_shared_expert_graphs: Vec<std::sync::Mutex<Option<GraphExec>>>,
    /// M15: per-layer captured graphs for the dGPU Q-chain prefix
    /// (6 kernels: q8_quantize → q8_matvec_qa → rms_w_qa → q8_quantize_qr
    /// → q8_matvec_qb → rms_nw_heads). Stops before rope_forward, which
    /// takes per-token `pos` and is launched directly.
    pub dgpu_q_chain_pre_rope_graphs: Vec<std::sync::Mutex<Option<GraphExec>>>,
    /// M15: per-layer captured graphs for the dGPU output-projection
    /// suffix (4 kernels: q8_quantize_heads → q8_grouped_matvec_a →
    /// q8_quantize_low → q8_matvec_b). Skips the leading rope_inverse,
    /// which takes per-token `pos`.
    pub dgpu_output_proj_post_rope_graphs: Vec<std::sync::Mutex<Option<GraphExec>>>,
    /// M15.1: per-layer captured graphs for the dGPU ffn_combine block
    /// (2 kernels: vec_add (ffn_moe_recv += ffn_shared) followed by
    /// hc_post writing residual_next). Safe to capture because the
    /// end-of-token extra swap keeps each layer's residual/residual_next
    /// pointing to stable physical buffers.
    pub dgpu_ffn_combine_graphs: Vec<std::sync::Mutex<Option<GraphExec>>>,
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
        use crate::forward::{HC_DIM, N_LAYER};
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

        self.dgpu.device.set_current()?;
        dgpu_scratch.residual.copy_from_host(input_hc_host)?;

        let token_start = std::time::Instant::now();
        for layer in 0..N_LAYER as usize {
            self.forward_layer(
                dgpu_scratch,
                igpu_scratch,
                &mut state.layers[layer],
                &weights.dgpu_layers[layer],
                &weights.igpu_layers[layer],
                pos,
                token_id,
            )?;
            std::mem::swap(&mut dgpu_scratch.residual, &mut dgpu_scratch.residual_next);
        }
        self.forward_head(dgpu_scratch, &weights.global)?;
        // M15.1: N_LAYER (43) is odd, so 43 in-loop swaps leave
        // residual/residual_next inverted from token start. Without an
        // extra swap, every token's layer 0 would read from a different
        // physical DeviceBuffer than the previous token's layer 0,
        // making it impossible to capture mhc_pre_attn / mhc_post_ffn
        // into HIP graphs (the captured pointer would be wrong on
        // alternating tokens). One extra swap here restores the
        // initial state so layer N always operates on the same
        // physical buffers across every token.
        std::mem::swap(&mut dgpu_scratch.residual, &mut dgpu_scratch.residual_next);
        self.dgpu.device.set_current()?;
        self.dgpu.compute.synchronize()?;
        let token_elapsed_us = token_start.elapsed().as_micros() as u64;

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
        }

        // Harvest per-kernel timings.
        let dgpu_timings = self.dgpu.events.harvest()?;
        let igpu_timings = self.igpu.events.harvest()?;

        let dgpu_busy_us: u64 = (dgpu_timings.iter().map(|t| t.ms as f64).sum::<f64>() * 1000.0) as u64;
        let igpu_busy_us: u64 = (igpu_timings.iter().map(|t| t.ms as f64).sum::<f64>() * 1000.0) as u64;

        let dgpu_idle_us = token_elapsed_us.saturating_sub(dgpu_busy_us);
        let igpu_idle_us = token_elapsed_us.saturating_sub(igpu_busy_us);

        // peer copies: ffn_input_norm (N_EMBD f32) + ffn_moe (N_EMBD f32) per layer.
        let peer_bytes = (N_LAYER as u64) * 2 * (crate::forward::N_EMBD as u64) * 4;

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

        let mut igpu_moe_graphs = Vec::with_capacity(N_LAYER as usize);
        let mut dgpu_mhc_pre_attn_graphs = Vec::with_capacity(N_LAYER as usize);
        let mut dgpu_mhc_pre_ffn_graphs = Vec::with_capacity(N_LAYER as usize);
        let mut dgpu_shared_expert_graphs = Vec::with_capacity(N_LAYER as usize);
        let mut dgpu_q_chain_pre_rope_graphs = Vec::with_capacity(N_LAYER as usize);
        let mut dgpu_output_proj_post_rope_graphs = Vec::with_capacity(N_LAYER as usize);
        let mut dgpu_ffn_combine_graphs = Vec::with_capacity(N_LAYER as usize);
        for _ in 0..N_LAYER {
            igpu_moe_graphs.push(std::sync::Mutex::new(None));
            dgpu_mhc_pre_attn_graphs.push(std::sync::Mutex::new(None));
            dgpu_mhc_pre_ffn_graphs.push(std::sync::Mutex::new(None));
            dgpu_shared_expert_graphs.push(std::sync::Mutex::new(None));
            dgpu_q_chain_pre_rope_graphs.push(std::sync::Mutex::new(None));
            dgpu_output_proj_post_rope_graphs.push(std::sync::Mutex::new(None));
            dgpu_ffn_combine_graphs.push(std::sync::Mutex::new(None));
        }

        Ok(Self {
            dgpu,
            igpu,
            mode,
            sync_events,
            perfetto: None,
            igpu_moe_graphs,
            dgpu_mhc_pre_attn_graphs,
            dgpu_mhc_pre_ffn_graphs,
            dgpu_shared_expert_graphs,
            dgpu_q_chain_pre_rope_graphs,
            dgpu_output_proj_post_rope_graphs,
            dgpu_ffn_combine_graphs,
        })
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
        Ok(())
    }
}
