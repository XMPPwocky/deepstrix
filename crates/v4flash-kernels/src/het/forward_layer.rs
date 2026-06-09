//! Per-layer dispatch for the het orchestrator.
//!
//! Pipeline (event-driven overlap mode — `ExecMode::HetParallel`):
//!
//!  1. **dGPU**: mHC pre-attn → attn_cur → attn_input_norm
//!  2. **dGPU**: Q chain + KV chain
//!  3. **dGPU**: KV cache push (FP8 + F16-roundtrip + SWA slide)
//!  4. **dGPU** (ratio>0): compressor step + boundary fire → comp_kv
//!  5. **dGPU**: attention (swa or mixed) → heads → attn_out
//!  6. **dGPU**: mHC post-attn → after_attn_hc
//!  7. **dGPU**: mHC pre-ffn → ffn_cur → ffn_input_norm
//!  8. **dGPU**: router → d_selected / d_ew
//!  9. **dGPU→iGPU**: peer push ffn_input_norm + d_selected/d_ew (on dGPU.xfer)
//! 10. **iGPU**: routed MoE → ffn_moe
//! 11. **iGPU→dGPU**: peer push ffn_moe → ffn_moe_recv
//! 12. **dGPU**: shared expert → ffn_shared (overlaps with 9-11)
//! 13. **dGPU**: ffn_moe_recv += ffn_shared then mHC post-ffn → residual_next
//!
//! `ExecMode::HetSingleStream` is a correctness oracle that runs the same
//! pipeline with `.synchronize()` between every kernel.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{Device, DeviceBuffer};

use crate::config::{
    BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, EXPERT_WEIGHT_SCALE, GROUP_DIM, HC_DIM, HC_MIX_DIM,
    INDEXER_COMP_WIDTH, INDEXER_TOP_K, N_EMBD, N_EXPERT, N_EXPERT_USED, N_FF_EXP, N_FF_SHARED,
    N_GROUPS, N_HC, N_HEAD, N_HEAD_DIM, N_INDEXER_HEAD, N_INDEXER_HEAD_DIM, N_LORA_Q, N_ROT,
    OUT_LOW, Q_FLAT, RANK, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS, SWA_WINDOW, SWIGLU_CLAMP_EXP,
};
use crate::routing::hash_router_select;
use crate::q8_k::BLOCK_Q8_K_BYTES;

/// Floor for the router-weight sum, mirroring the host topk path (f16
/// epsilon).
const ROUTER_WEIGHT_EPS: f32 = 6.103515625e-5;

use super::engine::{DeviceEngine, ExecMode, HeterogeneousEngine};
use super::scratch::{DgpuScratch, IgpuScratch};
use super::state::HetLayerState;
use super::sync::{peer_push_f32, peer_push_i32};
use super::weights::{DgpuLayerWeights, IgpuLayerWeights};
use tracing::debug_span;

impl HeterogeneousEngine {
    /// Run one layer in the het pipeline. Reads from
    /// `dgpu_scratch.residual` and writes to `dgpu_scratch.residual_next`.
    ///
    /// `next_dlw` is `Some` for all layers except the last. When present,
    /// layer N's ffn_combine is fused with layer N+1's mhc_pre_attn into
    /// one captured graph (the combined-transition graph that closes the
    /// ~115 µs/layer host-scheduling gap). Correspondingly, mhc_pre_attn
    /// is launched standalone ONLY for layer 0 — every other layer's
    /// mhc_pre_attn rides the previous layer's combined graph.
    pub fn forward_layer(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        next_dlw: Option<&DgpuLayerWeights>,
        ilw: &IgpuLayerWeights,
        pos: u32,
        token_id: i32,
    ) -> eyre::Result<()> {
        self.forward_layer_impl(
            dgpu_scratch,
            igpu_scratch,
            ls,
            dlw,
            next_dlw,
            ilw,
            pos,
            token_id,
            false,
        )
    }

    /// Run one layer with the cross-layer combined ffn_combine →
    /// next-mhc_pre_attn graph disabled. Each layer fires its own
    /// standalone mhc_pre_attn + standalone ffn_combine instead.
    ///
    /// Used by diagnostic tests that need per-layer control (e.g. running
    /// only layer N against the activation dump) — the combined graph
    /// would speculatively launch the NEXT layer's mhc_pre_attn before
    /// the test can read the layer-N output.
    pub fn forward_layer_standalone_graphs(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        ilw: &IgpuLayerWeights,
        pos: u32,
        token_id: i32,
    ) -> eyre::Result<()> {
        // next_dlw unused; the standalone flag forces is_first_layer +
        // is_last_layer to true regardless.
        self.forward_layer_impl(
            dgpu_scratch,
            igpu_scratch,
            ls,
            dlw,
            None,
            ilw,
            pos,
            token_id,
            true,
        )
    }

    /// Same as [`forward_layer`] but assumes the layer's iGPU MoE command
    /// sequence was already enqueued by [`issue_igpu_moe`] (M54 pre-issue
    /// mode) — the impl skips its inline iGPU section. Parallel mode only.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_layer_preissued_moe(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        next_dlw: Option<&DgpuLayerWeights>,
        ilw: &IgpuLayerWeights,
        pos: u32,
        token_id: i32,
    ) -> eyre::Result<()> {
        self.forward_layer_impl_inner(
            dgpu_scratch,
            igpu_scratch,
            ls,
            dlw,
            next_dlw,
            ilw,
            pos,
            token_id,
            false,
            true,
        )
    }

    /// M54: enqueue one layer's complete iGPU MoE sequence (wait on
    /// `selected_pushed` → routed_moe graph → record `moe_done` → peer-push
    /// ffn_moe → record `moe_arrived`). Every command is event-gated, so
    /// the whole lane for all 43 layers can be pre-issued at token start —
    /// the decode pftrace showed ~125 µs/layer of host-submission lag
    /// (`routed_moe → moe.wait` gaps) when the iGPU commands were
    /// interleaved with the ~25 dGPU submissions per layer, and that lag
    /// lands directly on the dGPU's MoE wait (the critical path).
    ///
    /// Safety relies on stream ordering: ffn_combine(L) precedes
    /// router(L+1) on the dGPU compute stream, so `selected_pushed(L+1)`
    /// implies ffn_moe_recv(L) was consumed; the activations push precedes
    /// the selected push on the dGPU xfer stream.
    pub fn issue_igpu_moe(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        ilw: &IgpuLayerWeights,
    ) -> eyre::Result<()> {
        use tracing::debug_span;
        let layer = ilw.layer_idx;
        let sev = &self.sync_events.layers[layer as usize];
        self.set_current_cached(self.igpu.device)?;
        let ie = &self.igpu;

        {
            // Value-wait, NOT event-wait: this lane is enqueued at token
            // start, before the dGPU's record/write calls for the token.
            // hipStreamWaitEvent snapshots at call time (would no-op);
            // hipStreamWaitValue32 compares at execution time.
            let _t_wait = ie.events.stage("igpu.moe.wait", &ie.compute)?;
            let seq = self
                .token_seq
                .load(std::sync::atomic::Ordering::Relaxed);
            let sig = unsafe {
                (self.moe_signal.as_slice().as_ptr() as *mut u32).add(layer as usize)
            };
            unsafe { ie.compute.wait_value32_gte(sig, seq)? };
            _t_wait.end()?;
        }

        let gbpe = ilw.routed.gate_bytes_per_expert;
        let ubpe = ilw.routed.up_bytes_per_expert;
        let dbpe = ilw.routed.down_bytes_per_expert;
        let mid_blocks_bytes = (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8_K_BYTES;

        let _t_moe = ie.events.stage("igpu.routed_moe", &ie.compute)?;
        let _s_moe = debug_span!("routed_moe").entered();
        self.igpu_graphs.run("routed_moe", layer as u32, &ie.compute, |s| {
            ie.q8k.launch(s, &mut igpu_scratch.d_xq_q8k, &igpu_scratch.ffn_input_norm_recv, BLOCKS_Q8K_GATE_IN)?;
            ie.iq2.launch_fused_swiglu_batch(s, &mut igpu_scratch.d_mid_cat, &ilw.routed.gate.buffer, &ilw.routed.up.buffer, &igpu_scratch.d_xq_q8k, &igpu_scratch.d_ew, &igpu_scratch.d_selected, gbpe as u32, ubpe as u32, N_EXPERT_USED as u32, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN)?;
            ie.q8k.launch(s, &mut igpu_scratch.d_midq_cat, &igpu_scratch.d_mid_cat, BLOCKS_Q8K_DOWN_IN * (N_EXPERT_USED as u32))?;
            ie.q2k.launch_batched(s, &mut igpu_scratch.ffn_moe, &ilw.routed.down.buffer, &igpu_scratch.d_midq_cat, &igpu_scratch.d_selected, dbpe as u32, mid_blocks_bytes as u32, N_EXPERT_USED as u32, N_EMBD, BLOCKS_Q8K_DOWN_IN)?;
            Ok(())
        })?;
        drop(_s_moe);
        _t_moe.end()?;

        sev.moe_done.record(&ie.compute)?;
        ie.xfer.wait_event(&sev.moe_done)?;
        let _t_peer_moe = ie.events.stage("igpu.peer_push_ffn_moe", &ie.xfer)?;
        peer_push_f32(
            &igpu_scratch.ffn_moe,
            &mut dgpu_scratch.ffn_moe_recv,
            &ie.xfer,
        )?;
        sev.moe_arrived.record(&ie.xfer)?;
        _t_peer_moe.end()?;
        Ok(())
    }

    fn forward_layer_impl(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        next_dlw: Option<&DgpuLayerWeights>,
        ilw: &IgpuLayerWeights,
        pos: u32,
        token_id: i32,
        standalone_graphs: bool,
    ) -> eyre::Result<()> {
        self.forward_layer_impl_inner(
            dgpu_scratch,
            igpu_scratch,
            ls,
            dlw,
            next_dlw,
            ilw,
            pos,
            token_id,
            standalone_graphs,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_impl_inner(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        next_dlw: Option<&DgpuLayerWeights>,
        ilw: &IgpuLayerWeights,
        pos: u32,
        token_id: i32,
        standalone_graphs: bool,
        igpu_moe_preissued: bool,
    ) -> eyre::Result<()> {
        let layer = dlw.layer_idx;
        if ilw.layer_idx != layer {
            return Err(eyre!(
                "het forward_layer: dgpu layer {} != igpu layer {}",
                layer,
                ilw.layer_idx
            ));
        }
        let ratio = dlw.ratio;
        // `standalone_graphs` forces standalone mhc_pre_attn + standalone
        // ffn_combine each layer (i.e. bypasses the cross-layer combined
        // graph). Required when the caller needs to observe layer-N
        // output before layer N+1 starts.
        let is_first_layer = standalone_graphs || layer == 0;
        let is_last_layer = standalone_graphs || next_dlw.is_none();

        let serial = matches!(self.mode, ExecMode::HetSingleStream);

        let _layer_span = debug_span!("het.layer", layer, pos).entered();

        // ============================================================
        // dGPU: mHC pre attn → attn_cur → attn_input_norm
        // ============================================================
        self.set_current_cached(self.dgpu.device)?;
        let de = &self.dgpu;

        // mhc_pre_attn for layer N is launched by the PREVIOUS layer's
        // combined ffn_combine→mhc_pre_attn graph, except for layer 0
        // which has no preceding ffn_combine. The standalone mhc_pre_attn
        // graph below only fires once per token (layer 0).
        if is_first_layer {
            let _t_mhc_pre = de.events.stage("dgpu.mhc_pre_attn", &de.compute)?;
            let _s_mhc_pre = debug_span!("mhc_pre_attn").entered();
            // Single fused kernel replaces the 5-kernel chain. Wrapped in
            // the same graph-replay path so launch is amortized identically
            // to the original chain — the win (if any) comes from the
            // single in-WG pipeline instead of 5 short kernels.
            // ENV MHC_FUSED=0 rolls back to the 5-kernel chain.
            // MHC_FUSED=1 enables the fused mhc_pre_fused kernel. Default
            // OFF: the 5-kernel chain in a captured graph is FASTER than
            // the fused single-WG version because the standalone f16.matvec
            // uses 24 WGs (~24 CUs) for the HC_MIX_DIM=24 outputs while the
            // single-WG fused kernel is limited to 1 CU. Graph capture
            // already amortizes launch overhead to ~1 µs/kernel. The
            // chain's bottleneck was matvec WORK, not launch overhead —
            // our "12 ms/tok launch-overhead-bound" estimate was wrong.
            // The kernel is kept in tree for the rollback and as documented
            // negative result.
            let mhc_fused = std::env::var("MHC_FUSED")
                .map(|v| v != "0").unwrap_or(false);
            if mhc_fused {
                self.dgpu_graphs.run("mhc_pre_attn_fused", layer as u32, &de.compute, |s| {
                    de.mhc_pre_fused.launch(
                        s,
                        &mut dgpu_scratch.attn_input_norm,
                        &dgpu_scratch.residual,
                        &dlw.hc_attn_fn.buffer,
                        &dlw.hc_attn_scale,
                        &dlw.hc_attn_base,
                        &dlw.attn_norm,
                        RMS_EPS,
                        SINKHORN_ITERS,
                    )?;
                    Ok(())
                })?;
            } else {
            self.dgpu_graphs.run("mhc_pre_attn", layer as u32, &de.compute, |s| {
                // RMS_NW_MW values:
                //   "fused" (default): compute inv_rms only, fold scale into
                //      next f16 matvec (no apply pass, no flat[] DRAM
                //      round-trip, one fewer launch).
                //   "split": multi-WG rms_nw + standalone matvec (the
                //      previous approach).
                //   "0"/"single": original Grid(1,1,1) single-WG kernel.
                let mode = std::env::var("RMS_NW_MW").unwrap_or_else(|_| "fused".into());
                match mode.as_str() {
                    "0" | "single" => {
                        de.rms_nw.launch(s, &mut dgpu_scratch.flat, &dgpu_scratch.residual, 1, HC_DIM, RMS_EPS)?;
                        de.f16.matvec(s, &mut dgpu_scratch.mix, &dlw.hc_attn_fn.buffer, &dgpu_scratch.flat, HC_MIX_DIM, HC_DIM)?;
                    }
                    "split" => {
                        de.rms_nw_mw.launch(s, &mut dgpu_scratch.flat, &dgpu_scratch.residual, &mut dgpu_scratch.rms_nw_partials, HC_DIM, 16, RMS_EPS)?;
                        de.f16.matvec(s, &mut dgpu_scratch.mix, &dlw.hc_attn_fn.buffer, &dgpu_scratch.flat, HC_MIX_DIM, HC_DIM)?;
                    }
                    _ => {
                        de.rms_nw_mw.launch_inv_only(s, &mut dgpu_scratch.rms_nw_inv_scalar, &dgpu_scratch.residual, &mut dgpu_scratch.rms_nw_partials, HC_DIM, 16, RMS_EPS)?;
                        // F16_KSPLIT=N (default 16 = K-split + u128 vector
                        // loads). N=0 rolls back to legacy narrow.
                        let ksplit: u32 = std::env::var("F16_KSPLIT")
                            .ok().and_then(|s| s.parse().ok()).unwrap_or(16);
                        if ksplit > 0 {
                            de.f16.matvec_narrow_ksplit_pre_scaled(
                                s, &mut dgpu_scratch.mix, &dlw.hc_attn_fn.buffer,
                                &dgpu_scratch.residual, &dgpu_scratch.rms_nw_inv_scalar,
                                &mut dgpu_scratch.mhc_matvec_partials,
                                HC_MIX_DIM, HC_DIM, ksplit,
                            )?;
                        } else {
                            de.f16.matvec_pre_scaled(s, &mut dgpu_scratch.mix, &dlw.hc_attn_fn.buffer, &dgpu_scratch.residual, &dgpu_scratch.rms_nw_inv_scalar, HC_MIX_DIM, HC_DIM)?;
                        }
                    }
                }
                de.hc_sinkhorn.launch(s, &mut dgpu_scratch.split, &dgpu_scratch.mix, &dlw.hc_attn_scale, &dlw.hc_attn_base, N_HC, SINKHORN_ITERS, SINKHORN_EPS)?;
                de.hc_weighted.launch(s, &mut dgpu_scratch.attn_cur, &dgpu_scratch.residual, &dgpu_scratch.split, N_EMBD, N_HC)?;
                // RMS_W_MW=1 enables multi-WG weighted RMS. Default OFF: at
                // N_EMBD=4096 the single-WG version is small enough that
                // multi-WG's extra kernel launch erases any parallelism
                // win — averaged across 256-token decode runs the variants
                // are statistical ties (~±2% within thermal noise).
                if std::env::var("RMS_W_MW").map(|v| v != "0").unwrap_or(false) {
                    de.rms_nw_mw.launch_weighted(s, &mut dgpu_scratch.attn_input_norm, &dgpu_scratch.attn_cur, &dlw.attn_norm, &mut dgpu_scratch.rms_nw_partials, N_EMBD, 16, RMS_EPS)?;
                } else {
                    de.rms_w.launch_weighted(s, &mut dgpu_scratch.attn_input_norm, &dgpu_scratch.attn_cur, &dlw.attn_norm, N_EMBD, RMS_EPS)?;
                }
                Ok(())
            })?;
            }
            drop(_s_mhc_pre);
            _t_mhc_pre.end()?;
        }

        // ============================================================
        // dGPU: Q LoRA chain → q_post_rope
        // ============================================================
        // q_chain prefix (6 kernels) captured into a graph — all params
        // are layer-constant and all I/O buffers are non-swapped scratch.
        // The trailing rope_forward takes per-token `pos` so stays a
        // direct launch.
        let _t_q = de.events.stage("dgpu.q_chain", &de.compute)?;
        let _s_q = debug_span!("q_chain").entered();
        self.dgpu_graphs.run("q_chain_pre_rope", layer as u32, &de.compute, |s| {
            de.q8.quantize_input(s, &mut dgpu_scratch.xq_n_embd, &mut dgpu_scratch.xscale_n_embd, &dgpu_scratch.attn_input_norm, N_EMBD)?;
            de.q8.matvec(s, &mut dgpu_scratch.qr, &dlw.attn_q_a.buffer, &dgpu_scratch.xq_n_embd, &dgpu_scratch.xscale_n_embd, N_LORA_Q, N_EMBD)?;
            // M55: fused rms_w + q8 quantize (was two single-WG kernels);
            // qr_normed is still written for the CSA indexer.
            de.rms_w.launch_weighted_quantize_q8(s, &mut dgpu_scratch.qr_normed, &mut dgpu_scratch.qr_xq, &mut dgpu_scratch.qr_xscale, &dgpu_scratch.qr, &dlw.q_a_norm, N_LORA_Q, RMS_EPS)?;
            de.q8.matvec(s, &mut dgpu_scratch.q, &dlw.attn_q_b.buffer, &dgpu_scratch.qr_xq, &dgpu_scratch.qr_xscale, Q_FLAT, N_LORA_Q)?;
            de.rms_nw.launch(s, &mut dgpu_scratch.q_normed, &dgpu_scratch.q, N_HEAD, N_HEAD_DIM, RMS_EPS)?;
            Ok(())
        })?;
        de.rope.launch_forward(
            &de.compute,
            &mut dgpu_scratch.q_normed,
            N_HEAD,
            N_HEAD_DIM,
            N_ROT,
            pos,
            &dlw.rope_params,
        )?;
        drop(_s_q);
        _t_q.end()?;

        // ============================================================
        // dGPU: KV chain → kv_post_rope → KV cache push
        // ============================================================
        let _t_kv = de.events.stage("dgpu.kv_chain", &de.compute)?;
        let _s_kv = debug_span!("kv_chain").entered();
        {
            let _t = de.events.stage("k.kv_chain.matvec", &de.compute)?;
            de.q8.matvec(
                &de.compute,
                &mut dgpu_scratch.kv_raw,
                &dlw.attn_kv.buffer,
                &dgpu_scratch.xq_n_embd,
                &dgpu_scratch.xscale_n_embd,
                N_HEAD_DIM,
                N_EMBD,
            )?;
        }
        {
            let _t = de.events.stage("k.kv_chain.rms_w", &de.compute)?;
            de.rms_w.launch_weighted(
                &de.compute,
                &mut dgpu_scratch.kv_normed,
                &dgpu_scratch.kv_raw,
                &dlw.kv_a_norm,
                N_HEAD_DIM,
                RMS_EPS,
            )?;
        }
        {
            let _t = de.events.stage("k.kv_chain.rope", &de.compute)?;
            de.rope.launch_forward(
                &de.compute,
                &mut dgpu_scratch.kv_normed,
                1,
                N_HEAD_DIM,
                N_ROT,
                pos,
                &dlw.rope_params,
            )?;
        }
        {
            let _t = de.events.stage("k.kv_chain.fp8", &de.compute)?;
            de.fp8
                .launch(&de.compute, &mut dgpu_scratch.kv_normed, N_HEAD_DIM - N_ROT)?;
        }
        {
            let _t = de.events.stage("k.kv_chain.f16rt", &de.compute)?;
            de.f16rt
                .launch(&de.compute, &mut dgpu_scratch.kv_normed, N_HEAD_DIM)?;
        }
        {
            // M55: MONOTONIC append at slot raw_off + n_raw — never the
            // kernel's evict-slide path (which moved 127 rows through 254
            // block-wide barriers per layer per token, ~25 µs each).
            // Passing the cache CAPACITY as the kernel's `swa_window`
            // guarantees the not-full branch. The window advances via
            // raw_off; readers below take a slice_view at raw_off.
            let _t = de.events.stage("k.kv_chain.kv_append", &de.compute)?;
            let slot = ls.raw_off + ls.n_raw;
            debug_assert!(
                (slot as usize) < super::state::KV_CACHE_ROWS,
                "kv monotonic append OOB: raw_off={} n_raw={}",
                ls.raw_off,
                ls.n_raw
            );
            de.kv_append.launch(
                &de.compute,
                &mut ls.kv_cache,
                &dgpu_scratch.kv_normed,
                slot,
                super::state::KV_CACHE_ROWS as u32,
                N_HEAD_DIM,
            )?;
        }
        if ls.n_raw < SWA_WINDOW {
            ls.n_raw += 1;
        } else {
            ls.raw_off += 1;
        }
        // Wrap (~once per B_MAX tokens per layer): eviction-down copy of
        // the live window to slots [0..W) via scratch (overlap-safe two-hop,
        // same pattern as prefill's post-chunk eviction), then reset.
        if (ls.raw_off + SWA_WINDOW) as usize >= super::state::KV_CACHE_ROWS {
            let head_dim = N_HEAD_DIM as usize;
            let win_len = (ls.n_raw as usize) * head_dim;
            let src_off = (ls.raw_off as usize) * head_dim;
            {
                let mut s = dgpu_scratch.kv_wrap_scratch.slice_view_mut(0, win_len);
                let src = ls.kv_cache.slice_view(src_off, win_len);
                s.copy_from_buffer_async(&src, &de.compute)?;
            }
            {
                let s = dgpu_scratch.kv_wrap_scratch.slice_view(0, win_len);
                let mut dst = ls.kv_cache.slice_view_mut(0, win_len);
                dst.copy_from_buffer_async(&s, &de.compute)?;
            }
            ls.raw_off = 0;
        }
        drop(_s_kv);
        _t_kv.end()?;

        // ============================================================
        // dGPU: Compressor (ratio>0 layers) — produces comp_kv rows on
        // boundary tokens. Reads attn_input_norm (computed above);
        // weights + state + scratch all live on dGPU so no peer push.
        // ============================================================
        let comp_fires_boundary = ratio > 0 && (pos + 1) % ratio == 0;
        let _parallel_pre = matches!(self.mode, ExecMode::HetParallel);
        let _sev_pre = &self.sync_events.layers[layer as usize];

        if ratio > 0 {
            let _t_comp = de.events.stage("dgpu.compressor", &de.compute)?;
            let _s_comp = debug_span!("compressor_dgpu", ratio).entered();

            let cw = dlw
                .compressor
                .as_ref()
                .ok_or_else(|| eyre!("L{layer}: missing compressor weights (dGPU)"))?;
            let comp_width = cw.width;
            let pos_mod = pos % ratio;
            let row = if ratio == 4 { 4 + pos_mod } else { pos_mod };
            {
                let _t = de.events.stage("k.compressor_d.f16_pair", &de.compute)?;
                de.f16.matvec_pair(
                    &de.compute,
                    &mut dgpu_scratch.kv_cur,
                    &mut dgpu_scratch.sc_cur,
                    &cw.wkv.buffer,
                    &cw.wgate.buffer,
                    &dgpu_scratch.attn_input_norm,
                    comp_width,
                    N_EMBD,
                )?;
            }
            let cs = ls
                .compressor
                .as_mut()
                .ok_or_else(|| eyre!("L{layer}: missing compressor state"))?;
            {
                let _t = de.events.stage("k.compressor_d.state_write", &de.compute)?;
                de.compressor_state_write.launch(
                    &de.compute,
                    &mut cs.state_kv,
                    &mut cs.state_score,
                    &dgpu_scratch.kv_cur,
                    &dgpu_scratch.sc_cur,
                    &cw.ape.buffer,
                    comp_width,
                    row,
                    pos_mod,
                )?;
            }

            if comp_fires_boundary {
                {
                    let _t = de.events.stage("k.compressor_d.pool", &de.compute)?;
                    de.compressor_pool.launch(
                        &de.compute,
                        &mut dgpu_scratch.pooled,
                        &cs.state_kv,
                        &cs.state_score,
                        N_HEAD_DIM,
                        ratio,
                    )?;
                }
                {
                    let _t = de.events.stage("k.compressor_d.rms_w", &de.compute)?;
                    de.rms_w.launch_weighted(
                        &de.compute,
                        &mut dgpu_scratch.comp_row,
                        &dgpu_scratch.pooled,
                        &cw.norm,
                        N_HEAD_DIM,
                        RMS_EPS,
                    )?;
                }
                let comp_pos = pos + 1 - ratio;
                {
                    let _t = de.events.stage("k.compressor_d.rope", &de.compute)?;
                    de.rope.launch_forward(
                        &de.compute,
                        &mut dgpu_scratch.comp_row,
                        1,
                        N_HEAD_DIM,
                        N_ROT,
                        comp_pos,
                        &dlw.rope_params,
                    )?;
                }
                {
                    let _t = de.events.stage("k.compressor_d.fp8", &de.compute)?;
                    de.fp8.launch(
                        &de.compute,
                        &mut dgpu_scratch.comp_row,
                        N_HEAD_DIM - N_ROT,
                    )?;
                }
                {
                    let _t = de.events.stage("k.compressor_d.f16rt", &de.compute)?;
                    de.f16rt.launch(
                        &de.compute,
                        &mut dgpu_scratch.comp_row,
                        N_HEAD_DIM,
                    )?;
                }
                if ratio == 4 {
                    let _t = de.events.stage("k.compressor_d.shuffle", &de.compute)?;
                    de.compressor_shuffle.launch(
                        &de.compute,
                        &mut cs.state_kv,
                        &mut cs.state_score,
                        comp_width,
                    )?;
                }

                // No peer push needed — append directly into local comp_kv.
                {
                    let _t = de.events.stage("k.compressor_d.comp_kv_append", &de.compute)?;
                    de.comp_kv_append.launch(
                        &de.compute,
                        &mut cs.comp_kv,
                        &dgpu_scratch.comp_row,
                        cs.n_comp,
                        N_HEAD_DIM,
                    )?;
                }
                cs.n_comp += 1;
            }
            drop(_s_comp);
            _t_comp.end()?;
        }

        // ============================================================
        // dGPU: CSA indexer compressor — second, parallel compressor at
        // head_dim=128, only on ratio==4 layers. Same kernel set as the
        // main compressor (matvec_pair → state_write → on boundary:
        // pool → rms_w → rope → f16rt → shuffle → comp_kv_append).
        // FP8 quantize is SKIPPED — only valid for head_dim=512.
        //
        // Scratch reuse: this block runs strictly after the main
        // compressor block on the same de.compute stream, so we can
        // reuse `kv_cur` / `sc_cur` / `pooled` / `comp_row` buffers via
        // slice views sized to the indexer's smaller dims.
        // ============================================================
        if ratio == 4 {
            let _t_icomp = de.events.stage("dgpu.indexer_compressor", &de.compute)?;
            let _s_icomp = debug_span!("indexer_compressor_dgpu").entered();

            let iw = dlw
                .indexer_compressor
                .as_ref()
                .ok_or_else(|| eyre!("L{layer}: missing indexer_compressor weights"))?;
            let ics = ls
                .indexer_compressor
                .as_mut()
                .ok_or_else(|| eyre!("L{layer}: missing indexer_compressor state"))?;
            let icw = INDEXER_COMP_WIDTH; // 256
            let ihd = N_INDEXER_HEAD_DIM; // 128
            let pos_mod = pos % ratio;
            let row = 4 + pos_mod; // ratio==4 always

            // matvec_pair writes [icw] into the head of kv_cur / sc_cur.
            {
                let _t = de.events.stage("k.indexer_compressor.f16_pair", &de.compute)?;
                let mut kv_view = dgpu_scratch.kv_cur.slice_view_mut(0, icw as usize);
                let mut sc_view = dgpu_scratch.sc_cur.slice_view_mut(0, icw as usize);
                de.f16.matvec_pair(
                    &de.compute,
                    &mut kv_view,
                    &mut sc_view,
                    &iw.wkv.buffer,
                    &iw.wgate.buffer,
                    &dgpu_scratch.attn_input_norm,
                    icw,
                    N_EMBD,
                )?;
            }
            {
                let _t = de.events.stage("k.indexer_compressor.state_write", &de.compute)?;
                let kv_view = dgpu_scratch.kv_cur.slice_view(0, icw as usize);
                let sc_view = dgpu_scratch.sc_cur.slice_view(0, icw as usize);
                de.compressor_state_write.launch(
                    &de.compute,
                    &mut ics.state_kv,
                    &mut ics.state_score,
                    &kv_view,
                    &sc_view,
                    &iw.ape.buffer,
                    icw,
                    row,
                    pos_mod,
                )?;
            }

            if comp_fires_boundary {
                {
                    let _t = de.events.stage("k.indexer_compressor.pool", &de.compute)?;
                    let mut pooled_view = dgpu_scratch.pooled.slice_view_mut(0, ihd as usize);
                    de.compressor_pool.launch(
                        &de.compute,
                        &mut pooled_view,
                        &ics.state_kv,
                        &ics.state_score,
                        ihd,
                        ratio,
                    )?;
                }
                {
                    let _t = de.events.stage("k.indexer_compressor.rms_w", &de.compute)?;
                    let mut row_view = dgpu_scratch.comp_row.slice_view_mut(0, ihd as usize);
                    let pooled_view = dgpu_scratch.pooled.slice_view(0, ihd as usize);
                    de.rms_w.launch_weighted(
                        &de.compute,
                        &mut row_view,
                        &pooled_view,
                        &iw.norm,
                        ihd,
                        RMS_EPS,
                    )?;
                }
                let comp_pos = pos + 1 - ratio;
                {
                    let _t = de.events.stage("k.indexer_compressor.rope", &de.compute)?;
                    let mut row_view = dgpu_scratch.comp_row.slice_view_mut(0, ihd as usize);
                    de.rope.launch_forward(
                        &de.compute,
                        &mut row_view,
                        1,
                        ihd,
                        N_ROT,
                        comp_pos,
                        &dlw.rope_params,
                    )?;
                }
                // No FP8 step — head_dim=128 ≠ 512 (ds4.c:6702 gates fp8 on head_dim==N_HEAD_DIM).
                {
                    let _t = de.events.stage("k.indexer_compressor.f16rt", &de.compute)?;
                    let mut row_view = dgpu_scratch.comp_row.slice_view_mut(0, ihd as usize);
                    de.f16rt.launch(&de.compute, &mut row_view, ihd)?;
                }
                {
                    let _t = de.events.stage("k.indexer_compressor.shuffle", &de.compute)?;
                    de.compressor_shuffle.launch(
                        &de.compute,
                        &mut ics.state_kv,
                        &mut ics.state_score,
                        icw,
                    )?;
                }
                {
                    let _t = de.events.stage("k.indexer_compressor.comp_kv_append", &de.compute)?;
                    let row_view = dgpu_scratch.comp_row.slice_view(0, ihd as usize);
                    de.comp_kv_append.launch(
                        &de.compute,
                        &mut ics.comp_kv,
                        &row_view,
                        ics.n_comp,
                        ihd,
                    )?;
                }
                ics.n_comp += 1;
            }
            drop(_s_icomp);
            _t_icomp.end()?;
        }

        // ============================================================
        // dGPU: Attention compute
        // ============================================================
        // M55: the live SWA window is rows [raw_off, raw_off + n_raw) of the
        // monotonic cache; every raw-KV reader gets this view (kernels are
        // unchanged — they read rows [0, n_raw) of their base pointer).
        let kv_win = ls.kv_cache.slice_view(
            (ls.raw_off as usize) * (N_HEAD_DIM as usize),
            (ls.n_raw as usize) * (N_HEAD_DIM as usize),
        );
        if ratio == 0 {
            let _t_attn = de.events.stage("dgpu.attn_compute", &de.compute)?;
            let _s_attn = debug_span!("attn_compute").entered();
            de.attn_swa.launch(
                &de.compute,
                &mut dgpu_scratch.heads,
                &dgpu_scratch.q_normed,
                &kv_win,
                &dlw.attn_sinks,
                N_HEAD,
                N_HEAD_DIM,
                ls.n_raw,
            )?;
            drop(_s_attn);
            _t_attn.end()?;
        } else {
            // CSA indexer: at ratio==4 layers with n_index_comp > 512, run
            // matvec(attn_q_b) → RoPE → matvec(proj) → scale →
            // IndexerScore → IndexerTopk → IndexerGather. Result is a
            // dense `active_comp_kv` of ≤512 rows for the attention kernels
            // to consume instead of the full `cs.comp_kv`.
            //
            // Below the early-permit boundary (n_index_comp ≤ 512), or for
            // ratio==128 layers (no indexer), the existing dense path runs
            // unchanged.
            let cs = ls.compressor.as_ref();
            let n_comp_full = cs.map(|c| c.n_comp).unwrap_or(0);
            let ics = ls.indexer_compressor.as_ref();
            let n_index_comp = ics.map(|c| c.n_comp).unwrap_or(0);
            let use_sparse = ratio == 4 && n_index_comp > INDEXER_TOP_K;
            let env_disable_sparse = std::env::var("DECODE_INDEXER")
                .map(|v| v == "off" || v == "0").unwrap_or(false);
            let use_sparse = use_sparse && !env_disable_sparse;

            if use_sparse {
                let _t_ix = de.events.stage("dgpu.indexer", &de.compute)?;
                let _s_ix = debug_span!("indexer_dgpu").entered();
                let iw = dlw
                    .indexer
                    .as_ref()
                    .ok_or_else(|| eyre!("L{layer}: ratio==4 but no indexer weights"))?;
                let ics_ref = ics.expect("ratio==4 must have indexer_compressor state");

                // 1. matvec(attn_q_b × qr_normed) → indexer_q [N_INDEXER_HEAD * N_INDEXER_HEAD_DIM]
                de.f16.matvec(
                    &de.compute,
                    &mut dgpu_scratch.indexer_q,
                    &iw.attn_q_b.buffer,
                    &dgpu_scratch.qr_normed,
                    N_INDEXER_HEAD * N_INDEXER_HEAD_DIM,
                    N_LORA_Q,
                )?;
                // 2. RoPE on indexer_q at this token's global position.
                de.rope.launch_forward(
                    &de.compute,
                    &mut dgpu_scratch.indexer_q,
                    N_INDEXER_HEAD,
                    N_INDEXER_HEAD_DIM,
                    N_ROT,
                    pos,
                    &dlw.rope_params,
                )?;
                // 3. matvec(indexer.proj × attn_input_norm) → head_weights [N_INDEXER_HEAD]
                de.f16.matvec(
                    &de.compute,
                    &mut dgpu_scratch.indexer_head_weights,
                    &iw.proj.buffer,
                    &dgpu_scratch.attn_input_norm,
                    N_INDEXER_HEAD,
                    N_EMBD,
                )?;
                // 4. Scale head_weights *= 1/sqrt(head_dim * n_head)
                let scale =
                    1.0f32 / ((N_INDEXER_HEAD_DIM as f32) * (N_INDEXER_HEAD as f32)).sqrt();
                de.vec_scale.launch(
                    &de.compute,
                    &mut dgpu_scratch.indexer_head_weights,
                    scale,
                    N_INDEXER_HEAD,
                )?;
                // 5. IndexerScore over the contiguous prefix of index_comp_kv.
                // Prefer the WMMA variant when available (28× faster at
                // production decode shape); fall back to the naive kernel
                // on iGPU or any arch without WMMA support.
                let kv_slice = ics_ref
                    .comp_kv
                    .slice_view(0, (n_index_comp * N_INDEXER_HEAD_DIM) as usize);
                if let Some(wmma) = de.indexer_score_wmma.as_ref() {
                    wmma.launch(
                        &de.compute,
                        &mut dgpu_scratch.indexer_scores,
                        &dgpu_scratch.indexer_q,
                        &dgpu_scratch.indexer_head_weights,
                        &kv_slice,
                        n_index_comp,
                    )?;
                } else {
                    de.indexer_score.launch(
                        &de.compute,
                        &mut dgpu_scratch.indexer_scores,
                        &dgpu_scratch.indexer_q,
                        &dgpu_scratch.indexer_head_weights,
                        &kv_slice,
                        n_index_comp,
                        N_INDEXER_HEAD,
                        N_INDEXER_HEAD_DIM,
                    )?;
                }
                // 6. IndexerTopk → sorted indices + bitmap. The bitonic
                // variant (ported from ds4) is 72× faster than the
                // greedy fallback at n_comp=16384.
                de.indexer_topk_bitonic.launch(
                    &de.compute,
                    &mut dgpu_scratch.indexer_selected,
                    &mut dgpu_scratch.indexer_allowed_bits,
                    &mut dgpu_scratch.indexer_topk_scratch,
                    &dgpu_scratch.indexer_scores,
                    n_index_comp,
                    INDEXER_TOP_K,
                )?;
                // 7. Gather selected rows of cs.comp_kv into active_comp_kv.
                let cs_ref = cs.expect("ratio==4 must have main compressor state");
                de.indexer_gather.launch(
                    &de.compute,
                    &mut dgpu_scratch.active_comp_kv,
                    &cs_ref.comp_kv,
                    &dgpu_scratch.indexer_selected,
                    INDEXER_TOP_K,
                    N_HEAD_DIM,
                )?;
                drop(_s_ix);
                _t_ix.end()?;
            }

            // For sparse-attn paths we read from active_comp_kv (≤512
            // rows). Otherwise we read from cs.comp_kv directly.
            let (attn_comp_kv, attn_n_comp) = if use_sparse {
                (Some(&dgpu_scratch.active_comp_kv), INDEXER_TOP_K)
            } else if n_comp_full > 0 {
                (cs.map(|c| &c.comp_kv), n_comp_full)
            } else {
                (None, 0)
            };
            {
                let _t = de.events.stage("dgpu.attn_score", &de.compute)?;
                let _s = debug_span!("attn_score").entered();
                // Default: B=1 head-tiled WMMA score (scalar-arg variant of
                // the prefill batched WMMA score). Isolated at ratio=4
                // n_comp=16384: 312 µs single-token → 58 µs = 5.4× faster.
                // Scalar args avoid the per-batch buffer reads that would
                // force ~86 copy_from_host calls/token via the batched API.
                // DECODE_SCORE=single rolls back to attention_mixed_score.
                let use_b1_wmma_score = std::env::var("DECODE_SCORE")
                    .map(|v| v != "single").unwrap_or(true);
                if use_b1_wmma_score {
                    let n_total_max = ls.n_raw + attn_n_comp;
                    de.attn_mixed.launch_score_b1_htiled_wmma(
                        &de.compute,
                        &mut dgpu_scratch.attn_scores,
                        &dgpu_scratch.q_normed,
                        &kv_win,
                        attn_comp_kv,
                        ls.n_raw, /*raw_off=*/0, attn_n_comp,
                        N_HEAD, N_HEAD_DIM, n_total_max,
                    )?;
                } else {
                    de.attn_mixed.launch_score(
                        &de.compute,
                        &mut dgpu_scratch.attn_scores,
                        &dgpu_scratch.q_normed,
                        &kv_win,
                        attn_comp_kv,
                        N_HEAD, N_HEAD_DIM, ls.n_raw, attn_n_comp,
                    )?;
                }
            }
            {
                let _t = de.events.stage("dgpu.attn_smwsum", &de.compute)?;
                let _s = debug_span!("attn_smwsum").entered();
                // Default: 3-pass K-split smwsum (head-tile=16 + k-split=16
                // + reduce). Recovers MLA V-share at B=1 by tiling 16 heads
                // per WG sharing V via LDS — the win the batched WMMA
                // kernel can't deliver at B=1 due to under-occupation.
                // Isolated bench at ratio=4 n_comp=16384:
                //   baseline      609 µs p50
                //   k-split=16    260 µs p50  (2.34× faster)
                // DECODE_SMWSUM=single rolls back to the existing kernel.
                let use_ksplit = std::env::var("DECODE_SMWSUM")
                    .map(|v| v != "single").unwrap_or(true);
                if use_ksplit {
                    const K_SPLIT: u32 = 16;
                    de.attn_mixed.launch_softmax_only(
                        &de.compute,
                        &mut dgpu_scratch.attn_scores,
                        &dlw.attn_sinks,
                        &mut dgpu_scratch.attn_inv_per_head,
                        N_HEAD, ls.n_raw, attn_n_comp,
                    )?;
                    de.attn_mixed.launch_wsum_b1_htiled_ksplit_ldsv(
                        &de.compute,
                        &mut dgpu_scratch.attn_partials,
                        &dgpu_scratch.attn_scores,
                        &kv_win,
                        attn_comp_kv,
                        N_HEAD, N_HEAD_DIM,
                        ls.n_raw, attn_n_comp,
                        K_SPLIT,
                    )?;
                    de.attn_mixed.launch_reduce_partials_apply_inv(
                        &de.compute,
                        &mut dgpu_scratch.heads,
                        &dgpu_scratch.attn_partials,
                        &dgpu_scratch.attn_inv_per_head,
                        N_HEAD, N_HEAD_DIM, K_SPLIT,
                    )?;
                } else {
                    de.attn_mixed.launch_softmax_wsum(
                        &de.compute,
                        &mut dgpu_scratch.heads,
                        &mut dgpu_scratch.attn_scores,
                        &dlw.attn_sinks,
                        &kv_win,
                        attn_comp_kv,
                        N_HEAD, N_HEAD_DIM, ls.n_raw, attn_n_comp,
                    )?;
                }
            }
        }

        // ============================================================
        // dGPU: Output projection
        // ============================================================
        // output_proj suffix (4 kernels after rope_inv) is captured into
        // a graph. rope_inv takes per-token `pos` and stays a direct launch.
        let _t_out = de.events.stage("dgpu.output_proj", &de.compute)?;
        let _s_out = debug_span!("output_proj").entered();
        de.rope.launch_inverse(
            &de.compute,
            &mut dgpu_scratch.heads,
            N_HEAD,
            N_HEAD_DIM,
            N_ROT,
            pos,
            &dlw.rope_params,
        )?;
        self.dgpu_graphs.run("output_proj_post_rope", layer as u32, &de.compute, |s| {
            de.q8.quantize_input(s, &mut dgpu_scratch.heads_xq, &mut dgpu_scratch.heads_xscale, &dgpu_scratch.heads, Q_FLAT)?;
            de.q8_grouped.matvec_grouped(s, &mut dgpu_scratch.low, &dlw.attn_output_a.buffer, &dgpu_scratch.heads_xq, &dgpu_scratch.heads_xscale, GROUP_DIM, RANK, N_GROUPS)?;
            de.q8.quantize_input(s, &mut dgpu_scratch.low_xq, &mut dgpu_scratch.low_xscale, &dgpu_scratch.low, OUT_LOW)?;
            de.q8.matvec(s, &mut dgpu_scratch.attn_out, &dlw.attn_output_b.buffer, &dgpu_scratch.low_xq, &dgpu_scratch.low_xscale, N_EMBD, OUT_LOW)?;
            Ok(())
        })?;
        drop(_s_out);
        _t_out.end()?;

        // ============================================================
        // dGPU: mHC post attn → after_attn_hc. hc_post reads post + comb
        // directly from the packed `split` buffer (no host roundtrip).
        // ============================================================
        let _t_mhc_post_attn = de.events.stage("dgpu.mhc_post_attn", &de.compute)?;
        let _s_mhc_post_attn = debug_span!("mhc_post_attn").entered();
        de.hc_post.launch_from_split(
            &de.compute,
            &mut dgpu_scratch.after_attn_hc,
            &dgpu_scratch.attn_out,
            &dgpu_scratch.residual,
            &dgpu_scratch.split,
            N_HC,   // n_w
            N_EMBD,
            N_HC,
        )?;
        drop(_s_mhc_post_attn);
        _t_mhc_post_attn.end()?;

        // ============================================================
        // dGPU: mHC pre ffn → ffn_input_norm
        // ============================================================
        // mhc_pre_ffn block (5 kernels, layer-constant params) is captured
        // into a HIP graph on first call and replayed thereafter.
        let _t_mhc_pre_ffn = de.events.stage("dgpu.mhc_pre_ffn", &de.compute)?;
        let _s_mhc_pre_ffn = debug_span!("mhc_pre_ffn").entered();
        let mhc_fused = std::env::var("MHC_FUSED")
            .map(|v| v != "0").unwrap_or(false);
        if mhc_fused {
            self.dgpu_graphs.run("mhc_pre_ffn_fused", layer as u32, &de.compute, |s| {
                de.mhc_pre_fused.launch(
                    s,
                    &mut dgpu_scratch.ffn_input_norm,
                    &dgpu_scratch.after_attn_hc,
                    &dlw.hc_ffn_fn.buffer,
                    &dlw.hc_ffn_scale,
                    &dlw.hc_ffn_base,
                    &dlw.ffn_norm,
                    RMS_EPS,
                    SINKHORN_ITERS,
                )?;
                Ok(())
            })?;
        } else {
        self.dgpu_graphs.run("mhc_pre_ffn", layer as u32, &de.compute, |s| {
            let mode = std::env::var("RMS_NW_MW").unwrap_or_else(|_| "fused".into());
            match mode.as_str() {
                "0" | "single" => {
                    de.rms_nw.launch(s, &mut dgpu_scratch.flat, &dgpu_scratch.after_attn_hc, 1, HC_DIM, RMS_EPS)?;
                    de.f16.matvec(s, &mut dgpu_scratch.mix, &dlw.hc_ffn_fn.buffer, &dgpu_scratch.flat, HC_MIX_DIM, HC_DIM)?;
                }
                "split" => {
                    de.rms_nw_mw.launch(s, &mut dgpu_scratch.flat, &dgpu_scratch.after_attn_hc, &mut dgpu_scratch.rms_nw_partials, HC_DIM, 16, RMS_EPS)?;
                    de.f16.matvec(s, &mut dgpu_scratch.mix, &dlw.hc_ffn_fn.buffer, &dgpu_scratch.flat, HC_MIX_DIM, HC_DIM)?;
                }
                _ => {
                    de.rms_nw_mw.launch_inv_only(s, &mut dgpu_scratch.rms_nw_inv_scalar, &dgpu_scratch.after_attn_hc, &mut dgpu_scratch.rms_nw_partials, HC_DIM, 16, RMS_EPS)?;
                    let ksplit: u32 = std::env::var("F16_KSPLIT")
                        .ok().and_then(|s| s.parse().ok()).unwrap_or(16);
                    if ksplit > 0 {
                        de.f16.matvec_narrow_ksplit_pre_scaled(
                            s, &mut dgpu_scratch.mix, &dlw.hc_ffn_fn.buffer,
                            &dgpu_scratch.after_attn_hc, &dgpu_scratch.rms_nw_inv_scalar,
                            &mut dgpu_scratch.mhc_matvec_partials,
                            HC_MIX_DIM, HC_DIM, ksplit,
                        )?;
                    } else {
                        de.f16.matvec_pre_scaled(s, &mut dgpu_scratch.mix, &dlw.hc_ffn_fn.buffer, &dgpu_scratch.after_attn_hc, &dgpu_scratch.rms_nw_inv_scalar, HC_MIX_DIM, HC_DIM)?;
                    }
                }
            }
            de.hc_sinkhorn.launch(s, &mut dgpu_scratch.split, &dgpu_scratch.mix, &dlw.hc_ffn_scale, &dlw.hc_ffn_base, N_HC, SINKHORN_ITERS, SINKHORN_EPS)?;
            de.hc_weighted.launch(s, &mut dgpu_scratch.ffn_cur, &dgpu_scratch.after_attn_hc, &dgpu_scratch.split, N_EMBD, N_HC)?;
            if std::env::var("RMS_W_MW").map(|v| v != "0").unwrap_or(false) {
                de.rms_nw_mw.launch_weighted(s, &mut dgpu_scratch.ffn_input_norm, &dgpu_scratch.ffn_cur, &dlw.ffn_norm, &mut dgpu_scratch.rms_nw_partials, N_EMBD, 16, RMS_EPS)?;
            } else {
                de.rms_w.launch_weighted(s, &mut dgpu_scratch.ffn_input_norm, &dgpu_scratch.ffn_cur, &dlw.ffn_norm, N_EMBD, RMS_EPS)?;
            }
            Ok(())
        })?;
        }
        drop(_s_mhc_pre_ffn);
        _t_mhc_pre_ffn.end()?;

        // ============================================================
        // dGPU: router. Runs on dGPU because (a) the f16 matvec is ~1.5 ms
        // faster on dGPU's 2.6× iGPU BW, and (b) keeping router off iGPU
        // lifts it from the iGPU MoE critical path.
        //
        // Order on dGPU (all on de.compute, FIFO):
        //   1. router_logits = matvec(ffn_gate_inp, ffn_input_norm)
        //   2. learned: router_topk → d_selected, d_ew
        //      hash:    host sync, readback router_logits, hash select,
        //               write back d_selected and d_ew.
        //   3. record selected_ready event.
        //
        // dGPU.xfer pushes both ffn_input_norm (for MoE q8k_xq) and
        // d_selected/d_ew (for iq2 + q2k) to iGPU in FIFO order:
        //   xfer: wait ain_ready → push ffn_input_norm
        //         wait selected_ready → push d_selected → push d_ew
        //         record selected_pushed
        //
        // iGPU.compute waits selected_pushed (which transitively
        // covers ain_pushed since both pushes are on the same xfer
        // stream), then runs the MoE graph.
        // ============================================================
        let parallel = matches!(self.mode, ExecMode::HetParallel);
        let sev = &self.sync_events.layers[layer as usize];

        // Mark ffn_input_norm ready, queue its peer push on xfer.
        if parallel {
            sev.ain_ready.record(&de.compute)?;
            let _t_wait = de
                .events
                .stage("dgpu.peer_push_ffn_input_norm.wait", &de.xfer)?;
            de.xfer.wait_event(&sev.ain_ready)?;
            _t_wait.end()?;
        } else {
            de.compute.synchronize()?;
        }
        let _t_peer_ain = de.events.stage("dgpu.peer_push_ffn_input_norm", &de.xfer)?;
        let _s_peer_ain = debug_span!("peer_push_ffn_input_norm").entered();
        peer_push_f32(
            &dgpu_scratch.ffn_input_norm,
            &mut igpu_scratch.ffn_input_norm_recv,
            &de.xfer,
        )?;
        if parallel {
            sev.ain_pushed.record(&de.xfer)?;
        } else {
            de.xfer.synchronize()?;
        }
        drop(_s_peer_ain);
        _t_peer_ain.end()?;

        // Router runs on dGPU.compute.
        let _t_router = de.events.stage("dgpu.router", &de.compute)?;
        let _s_router = debug_span!("router_dgpu").entered();
        {
            let _t = de.events.stage("k.router.f16_matvec", &de.compute)?;
            de.f16.matvec(
                &de.compute,
                &mut dgpu_scratch.router_logits,
                &dlw.ffn_gate_inp.buffer,
                &dgpu_scratch.ffn_input_norm,
                N_EXPERT,
                N_EMBD,
            )?;
        }
        if !dlw.is_hash_router {
            let _t = de.events.stage("k.router.topk", &de.compute)?;
            de.router_topk.launch(
                &de.compute,
                &mut dgpu_scratch.d_selected,
                &mut dgpu_scratch.d_ew,
                &dgpu_scratch.router_logits,
                dlw.router_bias_dev.as_ref(),
                N_EXPERT,
                N_EXPERT_USED as u32,
                EXPERT_WEIGHT_SCALE,
                ROUTER_WEIGHT_EPS,
            )?;
        } else {
            // Hash router: host sync the router matvec, read 6 chosen
            // logits, write back d_selected and d_ew on dGPU.compute.
            // The shared expert that runs after this will be FIFO-
            // serialized behind the copy_from_host; the iGPU MoE
            // wait depends on selected_pushed which depends on these
            // writes via stream-FIFO.
            de.compute.synchronize()?;
            dgpu_scratch
                .router_logits
                .copy_to_host(&mut dgpu_scratch.router_logits_host)?;
            let tid2eid = dlw
                .tid2eid
                .as_ref()
                .ok_or_else(|| eyre!("L{layer}: hash router but no tid2eid"))?;
            let (sel, w) =
                hash_router_select(tid2eid, token_id, &dgpu_scratch.router_logits_host);
            dgpu_scratch.d_selected.copy_from_host(&sel)?;
            dgpu_scratch.d_ew.copy_from_host(&w)?;
        }
        drop(_s_router);
        _t_router.end()?;

        // Push d_selected and d_ew to iGPU on dGPU.xfer (FIFO after the
        // ffn_input_norm push above).
        if parallel {
            sev.selected_ready.record(&de.compute)?;
            let _t_wait = de
                .events
                .stage("dgpu.peer_push_selected.wait", &de.xfer)?;
            de.xfer.wait_event(&sev.selected_ready)?;
            _t_wait.end()?;
        } else {
            de.compute.synchronize()?;
        }
        let _t_peer_sel = de.events.stage("dgpu.peer_push_selected", &de.xfer)?;
        let _s_peer_sel = debug_span!("peer_push_selected").entered();
        {
            let _t = de.events.stage("k.peer_push_selected.d_selected", &de.xfer)?;
            peer_push_i32(
                &dgpu_scratch.d_selected,
                &mut igpu_scratch.d_selected,
                &de.xfer,
            )?;
        }
        {
            let _t = de.events.stage("k.peer_push_selected.d_ew", &de.xfer)?;
            peer_push_f32(
                &dgpu_scratch.d_ew,
                &mut igpu_scratch.d_ew,
                &de.xfer,
            )?;
        }
        if parallel {
            sev.selected_pushed.record(&de.xfer)?;
            // M54: value-signal companion to selected_pushed, consumed by
            // the PRE-ISSUED iGPU lane (issue_igpu_moe). An event wait
            // enqueued before its record call is a no-op (snapshot
            // semantics); the 32-bit value wait compares at execution
            // time, which makes token-start pre-issue sound. One SDMA
            // dword write per layer — negligible when pre-issue is off.
            let seq = self
                .token_seq
                .load(std::sync::atomic::Ordering::Relaxed);
            // Const-cast: the engine only ever writes this slot from
            // device streams; the host slice is allocation-stable.
            let sig = unsafe {
                (self.moe_signal.as_slice().as_ptr() as *mut u32).add(layer as usize)
            };
            unsafe { de.xfer.write_value32(sig, seq)? };
        } else {
            de.xfer.synchronize()?;
        }
        drop(_s_peer_sel);
        _t_peer_sel.end()?;

        // dGPU shared expert can now run on de.compute (after router,
        // before / in parallel with iGPU MoE). It uses the same
        // ffn_input_norm input as the router.
        if parallel {
            let _t_shared = de.events.stage("dgpu.shared_expert", &de.compute)?;
            let _s_shared = debug_span!("shared_expert").entered();
            self.issue_shared_expert_graph(de, dgpu_scratch, dlw, layer)?;
            drop(_s_shared);
            _t_shared.end()?;
        }

        // M54 pre-issue mode: the whole iGPU MoE lane (wait → graph →
        // push) was enqueued at token start by issue_igpu_moe; skip the
        // inline iGPU section entirely (the dGPU's moe_arrived wait below
        // synchronizes against the pre-issued lane).
        if !igpu_moe_preissued {
        // Switch to iGPU for the MoE.
        self.set_current_cached(self.igpu.device)?;
        let ie = &self.igpu;

        if parallel {
            let _t_wait = ie.events.stage("igpu.moe.wait", &ie.compute)?;
            ie.compute.wait_event(&sev.selected_pushed)?;
            _t_wait.end()?;
        }

        let gbpe = ilw.routed.gate_bytes_per_expert;
        let ubpe = ilw.routed.up_bytes_per_expert;
        let dbpe = ilw.routed.down_bytes_per_expert;

        // ============================================================
        // iGPU: routed MoE → ffn_moe
        // (Per-slot host syncs inside still block iGPU compute, but
        //  dGPU's shared-expert kernels — already queued above in
        //  parallel mode — run concurrently.)
        // ============================================================
        // Fully device-side MoE pipeline (no per-slot host syncs). Each
        // iq2 writes directly into the cat positions, swiglu_cw fires
        // once, q8k_quantize + q2k accumulate then drain the cat via the
        // d_midq_cat staging buffer.
        //
        // The 4-kernel core (q8k_xq → iq2_fused → q8k_mid → q2k_down) is
        // captured into a per-layer HIP graph on first call and replayed
        // thereafter — all params are device pointers + layer-constant
        // scalars, so the graph stays valid for every subsequent token.
        //
        // d_selected and d_ew are device-side here: peer-pushed from
        // dGPU (router runs there) and arriving via the selected_pushed
        // event already waited on above.
        let mid_blocks_bytes = (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8_K_BYTES;
        let _t_moe = ie.events.stage("igpu.routed_moe", &ie.compute)?;
        let _s_moe = debug_span!("routed_moe").entered();
        // 4-kernel core (q8k_xq → iq2_fused → q8k_mid → q2k_down) captured
        // into a per-layer HIP graph on first call and replayed thereafter.
        // All kernel params are device pointers + layer-constant scalars.
        // Per-kernel event staging stays OUTSIDE the capture; the
        // event-record nodes would otherwise become part of the replayed
        // graph and corrupt the per-token harvest.
        self.igpu_graphs.run("routed_moe", layer as u32, &ie.compute, |s| {
            ie.q8k.launch(s, &mut igpu_scratch.d_xq_q8k, &igpu_scratch.ffn_input_norm_recv, BLOCKS_Q8K_GATE_IN)?;
            ie.iq2.launch_fused_swiglu_batch(s, &mut igpu_scratch.d_mid_cat, &ilw.routed.gate.buffer, &ilw.routed.up.buffer, &igpu_scratch.d_xq_q8k, &igpu_scratch.d_ew, &igpu_scratch.d_selected, gbpe as u32, ubpe as u32, N_EXPERT_USED as u32, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN)?;
            ie.q8k.launch(s, &mut igpu_scratch.d_midq_cat, &igpu_scratch.d_mid_cat, BLOCKS_Q8K_DOWN_IN * (N_EXPERT_USED as u32))?;
            ie.q2k.launch_batched(s, &mut igpu_scratch.ffn_moe, &ilw.routed.down.buffer, &igpu_scratch.d_midq_cat, &igpu_scratch.d_selected, dbpe as u32, mid_blocks_bytes as u32, N_EXPERT_USED as u32, N_EMBD, BLOCKS_Q8K_DOWN_IN)?;
            Ok(())
        })?;
        // `selected` is not materialized on host — d_selected stays
        // device-side, peer-pushed from dGPU above.
        drop(_s_moe);
        _t_moe.end()?;

        // ============================================================
        // iGPU → dGPU: peer push ffn_moe (16 KB)
        // ============================================================
        if parallel {
            sev.moe_done.record(&ie.compute)?;
            let _t_wait = ie
                .events
                .stage("igpu.peer_push_ffn_moe.wait", &ie.xfer)?;
            ie.xfer.wait_event(&sev.moe_done)?;
            _t_wait.end()?;
        } else {
            ie.compute.synchronize()?;
        }
        let _t_peer_moe = ie.events.stage("igpu.peer_push_ffn_moe", &ie.xfer)?;
        let _s_peer_moe = debug_span!("peer_push_ffn_moe").entered();
        peer_push_f32(
            &igpu_scratch.ffn_moe,
            &mut dgpu_scratch.ffn_moe_recv,
            &ie.xfer,
        )?;
        if parallel {
            sev.moe_arrived.record(&ie.xfer)?;
        } else {
            ie.xfer.synchronize()?;
        }
        drop(_s_peer_moe);
        _t_peer_moe.end()?;
        } // !igpu_moe_preissued

        // ============================================================
        // dGPU: in serial mode, issue shared expert NOW. In parallel
        // mode, it was issued earlier — just wait for ffn_moe to arrive.
        // ============================================================
        self.set_current_cached(self.dgpu.device)?;
        if !parallel {
            let _t_shared = de.events.stage("dgpu.shared_expert", &de.compute)?;
            let _s_shared = debug_span!("shared_expert").entered();
            self.issue_shared_expert_graph(de, dgpu_scratch, dlw, layer)?;
            drop(_s_shared);
            _t_shared.end()?;
        }

        // ffn_moe_recv += ffn_shared. In parallel mode wait for the
        // peer copy to land before doing the add.
        if parallel {
            let _t_wait = de.events.stage("dgpu.ffn_combine.wait", &de.compute)?;
            de.compute.wait_event(&sev.moe_arrived)?;
            _t_wait.end()?;
        }
        // Non-last layers launch the fused cross-layer graph (this
        // layer's ffn_combine + next layer's mhc_pre_attn) for a single
        // graph submit, eliminating the host-scheduling gap between
        // them. The last layer fires a pure combine.
        let combine_label = if is_last_layer {
            "dgpu.ffn_combine"
        } else {
            "dgpu.ffn_combine+next_pre_attn"
        };
        let _t_combine = de.events.stage(combine_label, &de.compute)?;
        let _s_combine = debug_span!("ffn_combine").entered();
        if is_last_layer {
            // (vec_add → hc_post writing residual_next). Stable per-layer
            // pointers thanks to the end-of-token extra swap. Used ONLY
            // for the last layer; non-last layers ride the combined graph.
            self.dgpu_graphs.run("ffn_combine", layer as u32, &de.compute, |s| {
                de.vec_add.launch(s, &mut dgpu_scratch.ffn_moe_recv, &dgpu_scratch.ffn_shared, N_EMBD)?;
                de.hc_post.launch_from_split(s, &mut dgpu_scratch.residual_next, &dgpu_scratch.ffn_moe_recv, &dgpu_scratch.after_attn_hc, &dgpu_scratch.split, N_HC, N_EMBD, N_HC)?;
                Ok(())
            })?;
        } else {
            // Combined ffn_combine_N + mhc_pre_attn_{N+1} graph. The
            // mhc_pre_attn block reads from layer N+1's `residual` which
            // is THIS layer's `residual_next` after the post-layer swap
            // — same physical buffer. We pass `residual_next.raw()`
            // throughout so the captured graph references one ptr.
            let next = next_dlw.expect("combined_ffn_pre_attn: next_dlw required for non-last layer");
            self.dgpu_graphs.run("combined_ffn_pre_attn", layer as u32, &de.compute, |s| {
                // ffn_combine half — writes residual_next (= layer N+1's residual).
                de.vec_add.launch(s, &mut dgpu_scratch.ffn_moe_recv, &dgpu_scratch.ffn_shared, N_EMBD)?;
                de.hc_post.launch_from_split(s, &mut dgpu_scratch.residual_next, &dgpu_scratch.ffn_moe_recv, &dgpu_scratch.after_attn_hc, &dgpu_scratch.split, N_HC, N_EMBD, N_HC)?;
                // mhc_pre_attn half — reads residual_next (= layer N+1's residual
                // after swap), uses layer N+1's hc/norm weights.
                let mode = std::env::var("RMS_NW_MW").unwrap_or_else(|_| "fused".into());
                match mode.as_str() {
                    "0" | "single" => {
                        de.rms_nw.launch(s, &mut dgpu_scratch.flat, &dgpu_scratch.residual_next, 1, HC_DIM, RMS_EPS)?;
                        de.f16.matvec(s, &mut dgpu_scratch.mix, &next.hc_attn_fn.buffer, &dgpu_scratch.flat, HC_MIX_DIM, HC_DIM)?;
                    }
                    "split" => {
                        de.rms_nw_mw.launch(s, &mut dgpu_scratch.flat, &dgpu_scratch.residual_next, &mut dgpu_scratch.rms_nw_partials, HC_DIM, 16, RMS_EPS)?;
                        de.f16.matvec(s, &mut dgpu_scratch.mix, &next.hc_attn_fn.buffer, &dgpu_scratch.flat, HC_MIX_DIM, HC_DIM)?;
                    }
                    _ => {
                        de.rms_nw_mw.launch_inv_only(s, &mut dgpu_scratch.rms_nw_inv_scalar, &dgpu_scratch.residual_next, &mut dgpu_scratch.rms_nw_partials, HC_DIM, 16, RMS_EPS)?;
                        let ksplit: u32 = std::env::var("F16_KSPLIT")
                            .ok().and_then(|s| s.parse().ok()).unwrap_or(16);
                        if ksplit > 0 {
                            de.f16.matvec_narrow_ksplit_pre_scaled(
                                s, &mut dgpu_scratch.mix, &next.hc_attn_fn.buffer,
                                &dgpu_scratch.residual_next, &dgpu_scratch.rms_nw_inv_scalar,
                                &mut dgpu_scratch.mhc_matvec_partials,
                                HC_MIX_DIM, HC_DIM, ksplit,
                            )?;
                        } else {
                            de.f16.matvec_pre_scaled(s, &mut dgpu_scratch.mix, &next.hc_attn_fn.buffer, &dgpu_scratch.residual_next, &dgpu_scratch.rms_nw_inv_scalar, HC_MIX_DIM, HC_DIM)?;
                        }
                    }
                }
                de.hc_sinkhorn.launch(s, &mut dgpu_scratch.split, &dgpu_scratch.mix, &next.hc_attn_scale, &next.hc_attn_base, N_HC, SINKHORN_ITERS, SINKHORN_EPS)?;
                de.hc_weighted.launch(s, &mut dgpu_scratch.attn_cur, &dgpu_scratch.residual_next, &dgpu_scratch.split, N_EMBD, N_HC)?;
                // RMS_W_MW=1 enables multi-WG weighted RMS. Default OFF: at
                // N_EMBD=4096 the single-WG version is small enough that
                // multi-WG's extra kernel launch erases any parallelism
                // win — averaged across 256-token decode runs the variants
                // are statistical ties (~±2% within thermal noise).
                if std::env::var("RMS_W_MW").map(|v| v != "0").unwrap_or(false) {
                    de.rms_nw_mw.launch_weighted(s, &mut dgpu_scratch.attn_input_norm, &dgpu_scratch.attn_cur, &next.attn_norm, &mut dgpu_scratch.rms_nw_partials, N_EMBD, 16, RMS_EPS)?;
                } else {
                    de.rms_w.launch_weighted(s, &mut dgpu_scratch.attn_input_norm, &dgpu_scratch.attn_cur, &next.attn_norm, N_EMBD, RMS_EPS)?;
                }
                Ok(())
            })?;
        }
        drop(_s_combine);
        _t_combine.end()?;
        if serial {
            de.compute.synchronize()?;
        }
        Ok(())
    }
}

impl HeterogeneousEngine {
    /// Capture the dGPU shared-expert chain (6 kernels, all layer-constant
    /// params) into a HIP graph on first call; replay thereafter. Caller is
    /// responsible for the surrounding events stage and span.
    fn issue_shared_expert_graph(
        &self,
        de: &DeviceEngine,
        dgpu_scratch: &mut DgpuScratch,
        dlw: &DgpuLayerWeights,
        layer: i32,
    ) -> eyre::Result<()> {
        self.dgpu_graphs.run("shared_expert", layer as u32, &de.compute, |s| {
            issue_shared_expert(de, dgpu_scratch, dlw, s)
        })
    }
}

/// Issue all dGPU shared-expert kernels on `stream` (= `de.compute` in
/// practice). Reads `ffn_input_norm`, writes `ffn_shared`. Used by both
/// modes; in `HetParallel` it's invoked earlier to overlap with iGPU MoE.
fn issue_shared_expert(
    de: &DeviceEngine,
    dgpu_scratch: &mut DgpuScratch,
    dlw: &DgpuLayerWeights,
    stream: &v4flash_hip::Stream,
) -> eyre::Result<()> {
    de.q8.quantize_input(stream, &mut dgpu_scratch.xq_n_embd, &mut dgpu_scratch.xscale_n_embd, &dgpu_scratch.ffn_input_norm, N_EMBD)?;
    de.q8.matvec(stream, &mut dgpu_scratch.gate_sh, &dlw.shared.gate.buffer, &dgpu_scratch.xq_n_embd, &dgpu_scratch.xscale_n_embd, N_FF_SHARED, N_EMBD)?;
    de.q8.matvec(stream, &mut dgpu_scratch.up_sh, &dlw.shared.up.buffer, &dgpu_scratch.xq_n_embd, &dgpu_scratch.xscale_n_embd, N_FF_SHARED, N_EMBD)?;
    de.swiglu.launch(stream, &mut dgpu_scratch.mid_sh, &dgpu_scratch.gate_sh, &dgpu_scratch.up_sh, N_FF_SHARED)?;
    de.q8.quantize_input(stream, &mut dgpu_scratch.mid_sh_xq, &mut dgpu_scratch.mid_sh_xscale, &dgpu_scratch.mid_sh, N_FF_SHARED)?;
    de.q8.matvec(stream, &mut dgpu_scratch.ffn_shared, &dlw.shared.down.buffer, &dgpu_scratch.mid_sh_xq, &dgpu_scratch.mid_sh_xscale, N_EMBD, N_FF_SHARED)?;
    Ok(())
}

#[allow(dead_code)]
fn softplus_stable(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0f32 + x.exp()).ln()
    }
}

#[allow(dead_code)]
fn topk_desc(score: &[f32], k: usize) -> [i32; 6] {
    let mut idx = [-1i32; 6];
    for i in 0..score.len() {
        for j in 0..k {
            if idx[j] < 0 || score[i] > score[idx[j] as usize] {
                for m in (j + 1..k).rev() {
                    idx[m] = idx[m - 1];
                }
                idx[j] = i as i32;
                break;
            }
        }
    }
    idx
}

// Suppress dead-code warning for now (used by forward_head + tests).
#[allow(dead_code)]
fn _unused_imports_warn_suppressor(_d: &Device, _b: &DeviceBuffer<f32>, _e: &DeviceEngine) {}
