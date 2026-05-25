//! M40-P4: per-token-stream substage-interleaved pair forward with
//! cross-layer pipelining.
//!
//! Mental model: the iGPU is an autonomous pipeline that does ONE job —
//! receive (ffn_input, d_selected, d_ew) for some (layer, token), produce
//! ffn_output. iGPU.compute serializes MoEs FIFO; iGPU.xfer pushes
//! results back FIFO. Nothing on the iGPU is per-token.
//!
//! The dGPU runs ALL the rest. To overlap, it has TWO compute streams
//! (de.compute_t0 / _t1) and TWO xfer streams (de.xfer_t0 / _t1). Each
//! token's mhc_pre_attn → q/kv → attn → output_proj → mhc_post →
//! mhc_pre_ffn → router → shared_expert lives on its own pair of streams.
//! Both streams write/read into per-token `TokenScratch` instances so
//! there's no aliasing. The dGPU scheduler co-schedules t0 and t1
//! kernels onto whatever CUs are free.
//!
//! Cross-token state dependency:
//!   * SHARED kv_cache: t0 writes row `pos`, t1 writes row `pos+1`
//!     (different rows; no aliasing for the writes). But t1's attn reads
//!     ALL rows [0..n_raw_t1] including t0's just-appended row → must
//!     wait for t0's kv_append before t1's attn. Event:
//!     `pair_t0_state_ready[L]` recorded on de.compute_t0 after t0's
//!     compressor stage, waited by de.compute_t1 before attn.
//!
//! Cross-layer pipeline:
//!   * Prologue: pre_moe for L=0 (both tokens) + queue iGPU MoEs
//!   * Loop L ∈ [0, N-2]:
//!       - post_moe (vec_add + hc_post → ts.residual_next) for L (both tokens)
//!       - residual ← residual_next (async copy)
//!       - pre_moe for L+1 (both tokens) + queue iGPU MoEs
//!     This loop body keeps de.compute_t0/_t1 continuously busy: while
//!     the dGPU is queueing L+1's pre_moe, the iGPU is still chewing L's
//!     MoEs. When L+1's wait_event(moe_arrived) fires, it's a no-op.
//!   * Epilogue: post_moe for L=N-1 (both tokens)
//!   * Head twice
//!
//! No host syncs anywhere except the final `compute_t0.synchronize()`
//! at the end of the function (to bound wall-time measurement).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, Event, Stream};

use crate::forward::{
    hash_router_select, BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, EXPERT_WEIGHT_SCALE, GROUP_DIM,
    HC_DIM, HC_MIX_DIM, N_EMBD, N_EXPERT, N_EXPERT_USED, N_FF_EXP, N_FF_SHARED, N_GROUPS, N_HC,
    N_HEAD, N_HEAD_DIM, N_LAYER, N_LORA_Q, N_ROT, OUT_LOW, Q_FLAT, RANK, RMS_EPS, SINKHORN_EPS,
    SINKHORN_ITERS, SWA_WINDOW, SWIGLU_CLAMP_EXP,
};
use crate::q8_k::BLOCK_Q8_K_BYTES;

use super::engine::{HeterogeneousEngine, LayerSyncEvents};
use super::scratch::{DgpuScratch, IgpuScratch, TokenScratch};
use super::state::HetLayerState;
use super::sync::{peer_push_f32, peer_push_i32};
use super::weights::{DgpuLayerWeights, HetModelWeights, IgpuLayerWeights};

const ROUTER_WEIGHT_EPS: f32 = 6.103515625e-5;

/// Pick a `&'static str` based on the token id at compile time. Used to
/// dispatch fine-grained perfetto stage names to the right per-token track
/// (events.stage requires &'static str so we can't fmt! at runtime).
macro_rules! stage_t {
    ($which:expr, $base:literal) => {
        match $which {
            TokenId::T0 => concat!($base, "_t0"),
            TokenId::T1 => concat!($base, "_t1"),
        }
    };
}

/// Direction of the cross-token state-ready event used in pre_moe.
enum KvEventDir<'a> {
    /// Token 0: record the event after our kv_append (+ optional comp_kv_append).
    Record(&'a Event),
    /// Token 1: wait on token 0's event before our attention reads kv_cache.
    Wait(&'a Event),
}

/// Which token's iGPU recv buffers to use.
enum TokenId {
    T0,
    T1,
}

impl HeterogeneousEngine {
    /// M40-P4: per-token-stream substage-interleaved pair forward with
    /// cross-layer pipelining. Same semantics as Phase 1 `forward_pair`
    /// (bit-equivalent to two sequential forward_token calls), much faster.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_pair_interleaved(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        state: &mut super::HetModelState,
        weights: &HetModelWeights,
        input_hc_0: &[f32],
        input_hc_1: &[f32],
        pos: u32,
        token_id_0: i32,
        token_id_1: i32,
    ) -> eyre::Result<()> {
        use tracing::debug_span;

        if input_hc_0.len() != HC_DIM as usize || input_hc_1.len() != HC_DIM as usize {
            return Err(eyre!(
                "forward_pair_interleaved: input_hc len mismatch ({}/{} vs HC_DIM {HC_DIM})",
                input_hc_0.len(),
                input_hc_1.len()
            ));
        }
        let _span = debug_span!("het.pair_interleaved", pos, token_id_0, token_id_1).entered();

        self.dgpu.events.reset();
        self.igpu.events.reset();
        self.set_current_cached(self.dgpu.device)?;

        // Streams (per-token streams allocated for dGPU in for_arch).
        let de_compute_t0 = self
            .dgpu
            .compute_t0
            .as_ref()
            .ok_or_else(|| eyre!("dGPU compute_t0 missing"))?;
        let de_compute_t1 = self
            .dgpu
            .compute_t1
            .as_ref()
            .ok_or_else(|| eyre!("dGPU compute_t1 missing"))?;
        let de_xfer_t0 = self
            .dgpu
            .xfer_t0
            .as_ref()
            .ok_or_else(|| eyre!("dGPU xfer_t0 missing"))?;
        let de_xfer_t1 = self
            .dgpu
            .xfer_t1
            .as_ref()
            .ok_or_else(|| eyre!("dGPU xfer_t1 missing"))?;

        // ===== Initial residual upload =====
        dgpu_scratch.t0.residual.copy_from_host(input_hc_0)?;
        dgpu_scratch.t1.residual.copy_from_host(input_hc_1)?;

        let pair_start = std::time::Instant::now();

        // ===== PROLOGUE: pre_moe + queue iGPU MoE for L=0 (both tokens) =====
        {
            let dlw = &weights.dgpu_layers[0];
            let ilw = &weights.igpu_layers[0];
            let ls = &mut state.layers[0];
            let evt_t0 = &self.sync_events.layers[0];
            let evt_t1 = &self.sync_events_t1.layers[0];
            let kv_evt = &self.pair_t0_state_ready[0];

            // t0 (on de_compute_t0 / de_xfer_t0) — records kv_evt after compressor.
            self.pair_pre_moe_one_token(
                &mut dgpu_scratch.t0,
                igpu_scratch,
                ls,
                dlw,
                ilw,
                pos,
                token_id_0,
                evt_t0,
                de_compute_t0,
                de_xfer_t0,
                KvEventDir::Record(kv_evt),
                TokenId::T0,
            )?;
            // t1 (on de_compute_t1 / de_xfer_t1) — waits on kv_evt before attn.
            self.pair_pre_moe_one_token(
                &mut dgpu_scratch.t1,
                igpu_scratch,
                ls,
                dlw,
                ilw,
                pos + 1,
                token_id_1,
                evt_t1,
                de_compute_t1,
                de_xfer_t1,
                KvEventDir::Wait(kv_evt),
                TokenId::T1,
            )?;
            // iGPU MoE for both tokens (FIFO on ie.compute).
            self.pair_igpu_moe_one_token(
                &mut dgpu_scratch.t0,
                igpu_scratch,
                ilw,
                evt_t0,
                TokenId::T0,
            )?;
            self.pair_igpu_moe_one_token(
                &mut dgpu_scratch.t1,
                igpu_scratch,
                ilw,
                evt_t1,
                TokenId::T1,
            )?;
        }

        // ===== STEADY STATE: L ∈ [0, N-2] — post_moe(L) + pre_moe(L+1) =====
        for layer in 0..(N_LAYER as usize - 1) {
            let dlw_l = &weights.dgpu_layers[layer];
            let dlw_l1 = &weights.dgpu_layers[layer + 1];
            let ilw_l1 = &weights.igpu_layers[layer + 1];
            let evt_t0_l = &self.sync_events.layers[layer];
            let evt_t1_l = &self.sync_events_t1.layers[layer];
            let evt_t0_l1 = &self.sync_events.layers[layer + 1];
            let evt_t1_l1 = &self.sync_events_t1.layers[layer + 1];
            let kv_evt_l1 = &self.pair_t0_state_ready[layer + 1];

            // post_moe for L on t0 (FIFO ordered on de_compute_t0).
            self.pair_post_moe_one_token(
                &mut dgpu_scratch.t0,
                dlw_l,
                evt_t0_l,
                de_compute_t0,
            )?;
            // post_moe for L on t1.
            self.pair_post_moe_one_token(
                &mut dgpu_scratch.t1,
                dlw_l,
                evt_t1_l,
                de_compute_t1,
            )?;

            // residual ← residual_next (async on the same per-token stream).
            dgpu_scratch
                .t0
                .residual
                .copy_from_buffer_async(&dgpu_scratch.t0.residual_next, de_compute_t0)?;
            dgpu_scratch
                .t1
                .residual
                .copy_from_buffer_async(&dgpu_scratch.t1.residual_next, de_compute_t1)?;

            // pre_moe for L+1 on both tokens. Note: ls for layer L+1.
            let ls_l1 = &mut state.layers[layer + 1];
            self.pair_pre_moe_one_token(
                &mut dgpu_scratch.t0,
                igpu_scratch,
                ls_l1,
                dlw_l1,
                ilw_l1,
                pos,
                token_id_0,
                evt_t0_l1,
                de_compute_t0,
                de_xfer_t0,
                KvEventDir::Record(kv_evt_l1),
                TokenId::T0,
            )?;
            self.pair_pre_moe_one_token(
                &mut dgpu_scratch.t1,
                igpu_scratch,
                ls_l1,
                dlw_l1,
                ilw_l1,
                pos + 1,
                token_id_1,
                evt_t1_l1,
                de_compute_t1,
                de_xfer_t1,
                KvEventDir::Wait(kv_evt_l1),
                TokenId::T1,
            )?;
            // Queue iGPU MoE for L+1.
            self.pair_igpu_moe_one_token(
                &mut dgpu_scratch.t0,
                igpu_scratch,
                ilw_l1,
                evt_t0_l1,
                TokenId::T0,
            )?;
            self.pair_igpu_moe_one_token(
                &mut dgpu_scratch.t1,
                igpu_scratch,
                ilw_l1,
                evt_t1_l1,
                TokenId::T1,
            )?;
        }

        // ===== EPILOGUE: post_moe for last layer =====
        {
            let last = (N_LAYER as usize) - 1;
            let dlw = &weights.dgpu_layers[last];
            let evt_t0 = &self.sync_events.layers[last];
            let evt_t1 = &self.sync_events_t1.layers[last];
            self.pair_post_moe_one_token(&mut dgpu_scratch.t0, dlw, evt_t0, de_compute_t0)?;
            self.pair_post_moe_one_token(&mut dgpu_scratch.t1, dlw, evt_t1, de_compute_t1)?;
        }

        // ===== HEAD x2 =====
        // forward_head reads dgpu_scratch.residual + writes dgpu_scratch.logits.
        // Stage t0's final HC into dgpu_scratch.residual, run head, save logits;
        // then do t1. Use the main de.compute stream for the head (single-stream
        // is fine here; the two heads run sequentially after the layers).
        // First make sure all per-token work is done.
        de_compute_t0.synchronize()?;
        de_compute_t1.synchronize()?;
        dgpu_scratch
            .residual
            .copy_from_buffer(&dgpu_scratch.t0.residual_next)?;
        self.forward_head(dgpu_scratch, &weights.global)?;
        dgpu_scratch
            .logits_token0
            .copy_from_buffer(&dgpu_scratch.logits)?;

        dgpu_scratch
            .residual
            .copy_from_buffer(&dgpu_scratch.t1.residual_next)?;
        self.forward_head(dgpu_scratch, &weights.global)?;

        self.set_current_cached(self.dgpu.device)?;
        let host_us = pair_start.elapsed().as_micros() as u64;
        self.dgpu.compute.synchronize()?;
        let pair_elapsed_us = pair_start.elapsed().as_micros() as u64;
        let sync_us = pair_elapsed_us.saturating_sub(host_us);
        use std::sync::atomic::Ordering;
        self.last_host_us.store(host_us, Ordering::Relaxed);
        self.last_sync_us.store(sync_us, Ordering::Relaxed);

        // ===== Perfetto device-time emit (M40-P4.1) =====
        // Route by name suffix to per-token tracks (4 dGPU + 2 iGPU).
        if let Some(exp_lock) = &self.perfetto {
            let mut exp = exp_lock.lock().unwrap();
            self.dgpu.events.for_each_pair(|name, s, e| {
                let track = if name.ends_with("_t0") {
                    // _t0 → dgpu_compute_t0 or dgpu_xfer_t0
                    if name.contains("peer_push") || name.contains(".xfer") {
                        exp.dgpu_xfer_t0.as_ref().unwrap_or(&exp.dgpu_xfer)
                    } else {
                        exp.dgpu_compute_t0.as_ref().unwrap_or(&exp.dgpu_compute)
                    }
                } else if name.ends_with("_t1") {
                    if name.contains("peer_push") || name.contains(".xfer") {
                        exp.dgpu_xfer_t1.as_ref().unwrap_or(&exp.dgpu_xfer)
                    } else {
                        exp.dgpu_compute_t1.as_ref().unwrap_or(&exp.dgpu_compute)
                    }
                } else if name.contains(".xfer") || name.contains(".peer_push") {
                    &exp.dgpu_xfer
                } else {
                    &exp.dgpu_compute
                };
                exp.emit_slice(track, name, s, e)
            })?;
            self.igpu.events.for_each_pair(|name, s, e| {
                let track = if name.contains("peer_push") || name.contains(".xfer") {
                    &exp.igpu_xfer
                } else {
                    &exp.igpu_compute
                };
                exp.emit_slice(track, name, s, e)
            })?;
            // Re-anchor for next pair.
            exp.re_anchor(
                self.dgpu.device,
                &self.dgpu.compute,
                &self.dgpu.xfer,
                self.igpu.device,
                &self.igpu.compute,
                &self.igpu.xfer,
            )?;
            if let (Some(ct0), Some(ct1), Some(xt0), Some(xt1)) = (
                self.dgpu.compute_t0.as_ref(),
                self.dgpu.compute_t1.as_ref(),
                self.dgpu.xfer_t0.as_ref(),
                self.dgpu.xfer_t1.as_ref(),
            ) {
                exp.re_anchor_pair_tracks(self.dgpu.device, ct0, ct1, xt0, xt1)?;
            }
        }

        Ok(())
    }

    /// pre_moe for ONE token — stages 1-12 of forward_layer, plus
    /// shared_expert at the end (so shared_expert overlaps with iGPU MoE
    /// instead of waiting after it). Uses caller-supplied per-token streams
    /// and TokenScratch. KvEventDir specifies whether this token records
    /// (t0) or waits (t1) on the cross-token kv state event.
    #[allow(clippy::too_many_arguments)]
    fn pair_pre_moe_one_token(
        &self,
        ts: &mut TokenScratch,
        igpu_scratch: &mut IgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        ilw: &IgpuLayerWeights,
        pos: u32,
        token_id: i32,
        events: &LayerSyncEvents,
        compute: &Stream,
        xfer: &Stream,
        kv_dir: KvEventDir<'_>,
        which: TokenId,
    ) -> eyre::Result<()> {
        let de = &self.dgpu;
        self.set_current_cached(self.dgpu.device)?;

        // M40-P4.1: coarse perfetto stage spans, routed to per-token tracks
        // by the "_t0" / "_t1" name suffix at emit time.
        let pre_span_name: &'static str = match which {
            TokenId::T0 => "dgpu.pre_moe_t0",
            TokenId::T1 => "dgpu.pre_moe_t1",
        };
        let _t_pre = de.events.stage(pre_span_name, compute)?;

        // ===== Stage 1: mhc_pre_attn =====
        {
            let _t = de.events.stage(stage_t!(which, "dgpu.mhc_pre_attn"), compute)?;
            de.rms_nw.launch(compute, &mut ts.flat, &ts.residual, 1, HC_DIM, RMS_EPS)?;
            de.f16.matvec(
                compute,
                &mut ts.mix,
                &dlw.hc_attn_fn.buffer,
                &ts.flat,
                HC_MIX_DIM,
                HC_DIM,
            )?;
            de.hc_sinkhorn.launch(
                compute,
                &mut ts.split,
                &ts.mix,
                &dlw.hc_attn_scale,
                &dlw.hc_attn_base,
                N_HC,
                SINKHORN_ITERS,
                SINKHORN_EPS,
            )?;
            de.hc_weighted.launch(
                compute,
                &mut ts.attn_cur,
                &ts.residual,
                &ts.split,
                N_EMBD,
                N_HC,
            )?;
            de.rms_w.launch_weighted(
                compute,
                &mut ts.attn_input_norm,
                &ts.attn_cur,
                &dlw.attn_norm,
                N_EMBD,
                RMS_EPS,
            )?;
        }

        // ===== Stage 2: Q chain =====
        {
            let _t = de.events.stage(stage_t!(which, "dgpu.q_chain"), compute)?;
            de.q8.quantize_input(
                compute,
                &mut ts.xq_n_embd,
                &mut ts.xscale_n_embd,
                &ts.attn_input_norm,
                N_EMBD,
            )?;
            de.q8.matvec(
                compute,
                &mut ts.qr,
                &dlw.attn_q_a.buffer,
                &ts.xq_n_embd,
                &ts.xscale_n_embd,
                N_LORA_Q,
                N_EMBD,
            )?;
            de.rms_w.launch_weighted(
                compute,
                &mut ts.qr_normed,
                &ts.qr,
                &dlw.q_a_norm,
                N_LORA_Q,
                RMS_EPS,
            )?;
            de.q8.quantize_input(
                compute,
                &mut ts.qr_xq,
                &mut ts.qr_xscale,
                &ts.qr_normed,
                N_LORA_Q,
            )?;
            de.q8.matvec(
                compute,
                &mut ts.q,
                &dlw.attn_q_b.buffer,
                &ts.qr_xq,
                &ts.qr_xscale,
                Q_FLAT,
                N_LORA_Q,
            )?;
            de.rms_nw
                .launch(compute, &mut ts.q_normed, &ts.q, N_HEAD, N_HEAD_DIM, RMS_EPS)?;
            // Stage 3: rope on q_normed (folded into q_chain span)
            de.rope.launch_forward(
                compute,
                &mut ts.q_normed,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                pos,
                &dlw.rope_params,
            )?;
        }

        // ===== Stage 4: KV chain + cache append =====
        {
            let _t = de.events.stage(stage_t!(which, "dgpu.kv_chain"), compute)?;
            de.q8.matvec(
                compute,
                &mut ts.kv_raw,
                &dlw.attn_kv.buffer,
                &ts.xq_n_embd,
                &ts.xscale_n_embd,
                N_HEAD_DIM,
                N_EMBD,
            )?;
            de.rms_w.launch_weighted(
                compute,
                &mut ts.kv_normed,
                &ts.kv_raw,
                &dlw.kv_a_norm,
                N_HEAD_DIM,
                RMS_EPS,
            )?;
            de.rope.launch_forward(
                compute,
                &mut ts.kv_normed,
                1,
                N_HEAD_DIM,
                N_ROT,
                pos,
                &dlw.rope_params,
            )?;
            de.fp8.launch(compute, &mut ts.kv_normed, N_HEAD_DIM - N_ROT)?;
            de.f16rt.launch(compute, &mut ts.kv_normed, N_HEAD_DIM)?;
            de.kv_append.launch(
                compute,
                &mut ls.kv_cache,
                &ts.kv_normed,
                pos,
                SWA_WINDOW,
                N_HEAD_DIM,
            )?;
        }
        ls.n_raw = (ls.n_raw + 1).min(SWA_WINDOW);

        // ===== Stage 5: compressor (ratio > 0) =====
        let ratio = dlw.ratio;
        if ratio > 0 {
            let _t = de.events.stage(stage_t!(which, "dgpu.compressor"), compute)?;
            let cw = dlw
                .compressor
                .as_ref()
                .ok_or_else(|| eyre!("L{}: missing compressor weights", dlw.layer_idx))?;
            let comp_width = cw.width;
            let pos_mod = pos % ratio;
            let row = if ratio == 4 { 4 + pos_mod } else { pos_mod };
            let cs = ls
                .compressor
                .as_mut()
                .ok_or_else(|| eyre!("L{}: missing compressor state", dlw.layer_idx))?;
            de.f16.matvec_pair(
                compute,
                &mut ts.kv_cur,
                &mut ts.sc_cur,
                &cw.wkv.buffer,
                &cw.wgate.buffer,
                &ts.attn_input_norm,
                comp_width,
                N_EMBD,
            )?;
            de.compressor_state_write.launch(
                compute,
                &mut cs.state_kv,
                &mut cs.state_score,
                &ts.kv_cur,
                &ts.sc_cur,
                &cw.ape.buffer,
                comp_width,
                row,
                pos_mod,
            )?;
            let comp_fires_boundary = (pos + 1) % ratio == 0;
            if comp_fires_boundary {
                de.compressor_pool.launch(
                    compute,
                    &mut ts.pooled,
                    &cs.state_kv,
                    &cs.state_score,
                    N_HEAD_DIM,
                    ratio,
                )?;
                de.rms_w.launch_weighted(
                    compute,
                    &mut ts.comp_row,
                    &ts.pooled,
                    &cw.norm,
                    N_HEAD_DIM,
                    RMS_EPS,
                )?;
                let comp_pos = pos + 1 - ratio;
                de.rope.launch_forward(
                    compute,
                    &mut ts.comp_row,
                    1,
                    N_HEAD_DIM,
                    N_ROT,
                    comp_pos,
                    &dlw.rope_params,
                )?;
                de.fp8.launch(compute, &mut ts.comp_row, N_HEAD_DIM - N_ROT)?;
                de.f16rt.launch(compute, &mut ts.comp_row, N_HEAD_DIM)?;
                if ratio == 4 {
                    de.compressor_shuffle.launch(
                        compute,
                        &mut cs.state_kv,
                        &mut cs.state_score,
                        comp_width,
                    )?;
                }
                de.comp_kv_append
                    .launch(compute, &mut cs.comp_kv, &ts.comp_row, cs.n_comp, N_HEAD_DIM)?;
                cs.n_comp += 1;
            }
        }

        // ===== Cross-token kv-state sync =====
        // t0: record event after kv_append + compressor are done.
        // t1: wait on it before our attn so kv_cache row `pos` is visible.
        match kv_dir {
            KvEventDir::Record(evt) => evt.record(compute)?,
            KvEventDir::Wait(evt) => compute.wait_event(evt)?,
        }

        // ===== Stage 6: attention =====
        {
            let _t = de.events.stage(stage_t!(which, "dgpu.attn_compute"), compute)?;
            let n_raw = ls.n_raw;
            if ratio == 0 {
                de.attn_swa.launch(
                    compute,
                    &mut ts.heads,
                    &ts.q_normed,
                    &ls.kv_cache,
                    &dlw.attn_sinks,
                    N_HEAD,
                    N_HEAD_DIM,
                    n_raw,
                )?;
            } else {
                let cs = ls.compressor.as_ref();
                let n_comp = cs.map(|c| c.n_comp).unwrap_or(0);
                let comp_kv_buf = if n_comp > 0 { cs.map(|c| &c.comp_kv) } else { None };
                de.attn_mixed.launch(
                    compute,
                    &mut ts.heads,
                    &ts.q_normed,
                    &ls.kv_cache,
                    comp_kv_buf,
                    None,
                    &dlw.attn_sinks,
                    N_HEAD,
                    N_HEAD_DIM,
                    n_raw,
                    n_comp,
                )?;
            }
        }

        // ===== Stage 7: output_proj =====
        {
            let _t = de.events.stage(stage_t!(which, "dgpu.output_proj"), compute)?;
            de.rope.launch_inverse(
                compute,
                &mut ts.heads,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                pos,
                &dlw.rope_params,
            )?;
            de.q8.quantize_input(
                compute,
                &mut ts.heads_xq,
                &mut ts.heads_xscale,
                &ts.heads,
                Q_FLAT,
            )?;
            de.q8_grouped.matvec_grouped(
                compute,
                &mut ts.low,
                &dlw.attn_output_a.buffer,
                &ts.heads_xq,
                &ts.heads_xscale,
                GROUP_DIM,
                RANK,
                N_GROUPS,
            )?;
            de.q8.quantize_input(
                compute,
                &mut ts.low_xq,
                &mut ts.low_xscale,
                &ts.low,
                OUT_LOW,
            )?;
            de.q8.matvec(
                compute,
                &mut ts.attn_out,
                &dlw.attn_output_b.buffer,
                &ts.low_xq,
                &ts.low_xscale,
                N_EMBD,
                OUT_LOW,
            )?;
        }

        // ===== Stage 8: mhc_post_attn =====
        {
            let _t = de.events.stage(stage_t!(which, "dgpu.mhc_post_attn"), compute)?;
            de.hc_post.launch_from_split(
                compute,
                &mut ts.after_attn_hc,
                &ts.attn_out,
                &ts.residual,
                &ts.split,
                N_HC,
                N_EMBD,
                N_HC,
            )?;
        }

        // ===== Stage 9: mhc_pre_ffn =====
        {
            let _t = de.events.stage(stage_t!(which, "dgpu.mhc_pre_ffn"), compute)?;
            de.rms_nw.launch(compute, &mut ts.flat, &ts.after_attn_hc, 1, HC_DIM, RMS_EPS)?;
            de.f16.matvec(
                compute,
                &mut ts.mix,
                &dlw.hc_ffn_fn.buffer,
                &ts.flat,
                HC_MIX_DIM,
                HC_DIM,
            )?;
            de.hc_sinkhorn.launch(
                compute,
                &mut ts.split,
                &ts.mix,
                &dlw.hc_ffn_scale,
                &dlw.hc_ffn_base,
                N_HC,
                SINKHORN_ITERS,
                SINKHORN_EPS,
            )?;
            de.hc_weighted.launch(
                compute,
                &mut ts.ffn_cur,
                &ts.after_attn_hc,
                &ts.split,
                N_EMBD,
                N_HC,
            )?;
            de.rms_w.launch_weighted(
                compute,
                &mut ts.ffn_input_norm,
                &ts.ffn_cur,
                &dlw.ffn_norm,
                N_EMBD,
                RMS_EPS,
            )?;
        }

        // ===== Stage 11: router on dGPU =====
        let _t_router = de.events.stage(stage_t!(which, "dgpu.router"), compute)?;
        if dlw.is_hash_router {
            de.f16.matvec(
                compute,
                &mut ts.router_logits,
                &dlw.ffn_gate_inp.buffer,
                &ts.ffn_input_norm,
                N_EXPERT,
                N_EMBD,
            )?;
            compute.synchronize()?;
            ts.router_logits.copy_to_host(&mut ts.router_logits_host)?;
            let tid2eid = dlw
                .tid2eid
                .as_ref()
                .ok_or_else(|| eyre!("hash router missing tid2eid"))?;
            let (sel, w) = hash_router_select(tid2eid, token_id, &ts.router_logits_host);
            ts.d_selected.copy_from_host(&sel)?;
            ts.d_ew.copy_from_host(&w)?;
        } else {
            de.f16.matvec(
                compute,
                &mut ts.router_logits,
                &dlw.ffn_gate_inp.buffer,
                &ts.ffn_input_norm,
                N_EXPERT,
                N_EMBD,
            )?;
            de.router_topk.launch(
                compute,
                &mut ts.d_selected,
                &mut ts.d_ew,
                &ts.router_logits,
                dlw.router_bias_dev.as_ref(),
                N_EXPERT,
                N_EXPERT_USED as u32,
                EXPERT_WEIGHT_SCALE,
                ROUTER_WEIGHT_EPS,
            )?;
        }
        drop(_t_router);

        // ===== Stages 10 + 12: peer-push ffn_input_norm + selected + d_ew to iGPU =====
        // RECORD selected_ready RIGHT AFTER router and queue the pushes BEFORE
        // shared_expert. Otherwise the pushes (and hence iGPU MoE start) get
        // pushed out behind shared_expert on de.compute_t0's FIFO — defeating
        // the whole point of overlapping shared_expert with iGPU MoE.
        events.selected_ready.record(compute)?;
        xfer.wait_event(&events.selected_ready)?;
        let push_span_name: &'static str = match which {
            TokenId::T0 => "dgpu.peer_push_t0",
            TokenId::T1 => "dgpu.peer_push_t1",
        };
        let _t_push = de.events.stage(push_span_name, xfer)?;
        // Per-token iGPU recv buffers (igpu_scratch).
        let (ig_ffn_in, ig_sel, ig_ew) = match which {
            TokenId::T0 => (
                &mut igpu_scratch.ffn_input_norm_recv_t0,
                &mut igpu_scratch.d_selected_t0,
                &mut igpu_scratch.d_ew_t0,
            ),
            TokenId::T1 => (
                &mut igpu_scratch.ffn_input_norm_recv_t1,
                &mut igpu_scratch.d_selected_t1,
                &mut igpu_scratch.d_ew_t1,
            ),
        };
        peer_push_f32(&ts.ffn_input_norm, ig_ffn_in, xfer)?;
        peer_push_i32(&ts.d_selected, ig_sel, xfer)?;
        peer_push_f32(&ts.d_ew, ig_ew, xfer)?;
        events.selected_pushed.record(xfer)?;
        drop(_t_push);

        // ===== Stage 13: shared_expert on dGPU — runs in parallel with iGPU MoE =====
        // Queued AFTER the peer push events were already recorded, so iGPU
        // MoE doesn't wait behind shared_expert. shared_expert reads
        // ts.ffn_input_norm (still valid — push above also reads it but
        // de.xfer is a separate stream and copies don't mutate the source)
        // and writes ts.ffn_shared (consumed by post_moe).
        let _t_shared = de.events.stage(stage_t!(which, "dgpu.shared_expert"), compute)?;
        de.q8.quantize_input(
            compute,
            &mut ts.xq_n_embd,
            &mut ts.xscale_n_embd,
            &ts.ffn_input_norm,
            N_EMBD,
        )?;
        de.q8.matvec(
            compute,
            &mut ts.gate_sh,
            &dlw.shared.gate.buffer,
            &ts.xq_n_embd,
            &ts.xscale_n_embd,
            N_FF_SHARED,
            N_EMBD,
        )?;
        de.q8.matvec(
            compute,
            &mut ts.up_sh,
            &dlw.shared.up.buffer,
            &ts.xq_n_embd,
            &ts.xscale_n_embd,
            N_FF_SHARED,
            N_EMBD,
        )?;
        de.swiglu
            .launch(compute, &mut ts.mid_sh, &ts.gate_sh, &ts.up_sh, N_FF_SHARED)?;
        de.q8.quantize_input(
            compute,
            &mut ts.mid_sh_xq,
            &mut ts.mid_sh_xscale,
            &ts.mid_sh,
            N_FF_SHARED,
        )?;
        de.q8.matvec(
            compute,
            &mut ts.ffn_shared,
            &dlw.shared.down.buffer,
            &ts.mid_sh_xq,
            &ts.mid_sh_xscale,
            N_EMBD,
            N_FF_SHARED,
        )?;
        drop(_t_shared);

        drop(_t_pre);

        Ok(())
    }

    /// Queue this token's iGPU MoE on ie.compute (gated on selected_pushed)
    /// + push ffn_moe back on ie.xfer (gated on moe_done) → ts.ffn_moe_recv,
    /// + record moe_arrived. All event-driven; host never blocks.
    fn pair_igpu_moe_one_token(
        &self,
        ts: &mut TokenScratch,
        igpu_scratch: &mut IgpuScratch,
        ilw: &IgpuLayerWeights,
        events: &LayerSyncEvents,
        which: TokenId,
    ) -> eyre::Result<()> {
        self.set_current_cached(self.igpu.device)?;
        let ie = &self.igpu;
        ie.compute.wait_event(&events.selected_pushed)?;

        let moe_span_name: &'static str = match which {
            TokenId::T0 => "igpu.moe_t0",
            TokenId::T1 => "igpu.moe_t1",
        };
        let _t_moe = ie.events.stage(moe_span_name, &ie.compute)?;

        let gbpe = ilw.routed.gate_bytes_per_expert;
        let ubpe = ilw.routed.up_bytes_per_expert;
        let dbpe = ilw.routed.down_bytes_per_expert;
        let mid_blocks_bytes = (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8_K_BYTES;

        let (ig_ffn_in, ig_sel, ig_ew, ig_ffn_moe) = match which {
            TokenId::T0 => (
                &igpu_scratch.ffn_input_norm_recv_t0,
                &igpu_scratch.d_selected_t0,
                &igpu_scratch.d_ew_t0,
                &mut igpu_scratch.ffn_moe_t0,
            ),
            TokenId::T1 => (
                &igpu_scratch.ffn_input_norm_recv_t1,
                &igpu_scratch.d_selected_t1,
                &igpu_scratch.d_ew_t1,
                &mut igpu_scratch.ffn_moe_t1,
            ),
        };

        ie.q8k
            .launch(&ie.compute, &mut igpu_scratch.d_xq_q8k, ig_ffn_in, BLOCKS_Q8K_GATE_IN)?;
        ie.iq2.launch_fused_swiglu_batch(
            &ie.compute,
            &mut igpu_scratch.d_mid_cat,
            &ilw.routed.gate.buffer,
            &ilw.routed.up.buffer,
            &igpu_scratch.d_xq_q8k,
            ig_ew,
            ig_sel,
            gbpe as u32,
            ubpe as u32,
            N_EXPERT_USED as u32,
            SWIGLU_CLAMP_EXP,
            N_FF_EXP,
            BLOCKS_Q8K_GATE_IN,
        )?;
        ie.q8k.launch(
            &ie.compute,
            &mut igpu_scratch.d_midq_cat,
            &igpu_scratch.d_mid_cat,
            BLOCKS_Q8K_DOWN_IN * (N_EXPERT_USED as u32),
        )?;
        ie.q2k.launch_batched(
            &ie.compute,
            ig_ffn_moe,
            &ilw.routed.down.buffer,
            &igpu_scratch.d_midq_cat,
            ig_sel,
            dbpe as u32,
            mid_blocks_bytes as u32,
            N_EXPERT_USED as u32,
            N_EMBD,
            BLOCKS_Q8K_DOWN_IN,
        )?;

        events.moe_done.record(&ie.compute)?;
        drop(_t_moe);
        let xfer_back_name: &'static str = match which {
            TokenId::T0 => "igpu.peer_push_back_t0",
            TokenId::T1 => "igpu.peer_push_back_t1",
        };
        let _t_back = ie.events.stage(xfer_back_name, &ie.xfer)?;
        ie.xfer.wait_event(&events.moe_done)?;
        peer_push_f32(ig_ffn_moe, &mut ts.ffn_moe_recv, &ie.xfer)?;
        events.moe_arrived.record(&ie.xfer)?;
        drop(_t_back);

        // Switch back to dGPU device so caller's subsequent dGPU work is on correct device.
        self.set_current_cached(self.dgpu.device)?;
        Ok(())
    }

    /// post_moe for ONE token: cross-stream wait on moe_arrived, then
    /// vec_add(ts.ffn_moe_recv, ts.ffn_shared) + hc_post → ts.residual_next.
    /// Tiny — most of the dGPU FFN work (shared_expert) was already done
    /// in pre_moe so we don't have to wait BEHIND it before vec_add.
    fn pair_post_moe_one_token(
        &self,
        ts: &mut TokenScratch,
        _dlw: &DgpuLayerWeights,
        events: &LayerSyncEvents,
        compute: &Stream,
    ) -> eyre::Result<()> {
        let de = &self.dgpu;
        // Span name picked by stream identity. We can't easily tell t0 vs t1
        // here without an extra param; route by stream pointer at emit time.
        // For simplicity, label by a wrapping span passed via debug_span.
        // Pick t0 if compute pointer matches dgpu.compute_t0.
        let is_t0 = std::ptr::eq(
            compute,
            self.dgpu
                .compute_t0
                .as_ref()
                .expect("compute_t0 missing"),
        );
        let (wait_name, combine_name): (&'static str, &'static str) = if is_t0 {
            ("dgpu.ffn_combine.wait_t0", "dgpu.ffn_combine_t0")
        } else {
            ("dgpu.ffn_combine.wait_t1", "dgpu.ffn_combine_t1")
        };
        // SEPARATE span around the wait so the trace shows exactly how long
        // dGPU.compute_tN sat idle waiting for moe_arrived (= the iGPU pipeline
        // for this layer's MoE). Then a separate span for the actual ffn_combine
        // kernels (vec_add + hc_post).
        {
            let _t_wait = de.events.stage(wait_name, compute)?;
            compute.wait_event(&events.moe_arrived)?;
        }
        {
            let _t_combine = de.events.stage(combine_name, compute)?;
            de.vec_add.launch(compute, &mut ts.ffn_moe_recv, &ts.ffn_shared, N_EMBD)?;
            de.hc_post.launch_from_split(
                compute,
                &mut ts.residual_next,
                &ts.ffn_moe_recv,
                &ts.after_attn_hc,
                &ts.split,
                N_HC,
                N_EMBD,
                N_HC,
            )?;
        }
        Ok(())
    }
}

// Silence unused-import lints if the file gets edited to remove some uses.
#[allow(dead_code)]
fn _silence_imports() {
    let _: Option<&DeviceBuffer<f32>> = None;
}
