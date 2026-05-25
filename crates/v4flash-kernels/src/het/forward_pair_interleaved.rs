//! M40-P4.5: pair forward with batched 2-wide kernels.
//!
//! Both tokens flow through ONE dGPU compute stream. Every big matvec is
//! a 2-wide pair kernel — loads W once, computes both columns. Small ops
//! (rms, sinkhorn, hc_weighted, rope, quantize, attn, vec_add, hc_post)
//! still run per-token; they're cheap and would only marginally benefit
//! from batching.
//!
//! Cross-layer pipelining: queue L+1's pre_moe immediately after L's
//! post_moe ffn_combine on the same de.compute stream — when L's post_moe
//! is blocked on wait_event(moe_arrived_L), L+1 pre_moe is already queued
//! ready to run. Meanwhile iGPU is busy with L's MoEs (and gets L+1's
//! MoEs queued the moment L+1's pushes complete).
//!
//! iGPU is per-token: ie.compute serializes MoE_t0 → MoE_t1 FIFO (one
//! pipeline). Per-token recv buffers (ffn_input_norm_recv_tN, d_selected_tN,
//! d_ew_tN, ffn_moe_tN) keep the two tokens' MoE inputs/outputs separate.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::Event;

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

impl HeterogeneousEngine {
    /// M40-P4.5: pair forward with batched 2-wide kernels.
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

        let de = &self.dgpu;
        // M40-P4.7: dropped per-token xfer streams. All dGPU work
        // (including peer pushes) goes on de.compute. See pair_pre_moe_batched
        // for rationale.

        // ===== Initial residual upload =====
        dgpu_scratch.t0.residual.copy_from_host(input_hc_0)?;
        dgpu_scratch.t1.residual.copy_from_host(input_hc_1)?;

        let pair_start = std::time::Instant::now();

        for layer in 0..N_LAYER as usize {
            let dlw = &weights.dgpu_layers[layer];
            let ilw = &weights.igpu_layers[layer];
            let ls = &mut state.layers[layer];
            let evt_t0 = &self.sync_events.layers[layer];
            let evt_t1 = &self.sync_events_t1.layers[layer];
            let kv_evt = &self.pair_t0_state_ready[layer];

            self.pair_pre_moe_batched(
                &mut dgpu_scratch.t0,
                &mut dgpu_scratch.t1,
                igpu_scratch,
                ls,
                dlw,
                ilw,
                pos,
                token_id_0,
                token_id_1,
                layer as u32,
                evt_t0,
                evt_t1,
                kv_evt,
            )?;

            // post_moe for both tokens (sequential on de.compute, each waits
            // for its own moe_arrived).
            self.pair_post_moe_batched(
                &mut dgpu_scratch.t0,
                &mut dgpu_scratch.t1,
                dlw,
                layer as u32,
                evt_t0,
                evt_t1,
            )?;

            // residual ← residual_next for both tokens (async on de.compute,
            // FIFO ordered after post_moe writes).
            dgpu_scratch
                .t0
                .residual
                .copy_from_buffer_async(&dgpu_scratch.t0.residual_next, &de.compute)?;
            dgpu_scratch
                .t1
                .residual
                .copy_from_buffer_async(&dgpu_scratch.t1.residual_next, &de.compute)?;
        }

        // ===== HEAD x2 =====
        de.compute.synchronize()?;
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
        de.compute.synchronize()?;
        let pair_elapsed_us = pair_start.elapsed().as_micros() as u64;
        let sync_us = pair_elapsed_us.saturating_sub(host_us);
        use std::sync::atomic::Ordering;
        self.last_host_us.store(host_us, Ordering::Relaxed);
        self.last_sync_us.store(sync_us, Ordering::Relaxed);

        // ===== Perfetto device-time emit =====
        if let Some(exp_lock) = &self.perfetto {
            let mut exp = exp_lock.lock().unwrap();
            self.dgpu.events.for_each_pair(|name, s, e| {
                let track = if name.ends_with("_t0") {
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

    /// pair_pre_moe_batched — runs stages 1-13 for BOTH tokens through ONE
    /// dGPU compute stream using 2-wide pair kernels for big matvecs.
    /// Small ops (rms, sinkhorn, hc_weighted, rope, fp8, kv_append, attn,
    /// compressor) run per-token, sequentially on the same stream.
    ///
    /// M40-P4.8: capturable blocks (mhc_pre_attn, q_chain, kv_chain,
    /// output_proj, mhc_post_attn, mhc_pre_ffn, router, shared_expert)
    /// captured into per-layer HIP graphs. Per-token rope/kv_append/attn
    /// kernels stay outside the graphs (per-token `pos` / `n_raw` params).
    #[allow(clippy::too_many_arguments)]
    fn pair_pre_moe_batched(
        &self,
        t0: &mut TokenScratch,
        t1: &mut TokenScratch,
        igpu_scratch: &mut IgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        ilw: &IgpuLayerWeights,
        pos: u32,
        token_id_0: i32,
        token_id_1: i32,
        layer: u32,
        evt_t0: &LayerSyncEvents,
        evt_t1: &LayerSyncEvents,
        _kv_evt: &Event,
    ) -> eyre::Result<()> {
        let de = &self.dgpu;
        let compute = &de.compute;
        self.set_current_cached(self.dgpu.device)?;
        let _t_pre = de.events.stage("dgpu.pair_pre_moe", compute)?;

        // ===== Stage 1: mhc_pre_attn — CAPTURED graph per layer =====
        {
            let _t = de.events.stage("dgpu.mhc_pre_attn_pair", compute)?;
            let graph_slot = &self.dgpu_pair_mhc_pre_attn_graphs[layer as usize];
            let mut guard = graph_slot.lock().unwrap();
            if guard.is_none() {
                compute.begin_capture(
                    v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )?;
                de.rms_nw.launch(compute, &mut t0.flat, &t0.residual, 1, HC_DIM, RMS_EPS)?;
                de.rms_nw.launch(compute, &mut t1.flat, &t1.residual, 1, HC_DIM, RMS_EPS)?;
                de.f16.matvec_two_inputs(
                    compute,
                    &mut t0.mix,
                    &mut t1.mix,
                    &dlw.hc_attn_fn.buffer,
                    &t0.flat,
                    &t1.flat,
                    HC_MIX_DIM,
                    HC_DIM,
                )?;
                de.hc_sinkhorn.launch(
                    compute,
                    &mut t0.split,
                    &t0.mix,
                    &dlw.hc_attn_scale,
                    &dlw.hc_attn_base,
                    N_HC,
                    SINKHORN_ITERS,
                    SINKHORN_EPS,
                )?;
                de.hc_sinkhorn.launch(
                    compute,
                    &mut t1.split,
                    &t1.mix,
                    &dlw.hc_attn_scale,
                    &dlw.hc_attn_base,
                    N_HC,
                    SINKHORN_ITERS,
                    SINKHORN_EPS,
                )?;
                de.hc_weighted.launch(
                    compute,
                    &mut t0.attn_cur,
                    &t0.residual,
                    &t0.split,
                    N_EMBD,
                    N_HC,
                )?;
                de.hc_weighted.launch(
                    compute,
                    &mut t1.attn_cur,
                    &t1.residual,
                    &t1.split,
                    N_EMBD,
                    N_HC,
                )?;
                de.rms_w.launch_weighted(
                    compute,
                    &mut t0.attn_input_norm,
                    &t0.attn_cur,
                    &dlw.attn_norm,
                    N_EMBD,
                    RMS_EPS,
                )?;
                de.rms_w.launch_weighted(
                    compute,
                    &mut t1.attn_input_norm,
                    &t1.attn_cur,
                    &dlw.attn_norm,
                    N_EMBD,
                    RMS_EPS,
                )?;
                let graph = compute.end_capture()?;
                let exec = graph.instantiate()?;
                exec.launch(compute)?;
                *guard = Some(exec);
            } else {
                guard.as_ref().unwrap().launch(compute)?;
            }
        }

        // ===== Stage 2: Q chain (batched) =====
        // Stage 2: Q chain — pre-rope CAPTURED (rope outside, per-token pos)
        {
            let _t = de.events.stage("dgpu.q_chain_pair", compute)?;
            let graph_slot = &self.dgpu_pair_q_chain_graphs[layer as usize];
            let mut guard = graph_slot.lock().unwrap();
            if guard.is_none() {
                compute.begin_capture(
                    v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )?;
                de.q8.quantize_input(
                    compute,
                    &mut t0.xq_n_embd,
                    &mut t0.xscale_n_embd,
                    &t0.attn_input_norm,
                    N_EMBD,
                )?;
                de.q8.quantize_input(
                    compute,
                    &mut t1.xq_n_embd,
                    &mut t1.xscale_n_embd,
                    &t1.attn_input_norm,
                    N_EMBD,
                )?;
                de.q8.matvec_pair(
                    compute,
                    &mut t0.qr,
                    &mut t1.qr,
                    &dlw.attn_q_a.buffer,
                    &t0.xq_n_embd,
                    &t1.xq_n_embd,
                    &t0.xscale_n_embd,
                    &t1.xscale_n_embd,
                    N_LORA_Q,
                    N_EMBD,
                )?;
                de.rms_w.launch_weighted(
                    compute,
                    &mut t0.qr_normed,
                    &t0.qr,
                    &dlw.q_a_norm,
                    N_LORA_Q,
                    RMS_EPS,
                )?;
                de.rms_w.launch_weighted(
                    compute,
                    &mut t1.qr_normed,
                    &t1.qr,
                    &dlw.q_a_norm,
                    N_LORA_Q,
                    RMS_EPS,
                )?;
                de.q8.quantize_input(
                    compute,
                    &mut t0.qr_xq,
                    &mut t0.qr_xscale,
                    &t0.qr_normed,
                    N_LORA_Q,
                )?;
                de.q8.quantize_input(
                    compute,
                    &mut t1.qr_xq,
                    &mut t1.qr_xscale,
                    &t1.qr_normed,
                    N_LORA_Q,
                )?;
                de.q8.matvec_pair(
                    compute,
                    &mut t0.q,
                    &mut t1.q,
                    &dlw.attn_q_b.buffer,
                    &t0.qr_xq,
                    &t1.qr_xq,
                    &t0.qr_xscale,
                    &t1.qr_xscale,
                    Q_FLAT,
                    N_LORA_Q,
                )?;
                de.rms_nw
                    .launch(compute, &mut t0.q_normed, &t0.q, N_HEAD, N_HEAD_DIM, RMS_EPS)?;
                de.rms_nw
                    .launch(compute, &mut t1.q_normed, &t1.q, N_HEAD, N_HEAD_DIM, RMS_EPS)?;
                let graph = compute.end_capture()?;
                let exec = graph.instantiate()?;
                exec.launch(compute)?;
                *guard = Some(exec);
            } else {
                guard.as_ref().unwrap().launch(compute)?;
            }
            // rope per-token (different pos!) — stays outside the captured graph.
            de.rope.launch_forward(
                compute,
                &mut t0.q_normed,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                pos,
                &dlw.rope_params,
            )?;
            de.rope.launch_forward(
                compute,
                &mut t1.q_normed,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                pos + 1,
                &dlw.rope_params,
            )?;
        }

        // Stage 4: KV chain — matvec+rms CAPTURED; rope+fp8+f16rt+kv_append
        // stay outside (per-token pos, and kv_append writes are tiny anyway).
        {
            let _t = de.events.stage("dgpu.kv_chain_pair", compute)?;
            let graph_slot = &self.dgpu_pair_kv_chain_graphs[layer as usize];
            let mut guard = graph_slot.lock().unwrap();
            if guard.is_none() {
                compute.begin_capture(
                    v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )?;
                de.q8.matvec_pair(
                    compute,
                    &mut t0.kv_raw,
                    &mut t1.kv_raw,
                    &dlw.attn_kv.buffer,
                    &t0.xq_n_embd,
                    &t1.xq_n_embd,
                    &t0.xscale_n_embd,
                    &t1.xscale_n_embd,
                    N_HEAD_DIM,
                    N_EMBD,
                )?;
                de.rms_w.launch_weighted(
                    compute,
                    &mut t0.kv_normed,
                    &t0.kv_raw,
                    &dlw.kv_a_norm,
                    N_HEAD_DIM,
                    RMS_EPS,
                )?;
                de.rms_w.launch_weighted(
                    compute,
                    &mut t1.kv_normed,
                    &t1.kv_raw,
                    &dlw.kv_a_norm,
                    N_HEAD_DIM,
                    RMS_EPS,
                )?;
                let graph = compute.end_capture()?;
                let exec = graph.instantiate()?;
                exec.launch(compute)?;
                *guard = Some(exec);
            } else {
                guard.as_ref().unwrap().launch(compute)?;
            }
            // rope/fp8/f16rt/kv_append: outside graph (per-token pos).
            de.rope.launch_forward(
                compute,
                &mut t0.kv_normed,
                1,
                N_HEAD_DIM,
                N_ROT,
                pos,
                &dlw.rope_params,
            )?;
            de.rope.launch_forward(
                compute,
                &mut t1.kv_normed,
                1,
                N_HEAD_DIM,
                N_ROT,
                pos + 1,
                &dlw.rope_params,
            )?;
            de.fp8.launch(compute, &mut t0.kv_normed, N_HEAD_DIM - N_ROT)?;
            de.fp8.launch(compute, &mut t1.kv_normed, N_HEAD_DIM - N_ROT)?;
            de.f16rt.launch(compute, &mut t0.kv_normed, N_HEAD_DIM)?;
            de.f16rt.launch(compute, &mut t1.kv_normed, N_HEAD_DIM)?;
            de.kv_append.launch(
                compute,
                &mut ls.kv_cache,
                &t0.kv_normed,
                pos,
                SWA_WINDOW,
                N_HEAD_DIM,
            )?;
            de.kv_append.launch(
                compute,
                &mut ls.kv_cache,
                &t1.kv_normed,
                pos + 1,
                SWA_WINDOW,
                N_HEAD_DIM,
            )?;
        }
        // n_raw advances by 2 (one per token).
        ls.n_raw = (ls.n_raw + 2).min(SWA_WINDOW);

        // ===== Stage 5: compressor (ratio > 0) =====
        let ratio = dlw.ratio;
        if ratio > 0 {
            let _t = de.events.stage("dgpu.compressor_pair", compute)?;
            let cw = dlw
                .compressor
                .as_ref()
                .ok_or_else(|| eyre!("L{}: missing compressor weights", dlw.layer_idx))?;
            let comp_width = cw.width;
            let cs = ls
                .compressor
                .as_mut()
                .ok_or_else(|| eyre!("L{}: missing compressor state", dlw.layer_idx))?;
            // Per-token compressor matvec_pair / state_write. The existing
            // f16.matvec_pair (1 input, 2 weights) computes kv+gate from one
            // attn_input_norm — not the pattern we need here (2 inputs, 1
            // weight). For now just run the compressor matvec_pair per token,
            // accepting the redundant W reads (compressor weight is tiny:
            // comp_width=1024, F16 → ~8 MB; per layer per pair 16 MB total).
            //
            // Unrolled t0 then t1 because borrow rules with `cs` (long-lived
            // &mut) preclude a clean iter over both TokenScratches in one go.
            // --- t0 ---
            {
                let p = pos;
                let pos_mod = p % ratio;
                let row = if ratio == 4 { 4 + pos_mod } else { pos_mod };
                de.f16.matvec_pair(
                    compute,
                    &mut t0.kv_cur,
                    &mut t0.sc_cur,
                    &cw.wkv.buffer,
                    &cw.wgate.buffer,
                    &t0.attn_input_norm,
                    comp_width,
                    N_EMBD,
                )?;
                de.compressor_state_write.launch(
                    compute,
                    &mut cs.state_kv,
                    &mut cs.state_score,
                    &t0.kv_cur,
                    &t0.sc_cur,
                    &cw.ape.buffer,
                    comp_width,
                    row,
                    pos_mod,
                )?;
                if (p + 1) % ratio == 0 {
                    de.compressor_pool.launch(
                        compute,
                        &mut t0.pooled,
                        &cs.state_kv,
                        &cs.state_score,
                        N_HEAD_DIM,
                        ratio,
                    )?;
                    de.rms_w.launch_weighted(
                        compute,
                        &mut t0.comp_row,
                        &t0.pooled,
                        &cw.norm,
                        N_HEAD_DIM,
                        RMS_EPS,
                    )?;
                    let comp_pos = p + 1 - ratio;
                    de.rope.launch_forward(
                        compute,
                        &mut t0.comp_row,
                        1,
                        N_HEAD_DIM,
                        N_ROT,
                        comp_pos,
                        &dlw.rope_params,
                    )?;
                    de.fp8.launch(compute, &mut t0.comp_row, N_HEAD_DIM - N_ROT)?;
                    de.f16rt.launch(compute, &mut t0.comp_row, N_HEAD_DIM)?;
                    if ratio == 4 {
                        de.compressor_shuffle.launch(
                            compute,
                            &mut cs.state_kv,
                            &mut cs.state_score,
                            comp_width,
                        )?;
                    }
                    de.comp_kv_append.launch(
                        compute,
                        &mut cs.comp_kv,
                        &t0.comp_row,
                        cs.n_comp,
                        N_HEAD_DIM,
                    )?;
                    cs.n_comp += 1;
                }
            }
            // --- t1 ---
            {
                let p = pos + 1;
                let pos_mod = p % ratio;
                let row = if ratio == 4 { 4 + pos_mod } else { pos_mod };
                de.f16.matvec_pair(
                    compute,
                    &mut t1.kv_cur,
                    &mut t1.sc_cur,
                    &cw.wkv.buffer,
                    &cw.wgate.buffer,
                    &t1.attn_input_norm,
                    comp_width,
                    N_EMBD,
                )?;
                de.compressor_state_write.launch(
                    compute,
                    &mut cs.state_kv,
                    &mut cs.state_score,
                    &t1.kv_cur,
                    &t1.sc_cur,
                    &cw.ape.buffer,
                    comp_width,
                    row,
                    pos_mod,
                )?;
                if (p + 1) % ratio == 0 {
                    de.compressor_pool.launch(
                        compute,
                        &mut t1.pooled,
                        &cs.state_kv,
                        &cs.state_score,
                        N_HEAD_DIM,
                        ratio,
                    )?;
                    de.rms_w.launch_weighted(
                        compute,
                        &mut t1.comp_row,
                        &t1.pooled,
                        &cw.norm,
                        N_HEAD_DIM,
                        RMS_EPS,
                    )?;
                    let comp_pos = p + 1 - ratio;
                    de.rope.launch_forward(
                        compute,
                        &mut t1.comp_row,
                        1,
                        N_HEAD_DIM,
                        N_ROT,
                        comp_pos,
                        &dlw.rope_params,
                    )?;
                    de.fp8.launch(compute, &mut t1.comp_row, N_HEAD_DIM - N_ROT)?;
                    de.f16rt.launch(compute, &mut t1.comp_row, N_HEAD_DIM)?;
                    if ratio == 4 {
                        de.compressor_shuffle.launch(
                            compute,
                            &mut cs.state_kv,
                            &mut cs.state_score,
                            comp_width,
                        )?;
                    }
                    de.comp_kv_append.launch(
                        compute,
                        &mut cs.comp_kv,
                        &t1.comp_row,
                        cs.n_comp,
                        N_HEAD_DIM,
                    )?;
                    cs.n_comp += 1;
                }
            }
        }

        // ===== Stage 6: attention (per token, sequential — kv_cache shared) =====
        {
            let _t = de.events.stage("dgpu.attn_compute_pair", compute)?;
            let n_raw_full = ls.n_raw;
            // t0 sees rows [0..n_raw_full-1]; t1 sees [0..n_raw_full].
            let n_raw_t0 = n_raw_full - 1;
            let n_raw_t1 = n_raw_full;
            if ratio == 0 {
                de.attn_swa.launch(
                    compute,
                    &mut t0.heads,
                    &t0.q_normed,
                    &ls.kv_cache,
                    &dlw.attn_sinks,
                    N_HEAD,
                    N_HEAD_DIM,
                    n_raw_t0,
                )?;
                de.attn_swa.launch(
                    compute,
                    &mut t1.heads,
                    &t1.q_normed,
                    &ls.kv_cache,
                    &dlw.attn_sinks,
                    N_HEAD,
                    N_HEAD_DIM,
                    n_raw_t1,
                )?;
            } else {
                let cs = ls.compressor.as_ref();
                let n_comp = cs.map(|c| c.n_comp).unwrap_or(0);
                let comp_kv_buf = if n_comp > 0 { cs.map(|c| &c.comp_kv) } else { None };
                de.attn_mixed.launch(
                    compute,
                    &mut t0.heads,
                    &t0.q_normed,
                    &ls.kv_cache,
                    comp_kv_buf,
                    None,
                    &dlw.attn_sinks,
                    N_HEAD,
                    N_HEAD_DIM,
                    n_raw_t0,
                    n_comp,
                )?;
                de.attn_mixed.launch(
                    compute,
                    &mut t1.heads,
                    &t1.q_normed,
                    &ls.kv_cache,
                    comp_kv_buf,
                    None,
                    &dlw.attn_sinks,
                    N_HEAD,
                    N_HEAD_DIM,
                    n_raw_t1,
                    n_comp,
                )?;
            }
        }

        // ===== Stage 7: output_proj (batched) =====
        {
            let _t = de.events.stage("dgpu.output_proj_pair", compute)?;
            // rope_inverse per-token (different pos) — OUTSIDE the captured graph.
            de.rope.launch_inverse(
                compute,
                &mut t0.heads,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                pos,
                &dlw.rope_params,
            )?;
            de.rope.launch_inverse(
                compute,
                &mut t1.heads,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                pos + 1,
                &dlw.rope_params,
            )?;
            // Post-rope output_proj — CAPTURED.
            let graph_slot = &self.dgpu_pair_output_proj_graphs[layer as usize];
            let mut guard = graph_slot.lock().unwrap();
            if guard.is_none() {
                compute.begin_capture(
                    v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )?;
                de.q8.quantize_input(
                    compute,
                    &mut t0.heads_xq,
                    &mut t0.heads_xscale,
                    &t0.heads,
                    Q_FLAT,
                )?;
                de.q8.quantize_input(
                    compute,
                    &mut t1.heads_xq,
                    &mut t1.heads_xscale,
                    &t1.heads,
                    Q_FLAT,
                )?;
                de.q8_grouped.matvec_grouped_pair(
                    compute,
                    &mut t0.low,
                    &mut t1.low,
                    &dlw.attn_output_a.buffer,
                    &t0.heads_xq,
                    &t1.heads_xq,
                    &t0.heads_xscale,
                    &t1.heads_xscale,
                    GROUP_DIM,
                    RANK,
                    N_GROUPS,
                )?;
                de.q8.quantize_input(
                    compute,
                    &mut t0.low_xq,
                    &mut t0.low_xscale,
                    &t0.low,
                    OUT_LOW,
                )?;
                de.q8.quantize_input(
                    compute,
                    &mut t1.low_xq,
                    &mut t1.low_xscale,
                    &t1.low,
                    OUT_LOW,
                )?;
                de.q8.matvec_pair(
                    compute,
                    &mut t0.attn_out,
                    &mut t1.attn_out,
                    &dlw.attn_output_b.buffer,
                    &t0.low_xq,
                    &t1.low_xq,
                    &t0.low_xscale,
                    &t1.low_xscale,
                    N_EMBD,
                    OUT_LOW,
                )?;
                let graph = compute.end_capture()?;
                let exec = graph.instantiate()?;
                exec.launch(compute)?;
                *guard = Some(exec);
            } else {
                guard.as_ref().unwrap().launch(compute)?;
            }
        }

        // Stage 8: mhc_post_attn — CAPTURED (2 hc_post calls).
        {
            let _t = de.events.stage("dgpu.mhc_post_attn_pair", compute)?;
            let graph_slot = &self.dgpu_pair_mhc_post_attn_graphs[layer as usize];
            let mut guard = graph_slot.lock().unwrap();
            if guard.is_none() {
                compute.begin_capture(
                    v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )?;
                de.hc_post.launch_from_split(
                    compute,
                    &mut t0.after_attn_hc,
                    &t0.attn_out,
                    &t0.residual,
                    &t0.split,
                    N_HC,
                    N_EMBD,
                    N_HC,
                )?;
                de.hc_post.launch_from_split(
                    compute,
                    &mut t1.after_attn_hc,
                    &t1.attn_out,
                    &t1.residual,
                    &t1.split,
                    N_HC,
                    N_EMBD,
                    N_HC,
                )?;
                let graph = compute.end_capture()?;
                let exec = graph.instantiate()?;
                exec.launch(compute)?;
                *guard = Some(exec);
            } else {
                guard.as_ref().unwrap().launch(compute)?;
            }
        }

        // Stage 9: mhc_pre_ffn — CAPTURED.
        {
            let _t = de.events.stage("dgpu.mhc_pre_ffn_pair", compute)?;
            let graph_slot = &self.dgpu_pair_mhc_pre_ffn_graphs[layer as usize];
            let mut guard = graph_slot.lock().unwrap();
            if guard.is_none() {
                compute.begin_capture(
                    v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )?;
                de.rms_nw.launch(
                    compute,
                    &mut t0.flat,
                    &t0.after_attn_hc,
                    1,
                    HC_DIM,
                    RMS_EPS,
                )?;
                de.rms_nw.launch(
                    compute,
                    &mut t1.flat,
                    &t1.after_attn_hc,
                    1,
                    HC_DIM,
                    RMS_EPS,
                )?;
                de.f16.matvec_two_inputs(
                    compute,
                    &mut t0.mix,
                    &mut t1.mix,
                    &dlw.hc_ffn_fn.buffer,
                    &t0.flat,
                    &t1.flat,
                    HC_MIX_DIM,
                    HC_DIM,
                )?;
                de.hc_sinkhorn.launch(
                    compute,
                    &mut t0.split,
                    &t0.mix,
                    &dlw.hc_ffn_scale,
                    &dlw.hc_ffn_base,
                    N_HC,
                    SINKHORN_ITERS,
                    SINKHORN_EPS,
                )?;
                de.hc_sinkhorn.launch(
                    compute,
                    &mut t1.split,
                    &t1.mix,
                    &dlw.hc_ffn_scale,
                    &dlw.hc_ffn_base,
                    N_HC,
                    SINKHORN_ITERS,
                    SINKHORN_EPS,
                )?;
                de.hc_weighted.launch(
                    compute,
                    &mut t0.ffn_cur,
                    &t0.after_attn_hc,
                    &t0.split,
                    N_EMBD,
                    N_HC,
                )?;
                de.hc_weighted.launch(
                    compute,
                    &mut t1.ffn_cur,
                    &t1.after_attn_hc,
                    &t1.split,
                    N_EMBD,
                    N_HC,
                )?;
                de.rms_w.launch_weighted(
                    compute,
                    &mut t0.ffn_input_norm,
                    &t0.ffn_cur,
                    &dlw.ffn_norm,
                    N_EMBD,
                    RMS_EPS,
                )?;
                de.rms_w.launch_weighted(
                    compute,
                    &mut t1.ffn_input_norm,
                    &t1.ffn_cur,
                    &dlw.ffn_norm,
                    N_EMBD,
                    RMS_EPS,
                )?;
                let graph = compute.end_capture()?;
                let exec = graph.instantiate()?;
                exec.launch(compute)?;
                *guard = Some(exec);
            } else {
                guard.as_ref().unwrap().launch(compute)?;
            }
        }

        // Stage 11: router — learned path CAPTURED. Hash router (L<3) skips
        // capture because it needs host sync + CPU work; runs direct each time.
        {
            let _t = de.events.stage("dgpu.router_pair", compute)?;
            if dlw.is_hash_router {
                de.f16.matvec_two_inputs(
                    compute,
                    &mut t0.router_logits,
                    &mut t1.router_logits,
                    &dlw.ffn_gate_inp.buffer,
                    &t0.ffn_input_norm,
                    &t1.ffn_input_norm,
                    N_EXPERT,
                    N_EMBD,
                )?;
                compute.synchronize()?;
                t0.router_logits.copy_to_host(&mut t0.router_logits_host)?;
                t1.router_logits.copy_to_host(&mut t1.router_logits_host)?;
                let tid2eid = dlw
                    .tid2eid
                    .as_ref()
                    .ok_or_else(|| eyre!("hash router missing tid2eid"))?;
                let (sel0, w0) = hash_router_select(tid2eid, token_id_0, &t0.router_logits_host);
                let (sel1, w1) = hash_router_select(tid2eid, token_id_1, &t1.router_logits_host);
                t0.d_selected.copy_from_host(&sel0)?;
                t0.d_ew.copy_from_host(&w0)?;
                t1.d_selected.copy_from_host(&sel1)?;
                t1.d_ew.copy_from_host(&w1)?;
            } else {
                let graph_slot = &self.dgpu_pair_router_graphs[layer as usize];
                let mut guard = graph_slot.lock().unwrap();
                if guard.is_none() {
                    compute.begin_capture(
                        v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                    )?;
                    de.f16.matvec_two_inputs(
                        compute,
                        &mut t0.router_logits,
                        &mut t1.router_logits,
                        &dlw.ffn_gate_inp.buffer,
                        &t0.ffn_input_norm,
                        &t1.ffn_input_norm,
                        N_EXPERT,
                        N_EMBD,
                    )?;
                    de.router_topk.launch(
                        compute,
                        &mut t0.d_selected,
                        &mut t0.d_ew,
                        &t0.router_logits,
                        dlw.router_bias_dev.as_ref(),
                        N_EXPERT,
                        N_EXPERT_USED as u32,
                        EXPERT_WEIGHT_SCALE,
                        ROUTER_WEIGHT_EPS,
                    )?;
                    de.router_topk.launch(
                        compute,
                        &mut t1.d_selected,
                        &mut t1.d_ew,
                        &t1.router_logits,
                        dlw.router_bias_dev.as_ref(),
                        N_EXPERT,
                        N_EXPERT_USED as u32,
                        EXPERT_WEIGHT_SCALE,
                        ROUTER_WEIGHT_EPS,
                    )?;
                    let graph = compute.end_capture()?;
                    let exec = graph.instantiate()?;
                    exec.launch(compute)?;
                    *guard = Some(exec);
                } else {
                    guard.as_ref().unwrap().launch(compute)?;
                }
            }
        }

        // ===== Stages 10 + 12 + 13 reordered + de.xfer dropped =====
        // M40-P4.7: peer pushes now happen on de.compute (the single dGPU
        // stream). FIFO ordering handles "wait for router writes" so we
        // dropped the selected_ready cross-stream event. The cross-device
        // selected_pushed event stays (iGPU has to wait for the push to
        // land before MoE).
        //
        // ORDER FIX: queue shared_expert_pair on compute BEFORE the iGPU
        // queueing (peer pushes + 8 iGPU launches + device switches). Without
        // this de.compute sits idle ~250 µs waiting for host to come back.
        // With the reorder de.compute FIFOs straight from router_pair →
        // shared_expert_pair → peer_pushes → selected_pushed.record.

        // Stage 13: shared_expert_pair — CAPTURED (queued NOW so de.compute
        // FIFOs straight from router_pair → shared_expert_pair → peer_pushes).
        {
            let _t = de.events.stage("dgpu.shared_expert_pair", compute)?;
            let graph_slot = &self.dgpu_pair_shared_expert_graphs[layer as usize];
            let mut guard = graph_slot.lock().unwrap();
            if guard.is_none() {
                compute.begin_capture(
                    v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )?;
                de.q8.quantize_input(
                    compute,
                    &mut t0.xq_n_embd,
                    &mut t0.xscale_n_embd,
                    &t0.ffn_input_norm,
                    N_EMBD,
                )?;
                de.q8.quantize_input(
                    compute,
                    &mut t1.xq_n_embd,
                    &mut t1.xscale_n_embd,
                    &t1.ffn_input_norm,
                    N_EMBD,
                )?;
                de.q8.matvec_pair(
                    compute,
                    &mut t0.gate_sh,
                    &mut t1.gate_sh,
                    &dlw.shared.gate.buffer,
                    &t0.xq_n_embd,
                    &t1.xq_n_embd,
                    &t0.xscale_n_embd,
                    &t1.xscale_n_embd,
                    N_FF_SHARED,
                    N_EMBD,
                )?;
                de.q8.matvec_pair(
                    compute,
                    &mut t0.up_sh,
                    &mut t1.up_sh,
                    &dlw.shared.up.buffer,
                    &t0.xq_n_embd,
                    &t1.xq_n_embd,
                    &t0.xscale_n_embd,
                    &t1.xscale_n_embd,
                    N_FF_SHARED,
                    N_EMBD,
                )?;
                de.swiglu
                    .launch(compute, &mut t0.mid_sh, &t0.gate_sh, &t0.up_sh, N_FF_SHARED)?;
                de.swiglu
                    .launch(compute, &mut t1.mid_sh, &t1.gate_sh, &t1.up_sh, N_FF_SHARED)?;
                de.q8.quantize_input(
                    compute,
                    &mut t0.mid_sh_xq,
                    &mut t0.mid_sh_xscale,
                    &t0.mid_sh,
                    N_FF_SHARED,
                )?;
                de.q8.quantize_input(
                    compute,
                    &mut t1.mid_sh_xq,
                    &mut t1.mid_sh_xscale,
                    &t1.mid_sh,
                    N_FF_SHARED,
                )?;
                de.q8.matvec_pair(
                    compute,
                    &mut t0.ffn_shared,
                    &mut t1.ffn_shared,
                    &dlw.shared.down.buffer,
                    &t0.mid_sh_xq,
                    &t1.mid_sh_xq,
                    &t0.mid_sh_xscale,
                    &t1.mid_sh_xscale,
                    N_EMBD,
                    N_FF_SHARED,
                )?;
                let graph = compute.end_capture()?;
                let exec = graph.instantiate()?;
                exec.launch(compute)?;
                *guard = Some(exec);
            } else {
                guard.as_ref().unwrap().launch(compute)?;
            }
        }

        // Peer pushes for both tokens on de.compute (FIFO with shared_expert).
        // Cost of the serialization: ~6 pushes × ~1-2 µs each = ~10 µs per
        // layer. In exchange: no cross-stream events, captured-graph friendly.
        {
            let _t = de.events.stage("dgpu.peer_push_pair", compute)?;
            peer_push_f32(&t0.ffn_input_norm, &mut igpu_scratch.ffn_input_norm_recv_t0, compute)?;
            peer_push_i32(&t0.d_selected, &mut igpu_scratch.d_selected_t0, compute)?;
            peer_push_f32(&t0.d_ew, &mut igpu_scratch.d_ew_t0, compute)?;
            peer_push_f32(&t1.ffn_input_norm, &mut igpu_scratch.ffn_input_norm_recv_t1, compute)?;
            peer_push_i32(&t1.d_selected, &mut igpu_scratch.d_selected_t1, compute)?;
            peer_push_f32(&t1.d_ew, &mut igpu_scratch.d_ew_t1, compute)?;
            // Cross-device events: iGPU MoEs wait on these before starting.
            evt_t0.selected_pushed.record(compute)?;
            evt_t1.selected_pushed.record(compute)?;
        }

        // ===== iGPU MoE for both tokens (FIFO on ie.compute) =====
        self.set_current_cached(self.igpu.device)?;
        let ie = &self.igpu;
        let gbpe = ilw.routed.gate_bytes_per_expert;
        let ubpe = ilw.routed.up_bytes_per_expert;
        let dbpe = ilw.routed.down_bytes_per_expert;
        let mid_blocks_bytes = (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8_K_BYTES;
        // t0 MoE
        ie.compute.wait_event(&evt_t0.selected_pushed)?;
        {
            let _t = ie.events.stage("igpu.moe_t0", &ie.compute)?;
            ie.q8k.launch(
                &ie.compute,
                &mut igpu_scratch.d_xq_q8k,
                &igpu_scratch.ffn_input_norm_recv_t0,
                BLOCKS_Q8K_GATE_IN,
            )?;
            ie.iq2.launch_fused_swiglu_batch(
                &ie.compute,
                &mut igpu_scratch.d_mid_cat,
                &ilw.routed.gate.buffer,
                &ilw.routed.up.buffer,
                &igpu_scratch.d_xq_q8k,
                &igpu_scratch.d_ew_t0,
                &igpu_scratch.d_selected_t0,
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
                &mut igpu_scratch.ffn_moe_t0,
                &ilw.routed.down.buffer,
                &igpu_scratch.d_midq_cat,
                &igpu_scratch.d_selected_t0,
                dbpe as u32,
                mid_blocks_bytes as u32,
                N_EXPERT_USED as u32,
                N_EMBD,
                BLOCKS_Q8K_DOWN_IN,
            )?;
            // M40-P4.7: push back on ie.compute (FIFO with MoE) — drops
            // ie.xfer and the cross-stream moe_done event. moe_arrived now
            // records on ie.compute. iGPU side becomes capture-friendly.
            peer_push_f32(&igpu_scratch.ffn_moe_t0, &mut t0.ffn_moe_recv, &ie.compute)?;
            evt_t0.moe_arrived.record(&ie.compute)?;
        }
        // t1 MoE
        ie.compute.wait_event(&evt_t1.selected_pushed)?;
        {
            let _t = ie.events.stage("igpu.moe_t1", &ie.compute)?;
            ie.q8k.launch(
                &ie.compute,
                &mut igpu_scratch.d_xq_q8k,
                &igpu_scratch.ffn_input_norm_recv_t1,
                BLOCKS_Q8K_GATE_IN,
            )?;
            ie.iq2.launch_fused_swiglu_batch(
                &ie.compute,
                &mut igpu_scratch.d_mid_cat,
                &ilw.routed.gate.buffer,
                &ilw.routed.up.buffer,
                &igpu_scratch.d_xq_q8k,
                &igpu_scratch.d_ew_t1,
                &igpu_scratch.d_selected_t1,
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
                &mut igpu_scratch.ffn_moe_t1,
                &ilw.routed.down.buffer,
                &igpu_scratch.d_midq_cat,
                &igpu_scratch.d_selected_t1,
                dbpe as u32,
                mid_blocks_bytes as u32,
                N_EXPERT_USED as u32,
                N_EMBD,
                BLOCKS_Q8K_DOWN_IN,
            )?;
            // Push back t1 on ie.compute (FIFO).
            peer_push_f32(&igpu_scratch.ffn_moe_t1, &mut t1.ffn_moe_recv, &ie.compute)?;
            evt_t1.moe_arrived.record(&ie.compute)?;
        }
        self.set_current_cached(self.dgpu.device)?;

        drop(_t_pre);
        Ok(())
    }

    /// pair_post_moe_batched — both tokens' ffn_combine on de.compute.
    /// Wait for moe_arrived (cross-device event, can't capture), then
    /// vec_add + hc_post per token. The vec_add+hc_post for BOTH tokens
    /// is captured into one graph (same buffer layout layer-after-layer).
    fn pair_post_moe_batched(
        &self,
        t0: &mut TokenScratch,
        t1: &mut TokenScratch,
        _dlw: &DgpuLayerWeights,
        layer: u32,
        evt_t0: &LayerSyncEvents,
        evt_t1: &LayerSyncEvents,
    ) -> eyre::Result<()> {
        let de = &self.dgpu;
        let compute = &de.compute;
        // Cross-device waits — can't go inside captured graph; stay direct.
        {
            let _t = de.events.stage("dgpu.ffn_combine.wait_t0", compute)?;
            compute.wait_event(&evt_t0.moe_arrived)?;
        }
        {
            let _t = de.events.stage("dgpu.ffn_combine.wait_t1", compute)?;
            compute.wait_event(&evt_t1.moe_arrived)?;
        }
        // Both ffn_combines CAPTURED into one graph per layer (4 kernels:
        // 2 vec_add + 2 hc_post).
        {
            let _t = de.events.stage("dgpu.ffn_combine_pair", compute)?;
            let graph_slot = &self.dgpu_pair_ffn_combine_graphs[layer as usize];
            let mut guard = graph_slot.lock().unwrap();
            if guard.is_none() {
                compute.begin_capture(
                    v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )?;
                de.vec_add.launch(compute, &mut t0.ffn_moe_recv, &t0.ffn_shared, N_EMBD)?;
                de.hc_post.launch_from_split(
                    compute,
                    &mut t0.residual_next,
                    &t0.ffn_moe_recv,
                    &t0.after_attn_hc,
                    &t0.split,
                    N_HC,
                    N_EMBD,
                    N_HC,
                )?;
                de.vec_add.launch(compute, &mut t1.ffn_moe_recv, &t1.ffn_shared, N_EMBD)?;
                de.hc_post.launch_from_split(
                    compute,
                    &mut t1.residual_next,
                    &t1.ffn_moe_recv,
                    &t1.after_attn_hc,
                    &t1.split,
                    N_HC,
                    N_EMBD,
                    N_HC,
                )?;
                let graph = compute.end_capture()?;
                let exec = graph.instantiate()?;
                exec.launch(compute)?;
                *guard = Some(exec);
            } else {
                guard.as_ref().unwrap().launch(compute)?;
            }
        }
        Ok(())
    }
}
