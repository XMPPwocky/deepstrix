//! M40-P3: substage-interleaved layer-major pair forward.
//!
//! Improves on Phase 1's `forward_pair` (which serializes token0's
//! full layer before token1's, giving no MoE-hiding parallelism) by
//! decomposing each layer into two halves:
//!
//!   * pre_moe: stages 1-12 of a normal forward_layer — everything
//!     through router output and the peer-push of (ffn_input_norm,
//!     selected, d_ew) to iGPU. Kicks off iGPU MoE for that token.
//!   * post_moe: stages 13-16 — shared_expert on dGPU, wait for
//!     iGPU MoE to land, ffn_combine writing residual_next.
//!
//! The interleaved per-layer flow:
//!   t0 pre_moe → push t0 → [iGPU MoE_t0 begins]
//!   t1 pre_moe → push t1 → [iGPU MoE_t1 queued behind MoE_t0]
//!   t0 post_moe → t1 post_moe
//!
//! Because t1's dGPU pre_moe (~250 µs) runs in parallel with iGPU
//! MoE_t0 (~220 µs), the per-layer wait_event(moe_arrived) is hidden.
//! Expected pair wall: ~30-40 ms vs Phase 1's ~180 ms.
//!
//! All launches are DIRECT (no captured graphs). The buffer-pointer
//! coupling that drives captured graphs would conflict with the
//! per-token stash/restore pattern; we trade ~5-10 ms of host enqueue
//! overhead for substage flexibility.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::DeviceBuffer;

use crate::forward::{
    hash_router_select, BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, EXPERT_WEIGHT_SCALE, GROUP_DIM,
    HC_DIM, HC_MIX_DIM, N_EMBD, N_EXPERT, N_EXPERT_USED, N_FF_EXP, N_FF_SHARED, N_GROUPS, N_HC,
    N_HEAD, N_HEAD_DIM, N_LAYER, N_LORA_Q, N_ROT, OUT_LOW, Q_FLAT, RANK, RMS_EPS, SINKHORN_EPS,
    SINKHORN_ITERS, SWA_WINDOW, SWIGLU_CLAMP_EXP,
};
use crate::q8_k::BLOCK_Q8_K_BYTES;

use super::engine::HeterogeneousEngine;
use super::scratch::{DgpuScratch, IgpuScratch};
use super::state::HetLayerState;
use super::sync::{peer_push_f32, peer_push_i32};
use super::weights::{DgpuLayerWeights, HetModelWeights, IgpuLayerWeights};

const ROUTER_WEIGHT_EPS: f32 = 6.103515625e-5;

impl HeterogeneousEngine {
    /// M40-P3: substage-interleaved pair forward. Same semantics as
    /// `forward_pair` (Phase 1) — produces `logits_token0` + `logits`
    /// bit-equivalent to two sequential `forward_token` calls — but
    /// reorders kernel launches so iGPU MoE for token0 hides behind
    /// token1's dGPU pre_moe.
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
    ) -> color_eyre::eyre::Result<()> {
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

        // Initialize per-token residual stashes.
        dgpu_scratch
            .residual_stash_token0
            .copy_from_host(input_hc_0)?;
        dgpu_scratch
            .residual_stash_token1
            .copy_from_host(input_hc_1)?;

        let pair_start = std::time::Instant::now();
        for layer in 0..N_LAYER as usize {
            let dlw = &weights.dgpu_layers[layer];
            let ilw = &weights.igpu_layers[layer];
            let ls = &mut state.layers[layer];

            // M40-P3.6: substage interleaving with EVENT-DRIVEN GPU pipeline.
            // t0_pre queues dGPU 1-12 + pushes + iGPU MoE_t0 (all event-gated).
            // Host returns before MoE_t0 runs. Snapshot async-copies state.
            // t1_pre queues t1's dGPU 1-12 + pushes + iGPU MoE_t1; t1's dGPU
            // pre work runs in parallel with t0's iGPU MoE (different devices).
            // t0_post queues shared_expert (independent of MoE), then waits
            // on moe_arrived_t0 (cross-stream event, no host block) before
            // ffn_combine. Same for t1_post with moe_arrived_t1.
            //
            // All stash/restore copies are async on de.compute → FIFO orders
            // them against the reading kernels. Host never blocks inside the
            // per-layer loop.
            let evt_t0 = &self.sync_events.layers[layer as usize];
            let evt_t1 = &self.sync_events_t1.layers[layer as usize];

            // ===== t0 pre_moe =====
            dgpu_scratch
                .residual
                .copy_from_buffer_async(&dgpu_scratch.residual_stash_token0, &self.dgpu.compute)?;
            self.forward_pair_pre_moe(
                dgpu_scratch,
                igpu_scratch,
                ls,
                dlw,
                ilw,
                pos,
                token_id_0,
                /* push_t1 */ false,
                evt_t0,
            )?;
            // Stash t0's pre_moe outputs async on de.compute (FIFO after
            // pre_moe writes, FIFO before t1's overwrites).
            dgpu_scratch
                .after_attn_hc_stash_t0
                .copy_from_buffer_async(&dgpu_scratch.after_attn_hc, &self.dgpu.compute)?;
            dgpu_scratch
                .ffn_input_norm_stash_t0
                .copy_from_buffer_async(&dgpu_scratch.ffn_input_norm, &self.dgpu.compute)?;
            dgpu_scratch
                .split_stash_t0
                .copy_from_buffer_async(&dgpu_scratch.split, &self.dgpu.compute)?;

            // ===== Snapshot per-layer state (post-t0, pre-t1) =====
            ls.snapshot_async(&self.dgpu.compute, &self.igpu.compute)?;
            self.invalidate_device_cache();
            self.set_current_cached(self.dgpu.device)?;

            // ===== t1 pre_moe (runs in parallel with t0's iGPU MoE) =====
            dgpu_scratch
                .residual
                .copy_from_buffer_async(&dgpu_scratch.residual_stash_token1, &self.dgpu.compute)?;
            self.forward_pair_pre_moe(
                dgpu_scratch,
                igpu_scratch,
                ls,
                dlw,
                ilw,
                pos + 1,
                token_id_1,
                /* push_t1 */ true,
                evt_t1,
            )?;
            dgpu_scratch
                .after_attn_hc_stash_t1
                .copy_from_buffer_async(&dgpu_scratch.after_attn_hc, &self.dgpu.compute)?;
            dgpu_scratch
                .ffn_input_norm_stash_t1
                .copy_from_buffer_async(&dgpu_scratch.ffn_input_norm, &self.dgpu.compute)?;
            dgpu_scratch
                .split_stash_t1
                .copy_from_buffer_async(&dgpu_scratch.split, &self.dgpu.compute)?;

            // ===== t0 post_moe — restore t0's pre_moe outputs, then run =====
            dgpu_scratch
                .after_attn_hc
                .copy_from_buffer_async(&dgpu_scratch.after_attn_hc_stash_t0, &self.dgpu.compute)?;
            dgpu_scratch
                .ffn_input_norm
                .copy_from_buffer_async(&dgpu_scratch.ffn_input_norm_stash_t0, &self.dgpu.compute)?;
            dgpu_scratch
                .split
                .copy_from_buffer_async(&dgpu_scratch.split_stash_t0, &self.dgpu.compute)?;
            self.forward_pair_post_moe(
                dgpu_scratch,
                dlw,
                /* token */ 0,
                evt_t0,
            )?;
            dgpu_scratch
                .residual_next_stash_t0
                .copy_from_buffer_async(&dgpu_scratch.residual_next, &self.dgpu.compute)?;

            // ===== t1 post_moe =====
            dgpu_scratch
                .after_attn_hc
                .copy_from_buffer_async(&dgpu_scratch.after_attn_hc_stash_t1, &self.dgpu.compute)?;
            dgpu_scratch
                .ffn_input_norm
                .copy_from_buffer_async(&dgpu_scratch.ffn_input_norm_stash_t1, &self.dgpu.compute)?;
            dgpu_scratch
                .split
                .copy_from_buffer_async(&dgpu_scratch.split_stash_t1, &self.dgpu.compute)?;
            self.forward_pair_post_moe(
                dgpu_scratch,
                dlw,
                /* token */ 1,
                evt_t1,
            )?;
            dgpu_scratch
                .residual_next_stash_t1
                .copy_from_buffer_async(&dgpu_scratch.residual_next, &self.dgpu.compute)?;

            // For next layer's input: copy residual_next_stash_tN → residual_stash_tokenN.
            // Async on de.compute so it FIFO-orders before next layer's pre_moe reads.
            dgpu_scratch
                .residual_stash_token0
                .copy_from_buffer_async(&dgpu_scratch.residual_next_stash_t0, &self.dgpu.compute)?;
            dgpu_scratch
                .residual_stash_token1
                .copy_from_buffer_async(&dgpu_scratch.residual_next_stash_t1, &self.dgpu.compute)?;
        }

        // ============ Head x2 ============
        // Head for token0: load its final HC into `residual` (forward_head consumes
        // dgpu_scratch.residual content; the pointer identity doesn't matter for the
        // direct-launch forward_head).
        dgpu_scratch
            .residual
            .copy_from_buffer(&dgpu_scratch.residual_stash_token0)?;
        self.forward_head(dgpu_scratch, &weights.global)?;
        dgpu_scratch
            .logits_token0
            .copy_from_buffer(&dgpu_scratch.logits)?;

        dgpu_scratch
            .residual
            .copy_from_buffer(&dgpu_scratch.residual_stash_token1)?;
        self.forward_head(dgpu_scratch, &weights.global)?;

        self.set_current_cached(self.dgpu.device)?;
        let host_us = pair_start.elapsed().as_micros() as u64;
        self.dgpu.compute.synchronize()?;
        let pair_elapsed_us = pair_start.elapsed().as_micros() as u64;
        let sync_us = pair_elapsed_us.saturating_sub(host_us);
        use std::sync::atomic::Ordering;
        self.last_host_us.store(host_us, Ordering::Relaxed);
        self.last_sync_us.store(sync_us, Ordering::Relaxed);

        Ok(())
    }

    // M40-P3.6: public debug wrappers removed (the pre/post split is now
    // tested end-to-end via the forward_pair_interleaved oracle).

    /// pre_moe: stages 1-12 inline, direct launches, no captured graphs.
    /// Reads `dgpu_scratch.residual` for this token's input HC; writes
    /// `after_attn_hc`, `ffn_input_norm`, `split`, plus pushes
    /// (ffn_input_norm, selected, d_ew) to the iGPU's per-token recv
    /// buffers (selected by `push_t1` flag).
    ///
    /// M40-P3.6: takes per-token `events` (sync_events for t0, sync_events_t1
    /// for t1) and uses cross-stream event gating instead of host syncs. The
    /// iGPU MoE is treated as an autonomous "ffn_in/selected → ffn_out" pipeline:
    /// dGPU pushes inputs, records selected_pushed; iGPU waits on it, runs MoE,
    /// records moe_done; iGPU.xfer waits on moe_done, pushes ffn_moe back to
    /// dGPU, records moe_arrived. Host never blocks.
    #[allow(clippy::too_many_arguments)]
    fn forward_pair_pre_moe(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        ilw: &IgpuLayerWeights,
        pos: u32,
        token_id: i32,
        push_t1: bool,
        events: &crate::het::engine::LayerSyncEvents,
    ) -> eyre::Result<()> {
        let de = &self.dgpu;
        self.set_current_cached(self.dgpu.device)?;

        // ===== Stage 1: mhc_pre_attn =====
        de.rms_nw.launch(
            &de.compute,
            &mut dgpu_scratch.flat,
            &dgpu_scratch.residual,
            1,
            HC_DIM,
            RMS_EPS,
        )?;
        de.f16.matvec(
            &de.compute,
            &mut dgpu_scratch.mix,
            &dlw.hc_attn_fn.buffer,
            &dgpu_scratch.flat,
            HC_MIX_DIM,
            HC_DIM,
        )?;
        de.hc_sinkhorn.launch(
            &de.compute,
            &mut dgpu_scratch.split,
            &dgpu_scratch.mix,
            &dlw.hc_attn_scale,
            &dlw.hc_attn_base,
            N_HC,
            SINKHORN_ITERS,
            SINKHORN_EPS,
        )?;
        de.hc_weighted.launch(
            &de.compute,
            &mut dgpu_scratch.attn_cur,
            &dgpu_scratch.residual,
            &dgpu_scratch.split,
            N_EMBD,
            N_HC,
        )?;
        de.rms_w.launch_weighted(
            &de.compute,
            &mut dgpu_scratch.attn_input_norm,
            &dgpu_scratch.attn_cur,
            &dlw.attn_norm,
            N_EMBD,
            RMS_EPS,
        )?;

        // ===== Stage 2: Q chain =====
        de.q8.quantize_input(
            &de.compute,
            &mut dgpu_scratch.xq_n_embd,
            &mut dgpu_scratch.xscale_n_embd,
            &dgpu_scratch.attn_input_norm,
            N_EMBD,
        )?;
        de.q8.matvec(
            &de.compute,
            &mut dgpu_scratch.qr,
            &dlw.attn_q_a.buffer,
            &dgpu_scratch.xq_n_embd,
            &dgpu_scratch.xscale_n_embd,
            N_LORA_Q,
            N_EMBD,
        )?;
        de.rms_w.launch_weighted(
            &de.compute,
            &mut dgpu_scratch.qr_normed,
            &dgpu_scratch.qr,
            &dlw.q_a_norm,
            N_LORA_Q,
            RMS_EPS,
        )?;
        de.q8.quantize_input(
            &de.compute,
            &mut dgpu_scratch.qr_xq,
            &mut dgpu_scratch.qr_xscale,
            &dgpu_scratch.qr_normed,
            N_LORA_Q,
        )?;
        de.q8.matvec(
            &de.compute,
            &mut dgpu_scratch.q,
            &dlw.attn_q_b.buffer,
            &dgpu_scratch.qr_xq,
            &dgpu_scratch.qr_xscale,
            Q_FLAT,
            N_LORA_Q,
        )?;
        de.rms_nw.launch(
            &de.compute,
            &mut dgpu_scratch.q_normed,
            &dgpu_scratch.q,
            N_HEAD,
            N_HEAD_DIM,
            RMS_EPS,
        )?;
        // Stage 3: rope
        de.rope.launch_forward(
            &de.compute,
            &mut dgpu_scratch.q_normed,
            N_HEAD,
            N_HEAD_DIM,
            N_ROT,
            pos,
            &dlw.rope_params,
        )?;

        // ===== Stage 4: KV chain + cache append =====
        de.q8.matvec(
            &de.compute,
            &mut dgpu_scratch.kv_raw,
            &dlw.attn_kv.buffer,
            &dgpu_scratch.xq_n_embd,
            &dgpu_scratch.xscale_n_embd,
            N_HEAD_DIM,
            N_EMBD,
        )?;
        de.rms_w.launch_weighted(
            &de.compute,
            &mut dgpu_scratch.kv_normed,
            &dgpu_scratch.kv_raw,
            &dlw.kv_a_norm,
            N_HEAD_DIM,
            RMS_EPS,
        )?;
        de.rope.launch_forward(
            &de.compute,
            &mut dgpu_scratch.kv_normed,
            1,
            N_HEAD_DIM,
            N_ROT,
            pos,
            &dlw.rope_params,
        )?;
        de.fp8
            .launch(&de.compute, &mut dgpu_scratch.kv_normed, N_HEAD_DIM - N_ROT)?;
        de.f16rt
            .launch(&de.compute, &mut dgpu_scratch.kv_normed, N_HEAD_DIM)?;
        de.kv_append.launch(
            &de.compute,
            &mut ls.kv_cache,
            &dgpu_scratch.kv_normed,
            pos,
            SWA_WINDOW,
            N_HEAD_DIM,
        )?;
        ls.n_raw = (ls.n_raw + 1).min(SWA_WINDOW);

        // ===== Stage 5: compressor (ratio > 0) =====
        let ratio = dlw.ratio;
        if ratio > 0 {
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
                &de.compute,
                &mut dgpu_scratch.kv_cur,
                &mut dgpu_scratch.sc_cur,
                &cw.wkv.buffer,
                &cw.wgate.buffer,
                &dgpu_scratch.attn_input_norm,
                comp_width,
                N_EMBD,
            )?;
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
            let comp_fires_boundary = (pos + 1) % ratio == 0;
            if comp_fires_boundary {
                de.compressor_pool.launch(
                    &de.compute,
                    &mut dgpu_scratch.pooled,
                    &cs.state_kv,
                    &cs.state_score,
                    N_HEAD_DIM,
                    ratio,
                )?;
                de.rms_w.launch_weighted(
                    &de.compute,
                    &mut dgpu_scratch.comp_row,
                    &dgpu_scratch.pooled,
                    &cw.norm,
                    N_HEAD_DIM,
                    RMS_EPS,
                )?;
                // Per forward_layer: boundary rope uses comp_pos = pos+1-ratio,
                // NOT pos (the pooled state represents the COMP block that just
                // closed at index (pos+1)/ratio - 1).
                let comp_pos = pos + 1 - ratio;
                de.rope.launch_forward(
                    &de.compute,
                    &mut dgpu_scratch.comp_row,
                    1,
                    N_HEAD_DIM,
                    N_ROT,
                    comp_pos,
                    &dlw.rope_params,
                )?;
                de.fp8.launch(
                    &de.compute,
                    &mut dgpu_scratch.comp_row,
                    N_HEAD_DIM - N_ROT,
                )?;
                de.f16rt
                    .launch(&de.compute, &mut dgpu_scratch.comp_row, N_HEAD_DIM)?;
                if ratio == 4 {
                    de.compressor_shuffle.launch(
                        &de.compute,
                        &mut cs.state_kv,
                        &mut cs.state_score,
                        comp_width,
                    )?;
                }
                de.comp_kv_append.launch(
                    &de.compute,
                    &mut cs.comp_kv,
                    &dgpu_scratch.comp_row,
                    cs.n_comp,
                    N_HEAD_DIM,
                )?;
                cs.n_comp += 1;
            }
        }

        // ===== Stage 6: attention =====
        // Match forward_layer's dispatch: ratio==0 → swa; ratio>0 → mixed
        // (even when n_comp==0 — empty comp_kv buffer). Not based on n_comp.
        let n_raw = ls.n_raw;
        if ratio == 0 {
            de.attn_swa.launch(
                &de.compute,
                &mut dgpu_scratch.heads,
                &dgpu_scratch.q_normed,
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
                &de.compute,
                &mut dgpu_scratch.heads,
                &dgpu_scratch.q_normed,
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

        // ===== Stage 7: output_proj =====
        de.rope.launch_inverse(
            &de.compute,
            &mut dgpu_scratch.heads,
            N_HEAD,
            N_HEAD_DIM,
            N_ROT,
            pos,
            &dlw.rope_params,
        )?;
        de.q8.quantize_input(
            &de.compute,
            &mut dgpu_scratch.heads_xq,
            &mut dgpu_scratch.heads_xscale,
            &dgpu_scratch.heads,
            Q_FLAT,
        )?;
        de.q8_grouped.matvec_grouped(
            &de.compute,
            &mut dgpu_scratch.low,
            &dlw.attn_output_a.buffer,
            &dgpu_scratch.heads_xq,
            &dgpu_scratch.heads_xscale,
            GROUP_DIM,
            RANK,
            N_GROUPS,
        )?;
        de.q8.quantize_input(
            &de.compute,
            &mut dgpu_scratch.low_xq,
            &mut dgpu_scratch.low_xscale,
            &dgpu_scratch.low,
            OUT_LOW,
        )?;
        de.q8.matvec(
            &de.compute,
            &mut dgpu_scratch.attn_out,
            &dlw.attn_output_b.buffer,
            &dgpu_scratch.low_xq,
            &dgpu_scratch.low_xscale,
            N_EMBD,
            OUT_LOW,
        )?;

        // ===== Stage 8: mhc_post_attn → after_attn_hc =====
        de.hc_post.launch_from_split(
            &de.compute,
            &mut dgpu_scratch.after_attn_hc,
            &dgpu_scratch.attn_out,
            &dgpu_scratch.residual,
            &dgpu_scratch.split,
            N_HC,
            N_EMBD,
            N_HC,
        )?;

        // ===== Stage 9: mhc_pre_ffn → ffn_input_norm + new split =====
        de.rms_nw.launch(
            &de.compute,
            &mut dgpu_scratch.flat,
            &dgpu_scratch.after_attn_hc,
            1,
            HC_DIM,
            RMS_EPS,
        )?;
        de.f16.matvec(
            &de.compute,
            &mut dgpu_scratch.mix,
            &dlw.hc_ffn_fn.buffer,
            &dgpu_scratch.flat,
            HC_MIX_DIM,
            HC_DIM,
        )?;
        de.hc_sinkhorn.launch(
            &de.compute,
            &mut dgpu_scratch.split,
            &dgpu_scratch.mix,
            &dlw.hc_ffn_scale,
            &dlw.hc_ffn_base,
            N_HC,
            SINKHORN_ITERS,
            SINKHORN_EPS,
        )?;
        de.hc_weighted.launch(
            &de.compute,
            &mut dgpu_scratch.ffn_cur,
            &dgpu_scratch.after_attn_hc,
            &dgpu_scratch.split,
            N_EMBD,
            N_HC,
        )?;
        de.rms_w.launch_weighted(
            &de.compute,
            &mut dgpu_scratch.ffn_input_norm,
            &dgpu_scratch.ffn_cur,
            &dlw.ffn_norm,
            N_EMBD,
            RMS_EPS,
        )?;

        // ===== Stage 11: router on dGPU =====
        if dlw.is_hash_router {
            // Hash router: matvec → host readback → CPU topk via tid2eid.
            de.f16.matvec(
                &de.compute,
                &mut dgpu_scratch.router_logits,
                &dlw.ffn_gate_inp.buffer,
                &dgpu_scratch.ffn_input_norm,
                N_EXPERT,
                N_EMBD,
            )?;
            de.compute.synchronize()?;
            dgpu_scratch
                .router_logits
                .copy_to_host(&mut dgpu_scratch.router_logits_host)?;
            let tid2eid = dlw
                .tid2eid
                .as_ref()
                .ok_or_else(|| eyre!("hash router missing tid2eid"))?;
            let (sel, w) = hash_router_select(tid2eid, token_id, &dgpu_scratch.router_logits_host);
            dgpu_scratch.d_selected.copy_from_host(&sel)?;
            dgpu_scratch.d_ew.copy_from_host(&w)?;
        } else {
            // Learned router: f16 matvec then topk.
            de.f16.matvec(
                &de.compute,
                &mut dgpu_scratch.router_logits,
                &dlw.ffn_gate_inp.buffer,
                &dgpu_scratch.ffn_input_norm,
                N_EXPERT,
                N_EMBD,
            )?;
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
        }

        // ===== Stages 10 + 12: peer-push ffn_input_norm + selected + d_ew to iGPU =====
        // EVENT-DRIVEN. The router (above) wrote d_selected/d_ew on de.compute;
        // mhc_pre_ffn (above) wrote ffn_input_norm on de.compute. We record
        // `selected_ready` once on de.compute — it covers BOTH writes (FIFO).
        // de.xfer waits on it, then queues all three pushes, then records
        // `selected_pushed`. iGPU.compute waits on selected_pushed and starts MoE.
        // No host blocks.
        events.selected_ready.record(&de.compute)?;
        de.xfer.wait_event(&events.selected_ready)?;
        if push_t1 {
            peer_push_f32(
                &dgpu_scratch.ffn_input_norm,
                &mut igpu_scratch.ffn_input_norm_recv_t1,
                &de.xfer,
            )?;
            peer_push_i32(
                &dgpu_scratch.d_selected,
                &mut igpu_scratch.d_selected_t1,
                &de.xfer,
            )?;
            peer_push_f32(&dgpu_scratch.d_ew, &mut igpu_scratch.d_ew_t1, &de.xfer)?;
        } else {
            peer_push_f32(
                &dgpu_scratch.ffn_input_norm,
                &mut igpu_scratch.ffn_input_norm_recv_t0,
                &de.xfer,
            )?;
            peer_push_i32(
                &dgpu_scratch.d_selected,
                &mut igpu_scratch.d_selected_t0,
                &de.xfer,
            )?;
            peer_push_f32(&dgpu_scratch.d_ew, &mut igpu_scratch.d_ew_t0, &de.xfer)?;
        }
        events.selected_pushed.record(&de.xfer)?;

        // ===== iGPU MoE for this token =====
        // EVENT-GATED: ie.compute waits on selected_pushed (which transitively
        // covers all three pushes since they're on the same xfer stream FIFO).
        // After MoE, ie.compute records moe_done; ie.xfer waits on it then
        // pushes ffn_moe back to dGPU and records moe_arrived. post_moe's
        // ffn_combine waits on moe_arrived. Host never blocks.
        self.set_current_cached(self.igpu.device)?;
        let ie = &self.igpu;
        ie.compute.wait_event(&events.selected_pushed)?;
        let gbpe = ilw.routed.gate_bytes_per_expert;
        let ubpe = ilw.routed.up_bytes_per_expert;
        let dbpe = ilw.routed.down_bytes_per_expert;
        let mid_blocks_bytes = (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8_K_BYTES;

        if push_t1 {
            // MoE for token1 → writes to ffn_moe_t1
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
            // Push back to per-token recv on dGPU. ie.xfer is a different
            // stream than ie.compute; event-gate the xfer on moe_done.
            events.moe_done.record(&ie.compute)?;
            ie.xfer.wait_event(&events.moe_done)?;
            peer_push_f32(
                &igpu_scratch.ffn_moe_t1,
                &mut dgpu_scratch.ffn_moe_recv_stash_t1,
                &ie.xfer,
            )?;
            events.moe_arrived.record(&ie.xfer)?;
        } else {
            // MoE for token0 → writes to ffn_moe_t0
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
            events.moe_done.record(&ie.compute)?;
            ie.xfer.wait_event(&events.moe_done)?;
            peer_push_f32(
                &igpu_scratch.ffn_moe_t0,
                &mut dgpu_scratch.ffn_moe_recv_stash_t0,
                &ie.xfer,
            )?;
            events.moe_arrived.record(&ie.xfer)?;
        }
        // Switch back to dGPU device for next launch.
        self.set_current_cached(self.dgpu.device)?;

        Ok(())
    }

    /// post_moe: stages 13 + 15 + 16 inline. Reads `dgpu_scratch.ffn_input_norm`,
    /// `after_attn_hc`, `split`, `residual` (all of which the caller has
    /// already restored from this token's stashes) plus the iGPU MoE output
    /// in `ffn_moe_recv_stash_t<token>`. Writes `residual_next`.
    ///
    /// M40-P3.6: shared_expert is queued on de.compute WITHOUT waiting for
    /// the iGPU MoE (they're independent). vec_add (which reads ffn_moe_recv)
    /// waits on moe_arrived via cross-stream event — no host blocks.
    fn forward_pair_post_moe(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        dlw: &DgpuLayerWeights,
        token: u32,
        events: &crate::het::engine::LayerSyncEvents,
    ) -> eyre::Result<()> {
        let de = &self.dgpu;
        self.set_current_cached(self.dgpu.device)?;

        // ===== Stage 13: shared_expert =====
        de.q8.quantize_input(
            &de.compute,
            &mut dgpu_scratch.xq_n_embd,
            &mut dgpu_scratch.xscale_n_embd,
            &dgpu_scratch.ffn_input_norm,
            N_EMBD,
        )?;
        de.q8.matvec(
            &de.compute,
            &mut dgpu_scratch.gate_sh,
            &dlw.shared.gate.buffer,
            &dgpu_scratch.xq_n_embd,
            &dgpu_scratch.xscale_n_embd,
            N_FF_SHARED,
            N_EMBD,
        )?;
        de.q8.matvec(
            &de.compute,
            &mut dgpu_scratch.up_sh,
            &dlw.shared.up.buffer,
            &dgpu_scratch.xq_n_embd,
            &dgpu_scratch.xscale_n_embd,
            N_FF_SHARED,
            N_EMBD,
        )?;
        de.swiglu.launch(
            &de.compute,
            &mut dgpu_scratch.mid_sh,
            &dgpu_scratch.gate_sh,
            &dgpu_scratch.up_sh,
            N_FF_SHARED,
        )?;
        de.q8.quantize_input(
            &de.compute,
            &mut dgpu_scratch.mid_sh_xq,
            &mut dgpu_scratch.mid_sh_xscale,
            &dgpu_scratch.mid_sh,
            N_FF_SHARED,
        )?;
        de.q8.matvec(
            &de.compute,
            &mut dgpu_scratch.ffn_shared,
            &dlw.shared.down.buffer,
            &dgpu_scratch.mid_sh_xq,
            &dgpu_scratch.mid_sh_xscale,
            N_EMBD,
            N_FF_SHARED,
        )?;

        // ===== Cross-stream event wait for this token's iGPU MoE =====
        // ffn_moe_recv_stash_t<token> was peer-pushed by ie.xfer; moe_arrived
        // was recorded after that push. dGPU.compute waits on it (no host
        // block), then does a device-to-device async copy from the stash into
        // ffn_moe_recv (which vec_add reads in-place below). The copy is
        // FIFO-ordered with the subsequent vec_add/hc_post on the same stream.
        de.compute.wait_event(&events.moe_arrived)?;
        if token == 0 {
            dgpu_scratch
                .ffn_moe_recv
                .copy_from_buffer_async(&dgpu_scratch.ffn_moe_recv_stash_t0, &de.compute)?;
        } else {
            dgpu_scratch
                .ffn_moe_recv
                .copy_from_buffer_async(&dgpu_scratch.ffn_moe_recv_stash_t1, &de.compute)?;
        }

        // ===== Stage 16: ffn_combine =====
        de.vec_add.launch(
            &de.compute,
            &mut dgpu_scratch.ffn_moe_recv,
            &dgpu_scratch.ffn_shared,
            N_EMBD,
        )?;
        de.hc_post.launch_from_split(
            &de.compute,
            &mut dgpu_scratch.residual_next,
            &dgpu_scratch.ffn_moe_recv,
            &dgpu_scratch.after_attn_hc,
            &dgpu_scratch.split,
            N_HC,
            N_EMBD,
            N_HC,
        )?;
        Ok(())
    }
}
