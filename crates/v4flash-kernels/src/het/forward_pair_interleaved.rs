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

            // M40-P3 DEBUG: serialized per-token decomposition (t0 pre+post,
            // then t1 pre+post). Lets us isolate whether the bug is in the
            // pre/post split itself (would also fail this serial version) vs
            // the substage interleaving (only fails interleaved).
            //
            // M40-P3: substage interleaving — t0_pre, snapshot, t1_pre,
            // t0_post, t1_post. t1's dGPU pre_moe runs while t0's iGPU MoE
            // is in flight; t0_post waits for t0's iGPU MoE result; same
            // for t1. The per-token stash buffers (after_attn_hc, ffn_input_norm,
            // split, ffn_moe_recv_stash) preserve t0's pre_moe outputs
            // across t1's overwrites.
            //
            // ===== t0 pre_moe =====
            dgpu_scratch
                .residual
                .copy_from_buffer(&dgpu_scratch.residual_stash_token0)?;
            self.forward_pair_pre_moe(
                dgpu_scratch,
                igpu_scratch,
                ls,
                dlw,
                ilw,
                pos,
                token_id_0,
                /* push_t1 */ false,
            )?;
            // Stash t0's pre_moe outputs (will be clobbered by t1_pre_moe).
            dgpu_scratch
                .after_attn_hc_stash_t0
                .copy_from_buffer(&dgpu_scratch.after_attn_hc)?;
            dgpu_scratch
                .ffn_input_norm_stash_t0
                .copy_from_buffer(&dgpu_scratch.ffn_input_norm)?;
            dgpu_scratch
                .split_stash_t0
                .copy_from_buffer(&dgpu_scratch.split)?;

            // ===== Snapshot per-layer state (post-t0, pre-t1) =====
            ls.snapshot_async(&self.dgpu.compute, &self.igpu.compute)?;
            self.invalidate_device_cache();
            self.set_current_cached(self.dgpu.device)?;

            // ===== t1 pre_moe (runs in parallel with t0's iGPU MoE) =====
            dgpu_scratch
                .residual
                .copy_from_buffer(&dgpu_scratch.residual_stash_token1)?;
            self.forward_pair_pre_moe(
                dgpu_scratch,
                igpu_scratch,
                ls,
                dlw,
                ilw,
                pos + 1,
                token_id_1,
                /* push_t1 */ true,
            )?;
            dgpu_scratch
                .after_attn_hc_stash_t1
                .copy_from_buffer(&dgpu_scratch.after_attn_hc)?;
            dgpu_scratch
                .ffn_input_norm_stash_t1
                .copy_from_buffer(&dgpu_scratch.ffn_input_norm)?;
            dgpu_scratch
                .split_stash_t1
                .copy_from_buffer(&dgpu_scratch.split)?;

            // ===== t0 post_moe — restore t0's pre_moe outputs, then run =====
            dgpu_scratch
                .after_attn_hc
                .copy_from_buffer(&dgpu_scratch.after_attn_hc_stash_t0)?;
            dgpu_scratch
                .ffn_input_norm
                .copy_from_buffer(&dgpu_scratch.ffn_input_norm_stash_t0)?;
            dgpu_scratch
                .split
                .copy_from_buffer(&dgpu_scratch.split_stash_t0)?;
            self.forward_pair_post_moe(
                dgpu_scratch,
                dlw,
                /* token */ 0,
            )?;
            dgpu_scratch
                .residual_next_stash_t0
                .copy_from_buffer(&dgpu_scratch.residual_next)?;

            // ===== t1 post_moe =====
            dgpu_scratch
                .after_attn_hc
                .copy_from_buffer(&dgpu_scratch.after_attn_hc_stash_t1)?;
            dgpu_scratch
                .ffn_input_norm
                .copy_from_buffer(&dgpu_scratch.ffn_input_norm_stash_t1)?;
            dgpu_scratch
                .split
                .copy_from_buffer(&dgpu_scratch.split_stash_t1)?;
            self.forward_pair_post_moe(
                dgpu_scratch,
                dlw,
                /* token */ 1,
            )?;
            dgpu_scratch
                .residual_next_stash_t1
                .copy_from_buffer(&dgpu_scratch.residual_next)?;

            // For next layer's input: swap stash slots.
            dgpu_scratch
                .residual_stash_token0
                .copy_from_buffer(&dgpu_scratch.residual_next_stash_t0)?;
            dgpu_scratch
                .residual_stash_token1
                .copy_from_buffer(&dgpu_scratch.residual_next_stash_t1)?;
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

    /// Test-only: public wrapper around the private `forward_pair_pre_moe`
    /// so tests/pre_post_layer_diff.rs can call it on a single layer.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_pair_pre_moe_public_debug(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        ilw: &IgpuLayerWeights,
        pos: u32,
        token_id: i32,
        push_t1: bool,
    ) -> eyre::Result<()> {
        self.forward_pair_pre_moe(
            dgpu_scratch,
            igpu_scratch,
            ls,
            dlw,
            ilw,
            pos,
            token_id,
            push_t1,
        )
    }

    /// Test-only: public wrapper around the private `forward_pair_post_moe`.
    #[doc(hidden)]
    pub fn forward_pair_post_moe_public_debug(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        dlw: &DgpuLayerWeights,
        token: u32,
    ) -> eyre::Result<()> {
        // post_moe expects ffn_moe_recv_stash_t<token> to hold the iGPU MoE
        // output. The pre_moe (with push_t0/push_t1 flag) sent it there
        // already; just need to ensure xfer has landed before post_moe reads.
        // The public pre_moe debug variant doesn't sync xfer at end, so do it here.
        self.igpu.xfer.synchronize()?;
        self.forward_pair_post_moe(dgpu_scratch, dlw, token)
    }

    /// pre_moe: stages 1-12 inline, direct launches, no captured graphs.
    /// Reads `dgpu_scratch.residual` for this token's input HC; writes
    /// `after_attn_hc`, `ffn_input_norm`, `split`, plus pushes
    /// (ffn_input_norm, selected, d_ew) to the iGPU's per-token recv
    /// buffers (selected by `push_t1` flag).
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
        // Per-token iGPU recv buffers so MoE_t0 and MoE_t1 don't race on the same inputs.
        //
        // CRITICAL: de.xfer is a DIFFERENT stream from de.compute. Without explicit
        // ordering, the peer push on de.xfer can race ahead of the router/mhc_pre_ffn
        // writes on de.compute and push STALE data from a previous iteration. Pair-mode
        // uses event-driven gating (sev.ain_ready / sev.selected_ready). Here we
        // pessimistically host-sync de.compute before queueing the pushes — simple
        // and bullet-proof. (Bug found: L0-L2 happened to work because compute
        // finished fast enough; L3+ raced because the ratio=128 compressor shifted
        // timing.)
        de.compute.synchronize()?;
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
        // Block dGPU compute on the xfer for now (simpler than events; the
        // host won't be doing more dGPU compute on this stream until t0_post_moe
        // runs anyway, but the xfer must land before we move on).
        de.xfer.synchronize()?;

        // ===== iGPU MoE for this token =====
        // Launch MoE work on iGPU. With direct launches, kernels enqueue on
        // ie.compute and run in FIFO. t0's MoE work is queued first; t1's
        // is queued second (after t1_pre_moe completes). They serialize on
        // ie.compute but run in parallel with the dGPU work between them.
        self.set_current_cached(self.igpu.device)?;
        let ie = &self.igpu;
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
            // stream than ie.compute; explicitly wait for the q2k write to
            // ffn_moe_t1 to complete on ie.compute before the xfer reads it.
            ie.compute.synchronize()?;
            peer_push_f32(
                &igpu_scratch.ffn_moe_t1,
                &mut dgpu_scratch.ffn_moe_recv_stash_t1,
                &ie.xfer,
            )?;
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
            ie.compute.synchronize()?;
            peer_push_f32(
                &igpu_scratch.ffn_moe_t0,
                &mut dgpu_scratch.ffn_moe_recv_stash_t0,
                &ie.xfer,
            )?;
        }
        // Switch back to dGPU device for next launch.
        self.set_current_cached(self.dgpu.device)?;

        Ok(())
    }

    /// post_moe: stages 13 + 15 + 16 inline. Reads `dgpu_scratch.ffn_input_norm`,
    /// `after_attn_hc`, `split`, `residual` (all of which the caller has
    /// already restored from this token's stashes) plus the iGPU MoE output
    /// in `ffn_moe_recv_stash_t<token>`. Writes `residual_next`.
    fn forward_pair_post_moe(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        dlw: &DgpuLayerWeights,
        token: u32,
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

        // ===== Wait for the iGPU MoE for THIS token to arrive =====
        // The peer push from iGPU.xfer landed in ffn_moe_recv_stash_t<token>.
        // Synchronize the iGPU streams to ensure the push has completed.
        // (Simpler than event-based; the post_moes serialize anyway.)
        self.igpu.compute.synchronize()?;
        self.igpu.xfer.synchronize()?;
        // Copy this token's MoE output into the shared ffn_moe_recv that
        // ffn_combine's vec_add reads.
        if token == 0 {
            dgpu_scratch
                .ffn_moe_recv
                .copy_from_buffer(&dgpu_scratch.ffn_moe_recv_stash_t0)?;
        } else {
            dgpu_scratch
                .ffn_moe_recv
                .copy_from_buffer(&dgpu_scratch.ffn_moe_recv_stash_t1)?;
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
