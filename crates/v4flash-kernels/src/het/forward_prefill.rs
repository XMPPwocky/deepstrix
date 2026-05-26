//! M50: layer-major batched prefill.
//!
//! ## Phase 1 — looped single-token kernels (`forward_prompt_batch`)
//!
//! Calls the existing `forward_layer` once per batch element, per layer.
//! Functionally equivalent to N sequential `forward_token` calls but
//! reorganized to dispatch layer-major. No batched kernels; no perf win.
//! Validates that the layer-major dispatch + state evolution is correct.
//!
//! ## Phase 2 — real batched kernels (`forward_prompt_batch_v2`)
//!
//! Uses [`BatchDgpuScratch`] (B-extended per-token buffers) and the
//! `*_batched` kernel wrappers. Stateless big matmuls and HC stages run
//! in single B-wide launches; stateful kernels (rope, KV append,
//! compressor, attention, iGPU MoE) stay in a serial inner loop using
//! `DeviceBuffer::slice_view` per batch element.
//!
//! ## Layer-major vs token-major
//!
//! Both produce identical state (per-layer KV cache + compressor are
//! commutative across batch elements only because layer N's per-position
//! state for token b at position pos0+b is only ever written by that
//! one call). Layer-major lets batched kernels amortize per-layer
//! weight reads across the batch.

use color_eyre::eyre::{self, eyre};

use crate::forward::{
    hash_router_select, BLOCKS_N_EMBD, BLOCKS_N_FF_SHARED, BLOCKS_N_LORA_Q, BLOCKS_OUT_LOW,
    EXPERT_WEIGHT_SCALE, GROUP_DIM, HC_DIM, HC_MIX_DIM, N_EMBD, N_EXPERT, N_EXPERT_USED,
    N_FF_SHARED, N_GROUPS, N_HC, N_HEAD, N_HEAD_DIM, N_LAYER, N_LORA_Q, N_ROT, OUT_LOW, Q_FLAT,
    RANK, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS, SWA_WINDOW,
};

use super::batch_scratch::{BatchDgpuScratch, BatchScratch};
use super::engine::HeterogeneousEngine;
use super::scratch::IgpuScratch;
use super::state::{HetLayerState, HetModelState};
use super::sync::{peer_push_f32, peer_push_i32};
use super::weights::{DgpuLayerWeights, HetModelWeights, IgpuLayerWeights};

const ROUTER_WEIGHT_EPS: f32 = 6.103515625e-5;

impl HeterogeneousEngine {
    /// Run a layer-major prefill over `B` tokens starting at `pos0`.
    ///
    /// `input_hcs[i]` is the layer-0 input HC for token `i`
    /// (broadcast of `embed(tokens[i])` to HC_DIM).
    /// `tokens[i]` is the token id at position `pos0 + i` (used by the
    /// hash router on bootstrap layers).
    ///
    /// Modifies: per-layer KV cache + compressor state in `state` for
    /// positions `pos0..pos0+B`. After return, `scratches.dgpu[b]
    /// .residual_next` holds the post-last-layer HC for token `b`.
    ///
    /// Does NOT compute logits / head — caller decides whether to run
    /// `forward_head` on the last batch element (typical prefill) or
    /// on every element (prompt-eval).
    pub fn forward_prompt_batch(
        &self,
        scratches: &mut BatchScratch,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        input_hcs: &[Vec<f32>],
        tokens: &[i32],
        pos0: u32,
    ) -> eyre::Result<()> {
        let b = tokens.len();
        if b == 0 {
            return Ok(());
        }
        if b > scratches.b_max() {
            return Err(eyre!(
                "forward_prompt_batch: B={b} exceeds B_MAX={}",
                scratches.b_max()
            ));
        }
        if input_hcs.len() != b {
            return Err(eyre!(
                "forward_prompt_batch: input_hcs len {} != tokens len {b}",
                input_hcs.len()
            ));
        }
        for (i, hc) in input_hcs.iter().enumerate() {
            if hc.len() != HC_DIM as usize {
                return Err(eyre!(
                    "forward_prompt_batch: input_hcs[{i}] len {} != HC_DIM {}",
                    hc.len(),
                    HC_DIM
                ));
            }
        }

        // Invalidate the engine's cached current_device — BatchScratch::alloc
        // may have left the driver pointing at iGPU after its last
        // IgpuScratch alloc. set_current_cached would skip the switch if
        // it still thinks we're on dGPU. Forcing -1 makes the next
        // set_current_cached actually call set_current.
        self.current_device
            .store(-1, std::sync::atomic::Ordering::Relaxed);
        self.set_current_cached(self.dgpu.device)?;

        // 1. Seed each token's per-token residual buffer with its
        //    layer-0 input HC.
        for i in 0..b {
            scratches.per_token_residual[i].copy_from_host(&input_hcs[i])?;
        }
        // residual_next per-token will hold the per-layer output after
        // we move into the layer loop. Initial value is don't-care.

        // 2. Layer-major dispatch: for each layer, for each token,
        //    swap the per-token residual into the SHARED scratch,
        //    run forward_layer, then swap residual_next out.
        //
        //    Why a shared scratch: forward_layer captures sub-blocks
        //    into per-layer HIP graphs that bake in buffer pointers.
        //    Replaying with a different scratch's pointers gives garbage
        //    output (or kernel errors). So Phase 1 reuses ONE scratch
        //    and pays the per-token-residual copy cost on every layer.
        //
        //    Use `forward_layer_pair_mode` to disable the M30 combined
        //    ffn_combine+next_mhc_pre_attn graph. The combined graph
        //    assumes the NEXT layer's mhc_pre_attn input is in the
        //    SAME scratch — which holds in single-token decode (one
        //    scratch flows through all layers) but NOT in batched
        //    prefill (token A's layer-N mhc_pre_attn output would be
        //    in shared_scratch, but at token B's call we've already
        //    moved on to a different residual). Standalone graphs per
        //    layer are correct.
        for layer in 0..N_LAYER as usize {
            for i in 0..b {
                let pos = pos0 + i as u32;
                // Move token i's residual into shared scratch.
                scratches
                    .shared_dgpu
                    .residual
                    .copy_from_buffer(&scratches.per_token_residual[i])?;
                self.forward_layer_pair_mode(
                    &mut scratches.shared_dgpu,
                    &mut scratches.shared_igpu,
                    &mut state.layers[layer],
                    &weights.dgpu_layers[layer],
                    &weights.igpu_layers[layer],
                    pos,
                    tokens[i],
                )?;
                // Move shared.residual_next out to token i's residual_next.
                scratches.per_token_residual_next[i]
                    .copy_from_buffer(&scratches.shared_dgpu.residual_next)?;
                // Per-token swap so layer N+1 reads from per_token_residual[i]
                // (= layer N's output).
                std::mem::swap(
                    &mut scratches.per_token_residual[i],
                    &mut scratches.per_token_residual_next[i],
                );
            }
        }

        // 3. Drain any pending async work before the epilogue swap.
        //    copy_from_buffer + the kernel writes may be queued; we
        //    need them all to land before the swap (which is CPU-side)
        //    is meaningful for any subsequent readback.
        self.dgpu.compute.synchronize()?;

        // 4. Epilogue swap per-token to restore parity (mirrors
        //    `forward_token`'s post-head swap). After 43 layers (odd)
        //    + this one extra swap = 44 total swaps, so
        //    `per_token_residual_next[i]` holds the post-last-layer HC.
        for i in 0..b {
            std::mem::swap(
                &mut scratches.per_token_residual[i],
                &mut scratches.per_token_residual_next[i],
            );
        }
        Ok(())
    }

    /// M50 Phase 2: layer-major batched prefill using batched kernels.
    ///
    /// Stateless big matmuls + HC stages run in single B-wide kernel
    /// launches against `batch_dgpu` (B-extended contiguous buffers).
    /// Stateful per-token kernels (rope, kv_append, compressor, attn,
    /// iGPU MoE) loop in a serial inner B loop using `slice_view`.
    ///
    /// After return, `batch_dgpu.residual` (or `residual_next` if the
    /// model layer count is odd — V4-Flash has 43, so `residual` after
    /// post-loop swap holds it) contains per-token post-last-layer HC.
    pub fn forward_prompt_batch_v2(
        &self,
        batch_dgpu: &mut BatchDgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        input_hcs: &[Vec<f32>],
        tokens: &[i32],
        pos0: u32,
    ) -> eyre::Result<()> {
        let b = tokens.len();
        if b == 0 {
            return Ok(());
        }
        if input_hcs.len() != b {
            return Err(eyre!(
                "forward_prompt_batch_v2: input_hcs len {} != tokens len {b}",
                input_hcs.len()
            ));
        }
        for (i, hc) in input_hcs.iter().enumerate() {
            if hc.len() != HC_DIM as usize {
                return Err(eyre!(
                    "forward_prompt_batch_v2: input_hcs[{i}] len {} != HC_DIM {}",
                    hc.len(),
                    HC_DIM
                ));
            }
        }

        self.current_device
            .store(-1, std::sync::atomic::Ordering::Relaxed);
        self.set_current_cached(self.dgpu.device)?;

        // 1. Seed per-token residual buffers in `batch_dgpu.residual`.
        //    `residual` is laid out [B, HC_DIM] contiguous. Each token's
        //    input HC is copied into its slot.
        for i in 0..b {
            let mut slot = batch_dgpu
                .residual
                .slice_view_mut(i * HC_DIM as usize, HC_DIM as usize);
            slot.copy_from_host(&input_hcs[i])?;
        }

        // 2. Layer loop: invoke forward_layer_batch_v2 once per layer.
        //    Each call swaps residual / residual_next internally (we do
        //    the swap here for clarity, mirroring forward_token's per-
        //    layer swap).
        for layer in 0..N_LAYER as usize {
            self.forward_layer_batch_v2(
                batch_dgpu,
                igpu_scratch,
                &mut state.layers[layer],
                &weights.dgpu_layers[layer],
                &weights.igpu_layers[layer],
                pos0,
                tokens,
            )?;
            // Swap residual / residual_next for the next layer: the
            // layer wrote residual_next; next layer reads residual.
            std::mem::swap(&mut batch_dgpu.residual, &mut batch_dgpu.residual_next);
        }

        // Drain any pending async work.
        self.dgpu.compute.synchronize()?;
        Ok(())
    }
}

impl HeterogeneousEngine {
    /// M50 Phase 2: one layer of batched prefill. Reads
    /// `batch_dgpu.residual` (per-token input HC), writes
    /// `batch_dgpu.residual_next` (per-token output HC). All other
    /// `batch_dgpu` fields are scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_layer_batch_v2(
        &self,
        bd: &mut BatchDgpuScratch,
        ig: &mut IgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        ilw: &IgpuLayerWeights,
        pos0: u32,
        tokens: &[i32],
    ) -> eyre::Result<()> {
        let layer = dlw.layer_idx;
        if ilw.layer_idx != layer {
            return Err(eyre!(
                "forward_layer_batch_v2: dgpu L{} != igpu L{}",
                layer,
                ilw.layer_idx
            ));
        }
        let ratio = dlw.ratio;
        let b = tokens.len() as u32;
        if b == 0 {
            return Ok(());
        }

        self.set_current_cached(self.dgpu.device)?;
        let de = &self.dgpu;
        let cs_n_embd = N_EMBD as usize;
        let cs_qflat = Q_FLAT as usize;
        let cs_kvhd = N_HEAD_DIM as usize;
        let cs_n_used = N_EXPERT_USED;

        // ========================================================
        // Stage 1: mhc_pre_attn (BATCHED)
        // rms_nw → f16_narrow → sinkhorn → hc_weighted → rms_w
        // ========================================================
        de.rms_nw
            .launch_batched(&de.compute, &mut bd.flat, &bd.residual, 1, HC_DIM, RMS_EPS, b)?;
        de.f16.matvec_narrow_batched(
            &de.compute,
            &mut bd.mix,
            &dlw.hc_attn_fn.buffer,
            &bd.flat,
            HC_MIX_DIM,
            HC_DIM,
            b,
        )?;
        de.hc_sinkhorn.launch_batched(
            &de.compute,
            &mut bd.split,
            &bd.mix,
            &dlw.hc_attn_scale,
            &dlw.hc_attn_base,
            N_HC,
            SINKHORN_ITERS,
            SINKHORN_EPS,
            b,
        )?;
        de.hc_weighted.launch_batched(
            &de.compute,
            &mut bd.attn_cur,
            &bd.residual,
            &bd.split,
            N_EMBD,
            N_HC,
            b,
        )?;
        de.rms_w.launch_weighted_batched(
            &de.compute,
            &mut bd.attn_input_norm,
            &bd.attn_cur,
            &dlw.attn_norm,
            N_EMBD,
            RMS_EPS,
            b,
        )?;

        // ========================================================
        // Stage 2: Q chain (BATCHED quantize + matvec + rms + ...)
        // ========================================================
        de.q8.quantize_input_batched(
            &de.compute,
            &mut bd.xq_n_embd,
            &mut bd.xscale_n_embd,
            &bd.attn_input_norm,
            N_EMBD,
            b,
        )?;
        de.q8.matvec_batched(
            &de.compute,
            &mut bd.qr,
            &dlw.attn_q_a.buffer,
            &bd.xq_n_embd,
            &bd.xscale_n_embd,
            N_LORA_Q,
            N_EMBD,
            b,
        )?;
        de.rms_w.launch_weighted_batched(
            &de.compute,
            &mut bd.qr_normed,
            &bd.qr,
            &dlw.q_a_norm,
            N_LORA_Q,
            RMS_EPS,
            b,
        )?;
        de.q8.quantize_input_batched(
            &de.compute,
            &mut bd.qr_xq,
            &mut bd.qr_xscale,
            &bd.qr_normed,
            N_LORA_Q,
            b,
        )?;
        de.q8.matvec_batched(
            &de.compute,
            &mut bd.q,
            &dlw.attn_q_b.buffer,
            &bd.qr_xq,
            &bd.qr_xscale,
            Q_FLAT,
            N_LORA_Q,
            b,
        )?;
        // rms_nw over batch: each batch has [N_HEAD, N_HEAD_DIM] rows.
        // batched API: grid (B, N_HEAD, 1), inner row of N_HEAD_DIM.
        de.rms_nw.launch_batched(
            &de.compute,
            &mut bd.q_normed,
            &bd.q,
            N_HEAD,
            N_HEAD_DIM,
            RMS_EPS,
            b,
        )?;
        // Per-token rope on q_normed (serial — pos differs per token).
        for i in 0..b as usize {
            let mut q_view = bd.q_normed.slice_view_mut(i * cs_qflat, cs_qflat);
            de.rope.launch_forward(
                &de.compute,
                &mut q_view,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                pos0 + i as u32,
                &dlw.rope_params,
            )?;
        }

        // ========================================================
        // Stage 3: KV chain (BATCHED matvec + rms; per-token rope/fp8/f16rt)
        // ========================================================
        de.q8.matvec_batched(
            &de.compute,
            &mut bd.kv_raw,
            &dlw.attn_kv.buffer,
            &bd.xq_n_embd,
            &bd.xscale_n_embd,
            N_HEAD_DIM,
            N_EMBD,
            b,
        )?;
        de.rms_w.launch_weighted_batched(
            &de.compute,
            &mut bd.kv_normed,
            &bd.kv_raw,
            &dlw.kv_a_norm,
            N_HEAD_DIM,
            RMS_EPS,
            b,
        )?;
        for i in 0..b as usize {
            let mut kv_view = bd.kv_normed.slice_view_mut(i * cs_kvhd, cs_kvhd);
            de.rope.launch_forward(
                &de.compute,
                &mut kv_view,
                1,
                N_HEAD_DIM,
                N_ROT,
                pos0 + i as u32,
                &dlw.rope_params,
            )?;
            de.fp8
                .launch(&de.compute, &mut kv_view, N_HEAD_DIM - N_ROT)?;
            de.f16rt.launch(&de.compute, &mut kv_view, N_HEAD_DIM)?;
        }

        // ========================================================
        // Stage 4: KV cache append + compressor (SERIAL per batch)
        // ========================================================
        for i in 0..b as usize {
            let pos = pos0 + i as u32;
            let kv_view = bd.kv_normed.slice_view(i * cs_kvhd, cs_kvhd);
            de.kv_append.launch(
                &de.compute,
                &mut ls.kv_cache,
                &kv_view,
                ls.n_raw,
                SWA_WINDOW,
                N_HEAD_DIM,
            )?;
            if ls.n_raw < SWA_WINDOW {
                ls.n_raw += 1;
            }
            if ratio > 0 {
                let cw = dlw
                    .compressor
                    .as_ref()
                    .ok_or_else(|| eyre!("L{layer}: missing compressor weights"))?;
                let comp_width = cw.width;
                let pos_mod = pos % ratio;
                let row = if ratio == 4 { 4 + pos_mod } else { pos_mod };
                let mut kv_cur_v = bd.kv_cur.slice_view_mut(i * 2 * cs_kvhd, 2 * cs_kvhd);
                let mut sc_cur_v = bd.sc_cur.slice_view_mut(i * 2 * cs_kvhd, 2 * cs_kvhd);
                let attn_norm_v = bd
                    .attn_input_norm
                    .slice_view(i * cs_n_embd, cs_n_embd);
                de.f16.matvec_pair(
                    &de.compute,
                    &mut kv_cur_v,
                    &mut sc_cur_v,
                    &cw.wkv.buffer,
                    &cw.wgate.buffer,
                    &attn_norm_v,
                    comp_width,
                    N_EMBD,
                )?;
                let cs = ls
                    .compressor
                    .as_mut()
                    .ok_or_else(|| eyre!("L{layer}: missing compressor state"))?;
                de.compressor_state_write.launch(
                    &de.compute,
                    &mut cs.state_kv,
                    &mut cs.state_score,
                    &kv_cur_v,
                    &sc_cur_v,
                    &cw.ape.buffer,
                    comp_width,
                    row,
                    pos_mod,
                )?;
                let comp_fires = (pos + 1) % ratio == 0;
                if comp_fires {
                    let mut pooled_v = bd.pooled.slice_view_mut(i * cs_kvhd, cs_kvhd);
                    let mut comp_row_v = bd.comp_row.slice_view_mut(i * cs_kvhd, cs_kvhd);
                    de.compressor_pool.launch(
                        &de.compute,
                        &mut pooled_v,
                        &cs.state_kv,
                        &cs.state_score,
                        N_HEAD_DIM,
                        ratio,
                    )?;
                    de.rms_w.launch_weighted(
                        &de.compute,
                        &mut comp_row_v,
                        &pooled_v,
                        &cw.norm,
                        N_HEAD_DIM,
                        RMS_EPS,
                    )?;
                    let comp_pos = pos + 1 - ratio;
                    de.rope.launch_forward(
                        &de.compute,
                        &mut comp_row_v,
                        1,
                        N_HEAD_DIM,
                        N_ROT,
                        comp_pos,
                        &dlw.rope_params,
                    )?;
                    de.fp8
                        .launch(&de.compute, &mut comp_row_v, N_HEAD_DIM - N_ROT)?;
                    de.f16rt
                        .launch(&de.compute, &mut comp_row_v, N_HEAD_DIM)?;
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
                        &comp_row_v,
                        cs.n_comp,
                        N_HEAD_DIM,
                    )?;
                    cs.n_comp += 1;
                }
            }
        }

        // ========================================================
        // Stage 5: Attention (SERIAL per batch — Phase 4 batches this)
        // ========================================================
        for i in 0..b as usize {
            let q_view = bd.q_normed.slice_view(i * cs_qflat, cs_qflat);
            let mut heads_view = bd.heads.slice_view_mut(i * cs_qflat, cs_qflat);
            if ratio == 0 {
                de.attn_swa.launch(
                    &de.compute,
                    &mut heads_view,
                    &q_view,
                    &ls.kv_cache,
                    &dlw.attn_sinks,
                    N_HEAD,
                    N_HEAD_DIM,
                    ls.n_raw,
                )?;
            } else {
                let cs = ls.compressor.as_ref();
                let n_comp = cs.map(|c| c.n_comp).unwrap_or(0);
                let comp_kv_buf = if n_comp > 0 { cs.map(|c| &c.comp_kv) } else { None };
                de.attn_mixed.launch(
                    &de.compute,
                    &mut heads_view,
                    &q_view,
                    &ls.kv_cache,
                    comp_kv_buf,
                    None,
                    &dlw.attn_sinks,
                    N_HEAD,
                    N_HEAD_DIM,
                    ls.n_raw,
                    n_comp,
                )?;
            }
        }

        // ========================================================
        // Stage 6: Output projection (rope_inv per b, then BATCHED q8)
        // ========================================================
        for i in 0..b as usize {
            let mut h_view = bd.heads.slice_view_mut(i * cs_qflat, cs_qflat);
            de.rope.launch_inverse(
                &de.compute,
                &mut h_view,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                pos0 + i as u32,
                &dlw.rope_params,
            )?;
        }
        de.q8.quantize_input_batched(
            &de.compute,
            &mut bd.heads_xq,
            &mut bd.heads_xscale,
            &bd.heads,
            Q_FLAT,
            b,
        )?;
        de.q8_grouped.matvec_grouped_batched(
            &de.compute,
            &mut bd.low,
            &dlw.attn_output_a.buffer,
            &bd.heads_xq,
            &bd.heads_xscale,
            GROUP_DIM,
            RANK,
            N_GROUPS,
            b,
        )?;
        de.q8.quantize_input_batched(
            &de.compute,
            &mut bd.low_xq,
            &mut bd.low_xscale,
            &bd.low,
            OUT_LOW,
            b,
        )?;
        de.q8.matvec_batched(
            &de.compute,
            &mut bd.attn_out,
            &dlw.attn_output_b.buffer,
            &bd.low_xq,
            &bd.low_xscale,
            N_EMBD,
            OUT_LOW,
            b,
        )?;

        // ========================================================
        // Stage 7: mhc_post_attn (BATCHED hc_post_from_split)
        // ========================================================
        de.hc_post.launch_from_split_batched(
            &de.compute,
            &mut bd.after_attn_hc,
            &bd.attn_out,
            &bd.residual,
            &bd.split,
            N_HC, // n_w (matches single-token path)
            N_EMBD,
            N_HC,
            b,
        )?;

        // ========================================================
        // Stage 8: mhc_pre_ffn (BATCHED, same shape as Stage 1)
        // ========================================================
        de.rms_nw.launch_batched(
            &de.compute,
            &mut bd.flat,
            &bd.after_attn_hc,
            1,
            HC_DIM,
            RMS_EPS,
            b,
        )?;
        de.f16.matvec_narrow_batched(
            &de.compute,
            &mut bd.mix,
            &dlw.hc_ffn_fn.buffer,
            &bd.flat,
            HC_MIX_DIM,
            HC_DIM,
            b,
        )?;
        de.hc_sinkhorn.launch_batched(
            &de.compute,
            &mut bd.split,
            &bd.mix,
            &dlw.hc_ffn_scale,
            &dlw.hc_ffn_base,
            N_HC,
            SINKHORN_ITERS,
            SINKHORN_EPS,
            b,
        )?;
        de.hc_weighted.launch_batched(
            &de.compute,
            &mut bd.ffn_cur,
            &bd.after_attn_hc,
            &bd.split,
            N_EMBD,
            N_HC,
            b,
        )?;
        de.rms_w.launch_weighted_batched(
            &de.compute,
            &mut bd.ffn_input_norm,
            &bd.ffn_cur,
            &dlw.ffn_norm,
            N_EMBD,
            RMS_EPS,
            b,
        )?;

        // ========================================================
        // Stage 9: Router (BATCHED matvec; per-batch topk OR hash select)
        // ========================================================
        de.f16.matvec_narrow_batched(
            &de.compute,
            &mut bd.router_logits,
            &dlw.ffn_gate_inp.buffer,
            &bd.ffn_input_norm,
            N_EXPERT,
            N_EMBD,
            b,
        )?;
        if !dlw.is_hash_router {
            for i in 0..b as usize {
                let logits_v = bd
                    .router_logits
                    .slice_view(i * (N_EXPERT as usize), N_EXPERT as usize);
                let mut sel_v = bd.d_selected.slice_view_mut(i * cs_n_used, cs_n_used);
                let mut ew_v = bd.d_ew.slice_view_mut(i * cs_n_used, cs_n_used);
                de.router_topk.launch(
                    &de.compute,
                    &mut sel_v,
                    &mut ew_v,
                    &logits_v,
                    dlw.router_bias_dev.as_ref(),
                    N_EXPERT,
                    cs_n_used as u32,
                    EXPERT_WEIGHT_SCALE,
                    ROUTER_WEIGHT_EPS,
                )?;
            }
        } else {
            // Hash router: readback all B × N_EXPERT logits, run host
            // select per batch element, upload d_selected + d_ew.
            de.compute.synchronize()?;
            bd.router_logits
                .copy_to_host(&mut bd.router_logits_host)?;
            let tid2eid = dlw
                .tid2eid
                .as_ref()
                .ok_or_else(|| eyre!("L{layer}: hash router but no tid2eid"))?;
            let mut all_sel: Vec<i32> = Vec::with_capacity(b as usize * cs_n_used);
            let mut all_ew: Vec<f32> = Vec::with_capacity(b as usize * cs_n_used);
            for i in 0..b as usize {
                let logit_slice = &bd.router_logits_host
                    [i * (N_EXPERT as usize)..(i + 1) * (N_EXPERT as usize)];
                let (sel, w) = hash_router_select(tid2eid, tokens[i], logit_slice);
                all_sel.extend_from_slice(&sel);
                all_ew.extend_from_slice(&w);
            }
            // d_selected / d_ew are B_MAX-sized; copy into [0..B*N_USED] view.
            let mut sel_v = bd
                .d_selected
                .slice_view_mut(0, b as usize * cs_n_used);
            sel_v.copy_from_host(&all_sel)?;
            let mut ew_v = bd.d_ew.slice_view_mut(0, b as usize * cs_n_used);
            ew_v.copy_from_host(&all_ew)?;
        }

        // ========================================================
        // Stage 10: Shared expert (BATCHED Q8_0 chains)
        // swiglu + vec_add are pure elementwise → stretch n by B
        // ========================================================
        de.q8.quantize_input_batched(
            &de.compute,
            &mut bd.xq_n_embd,
            &mut bd.xscale_n_embd,
            &bd.ffn_input_norm,
            N_EMBD,
            b,
        )?;
        de.q8.matvec_batched(
            &de.compute,
            &mut bd.gate_sh,
            &dlw.shared.gate.buffer,
            &bd.xq_n_embd,
            &bd.xscale_n_embd,
            N_FF_SHARED,
            N_EMBD,
            b,
        )?;
        de.q8.matvec_batched(
            &de.compute,
            &mut bd.up_sh,
            &dlw.shared.up.buffer,
            &bd.xq_n_embd,
            &bd.xscale_n_embd,
            N_FF_SHARED,
            N_EMBD,
            b,
        )?;
        // swiglu — elementwise; stretch n to B * N_FF_SHARED.
        de.swiglu.launch(
            &de.compute,
            &mut bd.mid_sh,
            &bd.gate_sh,
            &bd.up_sh,
            b * N_FF_SHARED,
        )?;
        de.q8.quantize_input_batched(
            &de.compute,
            &mut bd.mid_sh_xq,
            &mut bd.mid_sh_xscale,
            &bd.mid_sh,
            N_FF_SHARED,
            b,
        )?;
        de.q8.matvec_batched(
            &de.compute,
            &mut bd.ffn_shared,
            &dlw.shared.down.buffer,
            &bd.mid_sh_xq,
            &bd.mid_sh_xscale,
            N_EMBD,
            N_FF_SHARED,
            b,
        )?;

        // ========================================================
        // Stage 11: iGPU routed MoE (SERIAL per batch — Phase 3 batches this)
        //
        // For each batch element: peer-push the per-batch slice of
        // ffn_input_norm + d_selected + d_ew, run the existing single-
        // token iGPU MoE 4-kernel pipeline (uses captured graph),
        // peer-push ffn_moe back into the per-batch slot.
        // ========================================================
        let gbpe = ilw.routed.gate_bytes_per_expert as u32;
        let ubpe = ilw.routed.up_bytes_per_expert as u32;
        let dbpe = ilw.routed.down_bytes_per_expert as u32;
        let mid_blocks_bytes = (crate::forward::BLOCKS_Q8K_DOWN_IN as usize)
            * crate::q8_k::BLOCK_Q8_K_BYTES;
        for i in 0..b as usize {
            let ain_v = bd.ffn_input_norm.slice_view(i * cs_n_embd, cs_n_embd);
            let dsel_v = bd.d_selected.slice_view(i * cs_n_used, cs_n_used);
            let dew_v = bd.d_ew.slice_view(i * cs_n_used, cs_n_used);

            self.set_current_cached(self.dgpu.device)?;
            peer_push_f32(&ain_v, &mut ig.ffn_input_norm_recv, &de.xfer)?;
            peer_push_i32(&dsel_v, &mut ig.d_selected, &de.xfer)?;
            peer_push_f32(&dew_v, &mut ig.d_ew, &de.xfer)?;
            de.xfer.synchronize()?;

            self.set_current_cached(self.igpu.device)?;
            let ie = &self.igpu;
            // Replay the captured iGPU MoE graph if already captured
            // (single-token path captures it on first call); otherwise
            // capture here. We use the same shared scratch every call
            // so pointers in the capture remain valid.
            {
                let graph_slot = &self.igpu_moe_graphs[layer as usize];
                let mut guard = graph_slot.lock().unwrap();
                if guard.is_none() {
                    ie.compute.begin_capture(
                        v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                    )?;
                    ie.q8k.launch(
                        &ie.compute,
                        &mut ig.d_xq_q8k,
                        &ig.ffn_input_norm_recv,
                        crate::forward::BLOCKS_Q8K_GATE_IN,
                    )?;
                    ie.iq2.launch_fused_swiglu_batch(
                        &ie.compute,
                        &mut ig.d_mid_cat,
                        &ilw.routed.gate.buffer,
                        &ilw.routed.up.buffer,
                        &ig.d_xq_q8k,
                        &ig.d_ew,
                        &ig.d_selected,
                        gbpe,
                        ubpe,
                        cs_n_used as u32,
                        crate::forward::SWIGLU_CLAMP_EXP,
                        crate::forward::N_FF_EXP,
                        crate::forward::BLOCKS_Q8K_GATE_IN,
                    )?;
                    ie.q8k.launch(
                        &ie.compute,
                        &mut ig.d_midq_cat,
                        &ig.d_mid_cat,
                        crate::forward::BLOCKS_Q8K_DOWN_IN * (cs_n_used as u32),
                    )?;
                    ie.q2k.launch_batched(
                        &ie.compute,
                        &mut ig.ffn_moe,
                        &ilw.routed.down.buffer,
                        &ig.d_midq_cat,
                        &ig.d_selected,
                        dbpe,
                        mid_blocks_bytes as u32,
                        cs_n_used as u32,
                        N_EMBD,
                        crate::forward::BLOCKS_Q8K_DOWN_IN,
                    )?;
                    let graph = ie.compute.end_capture()?;
                    let exec = graph.instantiate()?;
                    exec.launch(&ie.compute)?;
                    *guard = Some(exec);
                } else {
                    guard.as_ref().unwrap().launch(&ie.compute)?;
                }
            }
            ie.compute.synchronize()?;

            // Push ffn_moe back to dGPU's per-batch slot.
            let mut moe_dst = bd.ffn_moe_recv.slice_view_mut(i * cs_n_embd, cs_n_embd);
            peer_push_f32(&ig.ffn_moe, &mut moe_dst, &ie.xfer)?;
            ie.xfer.synchronize()?;
        }
        self.set_current_cached(self.dgpu.device)?;

        // ========================================================
        // Stage 12: ffn_combine (vec_add stretched + BATCHED hc_post_from_split)
        // ========================================================
        de.vec_add.launch(
            &de.compute,
            &mut bd.ffn_moe_recv,
            &bd.ffn_shared,
            b * N_EMBD,
        )?;
        de.hc_post.launch_from_split_batched(
            &de.compute,
            &mut bd.residual_next,
            &bd.ffn_moe_recv,
            &bd.after_attn_hc,
            &bd.split,
            N_HC,
            N_EMBD,
            N_HC,
            b,
        )?;
        Ok(())
    }
}
