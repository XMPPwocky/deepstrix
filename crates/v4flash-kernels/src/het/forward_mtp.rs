//! MTP draft forward — the 6-stage speculative-token predictor.
//!
//! Mirrors antirez's `metal_graph_eval_mtp_draft_from_hc` in
//! `external/ds4/ds4.c:12884`. Given the last accepted token + the
//! HC state output of the main model's last layer at that token's
//! position, produces logits for the NEXT token (the draft).
//!
//! Stages (all on dGPU compute / iGPU compute streams of the engine):
//!   1. embed(token) into mtp_embed via base.token_embd (F16 → F32)
//!   2. rms_norm(mtp_embed, mtp.enorm) → mtp_enorm
//!   3. q8 matvec(mtp_enorm, mtp.e_proj) → mtp_eproj          [N_EMBD]
//!   4. broadcast mtp_eproj to N_HC rows → mtp_eproj_hc        [HC_DIM]
//!   5. rms_norm per-row(prev_hc, mtp.hnorm) → mtp_hnorm_hc    [HC_DIM]
//!   6. q8 matvec per-row(mtp_hnorm_hc, mtp.h_proj) → mtp_hproj_hc
//!   7. mtp_eproj_hc + mtp_hproj_hc → mtp_input_hc             [HC_DIM]
//!   8. MTP transformer layer (full attn + MoE with Q4_K experts +
//!      ffn_combine), using mtp_input_hc as residual input and
//!      state.mtp.kv_cache as its own raw KV cache → mtp_next_hc
//!   9. MTP head (mtp.norm + mtp.hc_head_* + base.output) → mtp_logits
//!  10. argmax(mtp_logits) → top_id (returned via copy_to_host)
//!
//! No HIP-graph captures — MTP fires at most once per spec_decode
//! round (~5 ms wall in the budget), so the per-launch host enqueue
//! cost is negligible.

use color_eyre::eyre::{self, eyre};
use v4flash_core::gguf::GgufType;
use v4flash_hip::DeviceBuffer;

use crate::forward::{
    EXPERT_WEIGHT_SCALE, BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, GROUP_DIM, HC_DIM, HC_MIX_DIM,
    N_EMBD, N_EXPERT, N_EXPERT_USED, N_FF_EXP, N_FF_SHARED, N_GROUPS, N_HC, N_HEAD, N_HEAD_DIM,
    N_LORA_Q, N_ROT, N_VOCAB, OUT_LOW, Q_FLAT, RANK, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS,
    SWIGLU_CLAMP_EXP, SWA_WINDOW,
};

use super::engine::HeterogeneousEngine;
use super::mtp_weights::MtpWeights;
use super::scratch::{DgpuScratch, IgpuScratch};
use super::state::HetModelState;
use super::sync::{peer_push_f32, peer_push_i32};
use super::weights::HetGlobalWeights;

/// Floor for the router-weight sum, mirroring the host topk path.
const ROUTER_WEIGHT_EPS: f32 = 6.103515625e-5;

impl HeterogeneousEngine {
    /// Run one MTP draft step. Writes the predicted-token argmax to
    /// `out_top_id` and the full logits to `dgpu_scratch.mtp_logits`.
    ///
    /// `prev_hc` is the HC state output by the main model's last layer
    /// at the position of `last_token`. For the standalone bench we
    /// supply this directly from a dump tensor; in real spec decode
    /// the caller copies `dgpu_scratch.residual` (after forward_token's
    /// final epilogue swap) into a stable buffer and passes it here.
    ///
    /// `last_token_embd_host` is the embedding row for `last_token`
    /// from the main model's token_embd. The caller looks it up host-
    /// side (one F16 row of N_EMBD = 8 KB, fast) and passes it in
    /// dequantized to F32. We keep this host-side because (a) we'd
    /// need a new embed_lookup kernel otherwise and (b) it's tiny.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_mtp_draft(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        state: &mut HetModelState,
        main_weights: &HetGlobalWeights,
        mtp_weights: &MtpWeights,
        prev_hc: &DeviceBuffer<f32>,
        last_token_embd_host: &[f32],
        pos: u32,
        token_id: i32,
    ) -> eyre::Result<()> {
        if prev_hc.len() != HC_DIM as usize {
            return Err(eyre!(
                "forward_mtp_draft: prev_hc len {} != HC_DIM {HC_DIM}",
                prev_hc.len()
            ));
        }
        if last_token_embd_host.len() != N_EMBD as usize {
            return Err(eyre!(
                "forward_mtp_draft: embd len {} != N_EMBD {N_EMBD}",
                last_token_embd_host.len()
            ));
        }
        let mtp_state = state
            .mtp
            .as_mut()
            .ok_or_else(|| eyre!("forward_mtp_draft: state.mtp not allocated; call alloc_mtp"))?;

        self.set_current_cached(self.dgpu.device)?;
        let de = &self.dgpu;
        let ie = &self.igpu;

        // ---------- Stages 1-7: HC combine ----------
        // 1. Upload embedded token.
        dgpu_scratch.mtp_embed.copy_from_host(last_token_embd_host)?;

        // 2. rms_norm(mtp_embed, enorm) → mtp_enorm
        de.rms_w.launch_weighted(
            &de.compute,
            &mut dgpu_scratch.mtp_enorm,
            &dgpu_scratch.mtp_embed,
            &mtp_weights.enorm,
            N_EMBD,
            RMS_EPS,
        )?;

        // 3. q8(mtp_enorm) → xq, matvec(e_proj, xq) → mtp_eproj
        //    Reuse xq_n_embd scratch (it's free here since we're not in main forward).
        de.q8.quantize_input(
            &de.compute,
            &mut dgpu_scratch.xq_n_embd,
            &mut dgpu_scratch.xscale_n_embd,
            &dgpu_scratch.mtp_enorm,
            N_EMBD,
        )?;
        de.q8.matvec(
            &de.compute,
            &mut dgpu_scratch.mtp_eproj,
            &mtp_weights.e_proj.buffer,
            &dgpu_scratch.xq_n_embd,
            &dgpu_scratch.xscale_n_embd,
            N_EMBD,
            N_EMBD,
        )?;

        // 4. broadcast mtp_eproj to N_HC rows → mtp_eproj_hc
        de.broadcast
            .launch(&de.compute, &mut dgpu_scratch.mtp_eproj_hc, &dgpu_scratch.mtp_eproj, N_EMBD, N_HC)?;

        // 5. Per-row rms_norm of prev_hc with hnorm. rms_w expects single
        //    row + dim ≤ 4096; loop over N_HC rows. With sub-buffer views
        //    not available, we'd need a multi-row variant or chunk the
        //    work. For prototype: use rms_norm_no_weight on rows (no
        //    scale applied), then multiply by hnorm via vec_mul. But we
        //    don't have vec_mul. Workaround: write to mtp_hnorm_hc row
        //    by row by quantizing each row, etc. Cleanest fix is a
        //    multi-row rms_w variant; for now we use a CPU fallback for
        //    this small (N_HC × N_EMBD = 16K floats) operation.
        //
        //    M40-P2-TODO: write a rms_w_multi_row kernel to keep this on device.
        {
            let mut prev_hc_host = vec![0f32; HC_DIM as usize];
            prev_hc.copy_to_host(&mut prev_hc_host)?;
            let mut hnorm_host = vec![0f32; N_EMBD as usize];
            mtp_weights.hnorm.copy_to_host(&mut hnorm_host)?;
            let mut hnorm_hc_host = vec![0f32; HC_DIM as usize];
            for row in 0..N_HC as usize {
                let row_start = row * N_EMBD as usize;
                let row_in = &prev_hc_host[row_start..row_start + N_EMBD as usize];
                let mut ssq: f64 = 0.0;
                for &v in row_in {
                    ssq += (v as f64) * (v as f64);
                }
                let mean = ssq / (N_EMBD as f64);
                let scale = 1.0 / (mean + RMS_EPS as f64).sqrt();
                for (i, &v) in row_in.iter().enumerate() {
                    hnorm_hc_host[row_start + i] = (v as f64 * scale) as f32 * hnorm_host[i];
                }
            }
            dgpu_scratch.mtp_hnorm_hc.copy_from_host(&hnorm_hc_host)?;
        }

        // 6. Per-row q8 matvec with h_proj → mtp_hproj_hc. Loop N_HC times.
        //    mtp_hnorm_hc is laid out as N_HC rows of N_EMBD. Each row
        //    needs its own quantize + matvec. Use the same xq_n_embd
        //    scratch; copy each row in by host roundtrip (small).
        {
            let mut row_host = vec![0f32; N_EMBD as usize];
            let mut hnorm_full = vec![0f32; HC_DIM as usize];
            dgpu_scratch.mtp_hnorm_hc.copy_to_host(&mut hnorm_full)?;
            let mut hproj_full = vec![0f32; HC_DIM as usize];
            let mut row_dev: DeviceBuffer<f32> =
                DeviceBuffer::new(self.dgpu.device.id, N_EMBD as usize)?;
            let mut row_out_dev: DeviceBuffer<f32> =
                DeviceBuffer::new(self.dgpu.device.id, N_EMBD as usize)?;
            let mut row_out_host = vec![0f32; N_EMBD as usize];
            for row in 0..N_HC as usize {
                let row_start = row * N_EMBD as usize;
                row_host.copy_from_slice(&hnorm_full[row_start..row_start + N_EMBD as usize]);
                row_dev.copy_from_host(&row_host)?;
                de.q8.quantize_input(
                    &de.compute,
                    &mut dgpu_scratch.xq_n_embd,
                    &mut dgpu_scratch.xscale_n_embd,
                    &row_dev,
                    N_EMBD,
                )?;
                de.q8.matvec(
                    &de.compute,
                    &mut row_out_dev,
                    &mtp_weights.h_proj.buffer,
                    &dgpu_scratch.xq_n_embd,
                    &dgpu_scratch.xscale_n_embd,
                    N_EMBD,
                    N_EMBD,
                )?;
                de.compute.synchronize()?;
                row_out_dev.copy_to_host(&mut row_out_host)?;
                hproj_full[row_start..row_start + N_EMBD as usize].copy_from_slice(&row_out_host);
            }
            dgpu_scratch.mtp_hproj_hc.copy_from_host(&hproj_full)?;
        }

        // 7. mtp_input_hc = mtp_eproj_hc + mtp_hproj_hc
        //    vec_add is in-place: a += b. Copy eproj_hc → input_hc first, then add hproj_hc.
        dgpu_scratch
            .mtp_input_hc
            .copy_from_buffer(&dgpu_scratch.mtp_eproj_hc)?;
        de.vec_add.launch(
            &de.compute,
            &mut dgpu_scratch.mtp_input_hc,
            &dgpu_scratch.mtp_hproj_hc,
            HC_DIM,
        )?;

        // ---------- Stage 8: MTP transformer layer ----------
        // Mirrors a main-model forward_layer but with MTP weights and
        // without compressor / indexer. Reads from mtp_input_hc, writes
        // to dgpu_scratch.mtp_next_hc.
        self.forward_mtp_layer(
            dgpu_scratch,
            igpu_scratch,
            mtp_state,
            mtp_weights,
            pos,
            token_id,
        )?;

        // ---------- Stage 9: MTP head ----------
        // mtp_next_hc → mtp_logits, using mtp.norm + mtp.hc_head_* + base.output
        self.forward_mtp_head(dgpu_scratch, main_weights, mtp_weights)?;

        // MTP raw-cache counter advance (cap at SWA_WINDOW).
        mtp_state.n_raw = mtp_state.n_raw.saturating_add(1).min(SWA_WINDOW);

        self.dgpu.compute.synchronize()?;
        Ok(())
    }

    /// Stage 8: MTP transformer layer. Inputs `dgpu_scratch.mtp_input_hc`
    /// as residual; outputs `dgpu_scratch.mtp_next_hc`. Uses MTP weights
    /// + MTP-specific KV cache. NO compressor / indexer. Routed MoE uses
    /// Q4_K kernels.
    #[allow(clippy::too_many_arguments)]
    fn forward_mtp_layer(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        mtp_state: &mut super::state::MtpLayerState,
        mw: &MtpWeights,
        pos: u32,
        token_id: i32,
    ) -> eyre::Result<()> {
        let de = &self.dgpu;
        let ie = &self.igpu;

        // ===== mhc_pre_attn (reads mtp_input_hc, writes attn_input_norm) =====
        de.rms_nw.launch(
            &de.compute,
            &mut dgpu_scratch.flat,
            &dgpu_scratch.mtp_input_hc,
            1,
            HC_DIM,
            RMS_EPS,
        )?;
        de.f16.matvec(
            &de.compute,
            &mut dgpu_scratch.mix,
            &mw.hc_attn_fn.buffer,
            &dgpu_scratch.flat,
            HC_MIX_DIM,
            HC_DIM,
        )?;
        de.hc_sinkhorn.launch(
            &de.compute,
            &mut dgpu_scratch.split,
            &dgpu_scratch.mix,
            &mw.hc_attn_scale,
            &mw.hc_attn_base,
            N_HC,
            SINKHORN_ITERS,
            SINKHORN_EPS,
        )?;
        de.hc_weighted.launch(
            &de.compute,
            &mut dgpu_scratch.attn_cur,
            &dgpu_scratch.mtp_input_hc,
            &dgpu_scratch.split,
            N_EMBD,
            N_HC,
        )?;
        de.rms_w.launch_weighted(
            &de.compute,
            &mut dgpu_scratch.attn_input_norm,
            &dgpu_scratch.attn_cur,
            &mw.attn_norm,
            N_EMBD,
            RMS_EPS,
        )?;

        // ===== Q LoRA chain → q_normed → rope =====
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
            &mw.attn_q_a.buffer,
            &dgpu_scratch.xq_n_embd,
            &dgpu_scratch.xscale_n_embd,
            N_LORA_Q,
            N_EMBD,
        )?;
        de.rms_w.launch_weighted(
            &de.compute,
            &mut dgpu_scratch.qr_normed,
            &dgpu_scratch.qr,
            &mw.q_a_norm,
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
            &mw.attn_q_b.buffer,
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
        de.rope.launch_forward(
            &de.compute,
            &mut dgpu_scratch.q_normed,
            N_HEAD,
            N_HEAD_DIM,
            N_ROT,
            pos,
            &mw.rope_params,
        )?;

        // ===== KV chain + cache append =====
        de.q8.matvec(
            &de.compute,
            &mut dgpu_scratch.kv_raw,
            &mw.attn_kv.buffer,
            &dgpu_scratch.xq_n_embd,
            &dgpu_scratch.xscale_n_embd,
            N_HEAD_DIM,
            N_EMBD,
        )?;
        de.rms_w.launch_weighted(
            &de.compute,
            &mut dgpu_scratch.kv_normed,
            &dgpu_scratch.kv_raw,
            &mw.kv_a_norm,
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
            &mw.rope_params,
        )?;
        de.fp8
            .launch(&de.compute, &mut dgpu_scratch.kv_normed, N_HEAD_DIM - N_ROT)?;
        de.f16rt
            .launch(&de.compute, &mut dgpu_scratch.kv_normed, N_HEAD_DIM)?;
        // Append to MTP's own KV cache (NOT the main-model layer kv_cache).
        de.kv_append.launch(
            &de.compute,
            &mut mtp_state.kv_cache,
            &dgpu_scratch.kv_normed,
            pos,
            SWA_WINDOW,
            N_HEAD_DIM,
        )?;

        // ===== Attention (SWA only — no compressor) =====
        // n_raw is the count of valid rows in MTP's KV cache up to and
        // including the just-appended pos.
        let n_raw = (mtp_state.n_raw + 1).min(SWA_WINDOW);
        de.attn_swa.launch(
            &de.compute,
            &mut dgpu_scratch.heads,
            &dgpu_scratch.q_normed,
            &mtp_state.kv_cache,
            &mw.attn_sinks,
            N_HEAD,
            N_HEAD_DIM,
            n_raw,
        )?;

        // ===== Output projection =====
        de.rope.launch_inverse(
            &de.compute,
            &mut dgpu_scratch.heads,
            N_HEAD,
            N_HEAD_DIM,
            N_ROT,
            pos,
            &mw.rope_params,
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
            &mw.attn_output_a.buffer,
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
            &mw.attn_output_b.buffer,
            &dgpu_scratch.low_xq,
            &dgpu_scratch.low_xscale,
            N_EMBD,
            OUT_LOW,
        )?;

        // ===== mhc_post_attn: blend mtp_input_hc + attn_out via split =====
        de.hc_post.launch_from_split(
            &de.compute,
            &mut dgpu_scratch.after_attn_hc,
            &dgpu_scratch.attn_out,
            &dgpu_scratch.mtp_input_hc,
            &dgpu_scratch.split,
            N_HC,
            N_EMBD,
            N_HC,
        )?;

        // ===== mhc_pre_ffn → ffn_input_norm =====
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
            &mw.hc_ffn_fn.buffer,
            &dgpu_scratch.flat,
            HC_MIX_DIM,
            HC_DIM,
        )?;
        de.hc_sinkhorn.launch(
            &de.compute,
            &mut dgpu_scratch.split,
            &dgpu_scratch.mix,
            &mw.hc_ffn_scale,
            &mw.hc_ffn_base,
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
            &mw.ffn_norm,
            N_EMBD,
            RMS_EPS,
        )?;

        // ===== Router (learned topk) =====
        // f16 matvec (router_logits = ffn_gate_inp @ ffn_input_norm)
        de.f16.matvec(
            &de.compute,
            &mut dgpu_scratch.router_logits,
            &mw.ffn_gate_inp.buffer,
            &dgpu_scratch.ffn_input_norm,
            N_EXPERT,
            N_EMBD,
        )?;
        de.router_topk.launch(
            &de.compute,
            &mut dgpu_scratch.d_selected,
            &mut dgpu_scratch.d_ew,
            &dgpu_scratch.router_logits,
            Some(&mw.router_bias_dev),
            N_EXPERT,
            N_EXPERT_USED as u32,
            EXPERT_WEIGHT_SCALE,
            ROUTER_WEIGHT_EPS,
        )?;

        // Peer-push selected + d_ew + ffn_input_norm to iGPU for MoE.
        peer_push_f32(
            &dgpu_scratch.ffn_input_norm,
            &mut igpu_scratch.ffn_input_norm_recv,
            &de.xfer,
        )?;
        peer_push_i32(&dgpu_scratch.d_selected, &mut igpu_scratch.d_selected, &de.xfer)?;
        peer_push_f32(&dgpu_scratch.d_ew, &mut igpu_scratch.d_ew, &de.xfer)?;

        // ===== Shared expert (dGPU, Q8_0) =====
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
            &mw.ffn_gate_shexp.buffer,
            &dgpu_scratch.xq_n_embd,
            &dgpu_scratch.xscale_n_embd,
            N_FF_SHARED,
            N_EMBD,
        )?;
        de.q8.matvec(
            &de.compute,
            &mut dgpu_scratch.up_sh,
            &mw.ffn_up_shexp.buffer,
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
            &mw.ffn_down_shexp.buffer,
            &dgpu_scratch.mid_sh_xq,
            &dgpu_scratch.mid_sh_xscale,
            N_EMBD,
            N_FF_SHARED,
        )?;

        // ===== Routed MoE on iGPU (Q4_K) =====
        // Synchronize transfers before iGPU reads them.
        self.set_current_cached(self.igpu.device)?;
        ie.q8k.launch(
            &ie.compute,
            &mut igpu_scratch.d_xq_q8k,
            &igpu_scratch.ffn_input_norm_recv,
            BLOCKS_Q8K_GATE_IN,
        )?;
        // gate+up+swiglu (Q4_K) → d_mid_cat [N_EXPERT_USED * N_FF_EXP]
        let gate_bpe = (N_EMBD as u64 * N_FF_EXP as u64
            * crate::q4_k::BLOCK_Q4_K_BYTES as u64
            / 256) as u32;
        let up_bpe = gate_bpe;
        // For MTP we use the same kernel as iq2 (different impl, same shape semantics).
        ie.q4k.launch_pair_swiglu_batched(
            &ie.compute,
            &mut igpu_scratch.d_mid_cat,
            &mw.routed.gate_exps.buffer,
            &mw.routed.up_exps.buffer,
            &igpu_scratch.d_xq_q8k,
            &igpu_scratch.d_ew,
            &igpu_scratch.d_selected,
            gate_bpe,
            up_bpe,
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
        let down_bpe = (N_FF_EXP as u64 * N_EMBD as u64
            * crate::q4_k::BLOCK_Q4_K_BYTES as u64
            / 256) as u32;
        let mid_blocks_bytes = (BLOCKS_Q8K_DOWN_IN as usize)
            * crate::q8_k::BLOCK_Q8_K_BYTES;
        ie.q4k.launch_batched(
            &ie.compute,
            &mut igpu_scratch.ffn_moe,
            &mw.routed.down_exps.buffer,
            &igpu_scratch.d_midq_cat,
            &igpu_scratch.d_selected,
            down_bpe,
            mid_blocks_bytes as u32,
            N_EXPERT_USED as u32,
            N_EMBD,
            BLOCKS_Q8K_DOWN_IN,
        )?;

        // Push ffn_moe back to dGPU.
        peer_push_f32(&igpu_scratch.ffn_moe, &mut dgpu_scratch.ffn_moe_recv, &ie.xfer)?;
        self.set_current_cached(self.dgpu.device)?;
        // Ensure iGPU work + xfer have completed before dGPU consumes ffn_moe_recv.
        ie.compute.synchronize()?;
        ie.xfer.synchronize()?;

        // ===== ffn_combine: ffn_moe_recv += ffn_shared; hc_post → mtp_next_hc =====
        de.vec_add.launch(
            &de.compute,
            &mut dgpu_scratch.ffn_moe_recv,
            &dgpu_scratch.ffn_shared,
            N_EMBD,
        )?;
        de.hc_post.launch_from_split(
            &de.compute,
            &mut dgpu_scratch.mtp_next_hc,
            &dgpu_scratch.ffn_moe_recv,
            &dgpu_scratch.after_attn_hc,
            &dgpu_scratch.split,
            N_HC,
            N_EMBD,
            N_HC,
        )?;

        let _ = token_id; // not used by MTP layer; learned router only
        Ok(())
    }

    /// Stage 9: MTP head. Reads `dgpu_scratch.mtp_next_hc` as the post-
    /// layer HC; writes `dgpu_scratch.mtp_logits`. Uses mtp.norm +
    /// mtp.hc_head_* + main_weights.output (shared vocab projection).
    fn forward_mtp_head(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        main: &HetGlobalWeights,
        mw: &MtpWeights,
    ) -> eyre::Result<()> {
        let de = &self.dgpu;
        // Mirror forward_head, swapping output_hc_* → mtp.hc_head_*
        // and output_norm → mtp.norm. Vocab matvec uses main.output.
        de.rms_nw.launch(
            &de.compute,
            &mut dgpu_scratch.head_flat,
            &dgpu_scratch.mtp_next_hc,
            1,
            HC_DIM,
            RMS_EPS,
        )?;
        de.f16.matvec(
            &de.compute,
            &mut dgpu_scratch.head_pre,
            &mw.hc_head_fn.buffer,
            &dgpu_scratch.head_flat,
            N_HC,
            HC_DIM,
        )?;
        de.hc_sigmoid.launch(
            &de.compute,
            &mut dgpu_scratch.head_w,
            &dgpu_scratch.head_pre,
            &mw.hc_head_scale,
            &mw.hc_head_base,
            N_HC,
        )?;
        de.hc_weighted.launch(
            &de.compute,
            &mut dgpu_scratch.head_embd,
            &dgpu_scratch.mtp_next_hc,
            &dgpu_scratch.head_w,
            N_EMBD,
            N_HC,
        )?;
        de.rms_w.launch_weighted(
            &de.compute,
            &mut dgpu_scratch.head_norm,
            &dgpu_scratch.head_embd,
            &mw.norm,
            N_EMBD,
            RMS_EPS,
        )?;
        de.q8.quantize_input(
            &de.compute,
            &mut dgpu_scratch.head_xq,
            &mut dgpu_scratch.head_xscale,
            &dgpu_scratch.head_norm,
            N_EMBD,
        )?;
        de.q8.matvec(
            &de.compute,
            &mut dgpu_scratch.mtp_logits,
            &main.output.buffer,
            &dgpu_scratch.head_xq,
            &dgpu_scratch.head_xscale,
            N_VOCAB,
            N_EMBD,
        )?;
        Ok(())
    }
}
