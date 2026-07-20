//! MTP draft forward — the speculative-token predictor (restored M40,
//! adapted to the current het engine 2026-07).
//!
//! Given the last accepted token + the HC state output of the main
//! model's last layer at that token's position, produces logits for the
//! NEXT token (the draft). Mirrors antirez's
//! `metal_graph_eval_mtp_draft_from_hc` in `external/ds4/ds4.c`.
//!
//! Stages (dGPU compute / iGPU compute):
//!   1. embed(token) → mtp_embed (host upload of the F16→F32 row)
//!   2. rms_norm(mtp_embed, mtp.enorm) → mtp_enorm
//!   3. q8 matvec(mtp_enorm, mtp.e_proj) → mtp_eproj            [N_EMBD]
//!   4. broadcast mtp_eproj to N_HC rows → mtp_eproj_hc          [HC_DIM]
//!   5. rms_norm per-row(prev_hc, mtp.hnorm) → mtp_hnorm_hc      [HC_DIM]
//!   6. q8 matvec per-row(mtp_hnorm_hc, mtp.h_proj) → mtp_hproj_hc
//!   7. mtp_eproj_hc + mtp_hproj_hc → mtp_input_hc              [HC_DIM]
//!   8. MTP transformer layer (SWA attn + Q4_K MoE) → mtp_next_hc
//!   9. MTP head (mtp.norm + mtp.hc_head_* + main.output) → mtp_logits
//!
//! Scope: DRAFTER ONLY. No verify forward, no accept/reject loop.
//! Additive — does not touch the B=1 decode path.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{Device, DeviceBuffer};

use crate::config::{
    EXPERT_WEIGHT_SCALE, BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, GROUP_DIM, HC_DIM, HC_MIX_DIM,
    N_EMBD, N_EXPERT, N_EXPERT_USED, N_FF_EXP, N_FF_SHARED, N_GROUPS, N_HC, N_HEAD, N_HEAD_DIM,
    N_LORA_Q, N_ROT, N_VOCAB, OUT_LOW, Q_FLAT, RANK, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS,
    SWIGLU_CLAMP_EXP, SWA_WINDOW,
};

use super::engine::HeterogeneousEngine;
use super::mtp_weights::MtpWeights;
use super::scratch::{DgpuScratch, IgpuScratch};
use super::state::MtpLayerState;
use super::sync::{peer_push_f32, peer_push_i32};
use super::weights::HetGlobalWeights;

/// Floor for the router-weight sum, mirroring the host topk path.
const ROUTER_WEIGHT_EPS: f32 = 6.103515625e-5;

/// dGPU-resident scratch specific to the MTP drafter. The MTP layer
/// otherwise reuses the main [`DgpuScratch`] fields (flat/mix/split,
/// attention setup, shared-expert, router, head) — only the HC-combine
/// intermediates and the MTP logits need their own buffers.
pub struct MtpScratch {
    pub mtp_embed: DeviceBuffer<f32>,      // [N_EMBD]
    pub mtp_enorm: DeviceBuffer<f32>,      // [N_EMBD]
    pub mtp_eproj: DeviceBuffer<f32>,      // [N_EMBD]
    pub mtp_eproj_hc: DeviceBuffer<f32>,   // [HC_DIM]
    pub mtp_hnorm_hc: DeviceBuffer<f32>,   // [HC_DIM]
    pub mtp_hproj_hc: DeviceBuffer<f32>,   // [HC_DIM]
    pub mtp_input_hc: DeviceBuffer<f32>,   // [HC_DIM]
    pub mtp_next_hc: DeviceBuffer<f32>,    // [HC_DIM]
    pub mtp_logits: DeviceBuffer<f32>,     // [N_VOCAB]
    /// Per-row temporaries for the host-assisted stage-6 h_proj matvec.
    row_dev: DeviceBuffer<f32>,            // [N_EMBD]
    row_out_dev: DeviceBuffer<f32>,        // [N_EMBD]
}

impl MtpScratch {
    pub fn alloc(dgpu_device: Device) -> eyre::Result<Self> {
        dgpu_device.set_current()?;
        let id = dgpu_device.id;
        Ok(Self {
            mtp_embed: DeviceBuffer::new(id, N_EMBD as usize)?,
            mtp_enorm: DeviceBuffer::new(id, N_EMBD as usize)?,
            mtp_eproj: DeviceBuffer::new(id, N_EMBD as usize)?,
            mtp_eproj_hc: DeviceBuffer::new(id, HC_DIM as usize)?,
            mtp_hnorm_hc: DeviceBuffer::new(id, HC_DIM as usize)?,
            mtp_hproj_hc: DeviceBuffer::new(id, HC_DIM as usize)?,
            mtp_input_hc: DeviceBuffer::new(id, HC_DIM as usize)?,
            mtp_next_hc: DeviceBuffer::new(id, HC_DIM as usize)?,
            mtp_logits: DeviceBuffer::new(id, N_VOCAB as usize)?,
            row_dev: DeviceBuffer::new(id, N_EMBD as usize)?,
            row_out_dev: DeviceBuffer::new(id, N_EMBD as usize)?,
        })
    }
}

impl HeterogeneousEngine {
    /// Run one MTP draft step. Writes the full logits to
    /// `mtp_scratch.mtp_logits`; the caller does argmax / sampling.
    ///
    /// `prev_hc` is the HC state output by the main model's last layer at
    /// the position of `last_token` (`HC_DIM` floats, dGPU-resident).
    /// `last_token_embd_host` is the dequantized F32 embedding row for
    /// `last_token` (looked up host-side from the main model's token_embd).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_mtp_draft(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        mtp_scratch: &mut MtpScratch,
        mtp_state: &mut MtpLayerState,
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

        self.set_current_cached(self.dgpu.device)?;
        let de = &self.dgpu;

        // ---------- Stages 1-7: HC combine ----------
        // 1. Upload embedded token.
        mtp_scratch.mtp_embed.copy_from_host(last_token_embd_host)?;

        // 2. rms_norm(mtp_embed, enorm) → mtp_enorm
        de.rms_w.launch_weighted(
            &de.compute,
            &mut mtp_scratch.mtp_enorm,
            &mtp_scratch.mtp_embed,
            &mtp_weights.enorm,
            N_EMBD,
            RMS_EPS,
        )?;

        // 3. q8(mtp_enorm) → xq, matvec(e_proj) → mtp_eproj
        de.q8.quantize_input(
            &de.compute,
            &mut dgpu_scratch.xq_n_embd,
            &mut dgpu_scratch.xscale_n_embd,
            &mtp_scratch.mtp_enorm,
            N_EMBD,
        )?;
        de.q8.matvec(
            &de.compute,
            &mut mtp_scratch.mtp_eproj,
            &mtp_weights.e_proj.buffer,
            &dgpu_scratch.xq_n_embd,
            &dgpu_scratch.xscale_n_embd,
            N_EMBD,
            N_EMBD,
        )?;

        // 4. broadcast mtp_eproj to N_HC rows → mtp_eproj_hc (host-side;
        //    N_HC × N_EMBD = 16K floats, negligible for a rare MTP call).
        de.compute.synchronize()?;
        {
            let mut eproj_host = vec![0f32; N_EMBD as usize];
            mtp_scratch.mtp_eproj.copy_to_host(&mut eproj_host)?;
            let mut eproj_hc_host = vec![0f32; HC_DIM as usize];
            for row in 0..N_HC as usize {
                let base = row * N_EMBD as usize;
                eproj_hc_host[base..base + N_EMBD as usize].copy_from_slice(&eproj_host);
            }
            mtp_scratch.mtp_eproj_hc.copy_from_host(&eproj_hc_host)?;
        }

        // 5. Per-row rms_norm of prev_hc with hnorm (host fallback — a
        //    multi-row rms_w kernel would keep this on device; deferred).
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
            mtp_scratch.mtp_hnorm_hc.copy_from_host(&hnorm_hc_host)?;
        }

        // 6. Per-row q8 matvec with h_proj → mtp_hproj_hc. Loop N_HC times.
        {
            let mut row_host = vec![0f32; N_EMBD as usize];
            let mut hnorm_full = vec![0f32; HC_DIM as usize];
            mtp_scratch.mtp_hnorm_hc.copy_to_host(&mut hnorm_full)?;
            let mut hproj_full = vec![0f32; HC_DIM as usize];
            let mut row_out_host = vec![0f32; N_EMBD as usize];
            for row in 0..N_HC as usize {
                let row_start = row * N_EMBD as usize;
                row_host.copy_from_slice(&hnorm_full[row_start..row_start + N_EMBD as usize]);
                mtp_scratch.row_dev.copy_from_host(&row_host)?;
                de.q8.quantize_input(
                    &de.compute,
                    &mut dgpu_scratch.xq_n_embd,
                    &mut dgpu_scratch.xscale_n_embd,
                    &mtp_scratch.row_dev,
                    N_EMBD,
                )?;
                de.q8.matvec(
                    &de.compute,
                    &mut mtp_scratch.row_out_dev,
                    &mtp_weights.h_proj.buffer,
                    &dgpu_scratch.xq_n_embd,
                    &dgpu_scratch.xscale_n_embd,
                    N_EMBD,
                    N_EMBD,
                )?;
                de.compute.synchronize()?;
                mtp_scratch.row_out_dev.copy_to_host(&mut row_out_host)?;
                hproj_full[row_start..row_start + N_EMBD as usize].copy_from_slice(&row_out_host);
            }
            mtp_scratch.mtp_hproj_hc.copy_from_host(&hproj_full)?;
        }

        // 7. mtp_input_hc = mtp_eproj_hc + mtp_hproj_hc
        mtp_scratch
            .mtp_input_hc
            .copy_from_buffer(&mtp_scratch.mtp_eproj_hc)?;
        de.vec_add.launch(
            &de.compute,
            &mut mtp_scratch.mtp_input_hc,
            &mtp_scratch.mtp_hproj_hc,
            HC_DIM,
        )?;

        // ---------- Stage 8: MTP transformer layer ----------
        self.forward_mtp_layer(
            dgpu_scratch,
            igpu_scratch,
            mtp_scratch,
            mtp_state,
            mtp_weights,
            pos,
            token_id,
        )?;

        // ---------- Stage 9: MTP head ----------
        self.forward_mtp_head(dgpu_scratch, mtp_scratch, main_weights, mtp_weights)?;

        // MTP raw-cache counter advance (cap at SWA_WINDOW).
        mtp_state.n_raw = (mtp_state.n_raw + 1).min(SWA_WINDOW);

        self.set_current_cached(self.dgpu.device)?;
        self.dgpu.compute.synchronize()?;
        Ok(())
    }

    /// Stage 8: MTP transformer layer. Reads `mtp_scratch.mtp_input_hc` as
    /// residual; writes `mtp_scratch.mtp_next_hc`. SWA attention only (no
    /// compressor / indexer); routed MoE uses Q4_K kernels on the iGPU.
    #[allow(clippy::too_many_arguments)]
    fn forward_mtp_layer(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        mtp_scratch: &mut MtpScratch,
        mtp_state: &mut MtpLayerState,
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
            &mtp_scratch.mtp_input_hc,
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
            &mtp_scratch.mtp_input_hc,
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
        // Append to MTP's own KV cache at slot n_raw (window not full for a
        // single cold draft); the append stores f32→f16 internally.
        let slot = mtp_state.n_raw.min(SWA_WINDOW - 1);
        de.kv_append.launch(
            &de.compute,
            &mut mtp_state.kv_cache,
            &dgpu_scratch.kv_normed,
            slot,
            SWA_WINDOW,
            N_HEAD_DIM,
        )?;

        // ===== Attention (SWA only — no compressor) =====
        let n_kv = (mtp_state.n_raw + 1).min(SWA_WINDOW);
        let kv_win = mtp_state
            .kv_cache
            .slice_view(0, (n_kv as usize) * (N_HEAD_DIM as usize));
        de.attn_swa.launch(
            &de.compute,
            &mut dgpu_scratch.heads,
            &dgpu_scratch.q_normed,
            &kv_win,
            &mw.attn_sinks,
            N_HEAD,
            N_HEAD_DIM,
            n_kv,
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
            &mtp_scratch.mtp_input_hc,
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

        // Ensure ffn_input_norm + selected/d_ew are computed before we peer-
        // push them; MTP fires rarely so a full sync here is fine.
        de.compute.synchronize()?;
        peer_push_f32(
            &dgpu_scratch.ffn_input_norm,
            &mut igpu_scratch.ffn_input_norm_recv,
            &de.xfer,
        )?;
        peer_push_i32(&dgpu_scratch.d_selected, &mut igpu_scratch.d_selected, &de.xfer)?;
        peer_push_f32(&dgpu_scratch.d_ew, &mut igpu_scratch.d_ew, &de.xfer)?;
        de.xfer.synchronize()?;

        // ===== Routed MoE on iGPU (Q4_K) =====
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
        let mid_blocks_bytes =
            (BLOCKS_Q8K_DOWN_IN as usize) * crate::q8_k::BLOCK_Q8_K_BYTES;
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
        ie.compute.synchronize()?;
        peer_push_f32(&igpu_scratch.ffn_moe, &mut dgpu_scratch.ffn_moe_recv, &ie.xfer)?;
        ie.xfer.synchronize()?;
        self.set_current_cached(self.dgpu.device)?;

        // ===== ffn_combine: ffn_moe_recv += ffn_shared; hc_post → mtp_next_hc =====
        de.vec_add.launch(
            &de.compute,
            &mut dgpu_scratch.ffn_moe_recv,
            &dgpu_scratch.ffn_shared,
            N_EMBD,
        )?;
        de.hc_post.launch_from_split(
            &de.compute,
            &mut mtp_scratch.mtp_next_hc,
            &dgpu_scratch.ffn_moe_recv,
            &dgpu_scratch.after_attn_hc,
            &dgpu_scratch.split,
            N_HC,
            N_EMBD,
            N_HC,
        )?;

        let _ = token_id; // MTP uses the learned router only
        Ok(())
    }

    /// Stage 9: MTP head. Reads `mtp_scratch.mtp_next_hc`; writes
    /// `mtp_scratch.mtp_logits`. Uses mtp.norm + mtp.hc_head_* + the main
    /// model's shared vocab projection (`main.output`).
    fn forward_mtp_head(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        mtp_scratch: &mut MtpScratch,
        main: &HetGlobalWeights,
        mw: &MtpWeights,
    ) -> eyre::Result<()> {
        let de = &self.dgpu;
        de.rms_nw.launch(
            &de.compute,
            &mut dgpu_scratch.head_flat,
            &mtp_scratch.mtp_next_hc,
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
            &mtp_scratch.mtp_next_hc,
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
            &mut mtp_scratch.mtp_logits,
            &main.output.buffer,
            &dgpu_scratch.head_xq,
            &dgpu_scratch.head_xscale,
            N_VOCAB,
            N_EMBD,
        )?;
        Ok(())
    }
}
