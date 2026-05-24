//! Per-layer dispatch for the het orchestrator.
//!
//! Pipeline (M13.1 — serial; events come in M13.4):
//!
//! 1. **dGPU**: mHC pre-attn → attn_cur → attn_input_norm
//! 2. **dGPU**: Q chain + KV chain
//! 3. **dGPU**: KV cache push (FP8 + F16-roundtrip + SWA slide)
//! 4. **dGPU** (ratio>0): compressor step + boundary fire → comp_kv
//! 5. **dGPU**: attention (swa or mixed) → heads → attn_out
//! 6. **dGPU**: mHC post-attn → after_attn_hc
//! 7. **dGPU**: mHC pre-ffn → ffn_cur → ffn_input_norm
//! 8. **dGPU→iGPU**: peer push ffn_input_norm
//! 9. **iGPU**: router (matvec + host topk) + routed MoE → ffn_moe
//! 10. **iGPU→dGPU**: peer push ffn_moe → ffn_moe_recv
//! 11. **dGPU**: shared expert → ffn_shared
//! 12. **dGPU**: ffn_moe_recv += ffn_shared (vec_add)
//! 13. **dGPU**: mHC post-ffn → residual_next

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{Device, DeviceBuffer};

use crate::forward::{
    hash_router_select, BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, EXPERT_WEIGHT_SCALE, GROUP_DIM,
    HC_DIM, HC_MIX_DIM, N_EMBD, N_EXPERT, N_EXPERT_USED, N_FF_EXP, N_FF_SHARED, N_GROUPS, N_HC,
    N_HEAD, N_HEAD_DIM, N_LORA_Q, N_ROT, OUT_LOW, Q_FLAT, RANK, RMS_EPS, SINKHORN_EPS,
    SINKHORN_ITERS, SWA_WINDOW, SWIGLU_CLAMP_EXP,
};
use crate::q8_k::BLOCK_Q8_K_BYTES;

/// Floor for the router-weight sum, mirroring the host topk path (f16
/// epsilon).
const ROUTER_WEIGHT_EPS: f32 = 6.103515625e-5;

use super::engine::{DeviceEngine, ExecMode, HeterogeneousEngine};
use super::scratch::{DgpuScratch, IgpuScratch};
use super::state::HetLayerState;
use super::sync::peer_push_f32;
use super::weights::{DgpuLayerWeights, IgpuLayerWeights};
use tracing::debug_span;

impl HeterogeneousEngine {
    /// Run one layer in the het pipeline. Reads from
    /// `dgpu_scratch.residual` and writes to `dgpu_scratch.residual_next`.
    pub fn forward_layer(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        ilw: &IgpuLayerWeights,
        pos: u32,
        token_id: i32,
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

        let serial = matches!(self.mode, ExecMode::HetSingleStream);

        let _layer_span = debug_span!("het.layer", layer, pos).entered();

        // ============================================================
        // dGPU: mHC pre attn → attn_cur → attn_input_norm
        // ============================================================
        self.dgpu.device.set_current()?;
        let de = &self.dgpu;

        let _t_mhc_pre = de.events.stage("dgpu.mhc_pre_attn", &de.compute)?;
        let _s_mhc_pre = debug_span!("mhc_pre_attn").entered();
        {
            let _t = de.events.stage("k.mhc_pre_attn.rms_nw", &de.compute)?;
            de.rms_nw.launch(
                &de.compute,
                &mut dgpu_scratch.flat,
                &dgpu_scratch.residual,
                1,
                HC_DIM,
                RMS_EPS,
            )?;
        }
        {
            let _t = de.events.stage("k.mhc_pre_attn.f16_matvec", &de.compute)?;
            de.f16.matvec(
                &de.compute,
                &mut dgpu_scratch.mix,
                &dlw.hc_attn_fn.buffer,
                &dgpu_scratch.flat,
                HC_MIX_DIM,
                HC_DIM,
            )?;
        }
        {
            let _t = de.events.stage("k.mhc_pre_attn.sinkhorn", &de.compute)?;
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
        }
        {
            let _t = de.events.stage("k.mhc_pre_attn.hc_weighted", &de.compute)?;
            de.hc_weighted.launch(
                &de.compute,
                &mut dgpu_scratch.attn_cur,
                &dgpu_scratch.residual,
                &dgpu_scratch.split,
                N_EMBD,
                N_HC,
            )?;
        }
        {
            let _t = de.events.stage("k.mhc_pre_attn.rms_w", &de.compute)?;
            de.rms_w.launch_weighted(
                &de.compute,
                &mut dgpu_scratch.attn_input_norm,
                &dgpu_scratch.attn_cur,
                &dlw.attn_norm,
                N_EMBD,
                RMS_EPS,
            )?;
        }
        drop(_s_mhc_pre);
        _t_mhc_pre.end()?;

        // ============================================================
        // dGPU: Q LoRA chain → q_post_rope
        // ============================================================
        let _t_q = de.events.stage("dgpu.q_chain", &de.compute)?;
        let _s_q = debug_span!("q_chain").entered();
        {
            let _t = de.events.stage("k.q_chain.q8_quantize_in", &de.compute)?;
            de.q8.quantize_input(
                &de.compute,
                &mut dgpu_scratch.xq_n_embd,
                &mut dgpu_scratch.xscale_n_embd,
                &dgpu_scratch.attn_input_norm,
                N_EMBD,
            )?;
        }
        {
            let _t = de.events.stage("k.q_chain.q8_matvec_qa", &de.compute)?;
            de.q8.matvec(
                &de.compute,
                &mut dgpu_scratch.qr,
                &dlw.attn_q_a.buffer,
                &dgpu_scratch.xq_n_embd,
                &dgpu_scratch.xscale_n_embd,
                N_LORA_Q,
                N_EMBD,
            )?;
        }
        {
            let _t = de.events.stage("k.q_chain.rms_w_qa", &de.compute)?;
            de.rms_w.launch_weighted(
                &de.compute,
                &mut dgpu_scratch.qr_normed,
                &dgpu_scratch.qr,
                &dlw.q_a_norm,
                N_LORA_Q,
                RMS_EPS,
            )?;
        }
        {
            let _t = de.events.stage("k.q_chain.q8_quantize_qr", &de.compute)?;
            de.q8.quantize_input(
                &de.compute,
                &mut dgpu_scratch.qr_xq,
                &mut dgpu_scratch.qr_xscale,
                &dgpu_scratch.qr_normed,
                N_LORA_Q,
            )?;
        }
        {
            let _t = de.events.stage("k.q_chain.q8_matvec_qb", &de.compute)?;
            de.q8.matvec(
                &de.compute,
                &mut dgpu_scratch.q,
                &dlw.attn_q_b.buffer,
                &dgpu_scratch.qr_xq,
                &dgpu_scratch.qr_xscale,
                Q_FLAT,
                N_LORA_Q,
            )?;
        }
        {
            let _t = de.events.stage("k.q_chain.rms_nw_heads", &de.compute)?;
            de.rms_nw.launch(
                &de.compute,
                &mut dgpu_scratch.q_normed,
                &dgpu_scratch.q,
                N_HEAD,
                N_HEAD_DIM,
                RMS_EPS,
            )?;
        }
        {
            let _t = de.events.stage("k.q_chain.rope_fwd", &de.compute)?;
            de.rope.launch_forward(
                &de.compute,
                &mut dgpu_scratch.q_normed,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                pos,
                &dlw.rope_params,
            )?;
        }
        drop(_s_q);
        _t_q.end()?;

        // ============================================================
        // dGPU: KV chain → kv_post_rope → KV cache push
        // ============================================================
        let _t_kv = de.events.stage("dgpu.kv_chain", &de.compute)?;
        let _s_kv = debug_span!("kv_chain").entered();
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

        // M13.3: device-side append + SWA slide. No sync, no host copies.
        de.kv_append.launch(
            &de.compute,
            &mut ls.kv_cache,
            &dgpu_scratch.kv_normed,
            ls.n_raw,
            SWA_WINDOW,
            N_HEAD_DIM,
        )?;
        if ls.n_raw < SWA_WINDOW {
            ls.n_raw += 1;
        }
        drop(_s_kv);
        _t_kv.end()?;

        // ============================================================
        // M13.5: Compressor migrated to iGPU.
        //
        // The dGPU already has `attn_input_norm` computed (above). We
        // peer-push it to the iGPU, run the compressor matvecs +
        // state-write there in parallel with the dGPU's Q/KV chain
        // work that follows, and on boundary tokens stream `comp_row`
        // back to the dGPU's `comp_kv` cache via a tiny peer-push +
        // `comp_kv_append` kernel.
        // ============================================================
        let comp_fires_boundary = ratio > 0 && (pos + 1) % ratio == 0;
        let parallel_pre = matches!(self.mode, ExecMode::HetParallel);
        let sev_pre = &self.sync_events.layers[layer as usize];

        if ratio > 0 {
            if parallel_pre {
                // Record the upstream event first, then bracket the wait
                // separately from the work so the perfetto timeline shows
                // wait time as its own slice instead of inflating the
                // peer_push stage with idle stream time.
                sev_pre.attn_in_ready.record(&de.compute)?;
                let _t_wait = de
                    .events
                    .stage("dgpu.peer_push_attn_input_norm.wait", &de.xfer)?;
                de.xfer.wait_event(&sev_pre.attn_in_ready)?;
                _t_wait.end()?;
            } else {
                de.compute.synchronize()?;
            }
            let _t_peer_ain_pre = de.events.stage(
                "dgpu.peer_push_attn_input_norm",
                &de.xfer,
            )?;
            let _s_peer_ain_pre = debug_span!("peer_push_attn_input_norm").entered();
            peer_push_f32(
                &dgpu_scratch.attn_input_norm,
                &mut igpu_scratch.attn_input_norm_recv,
                &de.xfer,
            )?;
            if parallel_pre {
                sev_pre.attn_in_pushed.record(&de.xfer)?;
            } else {
                de.xfer.synchronize()?;
            }
            drop(_s_peer_ain_pre);
            _t_peer_ain_pre.end()?;

            // iGPU compressor.
            self.igpu.device.set_current()?;
            let ie_c = &self.igpu;
            if parallel_pre {
                let _t_wait =
                    ie_c.events.stage("igpu.compressor.wait", &ie_c.compute)?;
                ie_c.compute.wait_event(&sev_pre.attn_in_pushed)?;
                _t_wait.end()?;
            }
            let _t_comp = ie_c.events.stage("igpu.compressor", &ie_c.compute)?;
            let _s_comp = debug_span!("compressor", ratio).entered();

            let cw = ilw
                .compressor
                .as_ref()
                .ok_or_else(|| eyre!("L{layer}: missing compressor weights (iGPU)"))?;
            let comp_width = cw.width;
            let pos_mod = pos % ratio;
            let row = if ratio == 4 { 4 + pos_mod } else { pos_mod };
            {
                // M14h: paired matvec — kv + gate fused into single launch,
                // sharing activation reads.
                let _t = ie_c.events.stage("k.compressor.f16_pair", &ie_c.compute)?;
                ie_c.f16.matvec_pair(
                    &ie_c.compute,
                    &mut igpu_scratch.kv_cur,
                    &mut igpu_scratch.sc_cur,
                    &cw.wkv.buffer,
                    &cw.wgate.buffer,
                    &igpu_scratch.attn_input_norm_recv,
                    comp_width,
                    N_EMBD,
                )?;
            }
            let cs = ls
                .compressor
                .as_mut()
                .ok_or_else(|| eyre!("L{layer}: missing compressor state"))?;
            {
                let _t = ie_c.events.stage("k.compressor.state_write", &ie_c.compute)?;
                ie_c.compressor_state_write.launch(
                    &ie_c.compute,
                    &mut cs.state_kv,
                    &mut cs.state_score,
                    &igpu_scratch.kv_cur,
                    &igpu_scratch.sc_cur,
                    &cw.ape.buffer,
                    comp_width,
                    row,
                    pos_mod,
                )?;
            }

            if comp_fires_boundary {
                ie_c.compressor_pool.launch(
                    &ie_c.compute,
                    &mut igpu_scratch.pooled,
                    &cs.state_kv,
                    &cs.state_score,
                    N_HEAD_DIM,
                    ratio,
                )?;
                ie_c.rms_w.launch_weighted(
                    &ie_c.compute,
                    &mut igpu_scratch.comp_row,
                    &igpu_scratch.pooled,
                    &cw.norm,
                    N_HEAD_DIM,
                    RMS_EPS,
                )?;
                let comp_pos = pos + 1 - ratio;
                ie_c.rope.launch_forward(
                    &ie_c.compute,
                    &mut igpu_scratch.comp_row,
                    1,
                    N_HEAD_DIM,
                    N_ROT,
                    comp_pos,
                    &ilw.rope_params,
                )?;
                ie_c.fp8.launch(
                    &ie_c.compute,
                    &mut igpu_scratch.comp_row,
                    N_HEAD_DIM - N_ROT,
                )?;
                ie_c.f16rt.launch(
                    &ie_c.compute,
                    &mut igpu_scratch.comp_row,
                    N_HEAD_DIM,
                )?;
                if ratio == 4 {
                    ie_c.compressor_shuffle.launch(
                        &ie_c.compute,
                        &mut cs.state_kv,
                        &mut cs.state_score,
                        comp_width,
                    )?;
                }

                // Peer push comp_row back to dGPU.
                if parallel_pre {
                    sev_pre.comp_row_ready.record(&ie_c.compute)?;
                    let _t_wait =
                        ie_c.events.stage("igpu.peer_push_comp_row.wait", &ie_c.xfer)?;
                    ie_c.xfer.wait_event(&sev_pre.comp_row_ready)?;
                    _t_wait.end()?;
                } else {
                    ie_c.compute.synchronize()?;
                }
                let _t_push = ie_c.events.stage("igpu.peer_push_comp_row", &ie_c.xfer)?;
                peer_push_f32(
                    &igpu_scratch.comp_row,
                    &mut dgpu_scratch.comp_row_recv,
                    &ie_c.xfer,
                )?;
                if parallel_pre {
                    sev_pre.comp_row_arrived.record(&ie_c.xfer)?;
                } else {
                    ie_c.xfer.synchronize()?;
                }
                _t_push.end()?;
            }
            drop(_s_comp);
            _t_comp.end()?;
            self.dgpu.device.set_current()?;

            // dGPU: append comp_row to comp_kv on boundary.
            if comp_fires_boundary {
                if parallel_pre {
                    let _t_wait = de
                        .events
                        .stage("dgpu.comp_kv_append.wait", &de.compute)?;
                    de.compute.wait_event(&sev_pre.comp_row_arrived)?;
                    _t_wait.end()?;
                }
                let _t_append = de.events.stage("dgpu.comp_kv_append", &de.compute)?;
                let cs = ls
                    .compressor
                    .as_mut()
                    .ok_or_else(|| eyre!("L{layer}: missing compressor state"))?;
                de.comp_kv_append.launch(
                    &de.compute,
                    &mut cs.comp_kv,
                    &dgpu_scratch.comp_row_recv,
                    cs.n_comp,
                    N_HEAD_DIM,
                )?;
                cs.n_comp += 1;
                _t_append.end()?;
            }
        }

        // ============================================================
        // dGPU: Attention compute
        // ============================================================
        let _t_attn = de.events.stage("dgpu.attn_compute", &de.compute)?;
        let _s_attn = debug_span!("attn_compute").entered();
        if ratio == 0 {
            de.attn_swa.launch(
                &de.compute,
                &mut dgpu_scratch.heads,
                &dgpu_scratch.q_normed,
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
                &mut dgpu_scratch.heads,
                &dgpu_scratch.q_normed,
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
        drop(_s_attn);
        _t_attn.end()?;

        // ============================================================
        // dGPU: Output projection
        // ============================================================
        let _t_out = de.events.stage("dgpu.output_proj", &de.compute)?;
        let _s_out = debug_span!("output_proj").entered();
        {
            let _t = de.events.stage("k.output_proj.rope_inv", &de.compute)?;
            de.rope.launch_inverse(
                &de.compute,
                &mut dgpu_scratch.heads,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                pos,
                &dlw.rope_params,
            )?;
        }
        {
            let _t = de.events.stage("k.output_proj.q8_quantize_heads", &de.compute)?;
            de.q8.quantize_input(
                &de.compute,
                &mut dgpu_scratch.heads_xq,
                &mut dgpu_scratch.heads_xscale,
                &dgpu_scratch.heads,
                Q_FLAT,
            )?;
        }
        {
            let _t = de.events.stage("k.output_proj.q8_grouped_a", &de.compute)?;
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
        }
        {
            let _t = de.events.stage("k.output_proj.q8_quantize_low", &de.compute)?;
            de.q8.quantize_input(
                &de.compute,
                &mut dgpu_scratch.low_xq,
                &mut dgpu_scratch.low_xscale,
                &dgpu_scratch.low,
                OUT_LOW,
            )?;
        }
        {
            let _t = de.events.stage("k.output_proj.q8_matvec_b", &de.compute)?;
            de.q8.matvec(
                &de.compute,
                &mut dgpu_scratch.attn_out,
                &dlw.attn_output_b.buffer,
                &dgpu_scratch.low_xq,
                &dgpu_scratch.low_xscale,
                N_EMBD,
                OUT_LOW,
            )?;
        }
        drop(_s_out);
        _t_out.end()?;

        // ============================================================
        // dGPU: mHC post attn → after_attn_hc (M13.3: hc_post reads
        // post + comb directly from the packed `split` buffer; no host
        // roundtrip)
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
        let _t_mhc_pre_ffn = de.events.stage("dgpu.mhc_pre_ffn", &de.compute)?;
        let _s_mhc_pre_ffn = debug_span!("mhc_pre_ffn").entered();
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
        // M13.3: split layout [w(4), post(4), comb(16)] stays on device.
        // The mhc_post_ffn step below reads post/comb directly from it.
        de.rms_w.launch_weighted(
            &de.compute,
            &mut dgpu_scratch.ffn_input_norm,
            &dgpu_scratch.ffn_cur,
            &dlw.ffn_norm,
            N_EMBD,
            RMS_EPS,
        )?;
        drop(_s_mhc_pre_ffn);
        _t_mhc_pre_ffn.end()?;

        // ============================================================
        // dGPU → iGPU: peer push ffn_input_norm (16 KB)
        //
        // HetSingleStream: synchronize → push → synchronize.
        // HetParallel: record event on compute, xfer stream waits and
        //   queues push, records "pushed" event for iGPU to wait on.
        // ============================================================
        let parallel = matches!(self.mode, ExecMode::HetParallel);
        let sev = &self.sync_events.layers[layer as usize];

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

        // ============================================================
        // iGPU: router → topk → selected (HOST READBACK REQUIRED)
        // ============================================================
        self.igpu.device.set_current()?;
        let ie = &self.igpu;

        if parallel {
            let _t_wait = ie.events.stage("igpu.router.wait", &ie.compute)?;
            ie.compute.wait_event(&sev.ain_pushed)?;
            _t_wait.end()?;
        }
        let _t_router = ie.events.stage("igpu.router", &ie.compute)?;
        let _s_router = debug_span!("router").entered();
        {
            let _t = ie.events.stage("k.router.f16_matvec", &ie.compute)?;
            ie.f16.matvec(
                &ie.compute,
                &mut igpu_scratch.router_logits,
                &ilw.ffn_gate_inp.buffer,
                &igpu_scratch.ffn_input_norm_recv,
                N_EXPERT,
                N_EMBD,
            )?;
        }
        if !ilw.is_hash_router {
            let _t = ie.events.stage("k.router.topk", &ie.compute)?;
            ie.router_topk.launch(
                &ie.compute,
                &mut igpu_scratch.d_selected,
                &mut igpu_scratch.d_ew,
                &igpu_scratch.router_logits,
                ilw.router_bias_dev.as_ref(),
                N_EXPERT,
                N_EXPERT_USED as u32,
                EXPERT_WEIGHT_SCALE,
                ROUTER_WEIGHT_EPS,
            )?;
        }
        drop(_s_router);
        _t_router.end()?;

        // M13.4 KEY REORDER: queue dGPU shared-expert kernels NOW —
        // BEFORE we host-block on iGPU compute for the topk readback.
        // The kernels are enqueued on de.compute (which is asynchronous
        // from the host's perspective), so dGPU starts executing the
        // shared expert in parallel with the iGPU MoE pipeline. This
        // is where the M13 overlap actually materializes.
        if parallel {
            self.dgpu.device.set_current()?;
            let _t_shared = de.events.stage("dgpu.shared_expert", &de.compute)?;
            let _s_shared = debug_span!("shared_expert").entered();
            issue_shared_expert(de, dgpu_scratch, dlw)?;
            drop(_s_shared);
            _t_shared.end()?;
            self.igpu.device.set_current()?;
        }

        // Now host-block on iGPU compute to get selected[].
        let selected: [i32; N_EXPERT_USED] = if ilw.is_hash_router {
            ie.compute.synchronize()?;
            igpu_scratch
                .router_logits
                .copy_to_host(&mut igpu_scratch.router_logits_host)?;
            let tid2eid = ilw
                .tid2eid
                .as_ref()
                .ok_or_else(|| eyre!("L{layer}: hash router but no tid2eid"))?;
            let (sel, w) =
                hash_router_select(tid2eid, token_id, &igpu_scratch.router_logits_host);
            igpu_scratch.d_ew.copy_from_host(&w)?;
            sel
        } else {
            ie.compute.synchronize()?;
            igpu_scratch
                .d_selected
                .copy_to_host(&mut igpu_scratch.host_selected)?;
            let mut arr = [0i32; N_EXPERT_USED];
            arr.copy_from_slice(&igpu_scratch.host_selected[..N_EXPERT_USED]);
            arr
        };

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
        // d_midq_cat staging buffer. Eliminates ~600μs/layer of host
        // round-trips (M13.4 inner-loop refactor).
        let _t_moe = ie.events.stage("igpu.routed_moe", &ie.compute)?;
        let _s_moe = debug_span!("routed_moe").entered();
        {
            let _t = ie.events.stage("k.moe.q8k_xq", &ie.compute)?;
            ie.q8k.launch(
                &ie.compute,
                &mut igpu_scratch.d_xq_q8k,
                &igpu_scratch.ffn_input_norm_recv,
                BLOCKS_Q8K_GATE_IN,
            )?;
        }
        // M14j: single batched iq2_fused launch handles all N_EXPERT_USED
        // slots via grid.y, reading expert indices from d_selected. For
        // the hash router we have to push the host-computed selection to
        // d_selected first (learned router already wrote it device-side).
        if ilw.is_hash_router {
            igpu_scratch.d_selected.copy_from_host(&selected)?;
        }
        {
            let _t = ie.events.stage("k.moe.iq2_fused_batch", &ie.compute)?;
            ie.iq2.launch_fused_swiglu_batch(
                &ie.compute,
                &mut igpu_scratch.d_mid_cat,
                &ilw.routed.gate.buffer,
                &ilw.routed.up.buffer,
                &igpu_scratch.d_xq_q8k,
                &igpu_scratch.d_ew,
                &igpu_scratch.d_selected,
                gbpe as u32,
                ubpe as u32,
                N_EXPERT_USED as u32,
                SWIGLU_CLAMP_EXP,
                N_FF_EXP,
                BLOCKS_Q8K_GATE_IN,
            )?;
        }
        let mid_blocks_bytes = (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8_K_BYTES;
        for slot in 0..N_EXPERT_USED {
            {
                let _t = ie.events.stage("k.moe.q8k_mid", &ie.compute)?;
                ie.q8k.launch_with_offsets(
                    &ie.compute,
                    &mut igpu_scratch.d_midq_cat,
                    slot * mid_blocks_bytes,
                    &igpu_scratch.d_mid_cat,
                    slot * (N_FF_EXP as usize),
                    BLOCKS_Q8K_DOWN_IN,
                )?;
            }
            let e = selected[slot] as usize;
            let _t = ie.events.stage("k.moe.q2k_down", &ie.compute)?;
            ie.q2k.launch_with_full_offsets(
                &ie.compute,
                &mut igpu_scratch.ffn_moe,
                &ilw.routed.down.buffer,
                e * dbpe,
                &igpu_scratch.d_midq_cat,
                slot * mid_blocks_bytes,
                N_EMBD,
                BLOCKS_Q8K_DOWN_IN,
                slot == 0,
            )?;
        }
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

        // ============================================================
        // dGPU: in serial mode, issue shared expert NOW. In parallel
        // mode, it was issued earlier — just wait for ffn_moe to arrive.
        // ============================================================
        self.dgpu.device.set_current()?;
        if !parallel {
            let _t_shared = de.events.stage("dgpu.shared_expert", &de.compute)?;
            let _s_shared = debug_span!("shared_expert").entered();
            issue_shared_expert(de, dgpu_scratch, dlw)?;
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
        let _t_combine = de.events.stage("dgpu.ffn_combine", &de.compute)?;
        let _s_combine = debug_span!("ffn_combine").entered();
        de.vec_add.launch(
            &de.compute,
            &mut dgpu_scratch.ffn_moe_recv,
            &dgpu_scratch.ffn_shared,
            N_EMBD,
        )?;

        // mHC post ffn → residual_next (M13.3: read post/comb from split)
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
        drop(_s_combine);
        _t_combine.end()?;
        if serial {
            de.compute.synchronize()?;
        }
        Ok(())
    }
}

/// Issue all dGPU shared-expert kernels on `de.compute`. Reads
/// `ffn_input_norm`, writes `ffn_shared`. Used by both modes; in
/// `HetParallel` it's invoked earlier to overlap with iGPU MoE.
fn issue_shared_expert(
    de: &DeviceEngine,
    dgpu_scratch: &mut DgpuScratch,
    dlw: &DgpuLayerWeights,
) -> eyre::Result<()> {
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
