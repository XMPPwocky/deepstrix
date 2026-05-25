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
use super::sync::{peer_push_f32, peer_push_i32};
use super::weights::{DgpuLayerWeights, IgpuLayerWeights};
use tracing::debug_span;

impl HeterogeneousEngine {
    /// Run one layer in the het pipeline. Reads from
    /// `dgpu_scratch.residual` and writes to `dgpu_scratch.residual_next`.
    ///
    /// M30: `next_dlw` is `Some` for all layers except the last. When
    /// present, layer N's ffn_combine is fused with layer N+1's
    /// mhc_pre_attn into one captured graph (the combined-transition
    /// graph). Correspondingly, mhc_pre_attn is launched standalone ONLY
    /// for layer 0 — every other layer's mhc_pre_attn rides the previous
    /// layer's combined graph.
    ///
    /// M40-P1: in pair-mode (`pair_mode=true`), the M30 cross-layer
    /// combined graph is bypassed — each layer uses standalone
    /// mhc_pre_attn + standalone ffn_combine. This is required because
    /// pair-mode interleaves two tokens at each layer boundary
    /// (token0's layer N → snapshot → token1's layer N → ...) and the
    /// combined graph would speculatively launch the NEXT layer's
    /// mhc_pre_attn before the snapshot point.
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

    /// M40-P1: pair-mode entry point. See `forward_layer` docs.
    pub fn forward_layer_pair_mode(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        ilw: &IgpuLayerWeights,
        pos: u32,
        token_id: i32,
    ) -> eyre::Result<()> {
        // pair-mode forces standalone graphs on every layer, so
        // next_dlw is unused. Pass None.
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
        pair_mode: bool,
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
        // M40-P1: pair-mode forces standalone graphs (no cross-layer
        // M30 combined graph). Set is_first_layer=true ALWAYS to fire
        // standalone mhc_pre_attn each layer, and is_last_layer=true
        // ALWAYS to fire standalone ffn_combine each layer.
        let is_first_layer = pair_mode || layer == 0;
        let is_last_layer = pair_mode || next_dlw.is_none();

        let serial = matches!(self.mode, ExecMode::HetSingleStream);

        let _layer_span = debug_span!("het.layer", layer, pos).entered();

        // ============================================================
        // dGPU: mHC pre attn → attn_cur → attn_input_norm
        // ============================================================
        self.set_current_cached(self.dgpu.device)?;
        let de = &self.dgpu;

        // M30: mhc_pre_attn for layer N is launched by the PREVIOUS
        // layer's combined ffn_combine→mhc_pre_attn graph, except for
        // layer 0 which has no preceding ffn_combine. The standalone
        // mhc_pre_attn graph below only fires once per token (layer 0).
        if is_first_layer {
            let _t_mhc_pre = de.events.stage("dgpu.mhc_pre_attn", &de.compute)?;
            let _s_mhc_pre = debug_span!("mhc_pre_attn").entered();
            let graph_slot = &self.dgpu_mhc_pre_attn_graphs[layer as usize];
            let mut guard = graph_slot.lock().unwrap();
            if guard.is_none() {
                de.compute
                    .begin_capture(v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
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
                let graph = de.compute.end_capture()?;
                let exec = graph.instantiate()?;
                exec.launch(&de.compute)?;
                *guard = Some(exec);
            } else {
                guard.as_ref().unwrap().launch(&de.compute)?;
            }
            drop(_s_mhc_pre);
            _t_mhc_pre.end()?;
        }

        // ============================================================
        // dGPU: Q LoRA chain → q_post_rope
        // ============================================================
        // M15: q_chain prefix (6 kernels) captured into a graph — all
        // params are layer-constant and all I/O buffers are non-swapped
        // scratch. The trailing rope_forward takes per-token `pos` so
        // stays a direct launch.
        let _t_q = de.events.stage("dgpu.q_chain", &de.compute)?;
        let _s_q = debug_span!("q_chain").entered();
        {
            let graph_slot = &self.dgpu_q_chain_pre_rope_graphs[layer as usize];
            let mut guard = graph_slot.lock().unwrap();
            if guard.is_none() {
                de.compute
                    .begin_capture(v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
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
                let graph = de.compute.end_capture()?;
                let exec = graph.instantiate()?;
                exec.launch(&de.compute)?;
                *guard = Some(exec);
            } else {
                guard.as_ref().unwrap().launch(&de.compute)?;
            }
        }
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
        let _parallel_pre = matches!(self.mode, ExecMode::HetParallel);
        let _sev_pre = &self.sync_events.layers[layer as usize];

        // M14L: compressor runs entirely on dGPU. attn_input_norm is
        // already local; weights + state + scratch all on dGPU; no peer
        // pushes (was: dGPU→iGPU attn_input_norm push, iGPU→dGPU
        // comp_row push).
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
                de.f16rt.launch(
                    &de.compute,
                    &mut dgpu_scratch.comp_row,
                    N_HEAD_DIM,
                )?;
                if ratio == 4 {
                    de.compressor_shuffle.launch(
                        &de.compute,
                        &mut cs.state_kv,
                        &mut cs.state_score,
                        comp_width,
                    )?;
                }

                // No peer push needed — append directly into local comp_kv.
                let _t_append = de.events.stage("dgpu.comp_kv_append", &de.compute)?;
                de.comp_kv_append.launch(
                    &de.compute,
                    &mut cs.comp_kv,
                    &dgpu_scratch.comp_row,
                    cs.n_comp,
                    N_HEAD_DIM,
                )?;
                cs.n_comp += 1;
                _t_append.end()?;
            }
            drop(_s_comp);
            _t_comp.end()?;
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
        // M15: output_proj suffix (4 kernels after rope_inv) captured.
        // rope_inv takes per-token `pos` and stays a direct launch.
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
        {
            let graph_slot = &self.dgpu_output_proj_post_rope_graphs[layer as usize];
            let mut guard = graph_slot.lock().unwrap();
            if guard.is_none() {
                de.compute
                    .begin_capture(v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
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
                let graph = de.compute.end_capture()?;
                let exec = graph.instantiate()?;
                exec.launch(&de.compute)?;
                *guard = Some(exec);
            } else {
                guard.as_ref().unwrap().launch(&de.compute)?;
            }
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
        // M15: capture mhc_pre_ffn block (5 kernels, layer-constant params)
        // into a HIP graph on first call; replay thereafter.
        let _t_mhc_pre_ffn = de.events.stage("dgpu.mhc_pre_ffn", &de.compute)?;
        let _s_mhc_pre_ffn = debug_span!("mhc_pre_ffn").entered();
        {
            let graph_slot = &self.dgpu_mhc_pre_ffn_graphs[layer as usize];
            let mut guard = graph_slot.lock().unwrap();
            if guard.is_none() {
                de.compute
                    .begin_capture(v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
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
                let graph = de.compute.end_capture()?;
                let exec = graph.instantiate()?;
                exec.launch(&de.compute)?;
                *guard = Some(exec);
            } else {
                guard.as_ref().unwrap().launch(&de.compute)?;
            }
        }
        drop(_s_mhc_pre_ffn);
        _t_mhc_pre_ffn.end()?;

        // ============================================================
        // M16: router → dGPU. dGPU has 2.6× iGPU BW so the matvec is
        // ~1.5 ms faster across the model, and running router on the
        // dGPU side lifts it off the iGPU's critical path.
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

        // M16: router on dGPU.compute.
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

        // M16: push d_selected and d_ew to iGPU on dGPU.xfer (FIFO
        // after ffn_input_norm push).
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
        peer_push_i32(
            &dgpu_scratch.d_selected,
            &mut igpu_scratch.d_selected,
            &de.xfer,
        )?;
        peer_push_f32(
            &dgpu_scratch.d_ew,
            &mut igpu_scratch.d_ew,
            &de.xfer,
        )?;
        if parallel {
            sev.selected_pushed.record(&de.xfer)?;
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
        // d_midq_cat staging buffer. Eliminates ~600μs/layer of host
        // round-trips (M13.4 inner-loop refactor).
        //
        // M15: the 4-kernel core (q8k_xq → iq2_fused → q8k_mid → q2k_down)
        // is captured into a per-layer HIP graph on the first call and
        // replayed thereafter. All kernel params are device pointers +
        // layer-constant scalars, so the captured graph stays valid for
        // every subsequent token.
        //
        // M16: d_selected and d_ew are now peer-pushed from dGPU (router
        // ran there) and arrive via the selected_pushed event already
        // waited on above — no more host copy_from_host here.
        let mid_blocks_bytes = (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8_K_BYTES;
        let _t_moe = ie.events.stage("igpu.routed_moe", &ie.compute)?;
        let _s_moe = debug_span!("routed_moe").entered();
        {
            let graph_slot = &self.igpu_moe_graphs[layer as usize];
            let mut guard = graph_slot.lock().unwrap();
            if guard.is_none() {
                // First call for this layer: capture the 4-kernel
                // sub-pipeline. begin_capture / end_capture bracket only
                // the launches we want in the graph; per-kernel event
                // staging is OUTSIDE the capture so the event-record
                // nodes don't become part of the replayed graph (they
                // would re-record the same pool events on every replay,
                // corrupting the per-token harvest).
                ie.compute
                    .begin_capture(v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                ie.q8k.launch(
                    &ie.compute,
                    &mut igpu_scratch.d_xq_q8k,
                    &igpu_scratch.ffn_input_norm_recv,
                    BLOCKS_Q8K_GATE_IN,
                )?;
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
                ie.q8k.launch(
                    &ie.compute,
                    &mut igpu_scratch.d_midq_cat,
                    &igpu_scratch.d_mid_cat,
                    BLOCKS_Q8K_DOWN_IN * (N_EXPERT_USED as u32),
                )?;
                ie.q2k.launch_batched(
                    &ie.compute,
                    &mut igpu_scratch.ffn_moe,
                    &ilw.routed.down.buffer,
                    &igpu_scratch.d_midq_cat,
                    &igpu_scratch.d_selected,
                    dbpe as u32,
                    mid_blocks_bytes as u32,
                    N_EXPERT_USED as u32,
                    N_EMBD,
                    BLOCKS_Q8K_DOWN_IN,
                )?;
                let graph = ie.compute.end_capture()?;
                let exec = graph.instantiate()?;
                // Stream-capture only RECORDS — it does not execute.
                // Launch the freshly instantiated graph now so the
                // first-token forward pass actually runs this layer's
                // MoE pipeline.
                exec.launch(&ie.compute)?;
                *guard = Some(exec);
            } else {
                guard
                    .as_ref()
                    .unwrap()
                    .launch(&ie.compute)?;
            }
        }
        // (M16: `selected` no longer materialized on host — d_selected
        // is fully device-side and peer-pushed from dGPU above.)
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
        let _t_combine = de.events.stage("dgpu.ffn_combine", &de.compute)?;
        let _s_combine = debug_span!("ffn_combine").entered();
        if is_last_layer {
            // M15.1: capture (vec_add → hc_post writing residual_next).
            // Stable per-layer pointers thanks to the end-of-token
            // extra swap. Used ONLY for the last layer under M30.
            let graph_slot = &self.dgpu_ffn_combine_graphs[layer as usize];
            let mut guard = graph_slot.lock().unwrap();
            if guard.is_none() {
                de.compute
                    .begin_capture(v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
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
                let graph = de.compute.end_capture()?;
                let exec = graph.instantiate()?;
                exec.launch(&de.compute)?;
                *guard = Some(exec);
            } else {
                guard.as_ref().unwrap().launch(&de.compute)?;
            }
        } else {
            // M30: combined ffn_combine_N + mhc_pre_attn_{N+1} graph.
            // The mhc_pre_attn block reads from layer N+1's `residual`
            // which is THIS layer's `residual_next` after the post-layer
            // swap — same physical buffer. We pass `residual_next.raw()`
            // throughout so the captured graph references one ptr.
            let next = next_dlw.expect("M30: next_dlw required for non-last layer");
            let graph_slot = &self.dgpu_combined_ffn_pre_attn_graphs[layer as usize];
            let mut guard = graph_slot.lock().unwrap();
            if guard.is_none() {
                de.compute
                    .begin_capture(v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                // ffn_combine half — writes residual_next (= layer N+1's residual).
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
                // mhc_pre_attn half — reads residual_next (= layer N+1's residual
                // after swap), uses layer N+1's hc/norm weights.
                de.rms_nw.launch(
                    &de.compute,
                    &mut dgpu_scratch.flat,
                    &dgpu_scratch.residual_next,
                    1,
                    HC_DIM,
                    RMS_EPS,
                )?;
                de.f16.matvec(
                    &de.compute,
                    &mut dgpu_scratch.mix,
                    &next.hc_attn_fn.buffer,
                    &dgpu_scratch.flat,
                    HC_MIX_DIM,
                    HC_DIM,
                )?;
                de.hc_sinkhorn.launch(
                    &de.compute,
                    &mut dgpu_scratch.split,
                    &dgpu_scratch.mix,
                    &next.hc_attn_scale,
                    &next.hc_attn_base,
                    N_HC,
                    SINKHORN_ITERS,
                    SINKHORN_EPS,
                )?;
                de.hc_weighted.launch(
                    &de.compute,
                    &mut dgpu_scratch.attn_cur,
                    &dgpu_scratch.residual_next,
                    &dgpu_scratch.split,
                    N_EMBD,
                    N_HC,
                )?;
                de.rms_w.launch_weighted(
                    &de.compute,
                    &mut dgpu_scratch.attn_input_norm,
                    &dgpu_scratch.attn_cur,
                    &next.attn_norm,
                    N_EMBD,
                    RMS_EPS,
                )?;
                let graph = de.compute.end_capture()?;
                let exec = graph.instantiate()?;
                exec.launch(&de.compute)?;
                *guard = Some(exec);
            } else {
                guard.as_ref().unwrap().launch(&de.compute)?;
            }
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
    /// M15: capture the dGPU shared-expert chain (6 kernels, all
    /// layer-constant params) into a HIP graph on first call; replay
    /// thereafter. Caller is responsible for the surrounding events
    /// stage and span.
    fn issue_shared_expert_graph(
        &self,
        de: &DeviceEngine,
        dgpu_scratch: &mut DgpuScratch,
        dlw: &DgpuLayerWeights,
        layer: i32,
    ) -> eyre::Result<()> {
        let graph_slot = &self.dgpu_shared_expert_graphs[layer as usize];
        let mut guard = graph_slot.lock().unwrap();
        if guard.is_none() {
            de.compute
                .begin_capture(v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            issue_shared_expert(de, dgpu_scratch, dlw)?;
            let graph = de.compute.end_capture()?;
            let exec = graph.instantiate()?;
            exec.launch(&de.compute)?;
            *guard = Some(exec);
        } else {
            guard.as_ref().unwrap().launch(&de.compute)?;
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
