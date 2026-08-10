//! Head dispatch (final RMS + logits matvec). Runs on dGPU.

use color_eyre::eyre;

use crate::config::{HC_DIM, N_EMBD, N_HC, N_VOCAB, RMS_EPS};

use super::engine::HeterogeneousEngine;
use super::scratch::DgpuScratch;
use super::weights::HetGlobalWeights;

impl HeterogeneousEngine {
    pub fn forward_head(
        &self,
        scratch: &mut DgpuScratch,
        weights: &HetGlobalWeights,
    ) -> eyre::Result<()> {
        self.set_current_cached(self.dgpu.device)?;
        let de = &self.dgpu;
        let _t_head = de.events.stage("dgpu.head", &de.compute)?;
        {
            let _t = de.events.stage("k.head.rms_nw", &de.compute)?;
            de.rms_nw.launch(
                &de.compute,
                &mut scratch.head_flat,
                &scratch.residual,
                1,
                HC_DIM,
                RMS_EPS,
            )?;
        }
        {
            let _t = de.events.stage("k.head.hc_fn", &de.compute)?;
            de.f16.matvec(
                &de.compute,
                &mut scratch.head_pre,
                &weights.output_hc_fn.buffer,
                &scratch.head_flat,
                N_HC,
                HC_DIM,
            )?;
        }
        {
            let _t = de.events.stage("k.head.sigmoid", &de.compute)?;
            de.hc_sigmoid.launch(
                &de.compute,
                &mut scratch.head_w,
                &scratch.head_pre,
                &weights.output_hc_scale,
                &weights.output_hc_base,
                N_HC,
            )?;
        }
        {
            let _t = de.events.stage("k.head.hc_weighted", &de.compute)?;
            de.hc_weighted.launch(
                &de.compute,
                &mut scratch.head_embd,
                &scratch.residual,
                &scratch.head_w,
                N_EMBD,
                N_HC,
            )?;
        }
        {
            let _t = de.events.stage("k.head.rms_w", &de.compute)?;
            de.rms_w.launch_weighted(
                &de.compute,
                &mut scratch.head_norm,
                &scratch.head_embd,
                &weights.output_norm,
                N_EMBD,
                RMS_EPS,
            )?;
        }
        // unsloth mix: output.weight is Q4_K (takes f32 directly, no
        // quantize step); antirez keeps Q8_0. Covers prefill too — the
        // prefill logits path loops per-token through this fn.
        if weights.output.dtype == v4flash_core::gguf::GgufType::Q8_0 {
            let _t = de.events.stage("k.head.quantize", &de.compute)?;
            de.q8.quantize_input(
                &de.compute,
                &mut scratch.head_xq,
                &mut scratch.head_xscale,
                &scratch.head_norm,
                N_EMBD,
            )?;
        }
        {
            let _t = de.events.stage("k.head.vocab_matvec", &de.compute)?;
            match weights.output.dtype {
                // dp4a GEMM@B=1 beats both the scalar Q4_K gemv (0.48 vs
                // 0.62 ms) and the Q8_0 dp4a matvec (0.90 ms) on the head
                // shape — bench_kquant_dense_isolated, 2026-08-10.
                v4flash_core::gguf::GgufType::Q4_K => {
                    de.q8k.launch(&de.compute, &mut scratch.head_q8k, &scratch.head_norm, 16)?;
                    de.dense_gemm.gemm(
                        &de.compute,
                        v4flash_core::gguf::GgufType::Q4_K,
                        &mut scratch.logits,
                        &weights.output.buffer,
                        &scratch.head_q8k,
                        1,
                        N_VOCAB,
                        16,
                    )?;
                }
                _ => super::dispatch::dense_matvec(
                    de,
                    &de.compute,
                    &mut scratch.logits,
                    &weights.output,
                    &scratch.head_norm,
                    &scratch.head_xq,
                    &scratch.head_xscale,
                    N_VOCAB,
                    N_EMBD,
                )?,
            }
        }
        drop(_t_head);
        Ok(())
    }
}
