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
        de.rms_nw.launch(
            &de.compute,
            &mut scratch.head_flat,
            &scratch.residual,
            1,
            HC_DIM,
            RMS_EPS,
        )?;
        de.f16.matvec(
            &de.compute,
            &mut scratch.head_pre,
            &weights.output_hc_fn.buffer,
            &scratch.head_flat,
            N_HC,
            HC_DIM,
        )?;
        de.hc_sigmoid.launch(
            &de.compute,
            &mut scratch.head_w,
            &scratch.head_pre,
            &weights.output_hc_scale,
            &weights.output_hc_base,
            N_HC,
        )?;
        de.hc_weighted.launch(
            &de.compute,
            &mut scratch.head_embd,
            &scratch.residual,
            &scratch.head_w,
            N_EMBD,
            N_HC,
        )?;
        de.rms_w.launch_weighted(
            &de.compute,
            &mut scratch.head_norm,
            &scratch.head_embd,
            &weights.output_norm,
            N_EMBD,
            RMS_EPS,
        )?;
        de.q8.quantize_input(
            &de.compute,
            &mut scratch.head_xq,
            &mut scratch.head_xscale,
            &scratch.head_norm,
            N_EMBD,
        )?;
        de.q8.matvec(
            &de.compute,
            &mut scratch.logits,
            &weights.output.buffer,
            &scratch.head_xq,
            &scratch.head_xscale,
            N_VOCAB,
            N_EMBD,
        )?;
        Ok(())
    }
}
