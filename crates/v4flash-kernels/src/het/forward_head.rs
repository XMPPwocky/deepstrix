//! Head dispatch (final RMS + logits matvec). Runs on dGPU.

use color_eyre::eyre;

use crate::config::{HC_DIM, N_EMBD, N_HC, N_VOCAB, RMS_EPS};

use v4flash_hip::DeviceBuffer;

use super::engine::HeterogeneousEngine;
use super::scratch::DgpuScratch;
use super::weights::HetGlobalWeights;

impl HeterogeneousEngine {
    /// Head over `b` contiguous rows, reading the vocab projection ONCE.
    ///
    /// The per-row `forward_head` chain ends in a matvec over the tied vocab
    /// projection: N_VOCAB x N_EMBD at Q8_0, ~700 MB, measured at 1112 us/call
    /// with mean == max, i.e. already at ~630 GB/s. It is purely bandwidth-bound,
    /// so running it per row re-reads those 700 MB once per row -- 6.7 ms for a
    /// B=6 verify to do six dot products.
    ///
    /// PREP STAYS PER ROW ON PURPOSE. Swapping hc_weighted / rms_w / quantize for
    /// their `_batched` twins changes their reduction order and MEASURED
    /// verify-vs-decode KLD 0.00551 -> 0.01068. Those steps are ~62 us/row
    /// combined, so there is nothing to win there and fidelity to lose. Only the
    /// matvec is batched, through `matvec_bpack`, which is bit-identical to
    /// `q8.matvec` per (row, b) and merely hoists the weight load out of the
    /// batch loop.
    ///
    /// Q8_0 output only; anything else returns `Ok(false)` and the caller keeps
    /// the per-row path.
    pub fn forward_head_batch(
        &self,
        scratch: &mut DgpuScratch,
        residual: &DeviceBuffer<f32>,
        carry: &DeviceBuffer<f32>,
        b: u32,
        weights: &HetGlobalWeights,
    ) -> eyre::Result<bool> {
        if b == 0 || b as usize > super::scratch::HEAD_BATCH_MAX {
            return Ok(false);
        }
        if weights.output.dtype != v4flash_core::gguf::GgufType::Q8_0 {
            return Ok(false);
        }
        self.set_current_cached(self.dgpu.device)?;
        let de = &self.dgpu;
        let _t_head = de.events.stage("dgpu.head_batch", &de.compute)?;
        let ne = N_EMBD as usize;
        let nblk = ne / 32;
        for i in 0..b as usize {
            // Identical kernels to the per-row path, just writing into row i of
            // the batched staging buffers.
            {
                let _t = de.events.stage("k.head.hc_weighted", &de.compute)?;
                de.hc_weighted.launch(
                    &de.compute,
                    &mut scratch.head_embd,
                    &residual.slice_view(i * HC_DIM as usize, HC_DIM as usize),
                    &carry.slice_view(i * crate::config::HC_MIX_DIM as usize,
                                      crate::config::HC_MIX_DIM as usize),
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
            {
                let _t = de.events.stage("k.head.quantize", &de.compute)?;
                let mut xq = scratch.head_xq_b.slice_view_mut(i * ne, ne);
                let mut xs = scratch.head_xscale_b.slice_view_mut(i * nblk, nblk);
                de.q8.quantize_input(
                    &de.compute,
                    &mut xq,
                    &mut xs,
                    &scratch.head_norm,
                    N_EMBD,
                )?;
            }
        }
        {
            // ONE read of the 700 MB projection for all `b` rows.
            let _t = de.events.stage("k.head.vocab_matvec_bpack", &de.compute)?;
            de.q8.matvec_bpack(
                &de.compute,
                &mut scratch.logits_b,
                &weights.output.buffer,
                &scratch.head_xq_b,
                &scratch.head_xscale_b,
                N_VOCAB,
                N_EMBD,
                b,
            )?;
        }
        Ok(true)
    }

    pub fn forward_head(
        &self,
        scratch: &mut DgpuScratch,
        weights: &HetGlobalWeights,
    ) -> eyre::Result<()> {
        self.set_current_cached(self.dgpu.device)?;
        let de = &self.dgpu;
        let _t_head = de.events.stage("dgpu.head", &de.compute)?;
        // V4.1 (CED / single-pass mHC): the final collapse reuses the pre_mix
        // CARRIED out of the last block — `blocks[-1].hc_pre(h, pre_mix)` in
        // model.py — so there is no head-level hc projection to evaluate and the
        // checkpoint has no `output_hc_*`. V4-Flash instead derives the collapse
        // weights here from `output_hc_fn` + a sigmoid. Validated against the
        // reference by the layer-major HEAD gate (argmax 11111 at T=6).
        let collapse_w = &scratch.hc_pre_carry;
        {
            let _t = de.events.stage("k.head.hc_weighted", &de.compute)?;
            de.hc_weighted.launch(
                &de.compute,
                &mut scratch.head_embd,
                &scratch.residual,
                collapse_w,
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
