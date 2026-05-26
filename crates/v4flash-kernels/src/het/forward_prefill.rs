//! M50: layer-major batched prefill.
//!
//! `forward_prompt_batch(scratches, state, weights, tokens[B], pos0)`
//! advances `B` tokens through every layer of the model in layer-major
//! order, sharing the per-layer state (KV cache, compressor) across the
//! batch but using a separate `DgpuScratch` / `IgpuScratch` per token.
//!
//! ## Phase 1 — looped single-token kernels
//!
//! This phase calls the existing `forward_layer` once per batch element,
//! per layer. Functionally equivalent to N sequential `forward_token`
//! calls but reorganized to dispatch layer-major. No batched kernels;
//! no perf win expected. Purpose:
//! 1. Get the prefill driver + per-batch scratch in place.
//! 2. Confirm KV cache + compressor state evolve identically to the
//!    sequential path (oracle test).
//! 3. Provide the wiring Phase 2 needs to swap looped kernels for
//!    real batched kernels.
//!
//! ## Layer-major vs token-major
//!
//! Both produce identical state (KV append + compressor write are
//! commutative across batch elements only when they touch *different*
//! per-layer state — which is true here because layer N's state for
//! token b at position pos0+b is only ever written by that one call).
//!
//! Layer-major lets a future batched-kernel phase amortize per-layer
//! weight reads across the batch.

use color_eyre::eyre::{self, eyre};

use crate::forward::{HC_DIM, N_LAYER};

use super::batch_scratch::BatchScratch;
use super::engine::HeterogeneousEngine;
use super::state::HetModelState;
use super::weights::HetModelWeights;

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
}
