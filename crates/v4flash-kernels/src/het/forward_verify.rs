//! MTP verify-decode forward (batched B=k).
//!
//! A verify step runs `k` candidate tokens (the accepted prefix + drafts)
//! through the full model in ONE batched forward, producing per-position
//! logits *and* per-position final-layer residual ("HC") so the spec loop
//! can (a) accept/reject each drafted position by argmax and (b) seed the
//! next draft from the last accepted position's HC.
//!
//! Dispatch:
//!   * **B == 1** → the DECODE per-stage graph-captured `forward_layer`
//!     loop (the same path `forward_token` drives). This is the ~34 ms/tok
//!     decode critical path; a verify-1 (B=2) that starts here keeps its
//!     B=1 anchor honest.
//!   * **B >= 2** → `forward_verify_batch`, the batched layer loop on the
//!     DECODE monotonic KV discipline (`forward_prompt_batch_v2_impl` with
//!     `verify_monotonic=true`). Its attention is the shared-window batched
//!     WMMA pair (`score_batched_htiled_wmma_f16s` +
//!     `softmax_wsum_batched_htiled_wmma_ldsv_f16s`), and its projection
//!     weights are read once per batch.
//!
//! Both paths advance `HetModelState` as a sequential decode would (k KV
//! appends, k routed MoE tokens, k comp boundaries) and produce logits
//! faithful to running `forward_token` k times in sequence — VALIDATED in
//! the MONOTONIC discipline for B up to 4 (argmax-exact vs a sequential
//! decode(B=1) reference primed the same way; HC drift ≤ vector-scale
//! reduction noise, no blow-up — see tests/bench_verify_forward.rs).
//!
//! ## KV-cache discipline (DONE — monotonic, decode-consistent)
//! The B>=2 path appends the k rows MONOTONICALLY at `raw_off + n_raw`
//! (contiguous, no eviction) and reads each token's attention window from
//! `raw_off`, advancing `raw_off`/`n_raw` exactly as k sequential decode
//! steps would (n_raw caps at SWA_WINDOW, then raw_off advances). When the
//! oversized cache (`KV_CACHE_ROWS = SWA_WINDOW + B_MAX`) would overflow the
//! append wraps the live window down to slot 0 first — the same two-hop copy
//! decode uses. This is what lets a mixed decode(B=1)+verify(B=k) loop run
//! on ONE evolving monotonic state (the O(1) MTP-rollback layout). The
//! PREFILL discipline (`forward_prompt_batch_v2`, append at `n_raw` + evict
//! the SWA window to slot 0 post-chunk) is used only by prompt prefill and
//! would corrupt a decode-maintained state — do not drive verify through it.
//!
//! The compressor boundary-fire loop stays host-side (boundary counts are
//! deterministic from `pos`, no device readback).
//!
//! ## Perf note (device-time, measured — where the floor actually lives)
//! The batched path amortizes FLAT across B (dGPU-busy 78.0 ms @ B=1 →
//! 81.6 ms @ B=3, +4.6% for 3× the tokens) but sits ~2.5× above the ~31 ms
//! graphed decode-B1 anchor. Per-stage attribution of the 78 ms floor
//! (batched B=1, depth 4K) shows the lever is NOT the MoE:
//!   k.output (proj GEMMs) 18.1 ms 23% | k.q_chain 13.4 ms 17% |
//!   k.shared 10.7 ms 14% | hot_moe_prefill 9.1 ms 12% | k.mhc/kv/router… rest
//! At B≤4 the WMMA GEMMs (`gemm_lds_tiled`, tuned for B≈512) are fixed-cost-
//! dominated. Setting QB_WMMA=0 Q8_OUT_VARIANT=dp4a Q8_GROUPED_VARIANT=dp4a
//! routes them to the per-row dp4a matvec (`q8_0_gemv_batched_warp8`, grid
//! (M/8,1,B)) — the batched twin of decode's `q8_0_gemv_warp8`, IDENTICAL
//! inner loop — and drops the floor 78.0 → 62.2 ms (−20%), argmax-exact.
//! After that, per-stage attribution of the 62 ms floor (batched B=1):
//!   k.shared 10.7 | hot_moe_prefill 9.3 | k.q_chain 8.0 | k.output 7.0 |
//!   k.mhc 7.0 | k.kv_chain 4.9 | k.router 4.3 | k.indexer 3.8 | rest
//! k.output (7.0 ms) is now AT the decode cost (6.5 ms) — a grid.z=B
//! by-expert/weight-once matvec cannot beat it (weight redundancy is 0 at
//! B=1, 2–4× at B≤4 on a ≤7 ms matmul; the partials+reduce inversion only
//! pays at B≈512). The residual 62-vs-31 ms gap is DIFFUSE: every stage
//! runs ~1.5–2× its decode kernel because the batched path issues more/
//! heavier launches than decode's tuned single-token GRAPH. The one stage
//! still notably above decode is k.q_chain (8.0 vs 4.6, ~3 ms). Conclusion:
//! authoring new grid.z=B projection kernels is NOT the lever (the per-row
//! dp4a batched matvec already exists and is at decode cost); the remaining
//! gap is graph/launch overhead + the sum of many small per-stage deltas.
//! NOTE: there is also NO grid.z=B *hetsplit* MoE kernel —
//! `launch_batched_hetsplit` is single-token (grid (n_rows/8,1,1)); the only
//! grid.z=B MoE kernels (`_bxn`) are non-hetsplit. Correctness is unaffected
//! by any of these levers. Default this path to dp4a for the −20% floor.

use color_eyre::eyre::{self, eyre};

use crate::config::{HC_DIM, N_LAYER, N_VOCAB};

use super::batch_scratch::{BatchDgpuScratch, BatchIgpuScratch};
use super::engine::HeterogeneousEngine;
use super::scratch::{DgpuScratch, IgpuScratch};
use super::state::HetModelState;
use super::weights::HetModelWeights;

/// Result of a batched verify forward.
pub struct VerifyOut {
    /// Per-position logits. `last_only` ⇒ `[N_VOCAB]` for the final
    /// position; else `[B * N_VOCAB]` row-major (position-major).
    pub logits: Vec<f32>,
    /// Per-position final-layer residual (HC). Always `[B * HC_DIM]`
    /// row-major — the spec loop seeds the next draft from the accepted
    /// position's slice, so we return every row regardless of `last_only`.
    pub hc: Vec<f32>,
}

impl HeterogeneousEngine {
    /// Batched verify-decode forward. See module docs.
    ///
    /// `input_hcs[i]` is the `HC_DIM` embedded-token residual for the token
    /// at absolute position `pos0 + i`; `tokens[i]` its id (hash router).
    /// `state` is advanced in place. On return `VerifyOut::hc` holds the
    /// per-position final residual and `::logits` the per-position (or
    /// last-only) vocabulary logits.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_verify(
        &self,
        bd: &mut BatchDgpuScratch,
        bi: &mut BatchIgpuScratch,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        head_scratch: &mut DgpuScratch,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        input_hcs: &[Vec<f32>],
        tokens: &[i32],
        pos0: u32,
        last_only: bool,
    ) -> eyre::Result<VerifyOut> {
        let b = tokens.len();
        if b == 0 {
            return Ok(VerifyOut {
                logits: Vec::new(),
                hc: Vec::new(),
            });
        }
        if input_hcs.len() != b {
            return Err(eyre!(
                "forward_verify: input_hcs len {} != tokens len {b}",
                input_hcs.len()
            ));
        }
        for (i, hc) in input_hcs.iter().enumerate() {
            if hc.len() != HC_DIM as usize {
                return Err(eyre!(
                    "forward_verify: input_hcs[{i}] len {} != HC_DIM {}",
                    hc.len(),
                    HC_DIM
                ));
            }
        }

        // VERIFY_FORCE_BATCHED=1: route B=1 through the batched-monotonic
        // path too (for apples-to-apples device-compute scaling benches —
        // NOT production, which wants the graphed decode critical path).
        let force_batched = std::env::var("VERIFY_FORCE_BATCHED").is_ok();
        if b == 1 && !force_batched {
            self.forward_verify_b1(
                dgpu_scratch,
                igpu_scratch,
                state,
                weights,
                &input_hcs[0],
                tokens[0],
                pos0,
                last_only,
            )
        } else {
            self.forward_verify_batched(
                bd,
                bi,
                head_scratch,
                state,
                weights,
                input_hcs,
                tokens,
                pos0,
                last_only,
            )
        }
    }

    /// B == 1 path: the decode `forward_layer` loop (graph-captured
    /// per-stage), then the head. Mirrors `forward_token` but also harvests
    /// the final-layer residual (HC) before the head consumes it.
    #[allow(clippy::too_many_arguments)]
    fn forward_verify_b1(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        input_hc: &[f32],
        token_id: i32,
        pos: u32,
        last_only: bool,
    ) -> eyre::Result<VerifyOut> {
        let _ = last_only; // B=1: single position, logits always returned.
        let cs_hc = HC_DIM as usize;
        let cs_vocab = N_VOCAB as usize;

        self.dgpu.events.reset();
        self.igpu.events.reset();

        self.set_current_cached(self.dgpu.device)?;
        dgpu_scratch.residual.copy_from_host(input_hc)?;

        // Publish per-token device scalars (rope pos + monotonic KV slot),
        // identical to forward_token — consumed by the merged qkv graphs.
        {
            let slot = state.layers[0].raw_off + state.layers[0].n_raw;
            let pos_ptr = dgpu_scratch.pos_dev.raw() as *mut u32;
            let slot_ptr = dgpu_scratch.kv_slot_dev.raw() as *mut u32;
            unsafe {
                self.dgpu.compute.write_value32(pos_ptr, pos)?;
                self.dgpu.compute.write_value32(slot_ptr, slot)?;
            }
        }
        // Keep the moe_signal token sequence in lockstep (the write side in
        // forward_layer reads it with `load`).
        self.token_seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        for layer in 0..N_LAYER as usize {
            let next_dlw = if layer + 1 < N_LAYER as usize {
                Some(&weights.dgpu_layers[layer + 1])
            } else {
                None
            };
            self.forward_layer(
                dgpu_scratch,
                igpu_scratch,
                &mut state.layers[layer],
                &weights.dgpu_layers[layer],
                next_dlw,
                &weights.igpu_layers[layer],
                pos,
                token_id,
            )?;
            std::mem::swap(&mut dgpu_scratch.residual, &mut dgpu_scratch.residual_next);
        }

        // `residual` now holds the layer-N output (HC). Harvest it before
        // the head consumes it.
        let mut hc = vec![0f32; cs_hc];
        self.dgpu.compute.synchronize()?;
        dgpu_scratch.residual.copy_to_host(&mut hc)?;

        self.forward_head(dgpu_scratch, &weights.global)?;
        let mut logits = vec![0f32; cs_vocab];
        self.dgpu.compute.synchronize()?;
        dgpu_scratch.logits.copy_to_host(&mut logits)?;

        // Restore buffer identity for the next call (43 odd swaps + this
        // one) so captured graphs replay against stable pointers.
        std::mem::swap(&mut dgpu_scratch.residual, &mut dgpu_scratch.residual_next);
        self.set_current_cached(self.dgpu.device)?;

        Ok(VerifyOut { logits, hc })
    }

    /// B >= 2 path: the batched prompt forward, then per-position head +
    /// HC harvest. `state` is advanced by the batched layer loop exactly as
    /// a k-token sequential decode would.
    #[allow(clippy::too_many_arguments)]
    fn forward_verify_batched(
        &self,
        bd: &mut BatchDgpuScratch,
        bi: &mut BatchIgpuScratch,
        head_scratch: &mut DgpuScratch,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        input_hcs: &[Vec<f32>],
        tokens: &[i32],
        pos0: u32,
        last_only: bool,
    ) -> eyre::Result<VerifyOut> {
        let b = tokens.len();
        let cs_hc = HC_DIM as usize;
        let cs_vocab = N_VOCAB as usize;

        self.dgpu.events.reset();
        self.igpu.events.reset();

        // DECODE monotonic KV discipline (not prefill's evict-to-slot-0) so a
        // decode(B=1)+verify(B=k) loop on ONE evolving state stays consistent.
        self.forward_verify_batch(
            bd, bi, state, weights, input_hcs, tokens, pos0,
        )?;

        // bd.residual now holds [B, HC_DIM] layer-N output (43 in-loop
        // swaps → residual = last written residual_next).
        self.set_current_cached(self.dgpu.device)?;
        let mut hc = vec![0f32; b * cs_hc];
        self.dgpu.compute.synchronize()?;
        bd.residual
            .slice_view(0, b * cs_hc)
            .copy_to_host(&mut hc)?;

        // Per-position head. Each position's residual runs through the
        // shared head graph in `head_scratch`.
        let mut logits: Vec<f32> = if last_only {
            Vec::with_capacity(cs_vocab)
        } else {
            Vec::with_capacity(b * cs_vocab)
        };
        let rows: &[usize] = if last_only { &[b - 1] } else { &[] };
        let iter: Box<dyn Iterator<Item = usize>> = if last_only {
            Box::new(rows.iter().copied())
        } else {
            Box::new(0..b)
        };
        for i in iter {
            head_scratch
                .residual
                .copy_from_buffer(&bd.residual.slice_view(i * cs_hc, cs_hc))?;
            self.forward_head(head_scratch, &weights.global)?;
            let mut row = vec![0f32; cs_vocab];
            head_scratch.logits.copy_to_host(&mut row)?;
            logits.extend_from_slice(&row);
        }

        Ok(VerifyOut { logits, hc })
    }
}
