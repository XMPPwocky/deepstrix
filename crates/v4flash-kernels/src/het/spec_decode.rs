//! M40-P6: K=1 speculative decode round.
//!
//! Structure (the correct one — earlier sketch had an extra forward_token
//! that's just wasted work):
//!
//! Each round we have:
//!   * `T_committed`        the just-accepted token at position `pos` (its
//!                          KV is NOT yet in cache; will be written by this
//!                          round's pair forward at t0)
//!   * `T_draft`            MTP's prediction for `pos + 1`, drafted last
//!                          round (or, for the first round, from an
//!                          initial forward_token + mtp_draft pair)
//!   * `prev_hc`            HC from last round's pair_t1 forward, used as
//!                          MTP's "prev_hc" input for this round's draft
//!
//! Per round:
//!   1. pair_forward_interleaved(T_committed at pos, T_draft at pos+1)
//!      → writes KV[pos] AND KV[pos+1]
//!      → gives logits[pos+1] (from pair_t0) and logits[pos+2] (from pair_t1)
//!      → captures HC at pos+1 (for next round's MTP draft)
//!   2. mtp_draft(captured_HC, T_draft_embd)
//!      → predicts T_next_draft for pos+2  (used by NEXT round)
//!   3. Verify: argmax(logits[pos+1]) == T_draft?
//!      * accept: BOTH commit. Next round T_committed = T_draft at pos+1,
//!        T_draft = T_next_draft, pos = pos+1, prev_hc = captured_HC.
//!      * reject: only T_committed at pos commits. Re-forward at pos+1
//!        with target's true argmax. Next round restarts from there.
//!
//! Per-round wall = pair_forward (~68 ms) + mtp_draft (~5 ms) = ~73 ms.
//! Commits per round: 1 (reject) or 2 (accept).
//!
//! At 100% accept: 2 / 73 ms = 27 tok/s — barely beats single (26 tok/s).
//! At 50% accept: 1.5 / 73 = 20.5 — worse than single.
//! Break-even with single-token decode: acceptance ≥ ~80%.
//!
//! For initialization: caller runs forward_token(prompt_last_token,
//! pos=N-1) followed by mtp_draft(initial_HC, prompt_last_embd) to seed
//! `T_committed` (= argmax of prompt's last logits) and `T_draft` (= MTP's
//! first prediction). Spec rounds then chain from there.

use color_eyre::eyre::{self, eyre};
use tracing::debug_span;
use v4flash_hip::DeviceBuffer;

use crate::forward::{HC_DIM, N_EMBD, N_VOCAB};

use super::engine::HeterogeneousEngine;
use super::mtp_weights::MtpWeights;
use super::scratch::{DgpuScratch, IgpuScratch};
use super::state::HetModelState;
use super::weights::HetModelWeights;

/// Result of one K=1 spec_decode round.
pub struct SpecDecodeStepResult {
    /// Tokens newly committed by this round: 1 (reject) or 2 (accept).
    pub committed: Vec<i32>,
    /// Did MTP's draft match target's argmax? (== accepted)
    pub accepted: bool,
    /// The token that becomes NEXT round's `T_committed`.
    pub next_t_committed: i32,
    /// The position NEXT round starts at (= where next_t_committed goes).
    pub next_pos: u32,
    /// MTP's draft for the token AFTER next_t_committed (becomes next round's
    /// `T_draft`).
    pub next_t_draft: i32,
}

impl HeterogeneousEngine {
    /// M40-P6: one K=1 spec_decode round.
    ///
    /// Inputs:
    ///   * `t_committed` — token at `pos` that's about to be committed by
    ///     this round's pair_t0 forward.
    ///   * `t_draft` — MTP's draft for pos+1 (predicted in last round).
    ///   * `input_hc_committed` — HC for layer-0 input of `t_committed`.
    ///   * `input_hc_draft` — HC for layer-0 input of `t_draft` at pos+1.
    ///   * `t_draft_embd_host` — F32 row of N_EMBD = base.token_embd[t_draft].
    ///     Used as MTP's `last_token_embd` for drafting NEXT round's token.
    ///
    /// The pair forward writes KV[pos] = t_committed and KV[pos+1] = t_draft.
    /// On reject, KV[pos+1] is stale (drafted token wrong) — for now this
    /// round leaves it as-is and the caller (or next round) is responsible
    /// for rollback if strict correctness is required. See the M40 design
    /// doc; rollback wiring is deferred (counter-only restore on reject).
    ///
    /// Perfetto spans (host + device):
    ///   * `spec.pair_verify` / `dgpu.spec.pair_verify`
    ///   * `spec.mtp_draft`   / `dgpu.spec.mtp_draft`
    ///   * `spec.decide`
    #[allow(clippy::too_many_arguments)]
    pub fn spec_decode_step(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        mtp_weights: &MtpWeights,
        t_committed: i32,
        t_draft: i32,
        pos: u32,
        input_hc_committed: &[f32],
        input_hc_draft: &[f32],
        t_draft_embd_host: &[f32],
    ) -> eyre::Result<SpecDecodeStepResult> {
        let _step_span = debug_span!("spec.step", pos, t_committed, t_draft).entered();

        if input_hc_committed.len() != HC_DIM as usize
            || input_hc_draft.len() != HC_DIM as usize
        {
            return Err(eyre!("spec_decode_step: input_hc len mismatch"));
        }
        if t_draft_embd_host.len() != N_EMBD as usize {
            return Err(eyre!("spec_decode_step: t_draft_embd len mismatch"));
        }

        // ===== 1) pair_forward(t_committed, t_draft) at (pos, pos+1) =====
        let (logits_at_pos1, logits_at_pos2) = {
            let _t_host = debug_span!("spec.pair_verify").entered();
            let _t_dev = self
                .dgpu
                .events
                .stage("dgpu.spec.pair_verify", &self.dgpu.compute)?;
            self.forward_pair_interleaved(
                dgpu_scratch,
                igpu_scratch,
                state,
                weights,
                input_hc_committed,
                input_hc_draft,
                pos,
                t_committed,
                t_draft,
            )?;
            let mut h0 = vec![0f32; N_VOCAB as usize];
            let mut h1 = vec![0f32; N_VOCAB as usize];
            dgpu_scratch.logits_token0.copy_to_host(&mut h0)?;
            dgpu_scratch.logits.copy_to_host(&mut h1)?;
            (h0, h1)
        };
        let target_at_pos1 = argmax(&logits_at_pos1) as i32;
        let target_at_pos2 = argmax(&logits_at_pos2) as i32;
        let accepted = target_at_pos1 == t_draft;

        // ===== 2) MTP draft for NEXT round (predicts T at pos+2 [accept]
        //         or T at pos+1 [reject], based on which HC we use) =====
        //
        // Strategy: we ALWAYS draft assuming accept (next pos = pos+2). On
        // reject the draft is wrong — the next round will overwrite t_draft
        // and discard this draft. Wasted MTP work on rejects (~5 ms) but
        // simpler control flow than skipping MTP conditionally.
        //
        // HC source: pair_t1's final HC (= HC at pos+1, after t_draft's
        // forward through all layers). On reject the HC came from the
        // wrong drafted token, but again — next round overrides.
        let mtp_hc_capture: DeviceBuffer<f32> = {
            let mut buf = DeviceBuffer::<f32>::new(self.dgpu.device.id, HC_DIM as usize)?;
            // forward_pair_interleaved leaves t1's final HC in
            // dgpu_scratch.t1.residual_next (the per-token scratch).
            buf.copy_from_buffer(&dgpu_scratch.t1.residual_next)?;
            buf
        };
        let next_t_draft = {
            let _t_host = debug_span!("spec.mtp_draft").entered();
            let _t_dev = self
                .dgpu
                .events
                .stage("dgpu.spec.mtp_draft", &self.dgpu.compute)?;
            self.forward_mtp_draft(
                dgpu_scratch,
                igpu_scratch,
                state,
                &weights.global,
                mtp_weights,
                &mtp_hc_capture,
                t_draft_embd_host,
                pos + 1,
                t_draft,
            )?;
            let mut h = vec![0f32; N_VOCAB as usize];
            dgpu_scratch.mtp_logits.copy_to_host(&mut h)?;
            argmax(&h) as i32
        };

        // ===== 3) Decide accept / reject =====
        let _t_host = debug_span!("spec.decide").entered();
        if accepted {
            // Both t_committed (at pos) and t_draft (at pos+1) commit.
            // Next round: T_committed = target_at_pos2 (= the token that
            // pair_t1's logits predict comes after t_draft).
            // pos advances by 2 (KV through pos+1 now populated).
            Ok(SpecDecodeStepResult {
                committed: vec![t_committed, t_draft],
                accepted: true,
                next_t_committed: target_at_pos2,
                next_pos: pos + 2,
                next_t_draft,
            })
        } else {
            // Only t_committed at pos commits. t_draft's pair_t1 forward
            // wrote KV[pos+1] using a wrong token — the proper fix is
            // rollback (snapshot/restore wiring deferred). For now we
            // leave it stale; next round overwrites at pos+1 with the
            // correct token. (For STRICT correctness this would need
            // snapshot/restore — counter-only restore is partially wired
            // via HetLayerState::snapshot_async but full rollback isn't.)
            //
            // Next round: T_committed = target_at_pos1 (target's correct
            // choice for pos+1 — we know it from pair_t0's logits).
            // pos advances by 1.
            Ok(SpecDecodeStepResult {
                committed: vec![t_committed],
                accepted: false,
                next_t_committed: target_at_pos1,
                next_pos: pos + 1,
                next_t_draft,
            })
        }
    }
}

fn argmax(x: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bestv = x[0];
    for (i, &v) in x.iter().enumerate().skip(1) {
        if v > bestv {
            bestv = v;
            best = i;
        }
    }
    best
}
