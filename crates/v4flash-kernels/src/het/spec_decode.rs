//! K=1 MTP speculative decode loop (draft 1, verify B=2) — greedy-exact.
//!
//! Assembles the shipped pieces: `forward_mtp_draft` (drafter),
//! `forward_verify` (batched B=k verify on the decode monotonic KV
//! discipline) and the `FrontierSnapshot` compressor-state rollback.
//!
//! ## Structure (ds4 `use_decode2_exact`, adapted)
//! The verify batch's FIRST position must be the *confirmed* main-greedy
//! token at P+1 and its SECOND a speculative draft for P+2. To produce a
//! draft for P+2 without the (unavailable) main HC at P+1, we draft
//! *recursively* through the MTP block, exactly like ds4's inner loop:
//!
//!   d0 = mtp_draft(main_HC@P, tok@P)              → guess for P+1  (+ mtp hid)
//!   d1 = mtp_draft(mtp_hid,   d0)                 → guess for P+2
//!
//! `d0` is checked *for free* against the already-known main argmax at P+1
//! (`c_next`). Only if `d0 == c_next` is the recursion's `d1` a valid draft
//! for P+2 — otherwise the recursion was conditioned on a wrong P+1 token,
//! so we skip speculation and advance one token via a B=1 verify.
//!
//! Per round (when `d0 == c_next`):
//!   verify B=2  [c_next @P+1, d1 @P+2]  → logits@P+2, logits@P+3, HC rows
//!   accept iff argmax(logits@P+2) == d1
//!     accept: commit both; advance P→P+2.
//!     reject: `restore_frontier` (undo KV@P+1, KV@P+2 + compressor), then
//!             re-apply c_next@P+1 via a B=1 verify ("replay one token").
//!
//! ## Greedy-exactness (the correctness gate)
//! Every emitted token is either `c_next` (the argmax of a real forward's
//! logits) or `d1` — and `d1` is emitted *only* when `d1 == argmax(logits@P+2)`.
//! So the committed stream is exactly the main-model greedy argmax chain, and
//! the KV/compressor state after each round reflects precisely the committed
//! prefix. Speculation changes speed, not output.
//!
//! ## Rollback
//! The raw-KV rewind is counter-only (`n_raw`/`raw_off` in the frontier),
//! but the compressor sliding state (ratio==4 `compressor_state_shuffle`
//! slide-down) is NOT counter-reversible, so the frontier also copies
//! `state_kv`/`state_score`. This mirrors ds4's `spec_frontier_snapshot`.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::DeviceBuffer;

use crate::config::{HC_DIM, N_HC, N_VOCAB, SWA_WINDOW};
use crate::sampler::SamplerRng;

use super::batch_scratch::{BatchDgpuScratch, BatchIgpuScratch};
use super::engine::HeterogeneousEngine;
use super::forward_mtp::MtpScratch;
use super::mtp_weights::MtpWeights;
use super::scratch::{DgpuScratch, IgpuScratch};
use super::state::{FrontierSnapshot, HetModelState, MtpLayerState};
use super::weights::HetModelWeights;

/// Loop configuration.
pub struct SpecDecodeConfig {
    /// Number of tokens to generate.
    pub n_tokens: usize,
    /// When `true` the loop is **greedy-exact** (== plain `forward_token`):
    /// the B=2 verify is used only to *decide* accept/reject and to exercise
    /// the frontier rollback, but every committed token's logits **and** KV
    /// are re-derived by an exact B=1 forward (`restore_frontier` + replay).
    /// This is the correctness deliverable; it does not bank the batched
    /// speedup because `forward_verify(B=2)`'s KV is not bit-identical to the
    /// B=1 decode path (accumulated batch-vs-seq drift flips near-tied
    /// argmaxes — see module docs / the report).
    ///
    /// When `false` the accept path *keeps* the batched B=2 KV/logits (the
    /// ~2× verify speedup) — faster, but **not** bit-exact greedy.
    pub bit_exact: bool,
}

/// Loop statistics for reporting.
#[derive(Default, Debug, Clone)]
pub struct SpecDecodeStats {
    /// Total loop rounds.
    pub rounds: usize,
    /// Rounds where the MTP top-1 matched the main argmax at P+1 (`d0 ==
    /// c_next`) — the free-check pass rate, the direct analogue of the
    /// mtp_oracle top-1 hit rate (~0.65).
    pub free_check_pass: usize,
    /// Rounds where a B=2 batched verify ran (== free_check_pass).
    pub b2_verifies: usize,
    /// B=2 verifies whose second (speculative) draft was accepted.
    pub b2_accepts: usize,
    /// Total committed tokens.
    pub committed: usize,
}

/// Committed tokens plus stats.
pub struct SpecDecodeOut {
    pub tokens: Vec<i32>,
    pub stats: SpecDecodeStats,
}

fn argmax(x: &[f32]) -> i32 {
    let mut best = 0i32;
    let mut bestv = x[0];
    for (i, &v) in x.iter().enumerate().skip(1) {
        if v > bestv {
            bestv = v;
            best = i as i32;
        }
    }
    best
}

/// Broadcast an `N_EMBD` embedding row to the `HC_DIM` layer-0 input residual
/// (`N_HC` identical rows) — the input_hc form `forward_verify` expects.
fn broadcast_hc(embd: &[f32]) -> Vec<f32> {
    let n = embd.len();
    let mut out = vec![0f32; (N_HC as usize) * n];
    for h in 0..N_HC as usize {
        out[h * n..(h + 1) * n].copy_from_slice(embd);
    }
    out
}

impl HeterogeneousEngine {
    /// Run the K=1 MTP speculative-decode loop.
    ///
    /// * `prime_tokens` — the full prompt; positions `0..prime_tokens.len()`
    ///   are primed through the model (B=1 verify) to build KV state.
    /// * `embed` — F32 `N_EMBD` embedding-row lookup for a token id
    ///   (host-side dequant of `token_embd`).
    /// * `frontier` — reusable snapshot, `state.alloc_frontier(dgpu)`.
    ///
    /// Returns the generated (committed) token stream + accept stats.
    #[allow(clippy::too_many_arguments)]
    pub fn spec_decode_run(
        &self,
        bd: &mut BatchDgpuScratch,
        bi: &mut BatchIgpuScratch,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        head_scratch: &mut DgpuScratch,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        mtp_scratch: &mut MtpScratch,
        mtp_state: &mut MtpLayerState,
        mtp_weights: &MtpWeights,
        frontier: &mut FrontierSnapshot,
        prime_tokens: &[i32],
        embed: &dyn Fn(i32) -> Vec<f32>,
        cfg: SpecDecodeConfig,
    ) -> eyre::Result<SpecDecodeOut> {
        if prime_tokens.is_empty() {
            return Err(eyre!("spec_decode_run: prime_tokens is empty"));
        }
        let dgpu = self.dgpu.device;
        let hc_dim = HC_DIM as usize;
        let vocab = N_VOCAB as usize;

        // Decode graphs bake this state's kv_cache pointers; clear so priming
        // captures against ITS OWN buffers.
        self.clear_graphs();

        // ---------- Prime the prompt (B=1 verify per position) ----------
        let mut last_hc = vec![0f32; hc_dim];
        let mut last_logits = vec![0f32; vocab];
        for (pos, &tok) in prime_tokens.iter().enumerate() {
            let inp = broadcast_hc(&embed(tok));
            let out = self.forward_verify(
                bd, bi, dgpu_scratch, igpu_scratch, head_scratch, state, weights,
                &[inp], &[tok], pos as u32, false,
            )?;
            if pos + 1 == prime_tokens.len() {
                last_hc.copy_from_slice(&out.hc);
                last_logits.copy_from_slice(&out.logits);
            }
        }

        // Round state after priming.
        let mut p: u32 = (prime_tokens.len() - 1) as u32; // last forwarded pos
        let mut tok_at_p: i32 = *prime_tokens.last().unwrap();
        let mut c_next: i32 = argmax(&last_logits); // confirmed token @ p+1
        let mut main_hc_dev = DeviceBuffer::<f32>::new(dgpu.id, hc_dim)?;
        main_hc_dev.copy_from_host(&last_hc)?; // HC @ p
        let mut hid0 = DeviceBuffer::<f32>::new(dgpu.id, hc_dim)?;

        let mut mtp_logits = vec![0f32; vocab];
        let mut out_tokens: Vec<i32> = Vec::with_capacity(cfg.n_tokens + 2);
        let mut stats = SpecDecodeStats::default();
        let debug = std::env::var("SPEC_DEBUG").is_ok();

        while out_tokens.len() < cfg.n_tokens {
            stats.rounds += 1;
            let mtp_base = mtp_state.n_raw;

            // ---- d0 = mtp_draft(HC@P, tok@P)  → guess for P+1 ----
            self.forward_mtp_draft(
                dgpu_scratch, igpu_scratch, mtp_scratch, mtp_state, &weights.global,
                mtp_weights, &main_hc_dev, &embed(tok_at_p), p, tok_at_p,
            )?;
            hid0.copy_from_buffer(&mtp_scratch.mtp_next_hc)?;
            mtp_scratch.mtp_logits.copy_to_host(&mut mtp_logits)?;
            let d0 = argmax(&mtp_logits);

            if debug {
                eprintln!(
                    "  [rnd {:>3}] p={p} tok@p={tok_at_p} c_next(@{})={c_next} d0={d0} {}",
                    stats.rounds,
                    p + 1,
                    if d0 == c_next { "HIT" } else { "miss" }
                );
            }

            if d0 != c_next {
                // ---- free-check miss: no speculation, advance one token ----
                mtp_state.n_raw = mtp_base; // drop d0's speculative MTP row
                let inp = broadcast_hc(&embed(c_next));
                let out = self.forward_verify(
                    bd, bi, dgpu_scratch, igpu_scratch, head_scratch, state, weights,
                    &[inp], &[c_next], p + 1, false,
                )?;
                out_tokens.push(c_next);
                stats.committed += 1;
                main_hc_dev.copy_from_host(&out.hc)?;
                tok_at_p = c_next;
                p += 1;
                c_next = argmax(&out.logits);
                continue;
            }
            stats.free_check_pass += 1;

            // ---- d1 = mtp_draft(mtp_hid, d0)  → guess for P+2 ----
            self.forward_mtp_draft(
                dgpu_scratch, igpu_scratch, mtp_scratch, mtp_state, &weights.global,
                mtp_weights, &hid0, &embed(d0), p + 1, d0,
            )?;
            mtp_scratch.mtp_logits.copy_to_host(&mut mtp_logits)?;
            let d1 = argmax(&mtp_logits);

            // ---- snapshot frontier, then B=2 verify [c_next@P+1, d1@P+2] ----
            state.capture_frontier(dgpu, frontier)?;
            let inp0 = broadcast_hc(&embed(c_next));
            let inp1 = broadcast_hc(&embed(d1));
            let out2 = self.forward_verify(
                bd, bi, dgpu_scratch, igpu_scratch, head_scratch, state, weights,
                &[inp0, inp1], &[c_next, d1], p + 1, false,
            )?;
            stats.b2_verifies += 1;

            let logits0 = &out2.logits[0..vocab];
            let logits1 = &out2.logits[vocab..2 * vocab];
            let hc1 = &out2.hc[hc_dim..2 * hc_dim];
            let batched_at_p2 = argmax(logits0); // batched main token @ P+2
            let batched_accept = batched_at_p2 == d1;
            if batched_accept {
                stats.b2_accepts += 1;
            }

            if debug {
                eprintln!(
                    "           d1(@{})={d1} bat_main@{}={batched_at_p2} {}",
                    p + 2,
                    p + 2,
                    if batched_accept { "ACCEPT" } else { "reject" }
                );
            }

            if !cfg.bit_exact {
                // ===== SPEEDUP PATH: bank the batched B=2 result (not
                // bit-exact greedy — accumulated batch-vs-seq KV drift). =====
                if batched_accept {
                    out_tokens.push(c_next);
                    out_tokens.push(d1);
                    stats.committed += 2;
                    mtp_state.n_raw = (mtp_base + 2).min(SWA_WINDOW);
                    main_hc_dev.copy_from_host(hc1)?; // HC @ P+2
                    tok_at_p = d1;
                    p += 2;
                    c_next = argmax(logits1);
                } else {
                    state.restore_frontier(dgpu, frontier)?;
                    let inp = broadcast_hc(&embed(c_next));
                    let out = self.forward_verify(
                        bd, bi, dgpu_scratch, igpu_scratch, head_scratch, state, weights,
                        &[inp], &[c_next], p + 1, false,
                    )?;
                    out_tokens.push(c_next);
                    stats.committed += 1;
                    mtp_state.n_raw = (mtp_base + 1).min(SWA_WINDOW);
                    main_hc_dev.copy_from_host(&out.hc)?; // HC @ P+1
                    tok_at_p = c_next;
                    p += 1;
                    c_next = argmax(&out.logits);
                }
                continue;
            }

            // ===== BIT-EXACT PATH =====
            // The B=2 verify + rollback above exercised the speculative path;
            // now discard its (drifted) KV and re-derive every committed token
            // from an EXACT B=1 forward, so the stream == plain forward_token.
            state.restore_frontier(dgpu, frontier)?;

            // Commit c_next@P+1 exactly.
            let inp_c = broadcast_hc(&embed(c_next));
            let out_c = self.forward_verify(
                bd, bi, dgpu_scratch, igpu_scratch, head_scratch, state, weights,
                &[inp_c], &[c_next], p + 1, false,
            )?;
            out_tokens.push(c_next);
            stats.committed += 1;
            let exact_at_p2 = argmax(&out_c.logits); // exact main token @ P+2

            if d1 == exact_at_p2 {
                // Exact accept: d1 is the true token@P+2 — forward it exactly.
                let inp_d = broadcast_hc(&embed(d1));
                let out_d = self.forward_verify(
                    bd, bi, dgpu_scratch, igpu_scratch, head_scratch, state, weights,
                    &[inp_d], &[d1], p + 2, false,
                )?;
                out_tokens.push(d1);
                stats.committed += 1;
                mtp_state.n_raw = (mtp_base + 2).min(SWA_WINDOW);
                main_hc_dev.copy_from_host(&out_d.hc)?; // HC @ P+2
                tok_at_p = d1;
                p += 2;
                c_next = argmax(&out_d.logits);
            } else {
                // Exact reject: only c_next committed.
                mtp_state.n_raw = (mtp_base + 1).min(SWA_WINDOW);
                main_hc_dev.copy_from_host(&out_c.hc)?; // HC @ P+1
                tok_at_p = c_next;
                p += 1;
                c_next = exact_at_p2;
            }
        }

        out_tokens.truncate(cfg.n_tokens);
        Ok(SpecDecodeOut {
            tokens: out_tokens,
            stats,
        })
    }
}

// =====================================================================
//  Rejection-sampling K=1 spec decode (the SAMPLING-path accept rule)
// =====================================================================
//
//  The production recipe is T=1.0 multinomial sampling (min_p=0), NOT
//  greedy. Greedy-argmax acceptance would reject almost every draft whose
//  top-1 differs from the target argmax even when the *sampled* token would
//  have matched — and, worse, it commits the argmax rather than a sample, so
//  the committed stream is NOT distributed as the target model. Rejection
//  sampling fixes both: every emitted token is drawn from the exact target
//  distribution `p` (Leviathan/Chen speculative sampling), so it is provably
//  distribution-correct and the batch-vs-seq KV drift is absorbed into the
//  sampling noise instead of flipping a hard argmax.
//
//  ## The accept rule (draft length 2: x1@P+1, x2@P+2)
//    p1 = softmax(logits(·|P)/T)              [target for P+1; from prev fwd]
//    q1 = softmax(mtp_draft(·|P)/T)           [draft dist for P+1]
//    x1 ~ q1  ;  accept x1 w.p. min(1, p1(x1)/q1(x1))
//      reject: emit r1 ~ norm(max(0, p1 - q1)) and STOP (draft x2 unused).
//      accept: commit x1. Verify B=2 [x1@P+1, x2@P+2] → p2 = softmax(row0/T)
//              (target for P+2 | x1), p3 = softmax(row1/T) (target for P+3).
//        q2 = softmax(mtp_draft(·|P+1,x1)/T) ; x2 ~ q2
//        accept x2 w.p. min(1, p2(x2)/q2(x2))
//          reject: emit r2 ~ norm(max(0, p2 - q2)); STOP.
//          accept: commit x2, then a BONUS token b ~ p3 (the free verify tail).
//
//  A draw `x` from `q` accepted w.p. min(1, p/q), OR from the residual
//  `norm(max(0,p−q))` on rejection, is exactly distributed as `p` — this is
//  the whole point. `p1`/`p2`/`p3` come from real target forwards, so the
//  committed stream is the target model's own sampled output.
//
//  ## KV discipline (correctness-first; perf is a later re-arch)
//  Every round ends with a B=1 `forward_verify` of the last new token, so the
//  next round's `p1` and `main_hc` are clean/sequential. Accepted speculative
//  pairs KEEP the batched B=2 KV (banks the verify; introduces the
//  batch-vs-seq drift we measure). Rejected positions roll back via the
//  existing `FrontierSnapshot`. MTP `n_raw` is kept gap-free: one drafter row
//  per committed position (an extra realign draft covers the bonus tail).

/// Config for the rejection-sampling spec loop.
pub struct SpecSampleConfig {
    /// Number of tokens to generate.
    pub n_tokens: usize,
    /// Sampling temperature (production default 1.0).
    pub temperature: f32,
    /// min-p threshold relative to the top token (production default 0.0).
    pub min_p: f32,
    /// RNG seed (reproducibility for the quality gate).
    pub seed: u64,
    /// When true, record the full-vocab target distribution each committed
    /// token was drawn from (for the KL/TVD quality gate). ~0.5 MB/token.
    pub collect_dists: bool,
}

/// Stats for the rejection-sampling loop.
#[derive(Default, Debug, Clone)]
pub struct SpecSampleStats {
    pub rounds: usize,
    /// x1 draws made (== rounds).
    pub draft1: usize,
    /// x1 draws accepted (target-consistent accept).
    pub draft1_accept: usize,
    /// x2 draws made (only when x1 accepted).
    pub draft2: usize,
    /// x2 draws accepted.
    pub draft2_accept: usize,
    /// Bonus tokens emitted (== draft2_accept).
    pub bonus: usize,
    /// Total committed tokens.
    pub committed: usize,
}

impl SpecSampleStats {
    /// Combined draft acceptance rate = accepted drafts / total drafts.
    pub fn accept_rate(&self) -> f64 {
        let drafts = self.draft1 + self.draft2;
        if drafts == 0 {
            0.0
        } else {
            (self.draft1_accept + self.draft2_accept) as f64 / drafts as f64
        }
    }
    pub fn tokens_per_round(&self) -> f64 {
        self.committed as f64 / self.rounds.max(1) as f64
    }
}

/// Output of the rejection-sampling loop.
pub struct SpecSampleOut {
    pub tokens: Vec<i32>,
    pub stats: SpecSampleStats,
    /// Per-committed-token full-vocab target distribution it was drawn from
    /// (only if `collect_dists`). `target_dists[i]` is P(token[i] | prefix).
    pub target_dists: Vec<Vec<f32>>,
}

/// Full-vocab softmax with temperature + min-p (matches ds4 `sample_full_vocab`,
/// top_p=1 path). Non-finite logits get probability 0; `min_p` prunes tokens
/// whose exp-value (relative to the top, whose exp is 1) is below `min_p`.
pub fn softmax_dist(logits: &[f32], temperature: f32, min_p: f32) -> Vec<f32> {
    let n = logits.len();
    let inv_t = 1.0f32 / temperature;
    let mut maxl = f32::NEG_INFINITY;
    let mut argm = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_finite() && v > maxl {
            maxl = v;
            argm = i;
        }
    }
    let mut d = vec![0f32; n];
    if !maxl.is_finite() {
        d[0] = 1.0;
        return d;
    }
    let min_rel = if min_p > 0.0 { min_p } else { 0.0 };
    let mut z = 0f64;
    for i in 0..n {
        let v = logits[i];
        if !v.is_finite() {
            continue;
        }
        let e = ((v - maxl) * inv_t).exp();
        if e < min_rel {
            continue;
        }
        d[i] = e;
        z += e as f64;
    }
    if z <= 0.0 {
        d.iter_mut().for_each(|x| *x = 0.0);
        d[argm] = 1.0;
        return d;
    }
    let inv_z = 1.0f64 / z;
    for x in d.iter_mut() {
        *x = (*x as f64 * inv_z) as f32;
    }
    d
}

/// Sample an index from a normalized distribution via cumulative walk with
/// `u ∈ [0,1)`.
fn sample_from_dist(d: &[f32], u: f32) -> i32 {
    let mut r = u as f64;
    for (i, &p) in d.iter().enumerate() {
        r -= p as f64;
        if r <= 0.0 {
            return i as i32;
        }
    }
    // FP slack fallback: last non-zero.
    for i in (0..d.len()).rev() {
        if d[i] > 0.0 {
            return i as i32;
        }
    }
    0
}

/// Residual distribution `norm(max(0, p - q))`. Returns `None` if the residual
/// mass is ~0 (p ≈ q), in which case the caller falls back to sampling from p.
fn residual_dist(p: &[f32], q: &[f32]) -> Option<Vec<f32>> {
    let n = p.len();
    let mut r = vec![0f32; n];
    let mut z = 0f64;
    for i in 0..n {
        let d = p[i] - q[i];
        if d > 0.0 {
            r[i] = d;
            z += d as f64;
        }
    }
    if z <= 1e-9 {
        return None;
    }
    let inv_z = 1.0f64 / z;
    for x in r.iter_mut() {
        *x = (*x as f64 * inv_z) as f32;
    }
    Some(r)
}

impl HeterogeneousEngine {
    /// Run the K=1 MTP **rejection-sampling** spec-decode loop.
    ///
    /// Same wiring as [`spec_decode_run`](Self::spec_decode_run) but the accept
    /// rule is speculative *sampling* (target-distribution-correct at T/min_p),
    /// not greedy argmax. See the module `SAMPLING` section for the algorithm.
    #[allow(clippy::too_many_arguments)]
    pub fn spec_decode_sample_run(
        &self,
        bd: &mut BatchDgpuScratch,
        bi: &mut BatchIgpuScratch,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        head_scratch: &mut DgpuScratch,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        mtp_scratch: &mut MtpScratch,
        mtp_state: &mut MtpLayerState,
        mtp_weights: &MtpWeights,
        frontier: &mut FrontierSnapshot,
        prime_tokens: &[i32],
        embed: &dyn Fn(i32) -> Vec<f32>,
        cfg: SpecSampleConfig,
    ) -> eyre::Result<SpecSampleOut> {
        if prime_tokens.is_empty() {
            return Err(eyre!("spec_decode_sample_run: prime_tokens is empty"));
        }
        if cfg.temperature <= 0.0 {
            return Err(eyre!(
                "spec_decode_sample_run: temperature must be > 0 (got {}); use the greedy loop for T=0",
                cfg.temperature
            ));
        }
        let dgpu = self.dgpu.device;
        let hc_dim = HC_DIM as usize;
        let vocab = N_VOCAB as usize;
        let t = cfg.temperature;
        let min_p = cfg.min_p;

        self.clear_graphs();

        // ---------- Prime the prompt (B=1 verify per position) ----------
        let mut last_hc = vec![0f32; hc_dim];
        let mut last_logits = vec![0f32; vocab];
        for (pos, &tok) in prime_tokens.iter().enumerate() {
            let inp = broadcast_hc(&embed(tok));
            let out = self.forward_verify(
                bd, bi, dgpu_scratch, igpu_scratch, head_scratch, state, weights,
                &[inp], &[tok], pos as u32, false,
            )?;
            if pos + 1 == prime_tokens.len() {
                last_hc.copy_from_slice(&out.hc);
                last_logits.copy_from_slice(&out.logits);
            }
        }

        let mut p: u32 = (prime_tokens.len() - 1) as u32; // last forwarded pos
        let mut tok_at_p: i32 = *prime_tokens.last().unwrap();
        // p1 = target dist for token @ P+1 (from the last clean forward's logits)
        let mut p1_dist = softmax_dist(&last_logits, t, min_p);
        let mut main_hc_dev = DeviceBuffer::<f32>::new(dgpu.id, hc_dim)?;
        main_hc_dev.copy_from_host(&last_hc)?; // HC @ p
        let mut hid1 = DeviceBuffer::<f32>::new(dgpu.id, hc_dim)?;
        let mut hc_tmp = DeviceBuffer::<f32>::new(dgpu.id, hc_dim)?;

        let mut rng = SamplerRng::new(cfg.seed);
        let mut mtp_logits = vec![0f32; vocab];
        let mut out_tokens: Vec<i32> = Vec::with_capacity(cfg.n_tokens + 2);
        let mut target_dists: Vec<Vec<f32>> = Vec::new();
        let mut stats = SpecSampleStats::default();
        let debug = std::env::var("SPEC_DEBUG").is_ok();

        // Commit a token whose target dist is `dist`; records + advances counts.
        macro_rules! record {
            ($tok:expr, $dist:expr) => {{
                let tk: i32 = $tok;
                out_tokens.push(tk);
                stats.committed += 1;
                if cfg.collect_dists {
                    target_dists.push($dist);
                }
            }};
        }

        while out_tokens.len() < cfg.n_tokens {
            stats.rounds += 1;
            let mtp_base = mtp_state.n_raw;

            // ---- draft x1 ~ q1 = softmax(mtp_draft(HC@P, tok@P)/T) ----
            self.forward_mtp_draft(
                dgpu_scratch, igpu_scratch, mtp_scratch, mtp_state, &weights.global,
                mtp_weights, &main_hc_dev, &embed(tok_at_p), p, tok_at_p,
            )?;
            hid1.copy_from_buffer(&mtp_scratch.mtp_next_hc)?;
            mtp_scratch.mtp_logits.copy_to_host(&mut mtp_logits)?;
            let q1 = softmax_dist(&mtp_logits, t, min_p);
            let x1 = sample_from_dist(&q1, rng.next_f32());
            stats.draft1 += 1;

            // accept x1 w.p. min(1, p1(x1)/q1(x1))
            let (px1, qx1) = (p1_dist[x1 as usize], q1[x1 as usize]);
            let ratio1 = if qx1 > 0.0 { (px1 as f64 / qx1 as f64) as f32 } else { f32::INFINITY };
            let accept1 = rng.next_f32() < ratio1.min(1.0);

            if debug {
                eprintln!(
                    "  [rnd {:>3}] p={p} x1={x1} p1={px1:.3e} q1={qx1:.3e} r={:.3} {}",
                    stats.rounds, ratio1.min(1.0), if accept1 { "ACC1" } else { "rej1" }
                );
            }

            if !accept1 {
                // ---- reject x1: emit residual token @ P+1, STOP ----
                let r1 = match residual_dist(&p1_dist, &q1) {
                    Some(res) => sample_from_dist(&res, rng.next_f32()),
                    None => sample_from_dist(&p1_dist, rng.next_f32()),
                };
                // Keep x1's MTP row (position P is valid). No frontier needed.
                mtp_state.n_raw = (mtp_base + 1).min(SWA_WINDOW);
                let dist_used = std::mem::replace(&mut p1_dist, Vec::new());
                let inp = broadcast_hc(&embed(r1));
                let out = self.forward_verify(
                    bd, bi, dgpu_scratch, igpu_scratch, head_scratch, state, weights,
                    &[inp], &[r1], p + 1, false,
                )?;
                record!(r1, dist_used);
                main_hc_dev.copy_from_host(&out.hc)?;
                tok_at_p = r1;
                p += 1;
                p1_dist = softmax_dist(&out.logits, t, min_p);
                continue;
            }
            stats.draft1_accept += 1;

            // ---- x1 accepted: draft x2 ~ q2 = softmax(mtp_draft(hid1, x1)/T) ----
            self.forward_mtp_draft(
                dgpu_scratch, igpu_scratch, mtp_scratch, mtp_state, &weights.global,
                mtp_weights, &hid1, &embed(x1), p + 1, x1,
            )?;
            mtp_scratch.mtp_logits.copy_to_host(&mut mtp_logits)?;
            let q2 = softmax_dist(&mtp_logits, t, min_p);
            let x2 = sample_from_dist(&q2, rng.next_f32());
            stats.draft2 += 1;

            // ---- snapshot frontier @P, then B=2 verify [x1@P+1, x2@P+2] ----
            state.capture_frontier(dgpu, frontier)?;
            let inp0 = broadcast_hc(&embed(x1));
            let inp1 = broadcast_hc(&embed(x2));
            let out2 = self.forward_verify(
                bd, bi, dgpu_scratch, igpu_scratch, head_scratch, state, weights,
                &[inp0, inp1], &[x1, x2], p + 1, false,
            )?;
            let p2_dist = softmax_dist(&out2.logits[0..vocab], t, min_p); // target @P+2 | x1
            let p3_logits = out2.logits[vocab..2 * vocab].to_vec();
            let hc1_row = out2.hc[hc_dim..2 * hc_dim].to_vec(); // main HC @ P+2

            // accept x2 w.p. min(1, p2(x2)/q2(x2))
            let (px2, qx2) = (p2_dist[x2 as usize], q2[x2 as usize]);
            let ratio2 = if qx2 > 0.0 { (px2 as f64 / qx2 as f64) as f32 } else { f32::INFINITY };
            let accept2 = rng.next_f32() < ratio2.min(1.0);

            if debug {
                eprintln!(
                    "           x2={x2} p2={px2:.3e} q2={qx2:.3e} r={:.3} {}",
                    ratio2.min(1.0), if accept2 { "ACC2" } else { "rej2" }
                );
            }

            if accept2 {
                stats.draft2_accept += 1;
                // ---- accept x1, x2; emit bonus b ~ p3. Bank batched KV. ----
                let p3_dist = softmax_dist(&p3_logits, t, min_p);
                let bonus = sample_from_dist(&p3_dist, rng.next_f32());
                let dist1 = std::mem::replace(&mut p1_dist, Vec::new());
                record!(x1, dist1);
                record!(x2, p2_dist);

                // Realign MTP KV: append a drafter row for position P+2 (x2) so
                // the drafter stays gap-free through the bonus. Uses main HC@P+2.
                hc_tmp.copy_from_host(&hc1_row)?;
                self.forward_mtp_draft(
                    dgpu_scratch, igpu_scratch, mtp_scratch, mtp_state, &weights.global,
                    mtp_weights, &hc_tmp, &embed(x2), p + 2, x2,
                )?;
                mtp_state.n_raw = (mtp_base + 3).min(SWA_WINDOW);

                // Forward bonus @ P+3 (B=1) for clean next-round p1 + HC.
                let inp_b = broadcast_hc(&embed(bonus));
                let out_b = self.forward_verify(
                    bd, bi, dgpu_scratch, igpu_scratch, head_scratch, state, weights,
                    &[inp_b], &[bonus], p + 3, false,
                )?;
                record!(bonus, p3_dist);
                stats.bonus += 1;
                main_hc_dev.copy_from_host(&out_b.hc)?;
                tok_at_p = bonus;
                p += 3;
                p1_dist = softmax_dist(&out_b.logits, t, min_p);
            } else {
                // ---- accept x1, reject x2: emit residual @P+2, STOP. ----
                // Roll back the batched P+1/P+2 KV, replay x1 then r2 as B=1.
                state.restore_frontier(dgpu, frontier)?;
                let r2 = match residual_dist(&p2_dist, &q2) {
                    Some(res) => sample_from_dist(&res, rng.next_f32()),
                    None => sample_from_dist(&p2_dist, rng.next_f32()),
                };
                // Keep both drafter rows (positions P=x1-cond, P+1=x1).
                mtp_state.n_raw = (mtp_base + 2).min(SWA_WINDOW);
                let dist1 = std::mem::replace(&mut p1_dist, Vec::new());
                let inp_x1 = broadcast_hc(&embed(x1));
                self.forward_verify(
                    bd, bi, dgpu_scratch, igpu_scratch, head_scratch, state, weights,
                    &[inp_x1], &[x1], p + 1, false,
                )?;
                record!(x1, dist1);
                let inp_r2 = broadcast_hc(&embed(r2));
                let out_r2 = self.forward_verify(
                    bd, bi, dgpu_scratch, igpu_scratch, head_scratch, state, weights,
                    &[inp_r2], &[r2], p + 2, false,
                )?;
                record!(r2, p2_dist);
                main_hc_dev.copy_from_host(&out_r2.hc)?;
                tok_at_p = r2;
                p += 2;
                p1_dist = softmax_dist(&out_r2.logits, t, min_p);
            }
        }

        out_tokens.truncate(cfg.n_tokens);
        if cfg.collect_dists {
            target_dists.truncate(cfg.n_tokens);
        }
        Ok(SpecSampleOut {
            tokens: out_tokens,
            stats,
            target_dists,
        })
    }
}
