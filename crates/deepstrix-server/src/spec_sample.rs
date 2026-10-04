//! Exact speculative sampling for DSpark (docs/v41/DSPARK_ARENA_PLAN.md
//! section 2, milestone M1). Host-only: the target distribution `p` of a verify
//! row, the per-position rejection-sampling test, and the block procedure.
//!
//! Ground rule (owner): ALWAYS rejection sample, never accept on
//! `argmax(p) == d` for a sampled request. A draft `d ~ q` is accepted with
//! probability `min(1, p(d) / q(d))`; on reject the emitted token is drawn from
//! `norm(max(0, p - q))`; after all K drafts are accepted a bonus token is drawn
//! from the last row's `p`. Whatever `q` is, the emitted sequence is distributed
//! exactly as plain sampling from the target, so the drafter can only change HOW
//! MANY tokens a step yields, never which (tests: G-RS1 below).
//!
//! `p` is defined ONCE, by [`TargetDist::from_logits`], which is the arena
//! sampler's own arithmetic (`multistream::sample_row` now calls it), so the
//! plain path and the speculative path cannot drift apart (G-RS2 below proves
//! the refactor emits the old sampler's tokens bit for bit).
//!
//! What may decide how many drafts to verify (plan 2.4): anything known before
//! the verify that does not depend on the DRAWN value of a sampled draft. The
//! point-mass test is exact for any draft value, so a position whose inclusion
//! looked past that rule must be tested with [`DraftDist::PointMass`]. The
//! negative-control test below shows the chi-square catches a violation.

use v4flash_kernels::het::SampleMode;
use v4flash_kernels::sampler::SamplerRng;

/// Weights below this (relative to the top token) cannot move a top-p cutoff
/// by more than `N_VOCAB * FLOOR` of the mass; see `from_logits`.
pub(crate) const FLOOR: f32 = 1e-10;

/// Below this total, `max(0, p - q)` is treated as empty (p ~= q to rounding):
/// draw from `p` instead of normalising ~0 (plan 2.3).
const RESIDUAL_EPS: f64 = 1e-12;

/// The target distribution `p` of one verify row: EXACTLY what the arena
/// sampler draws from (temperature, then top-p over the tempered weights, then
/// min-p). It keeps that sampler's own f32 arithmetic -- unnormalised weights,
/// the normaliser summed in the same order, the vocab-order cumulative walk --
/// so [`TargetDist::sample`] reproduces the plain sampler token for token.
#[derive(Clone, Debug, PartialEq)]
pub enum TargetDist {
    /// Temperature 0: a point mass on the FIRST maximal logit.
    Argmax(i32),
    /// The survivors (weight >= threshold), in VOCAB order.
    Weighted {
        ids: Vec<u32>,
        w: Vec<f32>,
        /// f32 normaliser, summed exactly as the plain sampler sums it.
        z: f32,
        /// What the walk returns if it visits no survivor. Unreachable for a
        /// valid mode (the top token always survives); kept so the refactor is
        /// bit-identical even for degenerate modes.
        fallback: i32,
    },
}

impl TargetDist {
    /// The distribution the arena sampler draws from for logit row `r`.
    ///
    /// Temperature, then min-p (survivors within `min_p_rel` of the top
    /// weight), then top-p over the SURVIVORS, threshold `max(top-p cut,
    /// min_p_rel)` -- the order the drafter's q follows (KNOWN_BUGS #50),
    /// without a 129K-entry f64 exp + full sort per row. (Corrected 2026-10-04:
    /// this said "same chain as `top_p_min_p_threshold`", which takes its top-p
    /// cut over ALL tempered weights.) The weight is `exp(logit/T - gmax)` in (0, 1]; entries
    /// below FLOOR cannot move a top-p cutoff by more than `N_VOCAB * FLOOR` of
    /// the mass (< 1e-5 of the total, which is >= 1), so only the survivors are
    /// sorted -- typically a few hundred.
    pub fn from_logits(r: &[f32], mode: &SampleMode) -> Self {
        match *mode {
            SampleMode::Argmax => {
                let mut best = 0usize;
                for (i, &x) in r.iter().enumerate() {
                    if x > r[best] {
                        best = i;
                    }
                }
                TargetDist::Argmax(best as i32)
            }
            SampleMode::Multinomial { temperature, min_p_rel, top_p } => {
                let inv_t = 1.0f32 / temperature;
                let gmax = r.iter().copied().fold(f32::NEG_INFINITY, f32::max) * inv_t;
                let lo = (min_p_rel.max(FLOOR)).ln(); // survivors: x*inv_t - gmax >= lo
                let mut cand: Vec<(f32, u32)> = Vec::with_capacity(512);
                for (i, &x) in r.iter().enumerate() {
                    let l = x * inv_t - gmax;
                    if l >= lo {
                        cand.push((l.exp(), i as u32));
                    }
                }
                // top-p cutoff over the survivors (sorted descending).
                let thr = if top_p >= 1.0 {
                    0.0f32
                } else {
                    cand.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    let z: f32 = cand.iter().map(|c| c.0).sum();
                    let target = top_p * z;
                    let mut cum = 0.0f32;
                    let mut t = cand.last().map(|c| c.0).unwrap_or(0.0);
                    for c in &cand {
                        cum += c.0;
                        if cum >= target {
                            t = c.0;
                            break;
                        }
                    }
                    t
                }
                .max(min_p_rel);
                // Normaliser in the order the plain sampler summed it (weight-
                // descending when top-p sorted, vocab order otherwise).
                let z: f32 = cand.iter().filter(|c| c.0 >= thr).map(|c| c.0).sum();
                // The plain sampler initialised its pick BEFORE re-sorting.
                let fallback = cand.first().map(|c| c.1 as i32).unwrap_or(0);
                // Cumulative walk in VOCAB order, over the survivors.
                if top_p < 1.0 {
                    cand.sort_unstable_by_key(|c| c.1);
                }
                let mut ids = Vec::with_capacity(cand.len());
                let mut w = Vec::with_capacity(cand.len());
                for c in &cand {
                    if c.0 >= thr {
                        ids.push(c.1);
                        w.push(c.0);
                    }
                }
                TargetDist::Weighted { ids, w, z, fallback }
            }
        }
    }

    /// `from_logits` from what a decode head ships INSTEAD of the 517 KB row
    /// (`HeterogeneousEngine::head_cands`): the row's `max`, `z_all` = the
    /// survivors' weight sum computed on the device, and `cands` = every
    /// `(logit, id)` within `band` nats of the max at the row's temperature,
    /// in any order. `None` when they cannot serve the row -- top-p >= 1 (the
    /// whole survivor set is the support), the nucleus reaching past the
    /// band, no candidate, a non-finite total -- and the caller uses the row.
    ///
    /// The same computation as `from_logits` over the same tokens, except the
    /// top-p target `top_p * z`: `from_logits` sums every survivor on the host
    /// in weight-descending order, this takes the device's sum of the same set
    /// (both f32, ~1e-7 relative apart), so the cutoff can only differ when
    /// the cumulative mass crosses the target within that much. When every
    /// survivor is a candidate (min-p above the band) the host sum is used and
    /// the result is bit-identical; Argmax is bit-identical (lowest id among
    /// the maxima). The cutoff must sit a margin inside the band, so a
    /// device/host rounding difference in a token's log-weight (the device
    /// fuses `x * inv_t - gmax`; ~ulp of `x * inv_t`, which grows with the
    /// logits' magnitude) cannot leave out a token that ties into the nucleus.
    pub fn from_cands(max: f32, z_all: f32, cands: &[(f32, u32)], band: f32, mode: &SampleMode) -> Option<Self> {
        let margin = |inv_t: f32| 1e-3 + 4.0 * (max.abs() + band / inv_t) * inv_t * f32::EPSILON;
        match *mode {
            SampleMode::Argmax => cands.iter().filter(|c| c.0 == max).map(|c| c.1).min().map(|id| TargetDist::Argmax(id as i32)),
            SampleMode::Multinomial { temperature, min_p_rel, top_p } => {
                if top_p >= 1.0 || !(z_all.is_finite() && z_all > 0.0) {
                    return None;
                }
                let inv_t = 1.0f32 / temperature;
                let gmax = max * inv_t;
                let lo = (min_p_rel.max(FLOOR)).ln();
                // Survivors among the candidates, pushed in VOCAB order as
                // `from_logits` pushes them.
                let mut by_id: Vec<(f32, u32)> = cands.to_vec();
                by_id.sort_unstable_by_key(|c| c.1);
                let mut cand: Vec<(f32, u32)> = Vec::with_capacity(by_id.len());
                for &(x, id) in &by_id {
                    let l = x * inv_t - gmax;
                    if l >= lo {
                        cand.push((l.exp(), id));
                    }
                }
                if cand.is_empty() {
                    return None;
                }
                let m = margin(inv_t);
                let all_survivors = lo >= -band + m;
                cand.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                let z = if all_survivors { cand.iter().map(|c| c.0).sum::<f32>() } else { z_all };
                let target = top_p * z;
                let mut cum = 0.0f32;
                let mut cut = None;
                for c in &cand {
                    cum += c.0;
                    if cum >= target {
                        cut = Some(c.0);
                        break;
                    }
                }
                let t = cut?;
                if !all_survivors && t < (-band + m).exp() {
                    return None;
                }
                let thr = t.max(min_p_rel);
                let z: f32 = cand.iter().filter(|c| c.0 >= thr).map(|c| c.0).sum();
                let fallback = cand.first().map(|c| c.1 as i32).unwrap_or(0);
                cand.sort_unstable_by_key(|c| c.1);
                let mut ids = Vec::with_capacity(cand.len());
                let mut w = Vec::with_capacity(cand.len());
                for c in &cand {
                    if c.0 >= thr {
                        ids.push(c.1);
                        w.push(c.0);
                    }
                }
                Some(TargetDist::Weighted { ids, w, z, fallback })
            }
        }
    }

    /// One draw with uniform `u` in [0, 1): the plain sampler's walk.
    pub fn draw(&self, u: f32) -> i32 {
        match self {
            TargetDist::Argmax(a) => *a,
            TargetDist::Weighted { ids, w, z, fallback } => {
                let target = u * *z;
                let mut acc = 0.0f32;
                let mut pick = *fallback;
                for (k, &wk) in w.iter().enumerate() {
                    acc += wk;
                    pick = ids[k] as i32;
                    if acc >= target {
                        break;
                    }
                }
                pick
            }
        }
    }

    /// What the arena sampler does for one row: one uniform for a sampled row,
    /// none at temperature 0.
    pub fn sample(&self, rng: &mut SamplerRng) -> i32 {
        match self {
            TargetDist::Argmax(a) => *a,
            TargetDist::Weighted { .. } => self.draw(rng.next_f32()),
        }
    }

    /// `p(x)` (0 outside the support).
    pub fn prob(&self, x: i32) -> f64 {
        match self {
            TargetDist::Argmax(a) => {
                if x == *a {
                    1.0
                } else {
                    0.0
                }
            }
            TargetDist::Weighted { ids, w, z, .. } => {
                if x < 0 {
                    return 0.0;
                }
                match ids.binary_search(&(x as u32)) {
                    Ok(k) => w[k] as f64 / *z as f64,
                    Err(_) => 0.0,
                }
            }
        }
    }
}

/// How a draft is tested.
#[derive(Clone, Debug, PartialEq)]
pub enum DraftDist {
    /// The point-mass test: draw `y ~ p`, keep the draft iff `y == d`, else emit
    /// `y`. Exact for ANY draft value drawn independently of the verifier's
    /// uniforms (plan 2.4), so it is both the test for argmax drafts and the
    /// safe test for a sampled draft whose inclusion looked past the stopping
    /// rule. Costs exactly the plain sampler's one uniform.
    PointMass,
    /// The exact distribution the draft was drawn from, as (token, prob) pairs
    /// (e.g. the drafter's tempered top-M, renormalised). Tested with
    /// `min(1, p(d)/q(d))`; on reject the token comes from `norm(max(0, p - q))`.
    Sampled(Vec<(i32, f64)>),
}

/// One draft token and how to test it.
#[derive(Clone, Debug, PartialEq)]
pub struct Draft {
    pub token: i32,
    pub q: DraftDist,
}

/// Outcome of testing one position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The draft is emitted; the block continues.
    Accept(i32),
    /// This token is emitted instead and the block ends.
    Reject(i32),
}

fn q_of(q: &[(i32, f64)], x: i32) -> f64 {
    q.iter().filter(|(t, _)| *t == x).map(|(_, pr)| *pr).sum()
}

/// Draw from `norm(max(0, p - q))`; `None` when that mass is ~0.
fn residual_draw(p: &TargetDist, q: &[(i32, f64)], u: f32) -> Option<i32> {
    match p {
        // p is a point mass on `a`: max(0, p - q) lives on `a` alone.
        TargetDist::Argmax(a) => {
            if 1.0 - q_of(q, *a) > RESIDUAL_EPS {
                Some(*a)
            } else {
                None
            }
        }
        TargetDist::Weighted { ids, w, z, .. } => {
            // Only p's support can carry positive residual mass.
            let zf = *z as f64;
            let r: Vec<f64> = ids
                .iter()
                .zip(w)
                .map(|(&id, &wk)| (wk as f64 / zf - q_of(q, id as i32)).max(0.0))
                .collect();
            let s: f64 = r.iter().sum();
            // `!(s > eps)` also catches a NaN.
            if !(s > RESIDUAL_EPS) {
                return None;
            }
            let target = u as f64 * s;
            let mut acc = 0.0f64;
            let mut pick = None;
            for (k, &rk) in r.iter().enumerate() {
                if rk <= 0.0 {
                    continue;
                }
                acc += rk;
                pick = Some(ids[k] as i32);
                if acc >= target {
                    break;
                }
            }
            pick
        }
    }
}

/// Test one draft against its verify row's target distribution.
pub fn verify_position(p: &TargetDist, d: &Draft, rng: &mut SamplerRng) -> Verdict {
    let point_mass = |rng: &mut SamplerRng| {
        let y = p.sample(rng);
        if y == d.token {
            Verdict::Accept(y)
        } else {
            Verdict::Reject(y)
        }
    };
    let q = match &d.q {
        DraftDist::PointMass => return point_mass(rng),
        DraftDist::Sampled(q) => q,
    };
    let qd = q_of(q, d.token);
    if !(qd > 0.0) {
        // A draft outside its own q cannot have been drawn from it, so the ratio
        // test does not apply; the point-mass test is exact for any draft.
        return point_mass(rng);
    }
    let a = (p.prob(d.token) / qd).min(1.0);
    if (rng.next_f32() as f64) < a {
        return Verdict::Accept(d.token);
    }
    match residual_draw(p, q, rng.next_f32()) {
        Some(y) => Verdict::Reject(y),
        // p ~= q: the residual is empty to rounding, so a reject here is a
        // rounding artefact; draw from p itself (never normalise ~0).
        None => Verdict::Reject(p.sample(rng)),
    }
}

/// What one verify block emits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockOutcome {
    /// In order: `accepted` drafts, then one token -- the correction after a
    /// rejection, or the bonus after all drafts were accepted. The caller emits
    /// them one by one and checks its stop conditions after EACH; the number it
    /// actually emits is the number of block rows whose KV it keeps (row 0's
    /// input is the previous step's token, row j's is draft j-1).
    pub tokens: Vec<i32>,
    pub accepted: usize,
}

/// The block procedure (plan 2.3). `rows[j]` is the target distribution
/// produced by verify row j, whose input was `next` (j = 0) or `drafts[j-1]`;
/// so `rows.len() == drafts.len() + 1`.
pub fn verify_block(rows: &[TargetDist], drafts: &[Draft], rng: &mut SamplerRng) -> BlockOutcome {
    assert_eq!(
        rows.len(),
        drafts.len() + 1,
        "verify_block: {} rows for {} drafts (need drafts + 1)",
        rows.len(),
        drafts.len()
    );
    let mut tokens = Vec::with_capacity(rows.len());
    for (j, d) in drafts.iter().enumerate() {
        match verify_position(&rows[j], d, rng) {
            Verdict::Accept(t) => tokens.push(t),
            Verdict::Reject(t) => {
                tokens.push(t);
                return BlockOutcome { tokens, accepted: j };
            }
        }
    }
    tokens.push(rows[drafts.len()].sample(rng));
    BlockOutcome { tokens, accepted: drafts.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The arena sampler BEFORE the refactor, verbatim (multistream.rs at
    /// d4f7e3d). G-RS2 holds the new path to it bit for bit.
    fn sample_row_reference(r: &[f32], mode: &SampleMode, rng: &mut SamplerRng) -> i32 {
        match *mode {
            SampleMode::Argmax => {
                let mut best = 0usize;
                for (i, &x) in r.iter().enumerate() {
                    if x > r[best] {
                        best = i;
                    }
                }
                best as i32
            }
            SampleMode::Multinomial { temperature, min_p_rel, top_p } => {
                const FLOOR: f32 = 1e-10;
                let inv_t = 1.0f32 / temperature;
                let gmax = r.iter().copied().fold(f32::NEG_INFINITY, f32::max) * inv_t;
                let lo = (min_p_rel.max(FLOOR)).ln();
                let mut cand: Vec<(f32, u32)> = Vec::with_capacity(512);
                for (i, &x) in r.iter().enumerate() {
                    let l = x * inv_t - gmax;
                    if l >= lo {
                        cand.push((l.exp(), i as u32));
                    }
                }
                let thr = if top_p >= 1.0 {
                    0.0f32
                } else {
                    cand.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    let z: f32 = cand.iter().map(|c| c.0).sum();
                    let target = top_p * z;
                    let mut cum = 0.0f32;
                    let mut t = cand.last().map(|c| c.0).unwrap_or(0.0);
                    for c in &cand {
                        cum += c.0;
                        if cum >= target {
                            t = c.0;
                            break;
                        }
                    }
                    t
                }
                .max(min_p_rel);
                let z: f32 = cand.iter().filter(|c| c.0 >= thr).map(|c| c.0).sum();
                let target = rng.next_f32() * z;
                let mut acc = 0.0f32;
                let mut pick = cand.first().map(|c| c.1).unwrap_or(0);
                if top_p < 1.0 {
                    cand.sort_unstable_by_key(|c| c.1);
                }
                for c in &cand {
                    if c.0 < thr {
                        continue;
                    }
                    acc += c.0;
                    pick = c.1;
                    if acc >= target {
                        break;
                    }
                }
                pick as i32
            }
        }
    }

    fn gauss(rng: &mut SamplerRng) -> f32 {
        // Box-Muller from two uniforms in (0, 1].
        let u1 = 1.0 - rng.next_f32();
        let u2 = rng.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// G-RS2: the refactored sampler emits exactly the old sampler's tokens,
    /// consuming the RNG identically, across logit shapes (peaked, flat, ties,
    /// huge negatives) and modes (argmax, temperatures, top-p, min-p).
    #[test]
    fn g_rs2_refactor_is_bit_identical_to_the_old_sampler() {
        let modes = {
            let mut m = vec![SampleMode::Argmax];
            for &temperature in &[0.3f32, 0.7, 1.0, 1.5] {
                for &top_p in &[1.0f32, 0.95, 0.8, 0.5] {
                    for &min_p_rel in &[0.0f32, 0.05] {
                        m.push(SampleMode::Multinomial { temperature, min_p_rel, top_p });
                    }
                }
            }
            m
        };
        let mut data = SamplerRng::new(7);
        let mut checked = 0usize;
        for shape in 0..6 {
            for _row in 0..60 {
                let v = 3000usize;
                let row: Vec<f32> = (0..v)
                    .map(|_| match shape {
                        0 => gauss(&mut data) * 2.0,             // ordinary
                        1 => gauss(&mut data) * 8.0,             // peaked
                        2 => gauss(&mut data) * 0.2,             // flat
                        3 => (gauss(&mut data) * 2.0).round(),   // many exact ties
                        4 => -1e4 + gauss(&mut data),            // huge negatives
                        _ => if data.next_f32() < 0.01 { 5.0 } else { -3.0 }, // tied top cluster
                    })
                    .collect();
                for mode in &modes {
                    let seed = (checked as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    let (mut a, mut b) = (SamplerRng::new(seed), SamplerRng::new(seed));
                    for _ in 0..4 {
                        let want = sample_row_reference(&row, mode, &mut a);
                        let got = TargetDist::from_logits(&row, mode).sample(&mut b);
                        assert_eq!(got, want, "shape {shape} mode {mode:?}");
                    }
                    // Same RNG consumption: the streams are still in lockstep.
                    assert_eq!(a.next_f32().to_bits(), b.next_f32().to_bits(), "RNG drifted, mode {mode:?}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 10_000);
    }

    /// What `logits_nucleus_cands` ships for row `r` (host emulation, its `z`
    /// summed in `from_logits`' order so equality can be exact).
    fn emulate_cands(r: &[f32], mode: &SampleMode, band: f32) -> (f32, f32, Vec<(f32, u32)>) {
        let max = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let (inv_t, lo, band) = match *mode {
            SampleMode::Argmax => (1.0f32, f32::INFINITY, 0.0f32),
            SampleMode::Multinomial { temperature, min_p_rel, .. } => (1.0 / temperature, min_p_rel.max(FLOOR).ln(), band),
        };
        let gmax = max * inv_t;
        let mut surv: Vec<f32> = Vec::new();
        let mut cands = Vec::new();
        for (i, &x) in r.iter().enumerate() {
            let l = x * inv_t - gmax;
            if l >= lo {
                surv.push(l.exp());
            }
            if l >= -band {
                cands.push((x, i as u32));
            }
        }
        surv.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        let z = surv.iter().sum::<f32>();
        cands.reverse(); // the device ships them in any order
        (max, z, cands)
    }

    fn same_dist(a: &TargetDist, b: &TargetDist) -> bool {
        match (a, b) {
            (TargetDist::Argmax(x), TargetDist::Argmax(y)) => x == y,
            (TargetDist::Weighted { ids: i1, w: w1, z: z1, .. }, TargetDist::Weighted { ids: i2, w: w2, z: z2, .. }) => {
                i1 == i2 && z1.to_bits() == z2.to_bits() && w1.iter().zip(w2).all(|(p, q)| p.to_bits() == q.to_bits())
            }
            _ => false,
        }
    }

    /// G-RS3: the decode head's candidates (`from_cands`) give the SAME
    /// distribution as the full row (`from_logits`) whenever they serve the
    /// row, given the same survivor total; they decline top-p 1.
    #[test]
    fn g_rs3_head_candidates_reproduce_the_row_sampler() {
        let mut modes = vec![SampleMode::Argmax];
        for &temperature in &[0.3f32, 0.7, 1.0, 1.5] {
            for &top_p in &[1.0f32, 0.95, 0.8, 0.5] {
                for &min_p_rel in &[0.0f32, 0.05] {
                    modes.push(SampleMode::Multinomial { temperature, min_p_rel, top_p });
                }
            }
        }
        let mut data = SamplerRng::new(11);
        let (mut served, mut declined) = (0usize, 0usize);
        for shape in 0..6 {
            for row_i in 0..40 {
                let v = if row_i % 2 == 0 { 3000 } else { 20_000 };
                let row: Vec<f32> = (0..v)
                    .map(|_| match shape {
                        0 => gauss(&mut data) * 2.0,
                        1 => gauss(&mut data) * 8.0,
                        2 => gauss(&mut data) * 0.2,
                        3 => (gauss(&mut data) * 2.0).round(),
                        4 => -1e4 + gauss(&mut data),
                        _ => if data.next_f32() < 0.01 { 5.0 } else { -3.0 },
                    })
                    .collect();
                for mode in &modes {
                    for band in [10.0f32, 4.0] {
                        let (max, z, cands) = emulate_cands(&row, mode, band);
                        match TargetDist::from_cands(max, z, &cands, band, mode) {
                            Some(d) => {
                                assert!(same_dist(&d, &TargetDist::from_logits(&row, mode)), "shape {shape} v {v} mode {mode:?} band {band}");
                                served += 1;
                            }
                            None => {
                                declined += 1;
                                // Declining is only ever about the support: top-p 1,
                                // or a nucleus holding a token outside (the margin of)
                                // the band.
                                match (*mode, TargetDist::from_logits(&row, mode)) {
                                    (SampleMode::Multinomial { top_p, temperature, .. }, TargetDist::Weighted { w, .. }) => assert!(
                                        top_p >= 1.0
                                            || w.iter().any(|&x| x < (-band + 1e-3 + 4.0 * (max.abs() + band * temperature) / temperature * f32::EPSILON).exp()),
                                        "declined a nucleus inside the band: shape {shape} mode {mode:?} band {band}"
                                    ),
                                    _ => panic!("declined an argmax row: shape {shape}"),
                                }
                            }
                        }
                    }
                    if let SampleMode::Multinomial { top_p, .. } = *mode {
                        if top_p >= 1.0 {
                            let (max, z, cands) = emulate_cands(&row, mode, 10.0);
                            assert!(TargetDist::from_cands(max, z, &cands, 10.0, mode).is_none());
                        }
                    }
                }
            }
        }
        assert!(served > 10_000, "served {served} declined {declined}");
    }

    /// The device sums the survivors in another order: a 1e-6 relative change
    /// of the total may move the top-p cutoff, but only for cumulative masses
    /// within that much of the target -- draws almost never change.
    #[test]
    fn g_rs3_a_device_order_total_barely_moves_draws() {
        let mode = SampleMode::Multinomial { temperature: 1.0, min_p_rel: 0.0, top_p: 0.95 };
        let mut data = SamplerRng::new(5);
        let (mut draws, mut moved) = (0usize, 0usize);
        for _ in 0..200 {
            let row: Vec<f32> = (0..20_000).map(|_| gauss(&mut data) * 3.0).collect();
            let (max, z, cands) = emulate_cands(&row, &mode, 10.0);
            let (Some(a), Some(b)) = (
                TargetDist::from_cands(max, z, &cands, 10.0, &mode),
                TargetDist::from_cands(max, z * (1.0 + 1e-6), &cands, 10.0, &mode),
            ) else {
                continue;
            };
            for k in 0..500 {
                let u = (k as f32 + 0.5) / 500.0;
                draws += 1;
                moved += usize::from(a.draw(u) != b.draw(u));
            }
        }
        assert!(draws > 50_000, "draws {draws}");
        assert!((moved as f64) < draws as f64 * 1e-3, "moved {moved} of {draws}");
    }

    /// A nucleus that reaches past the band is declined, not truncated.
    #[test]
    fn g_rs3_a_nucleus_past_the_band_is_declined() {
        let mode = SampleMode::Multinomial { temperature: 1.0, min_p_rel: 0.0, top_p: 0.95 };
        // One token at 0, 100,000 at -11 (each e^-11, 1.67 in all): the
        // nucleus needs them, the 10-nat band does not hold them.
        let mut row = vec![-11.0f32; 100_001];
        row[7] = 0.0;
        let (max, z, cands) = emulate_cands(&row, &mode, 10.0);
        assert_eq!(cands.len(), 1);
        assert!(TargetDist::from_cands(max, z, &cands, 10.0, &mode).is_none());
    }

    // ---- G-RS1: the emitted sequence is distributed as plain sampling ------

    const V: usize = 6;

    struct Toy {
        target: Vec<Vec<f32>>,
        drafter: Vec<Vec<f32>>,
        mode: SampleMode,
    }

    impl Toy {
        fn new(mode: SampleMode) -> Self {
            let mut rng = SamplerRng::new(11);
            let target: Vec<Vec<f32>> =
                (0..V).map(|_| (0..V).map(|_| gauss(&mut rng) * 1.5).collect()).collect();
            // A correlated but different drafter.
            let drafter: Vec<Vec<f32>> = target
                .iter()
                .map(|row| row.iter().map(|&x| x + gauss(&mut rng) * 1.2).collect())
                .collect();
            Toy { target, drafter, mode }
        }
        fn p(&self, prev: i32) -> TargetDist {
            TargetDist::from_logits(&self.target[prev as usize], &self.mode)
        }
        fn drafter_argmax(&self, prev: i32) -> i32 {
            let r = &self.drafter[prev as usize];
            (0..V).max_by(|&a, &b| r[a].partial_cmp(&r[b]).unwrap()).unwrap() as i32
        }
        /// The drafter's tempered top-`m`, renormalised: what a sampled draft is
        /// drawn from (plan 2.2), with support that can stray outside p's.
        fn drafter_q(&self, prev: i32, tau: f64, m: usize) -> Vec<(i32, f64)> {
            let r = &self.drafter[prev as usize];
            let mut idx: Vec<usize> = (0..V).collect();
            idx.sort_by(|&a, &b| r[b].partial_cmp(&r[a]).unwrap());
            idx.truncate(m);
            let mx = r[idx[0]] as f64;
            let w: Vec<f64> = idx.iter().map(|&i| ((r[i] as f64 - mx) / tau).exp()).collect();
            let s: f64 = w.iter().sum();
            idx.iter().zip(w).map(|(&i, wi)| (i as i32, wi / s)).collect()
        }
        /// q equal to p itself (every verified draft accepted).
        fn q_equals_p(&self, prev: i32) -> Vec<(i32, f64)> {
            let p = self.p(prev);
            (0..V as i32).map(|x| (x, p.prob(x))).filter(|(_, pr)| *pr > 0.0).collect()
        }
    }

    fn draw_sparse(q: &[(i32, f64)], u: f32) -> i32 {
        let target = u as f64;
        let mut acc = 0.0;
        for &(t, pr) in q {
            acc += pr;
            if acc >= target {
                return t;
            }
        }
        q.last().unwrap().0
    }

    #[derive(Clone, Copy)]
    enum Drafter {
        PointMass,
        Sampled { tau: f64, m: usize },
        QEqualsP,
    }

    /// The K policy of a run. `Fixed` / `Random` decide from things that do not
    /// depend on the drawn draft values (lossless); `PeekFirstDraft` verifies the
    /// first draft only when its DRAWN value is `a` (plan 2.4's counter-example).
    #[derive(Clone, Copy)]
    enum KPolicy {
        Fixed(usize),
        Random(usize),
        PeekFirstDraft { a: i32 },
        /// The production stopping rule (`ms_dspark::choose_k_stopping`) over
        /// a synthetic confidence that depends on the drafts through the
        /// markov-prev channel, as the real head does: conf_k is a function
        /// of d_{k-1}.
        #[cfg(feature = "v41")]
        Stopping,
    }

    /// The stopping rule's cost model, as configured (exactness must hold
    /// under any cost model).
    #[cfg(feature = "v41")]
    static STOPPING_COST: std::sync::LazyLock<crate::ms_dspark::StepCost> =
        std::sync::LazyLock::new(crate::ms_dspark::StepCost::from_env);

    /// Generate the first `len` tokens after `start` with speculative blocks.
    fn generate(toy: &Toy, start: i32, len: usize, drafter: Drafter, k: KPolicy, rng: &mut SamplerRng) -> Vec<i32> {
        let mut seq = Vec::with_capacity(len + 6);
        let mut prev = start;
        while seq.len() < len {
            let k_max = match k {
                KPolicy::Fixed(k) => k,
                KPolicy::Random(k) => (rng.next_f32() * (k + 1) as f32) as usize,
                KPolicy::PeekFirstDraft { .. } => 1,
                #[cfg(feature = "v41")]
                KPolicy::Stopping => 5,
            };
            let mut drafts = Vec::with_capacity(k_max);
            let mut dp = prev;
            for _ in 0..k_max {
                let d = match drafter {
                    Drafter::PointMass => Draft { token: toy.drafter_argmax(dp), q: DraftDist::PointMass },
                    Drafter::Sampled { tau, m } => {
                        let q = toy.drafter_q(dp, tau, m);
                        Draft { token: draw_sparse(&q, rng.next_f32()), q: DraftDist::Sampled(q) }
                    }
                    Drafter::QEqualsP => {
                        let q = toy.q_equals_p(dp);
                        Draft { token: draw_sparse(&q, rng.next_f32()), q: DraftDist::Sampled(q) }
                    }
                };
                dp = d.token;
                drafts.push(d);
            }
            if let KPolicy::PeekFirstDraft { a } = k {
                if drafts[0].token != a {
                    drafts.clear(); // the forbidden, draft-value-dependent truncation
                }
            }
            #[cfg(feature = "v41")]
            if let KPolicy::Stopping = k {
                // conf_j from the token BEFORE draft j (the stream's prev for
                // j = 0): strongly varying, so K really depends on the drafts.
                let mut conf = [0f32; 5];
                let mut before = prev;
                for (j, d) in drafts.iter().enumerate() {
                    let r = &toy.drafter[before as usize];
                    let mx = r.iter().cloned().fold(f32::MIN, f32::max);
                    conf[j] = (mx - r.iter().sum::<f32>() / r.len() as f32) * 2.0 - 2.0;
                    before = d.token;
                }
                let kk = crate::ms_dspark::choose_k_stopping(&conf, 5, &*STOPPING_COST);
                drafts.truncate(kk);
            }
            let mut rows = vec![toy.p(prev)];
            for d in &drafts {
                rows.push(toy.p(d.token));
            }
            let out = verify_block(&rows, &drafts, rng);
            assert_eq!(out.tokens.len(), out.accepted + 1);
            seq.extend_from_slice(&out.tokens);
            prev = *seq.last().unwrap();
        }
        seq.truncate(len);
        seq
    }

    /// Chi-square of the empirical `len`-token prefix distribution against the
    /// exact plain-sampling chain, with cells of expected count < 5 pooled.
    /// Returns (statistic, critical value at p = 1e-4).
    fn chi_square(toy: &Toy, start: i32, len: usize, drafter: Drafter, k: KPolicy, n: usize, seed: u64) -> (f64, f64) {
        let mut rng = SamplerRng::new(seed);
        let mut counts: HashMap<Vec<i32>, usize> = HashMap::new();
        for _ in 0..n {
            *counts.entry(generate(toy, start, len, drafter, k, &mut rng)).or_default() += 1;
        }
        // Exact probabilities of every prefix the chain can produce.
        let mut exact: Vec<(Vec<i32>, f64)> = vec![(vec![], 1.0)];
        for _ in 0..len {
            let mut next = Vec::new();
            for (prefix, pr) in &exact {
                let prev = *prefix.last().unwrap_or(&start);
                let p = toy.p(prev);
                for x in 0..V as i32 {
                    let px = p.prob(x);
                    if px > 0.0 {
                        let mut s = prefix.clone();
                        s.push(x);
                        next.push((s, pr * px));
                    }
                }
            }
            exact = next;
        }
        let mut stat = 0.0;
        let mut cells = 0usize;
        let (mut pooled_obs, mut pooled_exp) = (0.0f64, 0.0f64);
        let mut seen = 0usize;
        for (s, pr) in &exact {
            let e = pr * n as f64;
            let o = *counts.get(s).unwrap_or(&0) as f64;
            seen += o as usize;
            if e < 5.0 {
                pooled_obs += o;
                pooled_exp += e;
            } else {
                stat += (o - e) * (o - e) / e;
                cells += 1;
            }
        }
        // Any sequence the chain cannot produce lands in the pooled cell too.
        pooled_obs += (n - seen) as f64;
        if pooled_exp > 0.0 {
            stat += (pooled_obs - pooled_exp) * (pooled_obs - pooled_exp) / pooled_exp.max(1e-9);
            cells += 1;
        } else {
            assert_eq!(n - seen, 0, "sequences outside the target's support were emitted");
        }
        let dof = (cells - 1) as f64;
        // Wilson-Hilferty, z(1e-4) = 3.719.
        let h = 2.0 / (9.0 * dof);
        let crit = dof * (1.0 - h + 3.719 * h.sqrt()).powi(3);
        (stat, crit)
    }

    fn assert_exact(name: &str, toy: &Toy, drafter: Drafter, k: KPolicy, seed: u64) {
        let (stat, crit) = chi_square(toy, 2, 3, drafter, k, 120_000, seed);
        println!("{name}: chi2 {stat:.1} (critical at p=1e-4: {crit:.1})");
        assert!(stat < crit, "{name}: chi2 {stat:.1} >= critical {crit:.1}: the emitted sequence is NOT distributed as plain sampling");
    }

    fn sampled_mode(top_p: f32) -> SampleMode {
        SampleMode::Multinomial { temperature: 1.0, min_p_rel: 0.0, top_p }
    }

    #[cfg(feature = "v41")]
    #[test]
    fn g_rs1_sampled_drafts_with_the_production_stopping_rule_are_exact() {
        let toy = Toy::new(sampled_mode(0.9));
        assert_exact("sampled + stopping rule", &toy, Drafter::Sampled { tau: 1.0, m: 4 }, KPolicy::Stopping, 107);
        assert_exact("q = p + stopping rule", &toy, Drafter::QEqualsP, KPolicy::Stopping, 108);
    }

    /// TWO streams, each drafting 5 from its own previous token with its own
    /// RNG (production: `Stream::rng` / `draft_rng`), verified in one step with
    /// K per stream from the JOINT stopping rule (`ms_dspark::choose_ks_stopping`)
    /// over synthetic confidences that depend on each stream's drafts through
    /// the markov-prev channel. `peek_a`: the negative control, stream 0's
    /// first draft verified only when its drawn value is `a`.
    #[cfg(feature = "v41")]
    fn generate_two(toy: &Toy, starts: [i32; 2], len: usize, drafter: Drafter, peek_a: Option<i32>, rngs: &mut [SamplerRng; 2]) -> [Vec<i32>; 2] {
        use crate::ms_dspark::{choose_ks_stopping, BlockConf};
        let mut seqs: [Vec<i32>; 2] = [Vec::new(), Vec::new()];
        let mut prev = starts;
        while seqs.iter().any(|s| s.len() < len) {
            let mut drafts: [Vec<Draft>; 2] = [Vec::new(), Vec::new()];
            let mut conf = [[0f32; 5]; 2];
            for s in 0..2 {
                let mut dp = prev[s];
                for j in 0..5 {
                    let r = &toy.drafter[dp as usize];
                    let mx = r.iter().cloned().fold(f32::MIN, f32::max);
                    conf[s][j] = (mx - r.iter().sum::<f32>() / r.len() as f32) * 2.0 - 2.0;
                    let d = match drafter {
                        Drafter::PointMass => Draft { token: toy.drafter_argmax(dp), q: DraftDist::PointMass },
                        Drafter::Sampled { tau, m } => {
                            let q = toy.drafter_q(dp, tau, m);
                            Draft { token: draw_sparse(&q, rngs[s].next_f32()), q: DraftDist::Sampled(q) }
                        }
                        Drafter::QEqualsP => {
                            let q = toy.q_equals_p(dp);
                            Draft { token: draw_sparse(&q, rngs[s].next_f32()), q: DraftDist::Sampled(q) }
                        }
                    };
                    dp = d.token;
                    drafts[s].push(d);
                }
            }
            let blocks = [BlockConf { conf: conf[0], cap: 5 }, BlockConf { conf: conf[1], cap: 5 }];
            let mut ks = choose_ks_stopping(&blocks, 2, &*STOPPING_COST);
            if let Some(a) = peek_a {
                ks[0] = usize::from(drafts[0][0].token == a); // the forbidden, draft-value-dependent K
            }
            for s in 0..2 {
                drafts[s].truncate(ks[s]);
                let mut rows = vec![toy.p(prev[s])];
                for d in &drafts[s] {
                    rows.push(toy.p(d.token));
                }
                let out = verify_block(&rows, &drafts[s], &mut rngs[s]);
                seqs[s].extend_from_slice(&out.tokens);
                prev[s] = *seqs[s].last().unwrap();
            }
        }
        for s in seqs.iter_mut() {
            s.truncate(len);
        }
        seqs
    }

    /// Chi-square of the JOINT distribution of both streams' `len`-token
    /// prefixes against the product of the two exact plain-sampling chains
    /// (independent streams): catches a bias in either stream and any
    /// coupling between them. (statistic, critical value at p = 1e-4).
    #[cfg(feature = "v41")]
    fn chi_square_two(toy: &Toy, starts: [i32; 2], len: usize, drafter: Drafter, peek_a: Option<i32>, n: usize, seed: u64) -> (f64, f64) {
        let mut rngs = [SamplerRng::new(seed), SamplerRng::new(seed ^ 0x5DEE_CE66_D1CE_4E5B)];
        let mut counts: HashMap<(Vec<i32>, Vec<i32>), usize> = HashMap::new();
        for _ in 0..n {
            let [a, b] = generate_two(toy, starts, len, drafter, peek_a, &mut rngs);
            *counts.entry((a, b)).or_default() += 1;
        }
        let chain = |start: i32| {
            let mut exact: Vec<(Vec<i32>, f64)> = vec![(vec![], 1.0)];
            for _ in 0..len {
                let mut next = Vec::new();
                for (prefix, pr) in &exact {
                    let p = toy.p(*prefix.last().unwrap_or(&start));
                    for x in 0..V as i32 {
                        if p.prob(x) > 0.0 {
                            let mut s = prefix.clone();
                            s.push(x);
                            next.push((s, pr * p.prob(x)));
                        }
                    }
                }
                exact = next;
            }
            exact
        };
        let (ea, eb) = (chain(starts[0]), chain(starts[1]));
        let (mut stat, mut cells, mut seen) = (0.0f64, 0usize, 0usize);
        let (mut pooled_obs, mut pooled_exp) = (0.0f64, 0.0f64);
        for (sa, pa) in &ea {
            for (sb, pb) in &eb {
                let e = pa * pb * n as f64;
                let o = *counts.get(&(sa.clone(), sb.clone())).unwrap_or(&0) as f64;
                seen += o as usize;
                if e < 5.0 {
                    pooled_obs += o;
                    pooled_exp += e;
                } else {
                    stat += (o - e) * (o - e) / e;
                    cells += 1;
                }
            }
        }
        pooled_obs += (n - seen) as f64;
        if pooled_exp > 0.0 {
            stat += (pooled_obs - pooled_exp) * (pooled_obs - pooled_exp) / pooled_exp.max(1e-9);
            cells += 1;
        } else {
            assert_eq!(n - seen, 0, "sequences outside the target's support were emitted");
        }
        let dof = (cells - 1) as f64;
        let h = 2.0 / (9.0 * dof);
        (stat, dof * (1.0 - h + 3.719 * h.sqrt()).powi(3))
    }

    #[cfg(feature = "v41")]
    #[test]
    fn g_rs1_two_streams_with_the_joint_stopping_rule_are_exact() {
        let toy = Toy::new(sampled_mode(0.9));
        for (name, drafter, seed) in [
            ("two streams, sampled + joint stopping rule", Drafter::Sampled { tau: 1.0, m: 4 }, 111u64),
            ("two streams, q = p + joint stopping rule", Drafter::QEqualsP, 112),
            ("two streams, point-mass + joint stopping rule", Drafter::PointMass, 113),
        ] {
            let (stat, crit) = chi_square_two(&toy, [2, 4], 2, drafter, None, 150_000, seed);
            println!("{name}: chi2 {stat:.1} (critical at p=1e-4: {crit:.1})");
            assert!(stat < crit, "{name}: chi2 {stat:.1} >= critical {crit:.1}: NOT distributed as two independent plain samplers");
        }
    }

    /// The two-stream harness has power: stream 0 verifying its first SAMPLED
    /// draft only when the draft's drawn value is `a` must be caught.
    #[cfg(feature = "v41")]
    #[test]
    fn g_rs1_two_streams_negative_control_is_caught() {
        let toy = Toy::new(sampled_mode(1.0));
        let p = toy.p(2);
        let a = (0..V as i32).max_by(|&x, &y| p.prob(x).partial_cmp(&p.prob(y)).unwrap()).unwrap();
        let (stat, crit) = chi_square_two(&toy, [2, 4], 2, Drafter::QEqualsP, Some(a), 150_000, 511);
        println!("two-stream negative control: chi2 {stat:.1} (critical at p=1e-4: {crit:.1})");
        assert!(stat > 5.0 * crit, "two-stream negative control NOT detected: chi2 {stat:.1} vs critical {crit:.1}");
    }

    #[test]
    fn g_rs1_point_mass_drafts_are_exact() {
        let toy = Toy::new(sampled_mode(0.9));
        assert_exact("point-mass K=1", &toy, Drafter::PointMass, KPolicy::Fixed(1), 101);
        assert_exact("point-mass K=4", &toy, Drafter::PointMass, KPolicy::Fixed(4), 102);
    }

    #[test]
    fn g_rs1_sampled_drafts_are_exact() {
        let toy = Toy::new(sampled_mode(0.9));
        assert_exact("sampled tau=1 top-3 K=3", &toy, Drafter::Sampled { tau: 1.0, m: 3 }, KPolicy::Fixed(3), 201);
        assert_exact("sampled tau=0.5 top-4 K=4", &toy, Drafter::Sampled { tau: 0.5, m: 4 }, KPolicy::Fixed(4), 202);
        // q wider than p's truncated support (top-p 0.7 cuts more of p).
        let narrow = Toy::new(sampled_mode(0.7));
        assert_exact("sampled, q outside supp(p)", &narrow, Drafter::Sampled { tau: 1.5, m: 6 }, KPolicy::Fixed(2), 203);
    }

    #[test]
    fn g_rs1_q_equal_to_p_accepts_everything_and_stays_exact() {
        let toy = Toy::new(sampled_mode(1.0));
        assert_exact("q == p K=4", &toy, Drafter::QEqualsP, KPolicy::Fixed(4), 301);
    }

    #[test]
    fn g_rs1_draft_count_independent_of_drafts_is_lossless() {
        let toy = Toy::new(sampled_mode(0.9));
        assert_exact("random K in 0..=4, sampled", &toy, Drafter::Sampled { tau: 1.0, m: 3 }, KPolicy::Random(4), 401);
        assert_exact("random K in 0..=4, point-mass", &toy, Drafter::PointMass, KPolicy::Random(4), 402);
    }

    /// Negative control (plan 2.4): verifying a SAMPLED draft only when its
    /// drawn value is `a` breaks exactness (P(a) = p(a)(2 - p(a)) with q = p),
    /// and the chi-square must see it -- proof the test has power.
    #[test]
    fn g_rs1_negative_control_draft_dependent_k_is_caught() {
        let toy = Toy::new(sampled_mode(1.0));
        let start = 2;
        let p = toy.p(start);
        let a = (0..V as i32).max_by(|&x, &y| p.prob(x).partial_cmp(&p.prob(y)).unwrap()).unwrap();
        let (stat, crit) = chi_square(&toy, start, 3, Drafter::QEqualsP, KPolicy::PeekFirstDraft { a }, 120_000, 501);
        println!("negative control: chi2 {stat:.1} (critical at p=1e-4: {crit:.1})");
        assert!(stat > 10.0 * crit, "negative control NOT detected: chi2 {stat:.1} vs critical {crit:.1}");
    }

    /// Temperature 0: p is a point mass, so every scheme must emit exactly the
    /// greedy chain, whatever the drafts.
    #[test]
    fn g_rs1_temperature_zero_emits_the_greedy_chain() {
        let toy = Toy::new(SampleMode::Argmax);
        let greedy = {
            let mut s = vec![];
            let mut prev = 2;
            for _ in 0..8 {
                let TargetDist::Argmax(a) = toy.p(prev) else { unreachable!() };
                s.push(a);
                prev = a;
            }
            s
        };
        let mut rng = SamplerRng::new(601);
        for drafter in [Drafter::PointMass, Drafter::Sampled { tau: 1.0, m: 4 }] {
            for _ in 0..500 {
                assert_eq!(generate(&toy, 2, 8, drafter, KPolicy::Random(4), &mut rng), greedy);
            }
        }
    }

    /// p == q to rounding leaves `max(0, p - q)` empty: no NaN, and the fallback
    /// draws from p's support.
    #[test]
    fn zero_mass_residual_falls_back_to_p() {
        let toy = Toy::new(sampled_mode(0.9));
        let p = toy.p(1);
        let q = toy.q_equals_p(1);
        assert_eq!(residual_draw(&p, &q, 0.5), None);
        let mut rng = SamplerRng::new(701);
        for _ in 0..1000 {
            // q == p: the ratio is ~1, so a reject only happens through
            // rounding, and then the residual is empty and the fallback must
            // still draw from p. Every emitted token stays in supp(p).
            let d = Draft { token: q[0].0, q: DraftDist::Sampled(q.clone()) };
            match verify_position(&p, &d, &mut rng) {
                Verdict::Accept(t) | Verdict::Reject(t) => assert!(p.prob(t) > 0.0, "token {t} outside supp(p)"),
            }
        }
    }

    #[test]
    fn target_probabilities_sum_to_one() {
        let mut data = SamplerRng::new(9);
        let row: Vec<f32> = (0..500).map(|_| gauss(&mut data) * 3.0).collect();
        for top_p in [1.0f32, 0.9, 0.5] {
            let p = TargetDist::from_logits(&row, &sampled_mode(top_p));
            let s: f64 = (0..500).map(|x| p.prob(x)).sum();
            assert!((s - 1.0).abs() < 1e-5, "top_p {top_p}: sum {s}");
        }
    }
}
