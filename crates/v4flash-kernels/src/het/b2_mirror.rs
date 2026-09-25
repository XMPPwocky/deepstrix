//! Box 1's mirror of box 2's expert pool, and the route-time miss substitution
//! built on it (docs/v41/BOX2_MISS_SUBSTITUTION.md).
//!
//! Every box-2 reply to a request carrying `proto::REQ_FLAG_RESID` appends box
//! 2's residency map for that request's layer. `update` overwrites this
//! layer's row, so the mirror corrects itself and is never more than one
//! request per layer stale. There's no delta stream to reorder or drop.
//!
//! `V41_SUB` (default 0):
//! * 0 = off: no flag on the wire, no mirror, bit-identical to before.
//! * 1 = DRY RUN: mirror kept, the substitution is planned and counted
//!   (`take_sub_stats`) but NOT applied. Output is unchanged.
//! * 2 = ON: a decode row's box-2 pick that the mirror says box 2 does not hold
//!   is replaced by that row's best-ranked unused alternative (router
//!   `V41_ROUTER_ALTS`) held by either box. The row is renormalized exactly
//!   (ref.Gate).
//!
//! `V41_SUB_MIN_RANK` (1..=6, default 6): only picks at this rank or lower are
//! eligible (6 = the 6th pick only).

use crate::config::{N_EXPERT, N_LAYER};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const WORDS: usize = (N_EXPERT as usize).div_ceil(64);
const LAYERS: usize = N_LAYER as usize;

static BITS: [[AtomicU64; WORDS]; LAYERS] = [const { [const { AtomicU64::new(0) }; WORDS] }; LAYERS];
static SEEN: [AtomicBool; LAYERS] = [const { AtomicBool::new(false) }; LAYERS];

/// `V41_SUB`: 0 off, 1 dry run, 2 on.
pub fn mode() -> u32 {
    static M: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        let m = std::env::var("V41_SUB").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0).min(2);
        if m > 0 {
            eprintln!(
                "b2 mirror: V41_SUB={m} ({}), min rank {}",
                if m == 1 { "dry run" } else { "SUBSTITUTING" },
                min_rank()
            );
        }
        m
    });
    *M
}

/// Ask box 2 for residency maps (`proto::REQ_FLAG_RESID`)?
pub fn wanted() -> bool {
    mode() > 0
}

/// `V41_SUB_MIN_RANK`, 1-based: the highest-weight rank eligible for
/// substitution (6 = only the 6th pick).
pub fn min_rank() -> usize {
    static R: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("V41_SUB_MIN_RANK").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(6).clamp(1, 6)
    });
    *R
}

/// Overwrite `layer`'s row from a reply's residency map
/// (`proto::RESID_WORDS` u32s, bit e = expert e).
pub fn update(layer: u32, words: &[u32]) {
    let l = layer as usize;
    if l >= LAYERS {
        return;
    }
    for (i, slot) in BITS[l].iter().enumerate() {
        let lo = words.get(2 * i).copied().unwrap_or(0) as u64;
        let hi = words.get(2 * i + 1).copied().unwrap_or(0) as u64;
        slot.store(lo | (hi << 32), Ordering::Relaxed);
    }
    SEEN[l].store(true, Ordering::Release);
}

/// Does box 2 hold `(layer, e)`, as of its last reply for `layer`? `None`
/// until box 2 has reported `layer` at all.
pub fn resident(layer: i32, e: u32) -> Option<bool> {
    let l = layer as usize;
    if l >= LAYERS || e >= N_EXPERT || !SEEN[l].load(Ordering::Acquire) {
        return None;
    }
    let w = BITS[l][(e / 64) as usize].load(Ordering::Relaxed);
    Some((w >> (e % 64)) & 1 == 1)
}

// ---- per-step counters (drained by the multistream profile) ----

static N_PREDICTED: AtomicU64 = AtomicU64::new(0);
static N_AVOIDED: AtomicU64 = AtomicU64::new(0);
static N_SLOTS: AtomicU64 = AtomicU64::new(0);
static N_BLOCKED: AtomicU64 = AtomicU64::new(0);

/// `(predicted box-2 misses, reads avoided, picks substituted, misses that
/// could not be substituted)` since the last call, all counted per distinct
/// (layer, expert) per request except `picks substituted`. In dry-run mode
/// "avoided"/"substituted" are what WOULD have happened.
pub fn take_sub_stats() -> (u64, u64, u64, u64) {
    (
        N_PREDICTED.swap(0, Ordering::Relaxed),
        N_AVOIDED.swap(0, Ordering::Relaxed),
        N_SLOTS.swap(0, Ordering::Relaxed),
        N_BLOCKED.swap(0, Ordering::Relaxed),
    )
}

pub fn record(o: &SubOutcome) {
    N_PREDICTED.fetch_add(o.predicted as u64, Ordering::Relaxed);
    N_AVOIDED.fetch_add(o.avoided as u64, Ordering::Relaxed);
    N_SLOTS.fetch_add(o.slots as u64, Ordering::Relaxed);
    N_BLOCKED.fetch_add(o.blocked as u64, Ordering::Relaxed);
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SubOutcome {
    /// Distinct experts predicted to miss on box 2.
    pub predicted: u32,
    /// ... of which substituted in every row that picked them (read avoided).
    pub avoided: u32,
    /// Pick slots rewritten.
    pub slots: u32,
    /// ... predicted misses left alone: some row had no acceptable
    /// alternative, or picked it above `min_rank`.
    pub blocked: u32,
}

/// Plan (and apply) substitutions over a batch of rows.
///
/// * `sel` / `ew`: `[b, nu]` picks in rank order and their weights, rewritten
///   in place.
/// * `alts` / `alt_w`: `[b, na]` alternatives in rank order, with weights on
///   `ew`'s scale (router `alt_w`).
/// * `min_rank`: 1-based. A pick at rank < `min_rank` (a heavier pick) is
///   never substituted, and a predicted miss that any row picked at such a
///   rank is left alone everywhere (it is read anyway).
/// * `predicted_miss(e)`: will this pick make box 2 read from disk?
/// * `acceptable(a)`: is alternative `a` served without a read (resident on
///   whichever box computes it)?
///
/// A missing expert is substituted only if EVERY row that picked it can be;
/// otherwise box 2 reads it anyway and substituting the other rows would cost
/// quality for no time. Each substitution renormalizes its row exactly: with
/// the row's weights on the current sum's scale, swapping slot k for an
/// alternative whose current-scale weight is `wa` divides every weight by
/// `c = 1 - ew[k]/scale + wa/scale` and gives the substitute `wa / c`.
#[allow(clippy::too_many_arguments)]
pub fn substitute(
    sel: &mut [i32],
    ew: &mut [f32],
    alts: &[i32],
    alt_w: &[f32],
    nu: usize,
    na: usize,
    min_rank: usize,
    scale: f32,
    predicted_miss: impl Fn(i32) -> bool,
    acceptable: impl Fn(i32) -> bool,
) -> SubOutcome {
    let mut out = SubOutcome::default();
    if nu == 0 || na == 0 || sel.is_empty() {
        return out;
    }
    let b = sel.len() / nu;
    debug_assert_eq!(ew.len(), sel.len());
    debug_assert_eq!(alts.len(), b * na);
    debug_assert_eq!(alt_w.len(), b * na);

    // Distinct predicted misses.
    let mut missing: Vec<i32> = Vec::new();
    for &e in sel.iter() {
        if e >= 0 && !missing.contains(&e) && predicted_miss(e) {
            missing.push(e);
        }
    }
    out.predicted = missing.len() as u32;
    if missing.is_empty() {
        return out;
    }

    // Choose the alternative for (row r, slot k), given the alternatives this
    // row has already taken. Never an alternative the row already picks, never
    // one that is itself missing.
    let choose = |row: &[i32], r: usize, taken: &[bool]| -> Option<usize> {
        (0..na).find(|&j| {
            let a = alts[r * na + j];
            a >= 0 && !taken[j] && !row.contains(&a) && acceptable(a)
        })
    };

    // Pass 1: which missing experts can be substituted in every row?
    let mut blocked = vec![false; missing.len()];
    for r in 0..b {
        let row = &sel[r * nu..(r + 1) * nu];
        let mut taken = vec![false; na];
        for k in 0..nu {
            let Some(mi) = missing.iter().position(|&m| m == row[k]) else { continue };
            if k + 1 < min_rank {
                blocked[mi] = true;
                continue;
            }
            match choose(row, r, &taken) {
                Some(j) => taken[j] = true,
                None => blocked[mi] = true,
            }
        }
    }

    // Pass 2: apply. Fewer experts qualify than pass 1 tried, so every slot
    // that qualifies still finds an alternative.
    for r in 0..b {
        let mut taken = vec![false; na];
        let mut c_cum = 1.0f32; // current row sum / router's original sum
        for k in 0..nu {
            let e = sel[r * nu + k];
            let Some(mi) = missing.iter().position(|&m| m == e) else { continue };
            if blocked[mi] || k + 1 < min_rank {
                continue;
            }
            let row = &sel[r * nu..(r + 1) * nu];
            let Some(j) = choose(row, r, &taken) else { continue };
            taken[j] = true;
            let wa = alt_w[r * na + j] / c_cum;
            let c = (1.0 - ew[r * nu + k] / scale + wa / scale).max(1e-6);
            for w in &mut ew[r * nu..(r + 1) * nu] {
                *w /= c;
            }
            ew[r * nu + k] = wa / c;
            sel[r * nu + k] = alts[r * na + j];
            c_cum *= c;
            out.slots += 1;
        }
    }
    out.blocked = blocked.iter().filter(|&&x| x).count() as u32;
    out.avoided = out.predicted - out.blocked;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: f32 = 1.5;

    /// Weights like the router's: probs / sum * scale.
    fn weights(p: &[f32]) -> Vec<f32> {
        let s: f32 = p.iter().sum();
        p.iter().map(|x| x / s * S).collect()
    }

    #[test]
    fn swaps_the_6th_and_renormalizes_exactly() {
        let probs = [0.9f32, 0.8, 0.7, 0.6, 0.5, 0.4];
        let sum: f32 = probs.iter().sum();
        let mut sel = vec![10, 11, 12, 13, 14, 15];
        let mut ew = weights(&probs);
        // Alternative 20 has prob 0.39; alt_w is on the ORIGINAL sum's scale.
        let alts = vec![20, 21];
        let alt_w = vec![0.39 / sum * S, 0.38 / sum * S];
        let o = substitute(&mut sel, &mut ew, &alts, &alt_w, 6, 2, 6, S, |e| e == 15, |_| true);
        assert_eq!(o, SubOutcome { predicted: 1, avoided: 1, slots: 1, blocked: 0 });
        assert_eq!(sel, vec![10, 11, 12, 13, 14, 20]);
        let want = weights(&[0.9, 0.8, 0.7, 0.6, 0.5, 0.39]);
        for (g, w) in ew.iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "{ew:?} vs {want:?}");
        }
    }

    #[test]
    fn two_swaps_in_one_row_stay_exact() {
        let probs = [0.9f32, 0.8, 0.7, 0.6, 0.5, 0.4];
        let sum: f32 = probs.iter().sum();
        let mut sel = vec![10, 11, 12, 13, 14, 15];
        let mut ew = weights(&probs);
        let alts = vec![20, 21, 22];
        let alt_w: Vec<f32> = [0.39f32, 0.30, 0.2].iter().map(|p| p / sum * S).collect();
        let o = substitute(&mut sel, &mut ew, &alts, &alt_w, 6, 3, 5, S, |e| e == 14 || e == 15, |_| true);
        assert_eq!(o.slots, 2);
        assert_eq!(sel, vec![10, 11, 12, 13, 20, 21]);
        let want = weights(&[0.9, 0.8, 0.7, 0.6, 0.39, 0.30]);
        for (g, w) in ew.iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "{ew:?} vs {want:?}");
        }
    }

    #[test]
    fn min_rank_protects_heavier_picks_everywhere() {
        // Expert 15 is rank 6 in row 0 but rank 1 in row 1: row 1 must read
        // it, so row 0 keeps it too.
        let mut sel = vec![10, 11, 12, 13, 14, 15, 15, 1, 2, 3, 4, 5];
        let mut ew = vec![0.25; 12];
        let alts = vec![20, 21, 22, 23];
        let alt_w = vec![0.1; 4];
        let before = (sel.clone(), ew.clone());
        let o = substitute(&mut sel, &mut ew, &alts, &alt_w, 6, 2, 6, S, |e| e == 15, |_| true);
        assert_eq!(o, SubOutcome { predicted: 1, avoided: 0, slots: 0, blocked: 1 });
        assert_eq!((sel, ew), before);
    }

    #[test]
    fn all_rows_or_none() {
        // Both rows pick 15 at rank 6; row 1 has no acceptable alternative.
        let mut sel = vec![10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 15];
        let mut ew = vec![0.25; 12];
        let alts = vec![20, 21, 30, 31];
        let alt_w = vec![0.1; 4];
        let before = sel.clone();
        let o = substitute(&mut sel, &mut ew, &alts, &alt_w, 6, 2, 6, S, |e| e == 15, |a| a < 30);
        assert_eq!(o.blocked, 1);
        assert_eq!(o.slots, 0);
        assert_eq!(sel, before);
    }

    #[test]
    fn skips_unacceptable_and_duplicate_alternatives() {
        // 1st alt not resident, 2nd already a pick in the row, 3rd is used.
        let mut sel = vec![10, 11, 12, 13, 14, 15];
        let mut ew = vec![0.25; 6];
        let alts = vec![20, 11, 22];
        let alt_w = vec![0.1, 0.1, 0.1];
        let o = substitute(&mut sel, &mut ew, &alts, &alt_w, 6, 3, 6, S, |e| e == 15, |a| a != 20);
        assert_eq!(o.slots, 1);
        assert_eq!(sel[5], 22);
    }

    #[test]
    fn a_missing_alternative_is_never_chosen() {
        let mut sel = vec![10, 11, 12, 13, 14, 15];
        let mut ew = vec![0.25; 6];
        let alts = vec![20, 21];
        let alt_w = vec![0.1, 0.1];
        let miss = |e: i32| e == 15 || e == 20;
        let o = substitute(&mut sel, &mut ew, &alts, &alt_w, 6, 2, 6, S, miss, |a| !miss(a));
        assert_eq!(sel[5], 21);
        assert_eq!(o.predicted, 1, "20 is not a pick, so it is not a predicted miss");
    }

    #[test]
    fn nothing_missing_is_a_no_op() {
        let mut sel = vec![10, 11, 12, 13, 14, 15];
        let mut ew = vec![0.25; 6];
        let before = (sel.clone(), ew.clone());
        let o = substitute(&mut sel, &mut ew, &[20, 21], &[0.1, 0.1], 6, 2, 6, S, |_| false, |_| true);
        assert_eq!(o, SubOutcome::default());
        assert_eq!((sel, ew), before);
    }

    #[test]
    fn mirror_update_and_lookup() {
        let l = 7u32;
        assert_eq!(resident(l as i32, 3), None, "unseen layer is unknown");
        let mut w = vec![0u32; (N_EXPERT as usize).div_ceil(32)];
        w[0] = 1 << 3;
        w[11] = 1 << 31; // expert 383
        update(l, &w);
        assert_eq!(resident(l as i32, 3), Some(true));
        assert_eq!(resident(l as i32, 4), Some(false));
        assert_eq!(resident(l as i32, 383), Some(true));
        assert_eq!(resident(l as i32, 384), None, "out of range");
    }
}
