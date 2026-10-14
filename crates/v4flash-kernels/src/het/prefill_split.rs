//! Frozen per-job expert -> box split for LONG prefills' ENCODER layers (P1 of
//! docs/v41/LM_LAYER_OUTER_PREFILL_DESIGN.md, sec 4).
//!
//! WHY. `expert_pager::partition_box2` is tuned for DECODE: once the hot set is warm, box 1 owns each
//! layer's hottest ~103-115 ids by decode pick mass and EVERYTHING ELSE (~270/layer) is box 2's. Prefill touches
//! almost every expert, so that hands box 2 ~5,400 experts over the 20 encoder layers against ~4,570 usable pool
//! slots: ~1,146 re-reads per 4096-row window (MEASURED 2026-10-10, docs/v41/IGPU_MOE_F16LUT_PLAN.md).
//!
//! WHAT. For a long job (and `V41_PREFILL_SPLIT=1`), box 1 takes its decode-hot encoder head UNION a hash filler
//! sized from the hub's LIVE free LRU slots, so box 1's decode head is never displaced and box 2's share shrinks.
//! Built once at job start and FROZEN (a mid-job `hot_set` refresh does not move ids). Deterministic. Decoder layers
//! (the replay) keep `partition_box2`: `home2` returns `None` for them.
//!
//! PRIOR (`V41_PREFILL_SPLIT_PRIOR`, 2026-10-10). The filler -- and, while the hot set is COLD (the first minutes after a
//! hub restart), the head itself -- used to be the smallest-HASH ids: random. Measured on a 220K-token job built 2 min
//! after a restart (`warm=false`): box-2 requests touched 139-149 distinct experts vs 124-128 with a warm split, ~+20%
//! box-2 time (box 2 is weight-read bound, ~100 us per distinct expert). With the prior, ids are ranked by the server's
//! persisted PREFILL pick counts (`expert_stats.json` prefill bank, EWMA over runs; [`set_prior`]), hash as the tie-break:
//! a cold head is the historically prefill-hot ids, and the filler (bounded by free LRU slots, so it still displaces no
//! decode-hot id) soaks up the prefill-hot ids box 2 would otherwise serve. The warm head stays the decode-hot set.
//!
//! HOW IT IS CONSULTED. `pre_moe_route` has no job handle, so the object lives in a process-global
//! `RwLock<Option<Arc<..>>>` installed by a [`SplitGuard`] around each prefill unit; decode steps, DSpark verifies and
//! the replay between/after units see `None` and use `partition_box2` unchanged.
use std::sync::{Arc, RwLock};

use crate::config::{CED_DECODER_START, N_EXPERT, N_LAYER};

const NE: usize = N_EXPERT as usize;
/// Encoder layers the split covers: `0..CED_DECODER_START` (layer `CED_DECODER_START` has no MoE in the encoder pass).
const NL_ENC: usize = CED_DECODER_START;

/// `V41_PREFILL_SPLIT=1` (default off): use the frozen split for long jobs.
pub fn enabled() -> bool {
    crate::knobs::PREFILL_SPLIT.on()
}

/// `V41_PREFILL_B1_FILLER` (default 64): ceiling on the per-layer hash filler box 1 adds to its decode-hot head. The
/// effective filler is `min(this, free_lru_slots / encoder_layers)` from live counts (see [`SplitCfg::filler`]).
pub fn filler_max() -> usize {
    crate::knobs::PREFILL_B1_FILLER.usize()
}

/// `V41_PREFILL_SPLIT_PRIOR` (default off; LIVE): rank the filler (and a cold head) by the persisted prefill pick counts.
pub fn prior_enabled() -> bool {
    crate::knobs::PREFILL_SPLIT_PRIOR.on()
}

/// A prior with fewer prefill tokens than this is ignored (hash order, as before).
pub const PRIOR_MIN_TOKENS: u64 = 100_000;

/// Per-(encoder layer, expert) prefill pick counts (the server's `expert_stats` prefill bank).
pub struct Prior {
    /// `[NL_ENC][NE]`.
    counts: Vec<u64>,
    pub tokens: u64,
}

static PRIOR: RwLock<Option<Arc<Prior>>> = RwLock::new(None);

/// Publish the prefill prior: `counts` = the `[N_LAYER x N_EXPERT]` prefill bank (only the encoder layers are kept),
/// `tokens` = the tokens it accumulated over. Called at model load and after every stats flush.
pub fn set_prior(counts: &[u64], tokens: u64) {
    if counts.len() < NL_ENC * NE {
        return;
    }
    let p = Arc::new(Prior { counts: counts[..NL_ENC * NE].to_vec(), tokens });
    *PRIOR.write().unwrap_or_else(|e| e.into_inner()) = Some(p);
}

/// The prior to build with: `None` unless the knob is on and it has `PRIOR_MIN_TOKENS`.
pub fn prior() -> Option<Arc<Prior>> {
    if !prior_enabled() {
        return None;
    }
    PRIOR.read().unwrap_or_else(|e| e.into_inner()).clone().filter(|p| p.tokens >= PRIOR_MIN_TOKENS)
}

/// `V41_PREFILL_B2_USABLE` (default 4570): box 2's usable pool slots during a long prefill, used ONLY for the job-start
/// log's expected-re-read figure (pool 4,760 - ~136 pinned - nursery 48; the daemon's "pin budget" startup line).
pub fn b2_usable_estimate() -> usize {
    static N: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| std::env::var("V41_PREFILL_B2_USABLE").ok().and_then(|v| v.parse().ok()).unwrap_or(4570));
    *N
}

/// SplitMix64 finaliser: a stable, well-mixed order on `(layer, expert)`. Not `DefaultHasher` (not stable across
/// builds).
fn mix(layer: usize, e: usize) -> u64 {
    let mut z = ((layer as u64) << 20 | e as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Inputs of [`PrefillSplit::build`] that are not the hot set itself.
#[derive(Clone, Copy, Debug)]
pub struct SplitCfg {
    /// Hub LRU slots (`n_slots - lru_lo`): what decode-hot ids and the prefill share compete for.
    pub lru_slots: usize,
    /// Slots the filler must also leave alone (`V41_B1_PROTECT_OWNED`: box 1's resident NON-owned experts +
    /// `PROTECT_HEADROOM_PER_LAYER` per layer); 0 = the original sizing.
    pub reserve: usize,
    /// Ceiling on the per-layer filler (`filler_max()`).
    pub filler_max: usize,
    /// Per-layer head size to use when the hot set is NOT warm (`hot_set::per_layer()`).
    pub cold_head_per_layer: usize,
}

impl SplitCfg {
    /// `min(filler_max, (lru_slots - owned_now) / encoder_layers)`: the filler never exceeds the hub's live free
    /// slots, so it displaces ~no decode-hot id by construction (review r2 finding 1).
    pub fn filler(&self, owned_now: usize) -> usize {
        let free = self.lru_slots.saturating_sub(owned_now).saturating_sub(self.reserve);
        self.filler_max.min(free / NL_ENC)
    }
}

/// `V41_B1_PROTECT_OWNED`: headroom slots per layer (all layers) the filler leaves free.
pub const PROTECT_HEADROOM_PER_LAYER: usize = 2;

/// Sizes for the job-start log line and the gate (review r2).
#[derive(Clone, Debug, Default)]
pub struct SplitStats {
    pub warm: bool,
    pub owned_now: usize,
    pub filler: usize,
    /// Total ids box 1 / box 2 serve across the encoder layers.
    pub share1: usize,
    pub share2: usize,
    pub head_min: usize,
    pub head_max: usize,
    /// The ranking came from the prefill prior (else hash order).
    pub prior: bool,
}

impl SplitStats {
    /// Steady re-reads per window if every box-2 expert is touched each window: `max(0, share2 - usable2)`.
    pub fn expected_rereads(&self, usable2: usize) -> usize {
        self.share2.saturating_sub(usable2)
    }
}

pub struct PrefillSplit {
    /// `[NL_ENC][NE]`: true = box 2 serves this (layer, expert).
    box2: Vec<bool>,
    pub stats: SplitStats,
}

impl PrefillSplit {
    /// Build the frozen split. `head(layer, e)` = `hot_set::box1_owns` (`None` until warm). Hash order (no prior).
    pub fn build(cfg: &SplitCfg, head: &dyn Fn(i32, u32) -> Option<bool>) -> Self {
        Self::build_with_prior(cfg, head, None)
    }

    /// [`Self::build`] with the filler (and a cold head) ranked by `prior`'s prefill pick counts, hash as tie-break.
    pub fn build_with_prior(cfg: &SplitCfg, head: &dyn Fn(i32, u32) -> Option<bool>, prior: Option<&Prior>) -> Self {
        let warm = (0..NL_ENC).any(|l| (0..NE).any(|e| head(l as i32, e as u32).is_some()));
        // Live count of ids box 1 owns for decode across ALL layers (they compete for the same LRU slots).
        let owned_now = if warm {
            (0..N_LAYER as usize).map(|l| (0..NE).filter(|&e| head(l as i32, e as u32) == Some(true)).count()).sum()
        } else {
            cfg.cold_head_per_layer * N_LAYER as usize
        };
        let filler = cfg.filler(owned_now);
        let mut box2 = vec![true; NL_ENC * NE];
        let (mut share1, mut head_min, mut head_max) = (0usize, usize::MAX, 0usize);
        for l in 0..NL_ENC {
            let mut order: Vec<usize> = (0..NE).collect();
            match prior {
                Some(p) => order.sort_by_key(|&e| (std::cmp::Reverse(p.counts[l * NE + e]), mix(l, e))),
                None => order.sort_by_key(|&e| mix(l, e)),
            }
            // The head: the hot set's ids when warm, else the first `cold_head_per_layer` ids of `order` (prefill-hot with
            // the prior, smallest-hash without).
            let head_ids: Vec<usize> = if warm {
                (0..NE).filter(|&e| head(l as i32, e as u32) == Some(true)).collect()
            } else {
                order.iter().copied().take(cfg.cold_head_per_layer).collect()
            };
            for &e in &head_ids {
                box2[l * NE + e] = false;
            }
            // The filler: the first ids of `order` that are not already box 1's.
            let mut added = 0usize;
            for &e in &order {
                if added == filler {
                    break;
                }
                if box2[l * NE + e] {
                    box2[l * NE + e] = false;
                    added += 1;
                }
            }
            let n1 = head_ids.len() + added;
            share1 += n1;
            head_min = head_min.min(head_ids.len());
            head_max = head_max.max(head_ids.len());
        }
        let share2 = NL_ENC * NE - share1;
        PrefillSplit { box2, stats: SplitStats { warm, owned_now, filler, share1, share2, head_min, head_max, prior: prior.is_some() } }
    }

    /// `Some(true)` = box 2 serves `(layer, e)`; `None` for layers outside the encoder range (use `partition_box2`).
    #[inline]
    pub fn home2(&self, layer: i32, e: u32) -> Option<bool> {
        if layer < 0 || layer as usize >= NL_ENC || e as usize >= NE {
            return None;
        }
        Some(self.box2[layer as usize * NE + e as usize])
    }

    /// Per-layer box-2 count (tests, logs).
    pub fn box2_count(&self, layer: usize) -> usize {
        self.box2[layer * NE..(layer + 1) * NE].iter().filter(|&&b| b).count()
    }
}

static CURRENT: RwLock<Option<Arc<PrefillSplit>>> = RwLock::new(None);

/// The split installed for the prefill unit running on this process, if any. Clone ONCE per `pre_moe_route` call.
pub fn current() -> Option<Arc<PrefillSplit>> {
    CURRENT.read().unwrap_or_else(|p| p.into_inner()).clone()
}

/// Box-2 home of `(layer, e)` for a PREFILL consumer: the installed split if any covers the layer, else the decode
/// partition.
pub fn home2_or_partition(split: Option<&Arc<PrefillSplit>>, layer: i32, e: u32) -> bool {
    split.and_then(|s| s.home2(layer, e)).unwrap_or_else(|| super::expert_pager::partition_box2(layer, e))
}

/// RAII: installs `split` for the guard's lifetime and restores the previous value (so a nested or panicking unit
/// never leaks a split into a decode step).
pub struct SplitGuard {
    prev: Option<Arc<PrefillSplit>>,
}

pub fn install(split: Option<Arc<PrefillSplit>>) -> SplitGuard {
    let mut g = CURRENT.write().unwrap_or_else(|p| p.into_inner());
    let prev = std::mem::replace(&mut *g, split);
    SplitGuard { prev }
}

impl Drop for SplitGuard {
    fn drop(&mut self) {
        *CURRENT.write().unwrap_or_else(|p| p.into_inner()) = self.prev.take();
    }
}

/// COLD READS (`V41_PREFILL_COLD_B1`, knobs.rs): is the rule on for a prefill request with this split?
pub fn cold_b1_on(split: Option<&Arc<PrefillSplit>>) -> bool {
    match crate::knobs::PREFILL_COLD_B1.pick() {
        1 => split.is_none(),
        2 => true,
        _ => false,
    }
}

/// How many of a request's `n_cold` cold picks box 1 reads: `frac` of them, at most `max` (the rest stay box 2's).
pub fn cold_b1_take(n_cold: usize, frac: f64, max: usize) -> usize {
    ((n_cold as f64 * frac.clamp(0.0, 1.0)).floor() as usize).min(max).min(n_cold)
}

/// Cold picks since the start: computed on box 1 because it already held them / read on box 1 / left to box 2.
static COLD_B1_FREE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static COLD_B1_READ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static COLD_B2: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn note_cold(b1_free: u64, b1_read: u64, b2: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    COLD_B1_FREE.fetch_add(b1_free, Relaxed);
    COLD_B1_READ.fetch_add(b1_read, Relaxed);
    COLD_B2.fetch_add(b2, Relaxed);
}

/// LATE cold picks (`V41_PREFILL_COLD_ASYNC`), cumulative: joined an in-flight read (no new read) / new late reads /
/// refused for want of a free stage (left to box 2).
static COLD_LATE_SHARED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static COLD_LATE_NEW: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static COLD_LATE_REFUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn note_late(shared: u64, new: u64, refused: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    COLD_LATE_SHARED.fetch_add(shared, Relaxed);
    COLD_LATE_NEW.fetch_add(new, Relaxed);
    COLD_LATE_REFUSED.fetch_add(refused, Relaxed);
}

/// (shared, new, refused) late cold picks, cumulative.
pub fn late_totals() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (COLD_LATE_SHARED.load(Relaxed), COLD_LATE_NEW.load(Relaxed), COLD_LATE_REFUSED.load(Relaxed))
}

/// (box 1 already held, box-1 reads, box-2 reads) of the cold picks, cumulative.
pub fn cold_totals() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (COLD_B1_FREE.load(Relaxed), COLD_B1_READ.load(Relaxed), COLD_B2.load(Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_take_is_bounded() {
        assert_eq!(cold_b1_take(0, 0.5, 8), 0);
        assert_eq!(cold_b1_take(5, 0.5, 8), 2);
        assert_eq!(cold_b1_take(40, 0.5, 8), 8);
        assert_eq!(cold_b1_take(40, 0.0, 8), 0);
        assert_eq!(cold_b1_take(3, 1.0, 8), 3);
        assert_eq!(cold_b1_take(3, 7.0, 8), 3, "frac is clamped");
    }

    fn hot_head(per_layer: usize) -> impl Fn(i32, u32) -> Option<bool> {
        // A warm hot set: the `per_layer` smallest-hash ids of every layer.
        move |l, e| {
            let mut order: Vec<usize> = (0..NE).collect();
            order.sort_by_key(|&x| mix(l as usize, x) ^ 0xFFFF);
            Some(order.iter().take(per_layer).any(|&x| x == e as usize))
        }
    }

    #[test]
    fn reserve_shrinks_the_filler_and_zero_keeps_the_old_sizing() {
        // interleave 115 x 40 = 4,600 owned of 5,425 => 825 free => 41/layer (the production 41-43, 10-14) ...
        let base = SplitCfg { reserve: 0, lru_slots: 5425, filler_max: 64, cold_head_per_layer: 115 };
        assert_eq!(PrefillSplit::build(&base, &hot_head(115)).stats.filler, 41);
        // ... minus 745 resident non-owned + 2 x 40 headroom => 0 free: no filler (the share stays box 2's).
        let prot = SplitCfg { reserve: 745 + PROTECT_HEADROOM_PER_LAYER * N_LAYER as usize, ..base };
        assert_eq!(PrefillSplit::build(&prot, &hot_head(115)).stats.filler, 0);
        let some = SplitCfg { reserve: 400, ..base };
        assert_eq!(PrefillSplit::build(&some, &hot_head(115)).stats.filler, (825 - 400) / NL_ENC);
    }

    #[test]
    fn deterministic_and_frozen() {
        let cfg = SplitCfg { reserve: 0, lru_slots: 5041, filler_max: 64, cold_head_per_layer: 103 };
        let a = PrefillSplit::build(&cfg, &hot_head(103));
        let b = PrefillSplit::build(&cfg, &hot_head(103));
        assert_eq!(a.box2, b.box2);
        assert_eq!(a.stats.share1 + a.stats.share2, NL_ENC * NE);
    }

    #[test]
    fn filler_is_bounded_by_live_free_slots() {
        // top: 103 x 40 = 4,120 owned, 5,041 LRU slots => free 921 => filler 46/layer (review r2: NOT 64).
        let cfg = SplitCfg { reserve: 0, lru_slots: 5041, filler_max: 64, cold_head_per_layer: 103 };
        let s = PrefillSplit::build(&cfg, &hot_head(103));
        assert_eq!(s.stats.owned_now, 103 * N_LAYER as usize);
        assert_eq!(s.stats.filler, 46);
        // reclaimed dense window (5,425 slots): free 1,305 => capped at the knob (64) ... 1305/20 = 65 => 64.
        let cfg = SplitCfg { reserve: 0, lru_slots: 5425, filler_max: 64, cold_head_per_layer: 103 };
        let s = PrefillSplit::build(&cfg, &hot_head(103));
        assert_eq!(s.stats.filler, 64);
        // interleave 115 x 40 = 4,600 on 5,041 => 441 free => 22/layer.
        let cfg = SplitCfg { reserve: 0, lru_slots: 5041, filler_max: 64, cold_head_per_layer: 115 };
        let s = PrefillSplit::build(&cfg, &hot_head(115));
        assert_eq!(s.stats.filler, 22);
        // owned >= slots: no filler, never underflows.
        let cfg = SplitCfg { reserve: 0, lru_slots: 1000, filler_max: 64, cold_head_per_layer: 103 };
        assert_eq!(PrefillSplit::build(&cfg, &hot_head(103)).stats.filler, 0);
    }

    #[test]
    fn box1_keeps_its_decode_head_and_box2_gets_the_rest() {
        let cfg = SplitCfg { reserve: 0, lru_slots: 5041, filler_max: 64, cold_head_per_layer: 103 };
        let head = hot_head(103);
        let s = PrefillSplit::build(&cfg, &head);
        for l in 0..NL_ENC {
            for e in 0..NE {
                if head(l as i32, e as u32) == Some(true) {
                    assert_eq!(s.home2(l as i32, e as u32), Some(false), "decode-hot head must stay on box 1 (L{l} e{e})");
                }
            }
            // head 103 + filler 46 => box 2 serves 384 - 149 = 235 per layer (design sec 4).
            assert_eq!(s.box2_count(l), NE - 103 - 46);
        }
        assert_eq!(s.stats.share2, NL_ENC * (NE - 149));
        assert_eq!(s.stats.expected_rereads(4570), NL_ENC * 235 - 4570);
    }

    #[test]
    fn cold_hot_set_is_defined_not_n_slots_over_20() {
        // `owned_now = 0` would give n_slots/20 = 271/layer of filler (review r3 N1); the cold case must size from
        // `cold_head_per_layer`.
        let cfg = SplitCfg { reserve: 0, lru_slots: 5041, filler_max: 400, cold_head_per_layer: 103 };
        let s = PrefillSplit::build(&cfg, &|_, _| None);
        assert!(!s.stats.warm);
        assert_eq!(s.stats.owned_now, 103 * N_LAYER as usize);
        assert_eq!(s.stats.filler, (5041 - 103 * 40) / NL_ENC);
        assert!(s.stats.filler < 271);
        for l in 0..NL_ENC {
            assert_eq!(s.box2_count(l), NE - 103 - s.stats.filler);
        }
    }

    /// A prior where expert e's count in layer l is `(e * 7 + l) % NE` (distinct per layer): its top-k is known.
    fn test_prior() -> Prior {
        Prior { counts: (0..NL_ENC * NE).map(|i| ((i % NE) * 7 + i / NE) as u64 % NE as u64).collect(), tokens: PRIOR_MIN_TOKENS }
    }

    fn top_by_prior(p: &Prior, l: usize, k: usize) -> Vec<usize> {
        let mut o: Vec<usize> = (0..NE).collect();
        o.sort_by_key(|&e| (std::cmp::Reverse(p.counts[l * NE + e]), mix(l, e)));
        o.truncate(k);
        o
    }

    #[test]
    fn cold_head_with_a_prior_is_the_prefill_hot_ids() {
        let cfg = SplitCfg { reserve: 0, lru_slots: 5041, filler_max: 64, cold_head_per_layer: 115 };
        let p = test_prior();
        let s = PrefillSplit::build_with_prior(&cfg, &|_, _| None, Some(&p));
        assert!(!s.stats.warm && s.stats.prior);
        let f = s.stats.filler;
        for l in 0..NL_ENC {
            // head + filler = the top (115 + filler) ids by prior count: all box 1, everything else box 2
            let top = top_by_prior(&p, l, 115 + f);
            for e in 0..NE {
                assert_eq!(s.home2(l as i32, e as u32), Some(!top.contains(&e)), "L{l} e{e}");
            }
        }
    }

    #[test]
    fn warm_head_is_kept_and_the_filler_follows_the_prior() {
        let cfg = SplitCfg { reserve: 0, lru_slots: 5425, filler_max: 64, cold_head_per_layer: 103 };
        let head = hot_head(103);
        let p = test_prior();
        let s = PrefillSplit::build_with_prior(&cfg, &head, Some(&p));
        assert!(s.stats.warm && s.stats.prior);
        for l in 0..NL_ENC {
            let mut want_filler = Vec::new();
            for e in top_by_prior(&p, l, NE) {
                if want_filler.len() == s.stats.filler {
                    break;
                }
                if head(l as i32, e as u32) != Some(true) {
                    want_filler.push(e);
                }
            }
            for e in 0..NE {
                let box1 = head(l as i32, e as u32) == Some(true) || want_filler.contains(&e);
                assert_eq!(s.home2(l as i32, e as u32), Some(!box1), "L{l} e{e}");
            }
        }
        // same sizes as without the prior: only WHICH filler ids changes
        let h = PrefillSplit::build(&cfg, &head);
        assert_eq!((s.stats.share1, s.stats.share2, s.stats.filler), (h.stats.share1, h.stats.share2, h.stats.filler));
        assert!(!h.stats.prior);
    }

    #[test]
    fn prior_below_min_tokens_or_knob_off_is_ignored() {
        set_prior(&vec![1u64; N_LAYER as usize * NE], PRIOR_MIN_TOKENS - 1);
        crate::knobs::PREFILL_SPLIT_PRIOR.set("1");
        assert!(prior().is_none(), "too few tokens");
        set_prior(&vec![1u64; N_LAYER as usize * NE], PRIOR_MIN_TOKENS);
        assert!(prior().is_some());
        crate::knobs::PREFILL_SPLIT_PRIOR.set("0");
        assert!(prior().is_none(), "knob off");
        set_prior(&[1u64; 3], PRIOR_MIN_TOKENS); // short input is refused, previous prior kept
        crate::knobs::PREFILL_SPLIT_PRIOR.set("1");
        assert_eq!(prior().map(|p| p.counts.len()), Some(NL_ENC * NE));
        crate::knobs::PREFILL_SPLIT_PRIOR.set("0");
    }

    #[test]
    fn decoder_layers_are_not_covered() {
        let cfg = SplitCfg { reserve: 0, lru_slots: 5041, filler_max: 64, cold_head_per_layer: 103 };
        let s = PrefillSplit::build(&cfg, &hot_head(103));
        assert_eq!(s.home2(CED_DECODER_START as i32, 0), None);
        assert_eq!(s.home2(N_LAYER as i32 - 1, 7), None);
        assert_eq!(s.home2(-1, 0), None);
        assert!(s.home2(0, 0).is_some());
    }

    #[test]
    fn guard_installs_and_restores() {
        let cfg = SplitCfg { reserve: 0, lru_slots: 5041, filler_max: 64, cold_head_per_layer: 103 };
        let s = Arc::new(PrefillSplit::build(&cfg, &hot_head(103)));
        assert!(current().is_none());
        {
            let _g = install(Some(s.clone()));
            assert!(current().is_some());
            {
                let _inner = install(None);
                assert!(current().is_none());
            }
            assert!(current().is_some(), "inner guard restores the outer split");
        }
        assert!(current().is_none(), "dropped guard restores None");
    }
}
