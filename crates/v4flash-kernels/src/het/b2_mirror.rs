//! Box 1's mirror of box 2's expert pool, and the route-time miss substitution
//! built on it (docs/v41/BOX2_MISS_SUBSTITUTION.md).
//!
//! Every box-2 reply to a request carrying `proto::REQ_FLAG_RESID` appends box
//! 2's residency map for that request's layer. `update` overwrites this
//! layer's row, so the mirror corrects itself and is never more than one
//! request per layer stale. There's no delta stream to reorder or drop.
//!
//! Requests sent but not yet answered are overlaid as PENDING
//! (`note_submitted`): the picks a lane just asked box 2 for will be resident
//! by the time the other lane's request for the same layer is served, so the
//! other lane must neither avoid them (it would pay the quality cost of a
//! swap for a read box 2 is making anyway) nor count them as misses. `update`
//! clears the layer's pending row: under the lockstep and round-robin lane
//! drivers a layer's replies are consumed (`wait`) only after both lanes have
//! routed it, and the next route of that layer is the next step. (The
//! ready-first driver can consume one lane's reply before the other lane
//! routes; the map it brings then shows the read done, so the overlay is
//! merely not needed.) `V41_SUB_PENDING=0` turns the overlay off. Do that when box
//! 2 PARKs (`knobs::park`): there it serves the other lane while this lane's
//! read is parked, so an expert "being read" is NOT free for the other lane,
//! and swapping it away is what lets that lane skip the wait. Known leftovers:
//! a submit that fails after `note_submitted` leaves its bits until the
//! layer's next reply, and the map just after a prefill chunk includes its
//! scan admissions (which box 2 evicts first). Both err toward "resident": a
//! read box 2 makes anyway, never wrong output.
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
//! * 3 = CACHE-PRIOR (Skliar et al. 2024, arXiv 2412.00099): no host-side
//!   rewrite. The router itself adds `lambda * Delta_layer` to the selection
//!   score of every expert held by the box that computes it (box 2's mirror,
//!   box 1's pager), keeps the original top `V41_SUB_PROTECT` (default 2)
//!   picks, and renormalizes the weights over the final set. `Delta_layer` is
//!   a running average of each token's selection-score range (max - min), so
//!   one knob, `V41_SUB_LAMBDA`, is a per-layer gap gate: a missing pick is
//!   displaced only by a held expert within lambda * Delta of it. Displaced
//!   box-2 experts are admitted in the background as in mode 2.
//!
//! Swapped-away box-2 experts are still READ, just not on the critical path
//! (`V41_SUB_ADMIT`, default on; modes 2 and 3): the hub queues `layer << 16 |
//! e` as a box-2 PREFETCH word on the same request
//! (`remote_experts::push_prefetch_words`). Box 2's background readers fetch
//! it (yielding to demand misses) and admit it at that layer's next `ensure`,
//! so the mirror shows it resident soon after and the swap rate falls back to
//! the first-touch miss rate. Without it, a swapped expert is never read,
//! never admitted, and swapped again on every later pick (measured live: 3-5x
//! the swaps).
//!
//! INCOMING (`V41_SUB_INCOMING`, default 2 decode steps; 0 = off). Box 2's map
//! counts only LANDED slots, and the reply that follows an admission leaves
//! before its read lands, so at the layer's next route the mirror still says
//! "missing": an expert the router picks again on the next token was swapped
//! away again, for a read already under way (measured live 2026-09-25: 3.9% of
//! swapped-away experts are re-picked on the next token, 71% of those were
//! swapped again, ~2.8% of all swaps). `note_incoming` marks each admission
//! actually queued, and the mark counts as resident from the NEXT decode step
//! (`begin_step`, called first by every arena decode driver) for N steps. Not
//! in the same step: the other lane may route this layer after the mark --
//! under the ready-first driver even after this lane's reply is consumed --
//! and must not ride on a read queued a moment ago. The word leaves on this
//! lane's next submit, at the latest the first one of the next step, and box 2
//! handles a request's words before its picks. Any prefill pass ends every
//! mark (`expire_incoming`): a chunk churns box 2's pool. If
//! the read has not landed when a request needs it, box 2 promotes it and
//! waits (`ensure`'s in-flight wait): at most one read, already queued, never a
//! second. A mark is NOT cleared when a reply shows the expert held: under
//! PARK an older map can be consumed after a newer one. The price of a stale
//! mark is bounded by the window: an eviction inside it, or a word box 2
//! dropped (its speculative queue drops words when at most 2x the reserve of
//! staging sets is free, and it ignores them without a prefetch reader), costs
//! one demand read, and under mode 3 may steer a row onto that expert. Marked
//! experts are acceptable substitutes (mode 2) and are boosted by the prior
//! (mode 3), like any resident expert.
//!
//! Which picks may be swapped (modes 1-2):
//! * `V41_SUB_MIN_RANK` (1..=6, default 6): only picks at this rank or lower
//!   (6 = the 6th pick only).
//! * `V41_SUB_MAX_W` (unset = no cap): a swap is allowed only if the missing
//!   pick's weight AND the substitute's renormalized weight are both <= this
//!   (weights sum to 1.5). The local damage of a swap is about
//!   `w x ||E_sub - E_miss||`, so this caps the MASS moved whatever the rank:
//!   in a flat row every rank can qualify, in a peaked row only the small tail.
//!   ~0.2 is the 90th percentile of the 6th pick's weight in the golden data.

use crate::config::{N_EXPERT, N_LAYER};
use crate::router_topk::ROUTER_MAX_ALT;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

const WORDS: usize = (N_EXPERT as usize).div_ceil(64);
const LAYERS: usize = N_LAYER as usize;
const NE: usize = N_EXPERT as usize;
const MAX_ALT: usize = ROUTER_MAX_ALT as usize;

static BITS: [[AtomicU64; WORDS]; LAYERS] = [const { [const { AtomicU64::new(0) }; WORDS] }; LAYERS];
static PENDING: [[AtomicU64; WORDS]; LAYERS] = [const { [const { AtomicU64::new(0) }; WORDS] }; LAYERS];
static SEEN: [AtomicBool; LAYERS] = [const { AtomicBool::new(false) }; LAYERS];
/// Decode steps begun (`begin_step`): the INCOMING overlay's clock.
static STEP: AtomicU32 = AtomicU32::new(0);
/// Per (layer, expert): the `STEP` from which a queued admission counts as
/// resident (module doc, INCOMING); 0 = no mark.
static INCOMING: [[AtomicU32; NE]; LAYERS] = [const { [const { AtomicU32::new(0) }; NE] }; LAYERS];

/// `V41_SUB`: 0 off, 1 dry run, 2 host planner, 3 cache-prior.
pub fn mode() -> u32 {
    static M: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        let m = std::env::var("V41_SUB").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0).min(3);
        if m == 3 {
            eprintln!(
            "b2 mirror: V41_SUB=3 CACHE-PRIOR{}, lambda {}, protect top {}",
            if dry() { " (DRY RUN: counted, not applied)" } else { "" },
            lambda(),
            protect()
        );
        } else if m > 0 {
            eprintln!(
                "b2 mirror: V41_SUB={m} ({}), min rank {}, max weight {}",
                if m == 1 { "dry run" } else { "SUBSTITUTING" },
                min_rank(),
                max_w().map_or("none".to_string(), |w| format!("{w}")),
            );
            if super::forward_prefill::router_alts() == 0 {
                eprintln!("b2 mirror: WARNING V41_SUB={m} but V41_ROUTER_ALTS=0: there are no alternatives, nothing will be substituted");
            }
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

/// `V41_SUB_LAMBDA` (mode 3; default 0.1, clamped to [0, 1]): the cache-prior
/// strength, as a fraction of the layer's running selection-score range.
/// `V41_SUB_LAMBDA_FILE=<path>`: re-read the value from that file (a bare
/// number) at most once a second, so lambda can be swept live without a
/// restart (every hub restart cools box 1's pool). A missing or unparsable
/// file keeps the last value.
pub fn lambda() -> f32 {
    static BASE: std::sync::LazyLock<f32> = std::sync::LazyLock::new(|| {
        std::env::var("V41_SUB_LAMBDA").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.1).clamp(0.0, 1.0)
    });
    static FILE: std::sync::LazyLock<Option<String>> = std::sync::LazyLock::new(|| std::env::var("V41_SUB_LAMBDA_FILE").ok());
    static CUR: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
    static LAST: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
    let Some(path) = FILE.as_ref() else { return *BASE };
    if let Ok(mut last) = LAST.try_lock() {
        if last.is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(1)) {
            *last = Some(std::time::Instant::now());
            if let Some(v) = std::fs::read_to_string(path).ok().and_then(|s| s.trim().parse::<f32>().ok()) {
                let v = v.clamp(0.0, 1.0);
                if CUR.swap(v.to_bits(), Ordering::Relaxed) != v.to_bits() {
                    eprintln!("b2 mirror: cache-prior lambda = {v} (from {path})");
                }
            }
        }
    }
    match CUR.load(Ordering::Relaxed) {
        u32::MAX => *BASE,
        bits => f32::from_bits(bits),
    }
}

/// `V41_SUB_DRY=1` (mode 3): compute the cache-prior's selection but route
/// with the PLAIN picks; the would-be swaps are counted (`sub.*`) and traced
/// (`c` lines). Numerics unchanged. Use it to sweep lambda to a swap rate.
pub fn dry() -> bool {
    static D: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| matches!(std::env::var("V41_SUB_DRY").as_deref(), Ok("1") | Ok("on")));
    *D
}

/// `V41_SUB_PROTECT` (mode 3; default 2): the original top picks the prior
/// may never displace (the paper's J; 2 for fine-grained MoEs).
pub fn protect() -> u32 {
    static J: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        std::env::var("V41_SUB_PROTECT").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(2).min(6)
    });
    *J
}

/// Running per-layer average of the selection-score range (`Delta_layer`),
/// as f32 bits; 0 = not observed yet.
static DELTA_BITS: [std::sync::atomic::AtomicU32; LAYERS] = [const { std::sync::atomic::AtomicU32::new(0) }; LAYERS];

/// Fold one call's per-row ranges into the layer's running average
/// (exponential, alpha 0.05 per lane-layer call; the first call seeds it).
pub fn observe_range(layer: i32, ranges: &[f32]) {
    let l = layer as usize;
    if l >= LAYERS || ranges.is_empty() {
        return;
    }
    let kept: Vec<f32> = ranges.iter().copied().filter(|r| r.is_finite() && *r > 0.0).collect();
    if kept.is_empty() {
        return;
    }
    let mean = kept.iter().sum::<f32>() / kept.len() as f32;
    let old = f32::from_bits(DELTA_BITS[l].load(Ordering::Relaxed));
    let new = if old > 0.0 { old + 0.05 * (mean - old) } else { mean };
    DELTA_BITS[l].store(new.to_bits(), Ordering::Relaxed);
}

/// `Delta_layer`, once observed.
pub fn delta(layer: i32) -> Option<f32> {
    let l = layer as usize;
    if l >= LAYERS {
        return None;
    }
    let d = f32::from_bits(DELTA_BITS[l].load(Ordering::Relaxed));
    (d > 0.0).then_some(d)
}

/// `V41_SUB_PENDING` (default on): overlay picks of sent, unanswered requests
/// as resident (module doc; turn off under box-2 PARK).
pub fn pending_on() -> bool {
    static P: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_SUB_PENDING").as_deref() != Ok("0"));
    *P
}

/// `V41_SUB_ADMIT` (default on): queue swapped-away box-2 experts as box-2
/// prefetch words so they are read in the background (module doc).
pub fn admit_on() -> bool {
    static A: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_SUB_ADMIT").as_deref() != Ok("0"));
    *A
}

/// `V41_SUB_MAX_W`: cap on the weight moved by a swap (see the module doc).
pub fn max_w() -> Option<f32> {
    static W: std::sync::LazyLock<Option<f32>> = std::sync::LazyLock::new(|| {
        std::env::var("V41_SUB_MAX_W").ok().and_then(|v| v.parse::<f32>().ok()).filter(|w| *w > 0.0)
    });
    *W
}

/// `V41_SUB_INCOMING` (default 2; 0 = off): for how many decode steps a
/// queued background admission counts as resident (module doc, INCOMING).
pub fn incoming_steps() -> u32 {
    static N: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        std::env::var("V41_SUB_INCOMING").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(2)
    });
    *N
}

/// A decode step begins: advance the INCOMING clock. Every arena decode
/// driver calls this first; a path that never does leaves marks inactive.
pub fn begin_step() {
    STEP.fetch_add(1, Ordering::Relaxed);
}

/// A prefill-shaped pass begins (every prefill entry calls this after
/// switching the link to batch phase): end every live mark. A chunk pulls ~100
/// experts per layer through box 2's pool, so what was on its way before it is
/// likely evicted after. `+ N + 1`: a mark made at step s (`from = s + 1`) is
/// then at least N steps old.
pub fn expire_incoming() {
    STEP.fetch_add(incoming_steps() + 1, Ordering::Relaxed);
}

/// Admissions `(layer << 16) | e` were just queued for box 2: count them as
/// resident from the next decode step (module doc, INCOMING).
pub fn note_incoming(words: &[u32]) {
    if incoming_steps() == 0 {
        return;
    }
    let from = STEP.load(Ordering::Relaxed).wrapping_add(1).max(1);
    for &w in words {
        let (l, e) = ((w >> 16) as usize, (w & 0xffff) as usize);
        if l < LAYERS && e < NE {
            INCOMING[l][e].store(from, Ordering::Relaxed);
        }
    }
}

/// Is `(l, e)`'s admission mark inside its window?
fn incoming(l: usize, e: usize) -> bool {
    let n = incoming_steps();
    let from = INCOMING[l][e].load(Ordering::Relaxed);
    n > 0 && from != 0 && STEP.load(Ordering::Relaxed).wrapping_sub(from) < n
}

/// Overwrite `layer`'s row from a reply's residency map
/// (`proto::RESID_WORDS` u32s, bit e = expert e), and drop its pending row.
/// INCOMING marks are left alone (module doc).
pub fn update(layer: u32, words: &[u32]) {
    let l = layer as usize;
    if l >= LAYERS {
        return;
    }
    for (i, slot) in BITS[l].iter().enumerate() {
        let lo = words.get(2 * i).copied().unwrap_or(0) as u64;
        let hi = words.get(2 * i + 1).copied().unwrap_or(0) as u64;
        slot.store(lo | (hi << 32), Ordering::Relaxed);
        PENDING[l][i].store(0, Ordering::Relaxed);
    }
    SEEN[l].store(true, Ordering::Release);
}

/// A request for `layer` with these picks (`NO_PICK` / out of range ignored)
/// was just sent to box 2: overlay them as pending until the layer's next
/// reply.
pub fn note_submitted(layer: u32, sel: &[i32]) {
    let l = layer as usize;
    if l >= LAYERS {
        return;
    }
    for &e in sel {
        if (0..N_EXPERT as i32).contains(&e) {
            PENDING[l][(e / 64) as usize].fetch_or(1u64 << (e % 64), Ordering::Relaxed);
        }
    }
}

/// Will box 2 serve `(layer, e)` without a new read? Its last reply for `layer`
/// says it holds `e`, or a request already sent will bring it in (PENDING), or
/// a background read of it is already queued (INCOMING). `None` until box 2
/// has reported `layer` at all.
pub fn resident(layer: i32, e: u32) -> Option<bool> {
    lookup(layer, e).map(|r| r.held || (r.pending && pending_on()) || r.incoming)
}

/// Box 1's view of one box-2 expert, the sources kept apart (for the trace).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Residency {
    /// Held per box 2's last reply for the layer.
    pub held: bool,
    /// In a sent, unanswered request (`note_submitted`), whether or not
    /// `V41_SUB_PENDING` counts it.
    pub pending: bool,
    /// A queued background admission inside its step window
    /// (`note_incoming`; false when `V41_SUB_INCOMING=0`).
    pub incoming: bool,
}

/// `(layer, e)`'s residency sources; `None` until box 2 has reported `layer`.
pub fn lookup(layer: i32, e: u32) -> Option<Residency> {
    let l = layer as usize;
    if l >= LAYERS || e >= N_EXPERT || !SEEN[l].load(Ordering::Acquire) {
        return None;
    }
    let (w, b) = ((e / 64) as usize, e % 64);
    Some(Residency {
        held: (BITS[l][w].load(Ordering::Relaxed) >> b) & 1 == 1,
        pending: (PENDING[l][w].load(Ordering::Relaxed) >> b) & 1 == 1,
        incoming: incoming(l, e as usize),
    })
}

// ---- per-step counters (drained by the multistream profile) ----

static N_PREDICTED: AtomicU64 = AtomicU64::new(0);
static N_AVOIDED: AtomicU64 = AtomicU64::new(0);
static N_SLOTS: AtomicU64 = AtomicU64::new(0);
static N_BLOCKED: AtomicU64 = AtomicU64::new(0);
static N_FAILED: AtomicU64 = AtomicU64::new(0);
static N_ADMITS: AtomicU64 = AtomicU64::new(0);
static N_INCOMING: AtomicU64 = AtomicU64::new(0);

/// Background admissions queued (`V41_SUB_ADMIT`), for the profile.
pub fn note_admits(n: usize) {
    N_ADMITS.fetch_add(n as u64, Ordering::Relaxed);
}

/// Count one lane-layer's distinct box-2 picks (`is_box2`) that box 2's last
/// reply calls missing (and no counted PENDING covers) but a queued admission
/// does (INCOMING). Each would otherwise be a predicted miss, but not every
/// one would have been swapped (protected ranks, no held alternative within
/// reach), so this is an UPPER BOUND on the swaps the overlay prevented. Pass
/// the ROUTER's picks.
pub fn note_incoming_covered(layer: i32, picks: &[i32], is_box2: impl Fn(u32) -> bool) {
    if incoming_steps() == 0 {
        return;
    }
    let mut seen: Vec<i32> = Vec::new();
    for &e in picks {
        if !(0..N_EXPERT as i32).contains(&e) || seen.contains(&e) {
            continue;
        }
        seen.push(e);
        if let Some(r) = lookup(layer, e as u32) {
            if is_box2(e as u32) && !r.held && !(r.pending && pending_on()) && r.incoming {
                N_INCOMING.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// `(predicted box-2 misses, reads avoided, picks substituted, misses left
/// alone, planner failures, background admissions queued, picks covered by
/// INCOMING)` since the last call. Failures should be 0 (see
/// `SubOutcome::failed`). The first, second, fourth and last count distinct
/// experts per lane-layer; box 2 counts a miss once per (possibly merged)
/// pass, so compare with `box2.misses_x1e6` as an upper bound. In dry-run mode
/// "avoided" and "substituted" are what WOULD have happened.
pub fn take_sub_stats() -> (u64, u64, u64, u64, u64, u64, u64) {
    (
        N_PREDICTED.swap(0, Ordering::Relaxed),
        N_AVOIDED.swap(0, Ordering::Relaxed),
        N_SLOTS.swap(0, Ordering::Relaxed),
        N_BLOCKED.swap(0, Ordering::Relaxed),
        N_FAILED.swap(0, Ordering::Relaxed),
        N_ADMITS.swap(0, Ordering::Relaxed),
        N_INCOMING.swap(0, Ordering::Relaxed),
    )
}

pub fn record(o: &SubOutcome) {
    N_PREDICTED.fetch_add(o.predicted as u64, Ordering::Relaxed);
    N_AVOIDED.fetch_add(o.avoided as u64, Ordering::Relaxed);
    N_SLOTS.fetch_add(o.slots as u64, Ordering::Relaxed);
    N_BLOCKED.fetch_add(o.blocked as u64, Ordering::Relaxed);
    N_FAILED.fetch_add(o.failed as u64, Ordering::Relaxed);
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SubOutcome {
    /// Distinct experts predicted to miss on box 2.
    pub predicted: u32,
    /// ... of which substituted in every row that picked them (read avoided).
    pub avoided: u32,
    /// Pick slots rewritten.
    pub slots: u32,
    /// ... predicted misses left alone: some row picked them above `min_rank`,
    /// or had no alternative that is held and within the weight cap.
    pub blocked: u32,
    /// Slots the APPLY pass could not swap although the plan said it could.
    /// Must be 0: the apply pass repeats the fixpoint's last plan exactly. If
    /// it ever is not, the row is still consistent (unswapped, sums to the
    /// scale), but another row may have swapped the same expert for nothing.
    pub failed: u32,
}

/// Which picks may be swapped, and for what.
#[derive(Debug, Clone, Copy)]
pub struct SubRules {
    /// 1-based: picks at a rank < this (heavier) are never swapped.
    pub min_rank: usize,
    /// Cap on both the missing pick's weight and the substitute's
    /// renormalized weight.
    pub max_w: Option<f32>,
    /// The weights' sum (1.5).
    pub scale: f32,
}

impl SubRules {
    pub fn from_env(scale: f32) -> Self {
        Self { min_rank: min_rank(), max_w: max_w(), scale }
    }
}

/// Plan (and apply) substitutions over a batch of rows.
///
/// * `sel` / `ew`: `[b, nu]` picks in rank order and their weights, rewritten
///   in place.
/// * `alts` / `alt_w`: `[b, na]` alternatives in rank order, with weights on
///   `ew`'s ORIGINAL scale (router `alt_w`: prob / top-`nu` prob sum x scale).
/// * `predicted_miss(e)`: will this pick make box 2 read from disk?
/// * `acceptable(a)`: is alternative `a` served without a read (resident on
///   whichever box computes it)?
///
/// A missing expert is substituted only if EVERY row that picked it can be;
/// otherwise box 2 reads it anyway and substituting the other rows would cost
/// quality for no time. That is found by planning every row, blocking each
/// expert some row cannot swap, and re-planning until nothing new is blocked
/// (a blocked expert frees the alternatives it had taken). Each swap
/// renormalizes its row exactly: with the row's weights on the current sum's
/// scale, swapping slot k for an alternative whose current-scale weight is
/// `wa` divides every weight by `c = 1 - ew[k]/scale + wa/scale` and gives the
/// substitute `wa / c`.
#[allow(clippy::too_many_arguments)]
pub fn substitute(
    sel: &mut [i32],
    ew: &mut [f32],
    alts: &[i32],
    alt_w: &[f32],
    nu: usize,
    na: usize,
    rules: SubRules,
    predicted_miss: impl Fn(i32) -> bool,
    acceptable: impl Fn(i32) -> bool,
) -> SubOutcome {
    let mut out = SubOutcome::default();
    let na = na.min(MAX_ALT);
    if nu == 0 || na == 0 || sel.is_empty() {
        return out;
    }
    let b = sel.len() / nu;
    debug_assert_eq!(ew.len(), sel.len());
    debug_assert!(alts.len() >= b * na && alt_w.len() >= b * na);

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

    // Plan one row in place (on `row`/`w`), skipping blocked experts; returns
    // the swaps made and marks in `failed` every missing expert this row could
    // not swap.
    let plan_row = |row: &mut [i32], w: &mut [f32], r: usize, blocked: &[bool], failed: &mut [bool]| -> u32 {
        let mut taken = [false; MAX_ALT];
        let mut c_cum = 1.0f32; // this row's current sum / the router's original
        let mut swaps = 0u32;
        for k in 0..nu {
            let Some(mi) = missing.iter().position(|&m| m == row[k]) else { continue };
            if blocked[mi] {
                continue;
            }
            if k + 1 < rules.min_rank {
                failed[mi] = true;
                continue;
            }
            let pick = (0..na).find_map(|j| {
                let a = alts[r * na + j];
                if a < 0 || taken[j] || row.contains(&a) || !acceptable(a) {
                    return None;
                }
                let wa = alt_w[r * na + j] / c_cum;
                let c = (1.0 - w[k] / rules.scale + wa / rules.scale).max(1e-6);
                if let Some(cap) = rules.max_w {
                    if w[k] > cap || wa / c > cap {
                        return None;
                    }
                }
                Some((j, wa, c))
            });
            match pick {
                Some((j, wa, c)) => {
                    taken[j] = true;
                    for x in w.iter_mut() {
                        *x /= c;
                    }
                    w[k] = wa / c;
                    row[k] = alts[r * na + j];
                    c_cum *= c;
                    swaps += 1;
                }
                None => failed[mi] = true,
            }
        }
        swaps
    };

    // Block up front what fails regardless of competition for alternatives: a
    // row that picked the expert above `min_rank`, or a row with no held
    // alternative at all. Otherwise such an expert could take a row's only
    // alternative in the first round, fail another expert on contention, and be
    // blocked itself too late to free it (the blocked set only grows).
    let mut blocked = vec![false; missing.len()];
    for r in 0..b {
        let row = &sel[r * nu..(r + 1) * nu];
        for (k, &e) in row.iter().enumerate() {
            let Some(mi) = missing.iter().position(|&m| m == e) else { continue };
            let any_alt = (0..na).any(|j| {
                let a = alts[r * na + j];
                a >= 0 && !row.contains(&a) && acceptable(a)
            });
            if k + 1 < rules.min_rank || !any_alt {
                blocked[mi] = true;
            }
        }
    }
    // Fixpoint over what remains (contention, the weight cap), on scratch
    // copies.
    loop {
        let mut failed = vec![false; missing.len()];
        for r in 0..b {
            let mut row = sel[r * nu..(r + 1) * nu].to_vec();
            let mut w = ew[r * nu..(r + 1) * nu].to_vec();
            plan_row(&mut row, &mut w, r, &blocked, &mut failed);
        }
        let mut grew = false;
        for (bl, f) in blocked.iter_mut().zip(&failed) {
            if *f && !*bl {
                *bl = true;
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    // Apply: the same deterministic plan, now with nothing left to fail.
    let mut failed = vec![false; missing.len()];
    for r in 0..b {
        let (row, w) = (&mut sel[r * nu..(r + 1) * nu], &mut ew[r * nu..(r + 1) * nu]);
        out.slots += plan_row(row, w, r, &blocked, &mut failed);
    }
    out.failed = failed.iter().filter(|&&f| f).count() as u32;
    debug_assert_eq!(out.failed, 0, "the fixpoint left a failing swap");
    out.blocked = blocked.iter().filter(|&&x| x).count() as u32;
    out.avoided = out.predicted - out.blocked;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: f32 = 1.5;
    const R6: SubRules = SubRules { min_rank: 6, max_w: None, scale: S };

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
        let o = substitute(&mut sel, &mut ew, &alts, &alt_w, 6, 2, R6, |e| e == 15, |_| true);
        assert_eq!(o, SubOutcome { predicted: 1, avoided: 1, slots: 1, blocked: 0, failed: 0 });
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
        let r5 = SubRules { min_rank: 5, ..R6 };
        let o = substitute(&mut sel, &mut ew, &alts, &alt_w, 6, 3, r5, |e| e == 14 || e == 15, |_| true);
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
        let o = substitute(&mut sel, &mut ew, &alts, &alt_w, 6, 2, R6, |e| e == 15, |_| true);
        assert_eq!(o, SubOutcome { predicted: 1, avoided: 0, slots: 0, blocked: 1, failed: 0 });
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
        let o = substitute(&mut sel, &mut ew, &alts, &alt_w, 6, 2, R6, |e| e == 15, |a| a < 30);
        assert_eq!(o.blocked, 1);
        assert_eq!(o.slots, 0);
        assert_eq!(sel, before);
    }

    #[test]
    fn a_blocked_expert_frees_its_alternative() {
        // Row 0 misses Y=14 (rank 5) and X=15 (rank 6) with ONE usable
        // alternative. Row 1 picks Y at rank 1, so Y is blocked; X must then get
        // the alternative Y would have taken.
        let mut sel = vec![10, 11, 12, 13, 14, 15, 14, 1, 2, 3, 4, 5];
        let mut ew = vec![0.25; 12];
        let alts = vec![20, 21, 30, 31];
        let alt_w = vec![0.1; 4];
        let r5 = SubRules { min_rank: 5, ..R6 };
        let o = substitute(&mut sel, &mut ew, &alts, &alt_w, 6, 2, r5, |e| e == 14 || e == 15, |a| a == 20);
        assert_eq!((o.predicted, o.avoided, o.blocked, o.slots), (2, 1, 1, 1));
        assert_eq!(&sel[..6], &[10, 11, 12, 13, 14, 20]);
    }

    #[test]
    fn max_w_caps_the_mass_moved_at_any_rank() {
        // Flat row: every pick 0.25. Heavy row: rank 1 is 0.6. With the cap at
        // 0.3 and min_rank 1, the flat row's rank-1 miss swaps; the heavy row's
        // does not.
        let any = SubRules { min_rank: 1, max_w: Some(0.3), scale: S };
        let mut flat = vec![15, 11, 12, 13, 14, 16];
        let mut w_flat = vec![0.25f32; 6];
        let o = substitute(&mut flat, &mut w_flat, &[20], &[0.24], 6, 1, any, |e| e == 15, |_| true);
        assert_eq!(o.slots, 1);
        assert_eq!(flat[0], 20);

        let mut heavy = vec![15, 11, 12, 13, 14, 16];
        let mut w_heavy = vec![0.6f32, 0.3, 0.2, 0.15, 0.15, 0.1];
        let o = substitute(&mut heavy, &mut w_heavy, &[20], &[0.09], 6, 1, any, |e| e == 15, |_| true);
        assert_eq!((o.slots, o.blocked), (0, 1));
        assert_eq!(heavy[0], 15);

        // The SUBSTITUTE's renormalized weight is capped too.
        let mut s = vec![10, 11, 12, 13, 14, 15];
        let mut w = vec![0.4f32, 0.3, 0.3, 0.2, 0.2, 0.1];
        let o = substitute(&mut s, &mut w, &[20], &[0.5], 6, 1, SubRules { min_rank: 6, max_w: Some(0.3), scale: S }, |e| e == 15, |_| true);
        assert_eq!(o.slots, 0, "a weak pick replaced by a heavy substitute moves too much mass");
    }

    #[test]
    fn skips_unacceptable_and_duplicate_alternatives() {
        // 1st alt not resident, 2nd already a pick in the row, 3rd is used.
        let mut sel = vec![10, 11, 12, 13, 14, 15];
        let mut ew = vec![0.25; 6];
        let alts = vec![20, 11, 22];
        let alt_w = vec![0.1, 0.1, 0.1];
        let o = substitute(&mut sel, &mut ew, &alts, &alt_w, 6, 3, R6, |e| e == 15, |a| a != 20);
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
        let o = substitute(&mut sel, &mut ew, &alts, &alt_w, 6, 2, R6, miss, |a| !miss(a));
        assert_eq!(sel[5], 21);
        assert_eq!(o.predicted, 1, "20 is not a pick, so it is not a predicted miss");
    }

    #[test]
    fn nothing_missing_is_a_no_op() {
        let mut sel = vec![10, 11, 12, 13, 14, 15];
        let mut ew = vec![0.25; 6];
        let before = (sel.clone(), ew.clone());
        let o = substitute(&mut sel, &mut ew, &[20, 21], &[0.1, 0.1], 6, 2, R6, |_| false, |_| true);
        assert_eq!(o, SubOutcome::default());
        assert_eq!((sel, ew), before);
    }

    /// Randomized batches (fixed seed): the apply pass never fails, every
    /// swapped row still sums to the scale, no row ends up with a duplicate
    /// pick, and every swapped-in expert was acceptable. Runs the invariant
    /// the release-mode `debug_assert` cannot.
    #[test]
    fn randomized_plans_apply_cleanly() {
        let mut s = 0x2545f4914f6cdd1du64;
        let mut rnd = move |n: u32| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % n as u64) as u32
        };
        let (nu, na) = (6usize, 4usize);
        let (mut total_slots, mut multi_row_swaps) = (0u32, 0u32);
        for trial in 0..3000 {
            let b = 1 + rnd(6) as usize;
            let pool = 16 + rnd(24) as i32; // small id space => collisions
            let mut sel = Vec::with_capacity(b * nu);
            let mut alts = Vec::with_capacity(b * na);
            let mut ew = Vec::with_capacity(b * nu);
            let mut alt_w = Vec::with_capacity(b * na);
            let mut probs: Vec<std::collections::HashMap<i32, f32>> = Vec::with_capacity(b);
            for _ in 0..b {
                // 6 + 4 distinct ids per row, in rank order.
                let mut ids: Vec<i32> = Vec::new();
                while ids.len() < nu + na {
                    let e = rnd(pool as u32) as i32;
                    if !ids.contains(&e) {
                        ids.push(e);
                    }
                }
                let mut p: Vec<f32> = (0..nu + na).map(|_| 0.05 + rnd(1000) as f32 / 1000.0).collect();
                p.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let sum: f32 = p[..nu].iter().sum();
                probs.push(ids.iter().copied().zip(p.iter().copied()).collect());
                sel.extend_from_slice(&ids[..nu]);
                alts.extend_from_slice(&ids[nu..]);
                ew.extend(p[..nu].iter().map(|x| x / sum * S));
                alt_w.extend(p[nu..].iter().map(|x| x / sum * S));
            }
            let miss_mask: u64 = ((rnd(u32::MAX) as u64) << 32) | rnd(u32::MAX) as u64;
            let ok_mask: u64 = ((rnd(u32::MAX) as u64) << 32) | rnd(u32::MAX) as u64;
            let miss = |e: i32| (miss_mask >> (e % 64)) & 1 == 1;
            let acceptable = |a: i32| !miss(a) && (ok_mask >> (a % 64)) & 1 == 1;
            let rules = SubRules {
                min_rank: 1 + rnd(6) as usize,
                max_w: if rnd(2) == 0 { None } else { Some(0.1 + rnd(30) as f32 / 100.0) },
                scale: S,
            };
            let before = sel.clone();
            let o = substitute(&mut sel, &mut ew, &alts, &alt_w, nu, na, rules, miss, acceptable);
            assert_eq!(o.failed, 0, "trial {trial}: {o:?}");
            total_slots += o.slots;
            // Accounting: the predicted misses no longer picked anywhere are
            // exactly the avoided reads.
            let mut missing: Vec<i32> = before.iter().copied().filter(|&e| miss(e)).collect();
            missing.sort();
            missing.dedup();
            let gone = missing.iter().filter(|e| !sel.contains(e)).count() as u32;
            assert_eq!(gone, o.avoided, "trial {trial}: {o:?}");
            let mut rows_swapped = 0;
            for r in 0..b {
                let row = &sel[r * nu..(r + 1) * nu];
                let sum: f32 = ew[r * nu..(r + 1) * nu].iter().sum();
                assert!((sum - S).abs() < 1e-3, "trial {trial} row {r}: weights sum {sum}");
                // Exact against ref.Gate over the FINAL chosen set.
                let p_sum: f32 = row.iter().map(|e| probs[r][e]).sum();
                for k in 0..nu {
                    let want = probs[r][&row[k]] / p_sum * S;
                    assert!((ew[r * nu + k] - want).abs() < 1e-5, "trial {trial} row {r} slot {k}: {} vs ref.Gate {want}", ew[r * nu + k]);
                }
                if row != &before[r * nu..(r + 1) * nu] {
                    rows_swapped += 1;
                }
                for k in 0..nu {
                    assert!(!row[k + 1..].contains(&row[k]), "trial {trial} row {r}: duplicate pick");
                    if row[k] != before[r * nu + k] {
                        assert!(acceptable(row[k]), "trial {trial}: swapped in a non-acceptable expert");
                        assert!(k + 1 >= rules.min_rank, "trial {trial}: swapped above min_rank");
                        if let Some(cap) = rules.max_w {
                            // The cap holds at swap time. A later swap in the
                            // same row divides everything by its `c` again, which
                            // can lift an earlier substitute a little.
                            let p_before: f32 = before[r * nu..(r + 1) * nu].iter().map(|e| probs[r][e]).sum();
                            assert!(probs[r][&before[r * nu + k]] / p_before * S <= cap + 1e-4 || !miss(before[r * nu + k]), "trial {trial}: swapped a pick heavier than the cap");
                            assert!(ew[r * nu + k] <= cap * 1.25, "trial {trial}: substitute far above the cap: {}", ew[r * nu + k]);
                        }
                    }
                }
            }
            if rows_swapped > 1 {
                multi_row_swaps += 1;
            }
            // All-rows rule: an expert still picked anywhere was not swapped away
            // anywhere else.
            for (i, &e) in before.iter().enumerate() {
                if sel[i] != e {
                    assert!(!sel.contains(&e), "trial {trial}: {e} swapped in one row, kept in another");
                }
            }
        }
        // Not vacuous: a planner that never swaps would fail here.
        assert!(total_slots > 1000, "only {total_slots} swaps in 3000 trials");
        assert!(multi_row_swaps > 300, "only {multi_row_swaps} trials swapped in several rows");
    }

    /// Cache-prior Delta: the first call seeds it, later calls move it by 5%
    /// of the gap; non-finite and non-positive ranges are ignored.
    #[test]
    fn delta_running_average() {
        let l = (LAYERS - 2) as i32;
        assert_eq!(delta(l), None);
        observe_range(l, &[]);
        observe_range(l, &[f32::NAN, 0.0]);
        assert_eq!(delta(l), None, "no valid range yet");
        observe_range(l, &[2.0, 4.0]);
        assert_eq!(delta(l), Some(3.0));
        observe_range(l, &[5.0]);
        assert!((delta(l).unwrap() - 3.1).abs() < 1e-6);
    }

    /// The mirror is process-global; each mirror test uses a layer no other
    /// test touches.
    #[test]
    fn mirror_update_pending_and_lookup() {
        let l = (LAYERS - 1) as u32;
        assert_eq!(resident(l as i32, 3), None, "unseen layer is unknown");
        let mut w = vec![0u32; (N_EXPERT as usize).div_ceil(32)];
        w[0] = 1 << 3;
        w[(N_EXPERT as usize).div_ceil(32) - 1] = 1 << 31; // the last expert
        update(l, &w);
        assert_eq!(resident(l as i32, 3), Some(true));
        assert_eq!(resident(l as i32, 4), Some(false));
        assert_eq!(resident(l as i32, N_EXPERT - 1), Some(true));
        assert_eq!(resident(l as i32, N_EXPERT), None, "out of range");
        // A request already sent for this layer brings 4 in.
        note_submitted(l, &[4, -1, 9999]);
        assert_eq!(resident(l as i32, 4), Some(true));
        assert_eq!(
            lookup(l as i32, 4),
            Some(Residency { held: false, pending: true, incoming: false }),
            "pending, not held"
        );
        // The next reply is authoritative again.
        update(l, &w);
        assert_eq!(resident(l as i32, 4), Some(false));
    }

    /// INCOMING: a queued admission counts as resident from the NEXT decode
    /// step, for `incoming_steps()` steps, whatever replies arrive meanwhile.
    /// Own layer; the only test that advances the step clock.
    #[test]
    fn incoming_window_and_count() {
        let l = (LAYERS - 3) as u32;
        let n = incoming_steps();
        assert!(n >= 1, "test assumes V41_SUB_INCOMING is on (default 2)");
        let empty = vec![0u32; NE.div_ceil(32)];
        update(l, &empty);
        // Out-of-range words are ignored.
        note_incoming(&[(l << 16) | 7, (LAYERS as u32) << 16, (l << 16) | 9999]);
        assert_eq!(resident(l as i32, 7), Some(false), "not in the step that queued it");
        update(l, &empty);
        assert_eq!(resident(l as i32, 7), Some(false), "a reply in the same step does not activate it");
        for k in 0..n {
            begin_step();
            assert_eq!(resident(l as i32, 7), Some(true), "inside the window, step {k}");
            assert!(lookup(l as i32, 7).is_some_and(|r| r.incoming && !r.held));
        }
        begin_step();
        assert_eq!(resident(l as i32, 7), Some(false), "expired after {n} steps");

        // A reply showing it held does not clear the mark: an OLDER map
        // consumed after it (PARK) must not make it a miss again.
        note_incoming(&[(l << 16) | 8]);
        begin_step();
        let mut w = empty.clone();
        w[0] = 1 << 8;
        update(l, &w);
        assert_eq!(lookup(l as i32, 8), Some(Residency { held: true, pending: false, incoming: true }));
        update(l, &empty);
        assert_eq!(resident(l as i32, 8), Some(true), "still covered by its mark");
        // A prefill pass ends it at once.
        expire_incoming();
        assert_eq!(resident(l as i32, 8), Some(false), "expired by prefill");

        // Covered picks: distinct, box 2's only, in-window marks only.
        note_incoming(&[(l << 16) | 10]);
        begin_step();
        let _ = take_sub_stats();
        note_incoming_covered(l as i32, &[10, 10, 11, -1, 9999], |_| true);
        assert_eq!(take_sub_stats().6, 1);
        note_incoming_covered(l as i32, &[10], |_| false);
        assert_eq!(take_sub_stats().6, 0, "box-1 picks are not counted");
    }
}
