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
//! PINNING (`V41_B2_PIN=1`, default off; 2026-09-26): the hub is never
//! surprised. Without it "held" is box 2's landed map as of the layer's last
//! reply, and box 2 evicts ~20 experts between two replies of a layer: half of
//! decode's box-2 paging was on picks the mirror called held (measured
//! 2026-09-26), which the substitution never gets to avoid. With it:
//! * every request carries `proto::REQ_FLAG_PIN`; box 2 PINS what it reports
//!   (a reply's map is then its pinned set for the layer, never merely
//!   resident) and never evicts a pinned expert (`remote_experts`, the block
//!   above `PinBook`, has box 2's side and the no-deadlock budget);
//! * the hub RELEASES its coldest held experts (decayed, rank-weighted counts
//!   of the ROUTER's decode picks, `V41_B2_PIN_WANTS`)
//!   once box 2's pinned count nears its budget: `begin_step` clears them here
//!   at once and queues `layer << 16 | e` RELEASE words that ride on the next
//!   requests (`REQ_FLAG_RELEASE`). No request is routed-but-unsent at
//!   `begin_step`, so no pick can be routed as held and served released;
//! * box 2 applies release words in wire order and echoes how many it has
//!   applied (`epoch`); `update_pinned` masks every release queued after that
//!   epoch out of the map, so a map built before a release cannot bring the
//!   expert back. Hence `held ⊆ pinned ⊆ resident` at every submit;
//! * the reply also carries the pass's PAGED bits, and `check_surprises`
//!   counts `held at submit & paged` (must be 0): ERROR log, rate-limited,
//!   and a panic under `V41_B2_ASSERT_NO_SURPRISE=1`;
//! * negotiation: until a reply shows `RESP_FLAG_PIN` no release word is sent;
//!   a reply to a pin request without it (an older daemon) turns pinning off
//!   for the connection and the mirror works as before. A (re)connect resets
//!   the mirror (`on_connect`); box 2 drops a connection's pins with it.
//! `V41_SUB=0` + `V41_B2_PIN=1` keeps the mirror (maps are requested) but
//! routes exactly as without it: pinning only changes which slot box 2 evicts.
//! Knobs: `V41_B2_PIN_HEADROOM` (default 256 slots below box 2's budget before
//! releasing, down to twice that), `V41_B2_PIN_DECAY_STEPS` (default 256: pick
//! counts halve), `V41_B2_PIN_WANTS` (default on: rank by the router's picks,
//! not the prior's), `V41_B2_ASSERT_NO_SURPRISE`.
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

/// Ask box 2 for residency maps (`proto::REQ_FLAG_RESID`)? Pinning needs them
/// too, with or without a substitution mode.
pub fn wanted() -> bool {
    mode() > 0 || pin_wanted()
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
/// Pinning releases happen here too (module doc: nothing is routed but
/// unsent at this point).
pub fn begin_step() {
    STEP.fetch_add(1, Ordering::Relaxed);
    pin_begin_step();
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

// ---- pinning (module doc, PINNING) ----

use super::remote_experts::proto::{RESID_WORDS, REQ_FLAG_PIN};

/// `V41_B2_PIN` (0 = unread, 1 = off, 2 = on); `set_pin_wanted` overrides.
static PIN_KNOB: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// `V41_B2_PIN=1`: ask box 2 to pin what it reports (default off).
pub fn pin_wanted() -> bool {
    match PIN_KNOB.load(Ordering::Relaxed) {
        0 => {
            let on = matches!(std::env::var("V41_B2_PIN").as_deref(), Ok("1") | Ok("on"));
            if on {
                eprintln!(
                    "b2 mirror: V41_B2_PIN=1 PINNING (headroom {}, decay every {} steps, rank by {}{})",
                    pin_headroom(),
                    pin_decay_steps(),
                    if pin_wants() { "the router's picks (V41_B2_PIN_WANTS)" } else { "the picks sent (V41_B2_PIN_WANTS=0)" },
                    if assert_no_surprise() { ", V41_B2_ASSERT_NO_SURPRISE" } else { "" }
                );
            }
            let _ = PIN_KNOB.compare_exchange(0, if on { 2 } else { 1 }, Ordering::Relaxed, Ordering::Relaxed);
            PIN_KNOB.load(Ordering::Relaxed) == 2
        }
        k => k == 2,
    }
}

/// Override `V41_B2_PIN` (tests; takes effect at the next connect).
pub fn set_pin_wanted(on: bool) {
    PIN_KNOB.store(if on { 2 } else { 1 }, Ordering::Relaxed);
}

/// `V41_B2_PIN_HEADROOM` (default 256): release once box 2's pinned count
/// (net of releases in flight) exceeds `budget - headroom`, down to
/// `budget - 2 * headroom`.
pub fn pin_headroom() -> u32 {
    static H: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        std::env::var("V41_B2_PIN_HEADROOM").ok().and_then(|v| v.parse().ok()).unwrap_or(256)
    });
    *H
}

/// `V41_B2_PIN_DECAY_STEPS` (default 256; 0 = never): the decode pick counts
/// that rank experts for release halve every this many steps.
pub fn pin_decay_steps() -> u32 {
    static D: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        std::env::var("V41_B2_PIN_DECAY_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(256)
    });
    *D
}

/// `V41_B2_PIN_WANTS` (default ON; `0` = the ledger as deployed 2026-09-27):
/// rank held experts for release by the ROUTER's own decode picks (in its rank
/// order), not by the picks sent after the cache prior, and break count ties by
/// the last want (the longest-unwanted goes first) instead of by (layer, id).
/// Counting the sent picks let the held set earn its own credit: a held expert
/// the prior boosted into a row was counted, the pick it displaced was not. So
/// a displaced expert admitted in the background (`V41_SUB_ADMIT`) arrived with
/// a count of ~0, tied for coldest, and went at the next release: box 2 paid
/// the read, released it unused, and the prior swapped it away on its next
/// want. Box 1's hot set ranks by the router's picks for the same reason
/// (`forward_prefill`, "rank the hot set by the ROUTER's picks"). The wants
/// are masked to box 2's partition (`wants_for_box2`): an expert box 1 owns
/// must not earn box-2 pin credit from box 1's demand. And held experts box 2
/// no longer owns (a hot-set promotion moved them) are released first.
pub fn pin_wants() -> bool {
    static W: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_B2_PIN_WANTS").as_deref() != Ok("0"));
    *W
}

/// `V41_B2_ASSERT_NO_SURPRISE=1`: a surprise panics (verification runs).
/// Default: counted and logged. A surprise costs a few ms of box-2 paging; a
/// panic costs an outage for every agent on the server.
pub fn assert_no_surprise() -> bool {
    static A: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| matches!(std::env::var("V41_B2_ASSERT_NO_SURPRISE").as_deref(), Ok("1") | Ok("on")));
    *A
}

/// Most release words planned per step.
const PIN_RELEASE_MAX_PER_STEP: usize = 512;

/// Connection's pin state: 0 off (knob off), 1 asked (no reply yet), 2 on
/// (box 2 answered with a pin block), 3 unsupported (it answered without).
static PIN_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

static LEDGER: std::sync::LazyLock<std::sync::Mutex<PinLedger>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(PinLedger::new()));

static N_SURPRISES: AtomicU64 = AtomicU64::new(0);
static N_HELD_PICKS: AtomicU64 = AtomicU64::new(0);
static N_RELEASED: AtomicU64 = AtomicU64::new(0);
static N_RELEASED_UNUSED: AtomicU64 = AtomicU64::new(0);
/// Cumulative (never drained): for tests and the surprise log.
static TOT_SURPRISES: AtomicU64 = AtomicU64::new(0);
static TOT_HELD_PICKS: AtomicU64 = AtomicU64::new(0);
static TOT_RELEASED: AtomicU64 = AtomicU64::new(0);

/// Is release index `idx` later than box 2's `epoch` (wrapping)?
#[inline]
fn after(idx: u32, epoch: u32) -> bool {
    (idx.wrapping_sub(epoch) as i32) > 0
}

/// The hub's side of the pin contract: which box-2 experts it may treat as
/// HELD, the releases it has queued (and their wire indices), and the decayed
/// decode pick counts that choose what to release. Pure bookkeeping; the
/// process-wide instance is `LEDGER`, whose held rows are mirrored into `BITS`.
#[derive(Clone, Debug)]
pub struct PinLedger {
    /// Per `layer * NE + e`: index of the last release word queued for it (0
    /// = never).
    rel_idx: Vec<u32>,
    /// Release words queued so far (the last one's index; wrapping, skips 0).
    rel_sent: u32,
    /// Queued, not yet on the wire.
    queue: std::collections::VecDeque<u32>,
    /// Per layer, `WORDS` u64s: HELD.
    held: Vec<u64>,
    /// Per `layer * NE + e`: decode picks, WEIGHTED BY RANK (a rank-1 pick
    /// adds `N_EXPERT_USED`, a rank-6 pick adds 1), halved every `decay_steps`
    /// steps. The ROUTER's picks and ranks with `by_wants`, else the picks sent
    /// (see `pin_wants`). The reads box 2 still pays are the PROTECTED-rank
    /// picks the cache prior may not swap, so an expert that wins at rank 1-2
    /// outranks one wanted only at swappable ranks (2026-09-27: blocked picks
    /// were 4-6 of ~5 reads per 3-row step); a displaced want still counts.
    counts: Vec<u32>,
    steps: u32,
    /// `(epoch, pinned, budget)` of the reply with the newest epoch.
    last: Option<(u32, u32, u32)>,
    /// `pin_wants()` at creation: count the router's picks and break count
    /// ties by `last_want`. Off = the 2026-09-27 ledger exactly.
    by_wants: bool,
    /// Per `layer * NE + e`: `steps + 1` at the last want (0 = never).
    last_want: Vec<u32>,
    /// Per `layer * NE + e`: `steps + 1` when a background admission was last
    /// queued (`note_admits`); 0 = none since the last release.
    admitted: Vec<u32>,
    /// Per `layer * NE + e`: `steps + 1` when a request last SENT it to box 2
    /// (any shape; `note_sent`). A release with `admitted != 0 && last_sent <
    /// admitted` let go of an admission no request used since it was queued
    /// (a send in the same step counts as a use: the other lane's).
    last_sent: Vec<u32>,
    /// Such releases since the last `take_released_unused`.
    released_unused: u32,
}

impl Default for PinLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl PinLedger {
    pub fn new() -> Self {
        Self {
            rel_idx: vec![0; LAYERS * NE],
            rel_sent: 0,
            queue: Default::default(),
            held: vec![0; LAYERS * WORDS],
            counts: vec![0; LAYERS * NE],
            steps: 0,
            last: None,
            by_wants: pin_wants(),
            last_want: vec![0; LAYERS * NE],
            admitted: vec![0; LAYERS * NE],
            last_sent: vec![0; LAYERS * NE],
            released_unused: 0,
        }
    }

    /// A pin-mode reply's map for `layer` (box 2's PINNED set when it had
    /// applied `epoch` release words): the layer's held row is the map minus
    /// every expert released after that epoch. Returns the row.
    pub fn apply_map(&mut self, layer: u32, words: &[u32], epoch: u32) -> [u64; WORDS] {
        let mut row = [0u64; WORDS];
        let l = layer as usize;
        if l >= LAYERS {
            return row;
        }
        for (i, r) in row.iter_mut().enumerate() {
            let lo = words.get(2 * i).copied().unwrap_or(0) as u64;
            let hi = words.get(2 * i + 1).copied().unwrap_or(0) as u64;
            let mut m = lo | (hi << 32);
            let mut bits = m;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let e = i * 64 + b;
                if e >= NE || {
                    let idx = self.rel_idx[l * NE + e];
                    idx != 0 && after(idx, epoch)
                } {
                    m &= !(1u64 << b);
                }
            }
            *r = m;
        }
        self.held[l * WORDS..(l + 1) * WORDS].copy_from_slice(&row);
        row
    }

    /// Box 2's pin counters from a reply; the newest epoch wins.
    pub fn note_reply(&mut self, epoch: u32, pinned: u32, budget: u32) {
        if self.last.is_none_or(|(e, _, _)| !after(e, epoch)) {
            self.last = Some((epoch, pinned, budget));
        }
    }

    /// `(box 2's pinned count net of the releases it has not applied yet,
    /// its budget)`; `None` before any pin reply.
    pub fn est_pinned(&self) -> Option<(u32, u32)> {
        let (epoch, pinned, budget) = self.last?;
        let in_flight = if after(self.rel_sent, epoch) { self.rel_sent.wrapping_sub(epoch) } else { 0 };
        Some((pinned.saturating_sub(in_flight), budget))
    }

    pub fn held(&self, layer: u32, e: u32) -> bool {
        let (l, e) = (layer as usize, e as usize);
        l < LAYERS && e < NE && (self.held[l * WORDS + e / 64] >> (e % 64)) & 1 == 1
    }

    /// A decode pick sent to box 2 (ranks it for release), weight 1.
    pub fn note_pick(&mut self, layer: u32, e: u32) {
        self.note_pick_w(layer, e, 1);
    }

    /// A decode pick with a rank weight (`N_EXPERT_USED - rank`, rank 0-based).
    pub fn note_pick_w(&mut self, layer: u32, e: u32, w: u32) {
        let (l, e) = (layer as usize, e as usize);
        if l < LAYERS && e < NE {
            let c = &mut self.counts[l * NE + e];
            *c = c.saturating_add(w);
        }
    }

    /// The router wanted `(layer, e)`: stamp its recency.
    pub fn note_wanted(&mut self, layer: u32, e: u32) {
        let (l, e) = (layer as usize, e as usize);
        if l < LAYERS && e < NE {
            // +1 so that 0 keeps meaning "never".
            self.last_want[l * NE + e] = self.steps.wrapping_add(1);
        }
    }

    /// A request sent these picks to box 2 (any shape): stamp `last_sent`.
    pub fn note_sent(&mut self, layer: u32, sent: &[i32]) {
        let l = layer as usize;
        if l >= LAYERS {
            return;
        }
        let now = self.steps.wrapping_add(1);
        for &e in sent {
            if (0..N_EXPERT as i32).contains(&e) {
                self.last_sent[l * NE + e as usize] = now;
            }
        }
    }

    /// One decode-shaped request's picks, for the release ranking. `sent`: the
    /// picks sent to box 2 (`[b, nu]`, `NO_PICK` where box 1 computes).
    /// `wants`: the ROUTER's own picks for the same rows (`[b, nu]`, its rank
    /// order, masked to box 2's partition: `wants_for_box2`), when the caller
    /// has them; under a cache prior or a mode-2 substitution they differ from
    /// `sent`. With `by_wants` the counts come from `wants` (else `sent`);
    /// without, from `sent` (the 2026-09-27 ledger). Recency follows `wants`
    /// (else `sent`) in both modes. A `wants` of another shape is a wiring bug:
    /// it asserts in debug builds and is ignored in release.
    pub fn note_decode_picks(&mut self, layer: u32, sent: &[i32], wants: Option<&[i32]>) {
        let nu = crate::config::N_EXPERT_USED;
        debug_assert!(wants.is_none_or(|w| w.len() == sent.len()), "pin wants {:?} vs sent {}", wants.map(<[i32]>::len), sent.len());
        let wants = wants.filter(|w| w.len() == sent.len());
        let router = wants.unwrap_or(sent);
        let ranked = if self.by_wants { router } else { sent };
        for (i, &e) in ranked.iter().enumerate() {
            if (0..N_EXPERT as i32).contains(&e) {
                self.note_pick_w(layer, e as u32, (nu - i % nu) as u32);
            }
        }
        for &e in router {
            if (0..N_EXPERT as i32).contains(&e) {
                self.note_wanted(layer, e as u32);
            }
        }
    }

    /// A background admission of `(layer, e)` was queued: if it is released
    /// before the router wants it again, the read was paid for nothing.
    pub fn note_admitted(&mut self, layer: u32, e: u32) {
        let (l, e) = (layer as usize, e as usize);
        if l < LAYERS && e < NE {
            self.admitted[l * NE + e] = self.steps.wrapping_add(1);
        }
    }

    /// Releases of admitted-but-unwanted experts since the last call.
    pub fn take_released_unused(&mut self) -> u32 {
        std::mem::take(&mut self.released_unused)
    }

    /// One decode step: decay the pick counts, and if box 2 is within
    /// `headroom` of its budget, RELEASE the coldest held experts (at most
    /// `max`) down to `budget - 2 * headroom`: each is not held from now on,
    /// gets the next release index, and its word is queued. Returns the words.
    /// Count ties go to the longest-unwanted with `by_wants`, else to the
    /// lowest `(layer, e)`.
    pub fn step(&mut self, headroom: u32, decay_steps: u32, max: usize) -> Vec<u32> {
        self.step_ranked(headroom, decay_steps, max, |_, _| false)
    }

    /// `step`, where (with `by_wants` only) a held expert `stale(layer, e)`
    /// -- box 2 no longer owns it -- ranks below every other one.
    pub fn step_ranked(&mut self, headroom: u32, decay_steps: u32, max: usize, stale: impl Fn(u32, u32) -> bool) -> Vec<u32> {
        self.steps = self.steps.wrapping_add(1);
        if decay_steps > 0 && self.steps % decay_steps == 0 {
            for c in self.counts.iter_mut() {
                *c /= 2;
            }
        }
        let Some((est, budget)) = self.est_pinned() else { return Vec::new() };
        if est <= budget.saturating_sub(headroom) {
            return Vec::new();
        }
        let want = (est - budget.saturating_sub(2 * headroom)) as usize;
        // `(tier, count, recency, key)`; tier and recency are constant without
        // `by_wants`, so the order is exactly the old `(count, key)` one.
        let mut cand: Vec<(u32, u32, u32, u32)> = Vec::new();
        for (wi, &w) in self.held.iter().enumerate() {
            let mut bits = w;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let (l, e) = (wi / WORDS, (wi % WORDS) * 64 + b);
                let k = l * NE + e;
                if self.by_wants {
                    let tier = u32::from(!stale(l as u32, e as u32));
                    cand.push((tier, self.counts[k], self.last_want[k], k as u32));
                } else {
                    cand.push((1, self.counts[k], 0, k as u32));
                }
            }
        }
        let n = want.min(max).min(cand.len());
        if n == 0 {
            return Vec::new();
        }
        if n < cand.len() {
            cand.select_nth_unstable(n - 1);
        }
        let mut out = Vec::with_capacity(n);
        for &(_, _, _, k) in &cand[..n] {
            let (l, e) = (k as usize / NE, k as usize % NE);
            let adm = std::mem::take(&mut self.admitted[k as usize]);
            if adm != 0 && self.last_sent[k as usize] < adm {
                self.released_unused += 1;
            }
            self.held[l * WORDS + e / 64] &= !(1u64 << (e % 64));
            // Skips 0 (= never released): after 2^32 words the indices run one
            // ahead of box 2's epoch, which only masks a release one word
            // longer -- the safe direction.
            self.rel_sent = self.rel_sent.wrapping_add(1);
            if self.rel_sent == 0 {
                self.rel_sent = 1;
            }
            self.rel_idx[k as usize] = self.rel_sent;
            let w = ((l as u32) << 16) | e as u32;
            self.queue.push_back(w);
            out.push(w);
        }
        out
    }

    /// Up to `max` queued release words, in queue (= index) order.
    pub fn take_words(&mut self, max: usize) -> Vec<u32> {
        let n = self.queue.len().min(max);
        self.queue.drain(..n).collect()
    }

    pub fn queued(&self) -> usize {
        self.queue.len()
    }
}

/// `held & paged`, counted: the hub's surprises for one reply.
pub fn surprise_count(held: &[u32; RESID_WORDS], paged: &[u32; RESID_WORDS]) -> u32 {
    held.iter().zip(paged).map(|(h, p)| (h & p).count_ones()).sum()
}

/// A (re)connect: the new connection has no pins and no maps yet. Resets the
/// mirror (held, pending, seen) and the pin ledger; asks for pins again if the
/// knob is on.
pub fn on_connect() {
    let mut g = LEDGER.lock().unwrap_or_else(|p| p.into_inner());
    *g = PinLedger::new();
    for l in 0..LAYERS {
        SEEN[l].store(false, Ordering::Release);
        for i in 0..WORDS {
            BITS[l][i].store(0, Ordering::Relaxed);
            PENDING[l][i].store(0, Ordering::Relaxed);
        }
    }
    PIN_MODE.store(if pin_wanted() { 1 } else { 0 }, Ordering::Relaxed);
}

/// `REQ_FLAG_PIN` while pinning is asked for or on.
pub fn pin_request_flag() -> u32 {
    match PIN_MODE.load(Ordering::Relaxed) {
        1 | 2 => REQ_FLAG_PIN,
        _ => 0,
    }
}

/// Box 2 answered a pin request with a pin block: release words may be sent.
pub fn pin_active() -> bool {
    PIN_MODE.load(Ordering::Relaxed) == 2
}

/// A reply to a pin request did (`supported`) or did not carry a pin block.
pub fn pin_reply_seen(supported: bool) {
    let to = if supported { 2 } else { 3 };
    if PIN_MODE.compare_exchange(1, to, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
        if supported {
            eprintln!("b2 mirror: box 2 pins what it reports: the mirror's held set is exact");
        } else {
            eprintln!("b2 mirror: WARNING box 2 does not support pinning (older expertd): V41_B2_PIN falls back to the plain mirror");
        }
    }
}

/// A pin-mode reply: overwrite `layer`'s row with the pinned map minus the
/// releases box 2 had not applied (`epoch`), drop its pending row.
pub fn update_pinned(layer: u32, words: &[u32], epoch: u32, pinned: u32, budget: u32) {
    let l = layer as usize;
    if l >= LAYERS {
        return;
    }
    // BITS are written under the ledger lock, so a map and a release can
    // never interleave between the ledger and the mirror.
    let mut g = LEDGER.lock().unwrap_or_else(|p| p.into_inner());
    g.note_reply(epoch, pinned, budget);
    let row = g.apply_map(layer, words, epoch);
    for (i, slot) in BITS[l].iter().enumerate() {
        slot.store(row[i], Ordering::Relaxed);
        PENDING[l][i].store(0, Ordering::Relaxed);
    }
    SEEN[l].store(true, Ordering::Release);
}

/// At submit (pin mode): which of the sent picks `sel` the mirror HOLDS, as
/// `RESID_WORDS` bits, and how many distinct ones. `b <= 16` requests also
/// count picks for release ranking: the router's own `wants` (same `[b, nu]`
/// shape, its rank order) when the caller has them, else `sel`
/// (`PinLedger::note_decode_picks`).
pub fn pin_note_submit(layer: u32, sel: &[i32], wants: Option<&[i32]>, decode_shaped: bool) -> ([u32; RESID_WORDS], u32) {
    let mut held = [0u32; RESID_WORDS];
    let l = layer as usize;
    if l >= LAYERS {
        return (held, 0);
    }
    for &e in sel {
        if (0..N_EXPERT as i32).contains(&e) && (BITS[l][e as usize / 64].load(Ordering::Relaxed) >> (e % 64)) & 1 == 1 {
            held[e as usize / 32] |= 1 << (e % 32);
        }
    }
    if let Ok(mut g) = LEDGER.lock() {
        g.note_sent(layer, sel);
        if decode_shaped {
            g.note_decode_picks(layer, sel, wants);
        }
    }
    let n: u32 = held.iter().map(|w| w.count_ones()).sum();
    N_HELD_PICKS.fetch_add(u64::from(n), Ordering::Relaxed);
    TOT_HELD_PICKS.fetch_add(u64::from(n), Ordering::Relaxed);
    (held, n)
}

/// Up to `max` queued release words for the next request (pin mode on).
pub fn take_release_words(max: usize) -> Vec<u32> {
    if !pin_active() {
        return Vec::new();
    }
    match LEDGER.lock() {
        Ok(mut g) if g.queued() > 0 => g.take_words(max),
        _ => Vec::new(),
    }
}

/// `begin_step`'s pin maintenance: decay and plan releases, clearing the
/// released experts from the mirror at once.
fn pin_begin_step() {
    if !pin_active() {
        return;
    }
    let mut g = LEDGER.lock().unwrap_or_else(|p| p.into_inner());
    let partition = super::expert_pager::t2_partition();
    let words = g.step_ranked(pin_headroom(), pin_decay_steps(), PIN_RELEASE_MAX_PER_STEP, |l, e| {
        partition && !super::expert_pager::partition_box2(l as i32, e)
    });
    for &w in &words {
        let (l, e) = ((w >> 16) as usize, (w & 0xFFFF) as usize);
        BITS[l][e / 64].fetch_and(!(1u64 << (e % 64)), Ordering::Relaxed);
    }
    let unused = g.take_released_unused();
    drop(g);
    N_RELEASED.fetch_add(words.len() as u64, Ordering::Relaxed);
    TOT_RELEASED.fetch_add(words.len() as u64, Ordering::Relaxed);
    N_RELEASED_UNUSED.fetch_add(u64::from(unused), Ordering::Relaxed);
}

/// The end-to-end check on a pin-mode reply: sent picks the mirror held at
/// submit (`held`) that box 2 had to page or wait for anyway (`paged`).
/// Must be 0. Counted; ERROR-logged (the first 20, then at most every 10 s);
/// a panic under `V41_B2_ASSERT_NO_SURPRISE=1`. Returns the count.
pub fn check_surprises(layer: u32, seq: u32, held: &[u32; RESID_WORDS], paged: &[u32; RESID_WORDS]) -> u32 {
    let n = surprise_count(held, paged);
    if n == 0 {
        return 0;
    }
    N_SURPRISES.fetch_add(u64::from(n), Ordering::Relaxed);
    let tot = TOT_SURPRISES.fetch_add(u64::from(n), Ordering::Relaxed) + u64::from(n);
    let ids: Vec<usize> = (0..NE).filter(|&e| (held[e / 32] & paged[e / 32]) >> (e % 32) & 1 == 1).collect();
    static LAST_LOG: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
    let log = tot <= 20
        || LAST_LOG.lock().map(|g| g.is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(10))).unwrap_or(false);
    if log {
        if let Ok(mut g) = LAST_LOG.lock() {
            *g = Some(std::time::Instant::now());
        }
        tracing::error!(
            layer, seq, n, total = tot, experts = ?ids,
            "b2 pin SURPRISE: box 2 paged experts the hub mirror held (the pin invariant is broken)"
        );
    }
    if assert_no_surprise() {
        panic!("b2 pin SURPRISE (V41_B2_ASSERT_NO_SURPRISE): L{layer} seq {seq}: held experts {ids:?} were paged by box 2");
    }
    n
}

/// `(surprises, held picks sent, release words queued, box 2's pinned count
/// net of releases in flight, its budget, releases of background-admitted
/// experts no request sent to box 2 after the admission was queued)` since the
/// last call; `None` unless pinning is on. The last approximates box-2 reads
/// paid for nothing (an admission box 2 dropped counts too, if it was ever
/// pinned); `sub_admits_queued` is only a rough denominator (it counts each
/// lane's duplicates, and admissions while pinning was not yet active).
pub fn take_pin_stats() -> Option<[f64; 6]> {
    if !pin_wanted() {
        return None;
    }
    let (est, budget) = LEDGER
        .lock()
        .ok()
        .and_then(|g| g.est_pinned())
        .map_or((f64::NAN, f64::NAN), |(p, b)| (f64::from(p), f64::from(b)));
    Some([
        N_SURPRISES.swap(0, Ordering::Relaxed) as f64,
        N_HELD_PICKS.swap(0, Ordering::Relaxed) as f64,
        N_RELEASED.swap(0, Ordering::Relaxed) as f64,
        est,
        budget,
        N_RELEASED_UNUSED.swap(0, Ordering::Relaxed) as f64,
    ])
}

/// Cumulative `(surprises, held picks sent, release words queued)` (tests).
pub fn pin_totals() -> (u64, u64, u64) {
    (
        TOT_SURPRISES.load(Ordering::Relaxed),
        TOT_HELD_PICKS.load(Ordering::Relaxed),
        TOT_RELEASED.load(Ordering::Relaxed),
    )
}

/// Is `(layer, e)` HELD by the mirror (pin mode: pinned on box 2)?
pub fn held(layer: u32, e: u32) -> bool {
    let l = layer as usize;
    l < LAYERS && e < N_EXPERT && (BITS[l][(e / 64) as usize].load(Ordering::Relaxed) >> (e % 64)) & 1 == 1
}

// ---- per-step counters (drained by the multistream profile) ----

static N_PREDICTED: AtomicU64 = AtomicU64::new(0);
static N_AVOIDED: AtomicU64 = AtomicU64::new(0);
static N_SLOTS: AtomicU64 = AtomicU64::new(0);
static N_BLOCKED: AtomicU64 = AtomicU64::new(0);
static N_FAILED: AtomicU64 = AtomicU64::new(0);
static N_ADMITS: AtomicU64 = AtomicU64::new(0);
static N_INCOMING: AtomicU64 = AtomicU64::new(0);

/// The router's picks `router` (`[b, nu]`) with every expert box 2 does not
/// own replaced by `NO_PICK`, positions kept (the ledger's rank weights come
/// from them): the pin ledger's `wants`. Box 1's own experts must not earn
/// box-2 pin credit.
pub fn wants_for_box2(router: &[i32], is_box2: impl Fn(u32) -> bool) -> Vec<i32> {
    router
        .iter()
        .map(|&e| if (0..N_EXPERT as i32).contains(&e) && is_box2(e as u32) { e } else { super::remote_experts::NO_PICK })
        .collect()
}

/// Background admissions queued (`V41_SUB_ADMIT`, words `layer << 16 | e`),
/// for the profile and, in pin mode, the ledger's admitted-unused marks.
pub fn note_admits(words: &[u32]) {
    N_ADMITS.fetch_add(words.len() as u64, Ordering::Relaxed);
    if pin_active() && !words.is_empty() {
        let mut g = LEDGER.lock().unwrap_or_else(|p| p.into_inner());
        for &w in words {
            g.note_admitted(w >> 16, w & 0xFFFF);
        }
    }
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

    /// The mirror's rows, marks and the pin ledger are process-wide statics:
    /// tests that touch them run one at a time.
    static STATICS: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
        let _g = STATICS.lock().unwrap_or_else(|p| p.into_inner());
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
        let _g = STATICS.lock().unwrap_or_else(|p| p.into_inner());
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

    fn map(ids: &[u32]) -> Vec<u32> {
        let mut w = vec![0u32; RESID_WORDS];
        for &e in ids {
            w[e as usize / 32] |= 1 << (e % 32);
        }
        w
    }

    /// PinLedger: releases pick the coldest held experts, clear them at once
    /// and get wire indices; a map built before a release (older epoch)
    /// cannot bring the expert back, one built after it can; the pinned
    /// estimate nets out releases in flight; words leave in index order.
    #[test]
    fn pin_ledger_masks_releases_after_epoch() {
        let mut g = PinLedger::new();
        assert_eq!(g.step(1, 0, 512), Vec::<u32>::new(), "no reply yet: no releases");
        g.apply_map(2, &map(&[3, 5, 7]), 0);
        assert!(g.held(2, 3) && g.held(2, 5) && g.held(2, 7) && !g.held(2, 4));
        g.note_reply(0, 3, 3);
        for _ in 0..5 {
            g.note_pick(2, 5);
        }
        g.note_pick(2, 7);
        // est 3 > budget 3 - headroom 1: release down to 3 - 2 = 1, coldest
        // first (3 has no picks, then 7).
        let words = g.step(1, 0, 512);
        let mut sorted = words.clone();
        sorted.sort();
        assert_eq!(sorted, vec![(2 << 16) | 3, (2 << 16) | 7]);
        assert!(!g.held(2, 3) && g.held(2, 5) && !g.held(2, 7));
        assert_eq!(g.est_pinned(), Some((1, 3)), "3 pinned minus 2 releases in flight");
        assert_eq!(g.step(1, 0, 512), Vec::<u32>::new(), "at target: nothing more");
        // A stale map (epoch 0: box 2 applied neither release) must not
        // resurrect them.
        g.apply_map(2, &map(&[3, 5, 7]), 0);
        assert!(!g.held(2, 3) && g.held(2, 5) && !g.held(2, 7));
        // Box 2 applied the first release (epoch 1), and the expert released
        // FIRST was re-pinned since: held again; the second is still masked.
        let first = words[0] & 0xFFFF;
        let second = words[1] & 0xFFFF;
        g.apply_map(2, &map(&[first, second, 5]), 1);
        assert!(g.held(2, first) && g.held(2, 5) && !g.held(2, second));
        g.note_reply(1, 2, 3);
        assert_eq!(g.est_pinned(), Some((1, 3)), "one release still in flight");
        // An older reply never replaces the newest epoch's counters.
        g.note_reply(0, 3, 3);
        assert_eq!(g.est_pinned(), Some((1, 3)));
        assert_eq!(g.take_words(1), vec![words[0]]);
        assert_eq!(g.take_words(10), vec![words[1]]);
        assert!(g.take_words(10).is_empty());
        // Decay halves the counts.
        let mut d = PinLedger::new();
        d.note_pick(1, 1);
        d.note_pick(1, 1);
        let _ = d.step(0, 1, 0);
        assert_eq!(d.counts[NE + 1], 1);
        // Surprises are `held & paged`.
        let mut h = [0u32; RESID_WORDS];
        let mut p = [0u32; RESID_WORDS];
        h[0] = 0b1110;
        p[0] = 0b0111;
        p[11] = 1;
        assert_eq!(surprise_count(&h, &p), 2);
    }

    /// The hub's REAL pin glue, end to end on the statics (the randomized
    /// protocol test in `remote_experts` drives a bare `PinLedger`): connect
    /// -> negotiation -> a pin reply holds -> `begin_step` releases the
    /// coldest, clearing the mirror's BITS at once -> the words leave in
    /// index order, 128 per request -> a stale map (old epoch) cannot bring a
    /// released expert back, a later one can -> the surprise check counts
    /// `held & paged` and the per-step stats drain -> an older daemon (no pin
    /// block) turns pinning off for the connection.
    #[test]
    fn pin_statics_end_to_end() {
        let _g = STATICS.lock().unwrap_or_else(|p| p.into_inner());
        let l = (LAYERS - 2) as u32;
        set_pin_wanted(true);
        on_connect();
        assert_eq!(pin_request_flag(), REQ_FLAG_PIN, "asked from the first request");
        assert!(!pin_active() && take_release_words(8).is_empty(), "no words before box 2 answers");
        // Box 2 answers with a pin block: 300 experts of the layer pinned,
        // 745 of 1000 overall (above budget - headroom 256: a release is due).
        let all: Vec<u32> = (0..300).collect();
        update_pinned(l, &map(&all), 0, 745, 1000);
        pin_reply_seen(true);
        assert!(pin_active() && pin_request_flag() == REQ_FLAG_PIN);
        let (h, n) = pin_note_submit(l, &[3, 5, 301, -1], None, true);
        assert_eq!(n, 2);
        assert!(h[0] & (1 << 3) != 0 && h[0] & (1 << 5) != 0 && h[9] & (1 << (301 % 32)) == 0);
        assert!(held(l, 7) && !held(l, 301));
        // 3, 5, 7 are hot; everything else in the layer is cold.
        for _ in 0..4 {
            let _ = pin_note_submit(l, &[3, 5, 7], None, true);
        }
        let _ = take_pin_stats();
        // Background admissions of 0 (cold: released below, never wanted
        // again = a read paid for nothing) and 3 (hot: kept).
        note_admits(&[l << 16, (l << 16) | 3]);
        // A step: release the 257 coldest (745 - (1000 - 512)) at once.
        begin_step();
        let mut words = Vec::new();
        for want in [128usize, 128, 1, 0] {
            let w = take_release_words(128);
            assert_eq!(w.len(), want);
            words.extend(w);
        }
        assert_eq!(words.len(), 257);
        let released: Vec<u32> = words.iter().map(|w| w & 0xFFFF).collect();
        assert!(words.iter().all(|w| w >> 16 == l));
        assert!(!released.iter().any(|e| [3, 5, 7].contains(e)), "the hot ones stay");
        for e in 0..300u32 {
            assert_eq!(held(l, e), !released.contains(&e), "BITS cleared at release for e{e}");
            assert_eq!(pin_note_submit(l, &[e as i32], None, false).1, u32::from(!released.contains(&e)));
        }
        let st = take_pin_stats().unwrap();
        assert_eq!((st[2], st[3], st[4]), (257.0, 488.0, 1000.0), "released, est pinned net of in-flight, budget");
        assert!(released.contains(&0) && !released.contains(&3));
        assert_eq!(st[5], 1.0, "one admitted expert released before any later want");
        // Nothing more to release now that the estimate is at target.
        begin_step();
        assert!(take_release_words(128).is_empty());
        // A map built before box 2 applied the releases (epoch 0) must not
        // resurrect them; one built after all 257 (epoch 257) does.
        update_pinned(l, &map(&all), 0, 745, 1000);
        assert!(!held(l, released[0]) && held(l, 3));
        update_pinned(l, &map(&all), 257, 745, 1000);
        assert!(held(l, released[0]) && held(l, released[256]) && held(l, 3));
        // Surprise: held-at-submit 3 paged anyway (with 301, which was not held).
        let (h, _) = pin_note_submit(l, &[3, 301], None, true);
        let mut paged = [0u32; RESID_WORDS];
        paged[0] = 1 << 3;
        paged[301 / 32] = 1 << (301 % 32);
        assert_eq!(check_surprises(l, 42, &h, &paged), 1);
        let st = take_pin_stats().unwrap();
        assert_eq!(st[0], 1.0, "one surprise drained");
        assert_eq!(take_pin_stats().unwrap()[0], 0.0);
        // Reconnect: everything forgotten; an older daemon turns pinning off.
        on_connect();
        assert!(!held(l, 3) && !pin_active() && take_release_words(8).is_empty());
        pin_reply_seen(false);
        assert!(!pin_active() && pin_request_flag() == 0, "fell back to the plain mirror");
        set_pin_wanted(false);
        on_connect();
        assert_eq!(pin_request_flag(), 0);
    }

    /// Rank-weighted ranking: an expert picked once at rank 1 outranks one
    /// picked three times at rank 6, so a release takes the latter first.
    #[test]
    fn pin_release_prefers_rank1_winners() {
        let nu = crate::config::N_EXPERT_USED as u32;
        let mut g = PinLedger::new();
        g.apply_map(4, &map(&[1, 2, 3]), 0);
        g.note_reply(0, 3, 3);
        g.note_pick_w(4, 1, nu); // one rank-1 pick
        for _ in 0..3 {
            g.note_pick_w(4, 2, 1); // three rank-6 picks
        }
        // est 3 > 3 - 1: release down to 3 - 2 = 1 -> the two coldest: 3 (0) and 2 (3).
        let mut w = g.step(1, 0, 512);
        w.sort();
        assert_eq!(w, vec![(4 << 16) | 2, (4 << 16) | 3]);
        assert!(g.held(4, 1) && !g.held(4, 2) && !g.held(4, 3));
        // `pin_note_submit` applies the weights from the row position.
        let _guard = STATICS.lock().unwrap_or_else(|p| p.into_inner());
        let mut l = LEDGER.lock().unwrap_or_else(|p| p.into_inner());
        *l = PinLedger::new();
        drop(l);
        let nu_us = crate::config::N_EXPERT_USED;
        let mut sel = vec![-1i32; nu_us];
        sel[0] = 7;
        sel[nu_us - 1] = 8;
        let _ = pin_note_submit(5, &sel, None, true);
        let l = LEDGER.lock().unwrap_or_else(|p| p.into_inner());
        assert_eq!((l.counts[5 * NE + 7], l.counts[5 * NE + 8]), (nu, 1));
    }

    /// `V41_B2_PIN_WANTS`: the ranking counts the ROUTER's picks at the
    /// router's ranks. A pick only the prior put in a row earns nothing and the
    /// pick it displaced is credited; off, the sent picks count exactly as
    /// before. Without router picks (or with a mis-shaped slice) the sent
    /// picks count either way.
    #[test]
    fn pin_ledger_counts_router_wants() {
        let nu = crate::config::N_EXPERT_USED;
        // One row: the router wanted 10.. in rank order; the prior replaced
        // its rank-3 pick (12) with a held 40, in the same slot.
        let wants: Vec<i32> = (10..10 + nu as i32).collect();
        let mut sent = wants.clone();
        sent[2] = 40;
        let mut on = PinLedger::new();
        on.by_wants = true;
        on.note_decode_picks(3, &sent, Some(&wants));
        assert_eq!(on.counts[3 * NE + 12], (nu - 2) as u32, "the displaced want is credited at its rank");
        assert_eq!(on.counts[3 * NE + 40], 0, "a pick only the prior made earns nothing");
        assert_eq!(on.counts[3 * NE + 10], nu as u32);
        let mut off = PinLedger::new();
        off.by_wants = false;
        off.note_decode_picks(3, &sent, Some(&wants));
        assert_eq!((off.counts[3 * NE + 40], off.counts[3 * NE + 12]), ((nu - 2) as u32, 0), "off: the sent picks");
        let mut none = PinLedger::new();
        none.by_wants = true;
        none.note_decode_picks(3, &sent, None);
        // A mis-shaped `wants` asserts in debug builds and falls back to the
        // sent picks in release.
        let short = if cfg!(debug_assertions) { None } else { Some(&wants[..nu - 1]) };
        none.note_decode_picks(3, &sent, short);
        assert_eq!((none.counts[3 * NE + 40], none.counts[3 * NE + 12]), (2 * (nu - 2) as u32, 0));
        // Recency follows the router's picks in both modes.
        assert!(on.last_want[3 * NE + 12] != 0 && off.last_want[3 * NE + 12] != 0);
        assert_eq!((on.last_want[3 * NE + 40], off.last_want[3 * NE + 40]), (0, 0));
    }

    /// Count ties go to the longest-unwanted with `by_wants` (never-wanted
    /// first), to the lowest `(layer, e)` without.
    #[test]
    fn pin_release_ties_go_to_the_longest_unwanted() {
        for by_wants in [true, false] {
            let mut g = PinLedger::new();
            g.by_wants = by_wants;
            g.apply_map(1, &map(&[4, 5, 6]), 0);
            g.note_reply(0, 3, 3);
            g.steps = 10;
            g.note_wanted(1, 6);
            g.steps = 20;
            g.note_wanted(1, 4);
            // All counts 0; est 3 > 3 - 1: release down to 3 - 2 = 1.
            let mut w = g.step(1, 0, 512);
            w.sort();
            let kept = if by_wants { 4 } else { 6 };
            let gone: Vec<u32> = [4, 5, 6].into_iter().filter(|&e| e != kept).map(|e| (1 << 16) | e).collect();
            assert_eq!(w, gone, "by_wants {by_wants}");
            assert!(g.held(1, kept));
        }
    }

    /// The bug `V41_B2_PIN_WANTS` fixes, on a bare ledger in production order
    /// (admission queued at route time, then the submit notes the picks): the
    /// prior displaces a want for 3 and box 2 admits it; a stale held 9 was
    /// never wanted. Off, 3 has no count, loses the (layer, e) tie and is
    /// released unused. On, its displaced want keeps it, and 9 goes.
    #[test]
    fn pin_admitted_displaced_want_survives() {
        let nu = crate::config::N_EXPERT_USED;
        let mut wants: Vec<i32> = (0..nu as i32).map(|i| 20 + i).collect();
        wants[nu - 1] = 3; // the router's rank-6 pick
        let mut sent = wants.clone();
        sent[nu - 1] = 21 + nu as i32; // the prior's held substitute
        for by_wants in [true, false] {
            let mut g = PinLedger::new();
            g.by_wants = by_wants;
            g.note_admitted(2, 3);
            g.note_sent(2, &sent);
            g.note_decode_picks(2, &sent, Some(&wants));
            // Box 2 lands and pins 3; 9 is stale. Over budget by one.
            g.apply_map(2, &map(&[3, 9]), 0);
            g.note_reply(0, 2, 2);
            let w = g.step(1, 0, 1);
            if by_wants {
                assert_eq!(w, vec![(2 << 16) | 9]);
                assert_eq!(g.take_released_unused(), 0);
            } else {
                assert_eq!(w, vec![(2 << 16) | 3], "the old ledger releases the fresh admission first");
                assert_eq!(g.take_released_unused(), 1);
                assert_eq!(g.take_released_unused(), 0, "drained");
            }
        }
    }

    /// `released_unused` counts a release of an admitted expert that no
    /// request sent to box 2 since the admission was queued. A send in the
    /// same step (the other lane's, before or after it) is a use; a send in an
    /// earlier step is not; a later want the prior displaced again is not.
    #[test]
    fn pin_released_unused_means_never_sent_since_admission() {
        // (when box 2 was sent the expert, relative to the admission's step)
        for (sent_at, unused) in [(None, 1u32), (Some(0i32), 0), (Some(-1), 1), (Some(1), 0)] {
            let mut g = PinLedger::new();
            if sent_at == Some(-1) {
                g.steps = 9;
                g.note_sent(0, &[5]);
            }
            g.steps = 10;
            if sent_at == Some(0) {
                g.note_sent(0, &[5]);
            }
            g.note_admitted(0, 5);
            g.note_wanted(0, 5);
            if sent_at == Some(1) {
                g.steps = 11;
                g.note_sent(0, &[5]);
            }
            g.apply_map(0, &map(&[5]), 0);
            g.note_reply(0, 1, 1);
            assert_eq!(g.step(1, 0, 512), vec![5], "{sent_at:?}");
            assert_eq!(g.take_released_unused(), unused, "{sent_at:?}");
        }
    }

    /// The wants keep their positions (rank weights) with box 1's experts,
    /// invalid ids and out-of-range ids masked out.
    #[test]
    fn wants_for_box2_masks_box1_experts() {
        let w = wants_for_box2(&[7, 300, -1, 12, 384, 250], |e| e >= 200);
        assert_eq!(w, vec![-1, 300, -1, -1, -1, 250]);
    }

    /// With `by_wants`, a held expert box 2 no longer owns is released first,
    /// even when it is the hottest; without, the closure is ignored.
    #[test]
    fn pin_release_takes_experts_box2_no_longer_owns_first() {
        for by_wants in [true, false] {
            let mut g = PinLedger::new();
            g.by_wants = by_wants;
            g.apply_map(3, &map(&[1, 2, 3]), 0);
            g.note_reply(0, 3, 3);
            g.note_pick_w(3, 2, 50);
            // est 3 > 3 - 1: release down to 3 - 2 = 1.
            let mut w = g.step_ranked(1, 0, 512, |l, e| l == 3 && e == 2);
            w.sort();
            let gone: &[u32] = if by_wants { &[1, 2] } else { &[1, 3] };
            assert_eq!(w, gone.iter().map(|&e| (3 << 16) | e).collect::<Vec<_>>(), "by_wants {by_wants}");
        }
    }

    /// Rank weights come from each row's own positions.
    #[test]
    fn pin_ledger_weights_wants_per_row() {
        let nu = crate::config::N_EXPERT_USED;
        let mut wants = vec![-1i32; 2 * nu];
        wants[0] = 40; // row 0, rank 1
        wants[nu + 1] = 41; // row 1, rank 2
        wants[2 * nu - 1] = 40; // row 1, rank 6
        let mut g = PinLedger::new();
        g.by_wants = true;
        g.note_decode_picks(6, &vec![-1i32; 2 * nu], Some(&wants));
        assert_eq!(g.counts[6 * NE + 40], nu as u32 + 1);
        assert_eq!(g.counts[6 * NE + 41], (nu - 1) as u32);
    }

    /// The real glue: `pin_note_submit` counts the wants it is handed and
    /// stamps what it sends.
    #[test]
    fn pin_note_submit_forwards_wants() {
        let _g = STATICS.lock().unwrap_or_else(|p| p.into_inner());
        {
            let mut l = LEDGER.lock().unwrap_or_else(|p| p.into_inner());
            *l = PinLedger::new();
            l.by_wants = true;
        }
        let nu = crate::config::N_EXPERT_USED;
        let mut sent = vec![-1i32; nu];
        sent[0] = 50;
        let mut wants = vec![-1i32; nu];
        wants[0] = 60;
        let _ = pin_note_submit(7, &sent, Some(&wants), true);
        let mut l = LEDGER.lock().unwrap_or_else(|p| p.into_inner());
        assert_eq!((l.counts[7 * NE + 60], l.counts[7 * NE + 50]), (nu as u32, 0));
        assert!(l.last_sent[7 * NE + 50] != 0 && l.last_sent[7 * NE + 60] == 0);
        *l = PinLedger::new();
    }

    /// `V41_B2_PIN_WANTS=0` is the 2026-09-27 ledger exactly: random maps,
    /// replies, submits (with and without wants), admissions and steps drive a
    /// knob-off ledger and a verbatim copy of the old counting and `step`; the
    /// release words (in order) and the old state agree after every operation.
    #[test]
    fn pin_ledger_knob_off_is_the_0927_ledger() {
        fn old_note(g: &mut PinLedger, layer: u32, sel: &[i32]) {
            let nu = crate::config::N_EXPERT_USED;
            for (i, &e) in sel.iter().enumerate() {
                if (0..N_EXPERT as i32).contains(&e) {
                    g.note_pick_w(layer, e as u32, (nu - i % nu) as u32);
                }
            }
        }
        fn old_step(g: &mut PinLedger, headroom: u32, decay_steps: u32, max: usize) -> Vec<u32> {
            g.steps = g.steps.wrapping_add(1);
            if decay_steps > 0 && g.steps % decay_steps == 0 {
                for c in g.counts.iter_mut() {
                    *c /= 2;
                }
            }
            let Some((est, budget)) = g.est_pinned() else { return Vec::new() };
            if est <= budget.saturating_sub(headroom) {
                return Vec::new();
            }
            let want = (est - budget.saturating_sub(2 * headroom)) as usize;
            let mut cand: Vec<(u32, u32)> = Vec::new();
            for (wi, &w) in g.held.iter().enumerate() {
                let mut bits = w;
                while bits != 0 {
                    let b = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let (l, e) = (wi / WORDS, (wi % WORDS) * 64 + b);
                    cand.push((g.counts[l * NE + e], (l * NE + e) as u32));
                }
            }
            let n = want.min(max).min(cand.len());
            if n == 0 {
                return Vec::new();
            }
            if n < cand.len() {
                cand.select_nth_unstable(n - 1);
            }
            let mut out = Vec::with_capacity(n);
            for &(_, k) in &cand[..n] {
                let (l, e) = (k as usize / NE, k as usize % NE);
                g.held[l * WORDS + e / 64] &= !(1u64 << (e % 64));
                g.rel_sent = g.rel_sent.wrapping_add(1);
                if g.rel_sent == 0 {
                    g.rel_sent = 1;
                }
                g.rel_idx[k as usize] = g.rel_sent;
                let w = ((l as u32) << 16) | e as u32;
                g.queue.push_back(w);
                out.push(w);
            }
            out
        }
        struct Rng(u64);
        impl Rng {
            fn next(&mut self, n: u64) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0 % n.max(1)
            }
        }
        let nu = crate::config::N_EXPERT_USED;
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut releases = 0usize;
        for trial in 0..100 {
            let (mut new, mut old) = (PinLedger::new(), PinLedger::new());
            new.by_wants = false;
            let layers = 1 + rng.next(3);
            let mut epoch = 0u32;
            for op in 0..200 {
                match rng.next(10) {
                    0 | 1 => {
                        let l = rng.next(layers) as u32;
                        let ids: Vec<u32> = (0..rng.next(48)).map(|_| rng.next(64) as u32).collect();
                        assert_eq!(new.apply_map(l, &map(&ids), epoch), old.apply_map(l, &map(&ids), epoch));
                    }
                    2 => {
                        epoch = epoch.wrapping_add(rng.next(3) as u32);
                        let (pinned, budget) = (rng.next(120) as u32, 40 + rng.next(80) as u32);
                        new.note_reply(epoch, pinned, budget);
                        old.note_reply(epoch, pinned, budget);
                    }
                    3..=6 => {
                        let l = rng.next(layers) as u32;
                        let n = (1 + rng.next(3) as usize) * nu;
                        let sent: Vec<i32> = (0..n).map(|_| if rng.next(4) == 0 { -1 } else { rng.next(64) as i32 }).collect();
                        let wants: Vec<i32> = (0..n).map(|_| if rng.next(4) == 0 { -1 } else { rng.next(64) as i32 }).collect();
                        new.note_sent(l, &sent);
                        new.note_decode_picks(l, &sent, (rng.next(2) == 0).then_some(&wants[..]));
                        old_note(&mut old, l, &sent);
                    }
                    7 => new.note_admitted(rng.next(layers) as u32, rng.next(64) as u32),
                    _ => {
                        let (h, d, m) = (rng.next(20) as u32, rng.next(4) as u32, 1 + rng.next(64) as usize);
                        let a = new.step_ranked(h, d, m, |_, e| e % 2 == 0);
                        let b = old_step(&mut old, h, d, m);
                        releases += a.len();
                        assert_eq!(a, b, "trial {trial} op {op}");
                    }
                }
                assert!(new.held == old.held && new.counts == old.counts && new.rel_idx == old.rel_idx, "trial {trial} op {op}");
                assert_eq!((new.rel_sent, new.steps, new.last, &new.queue), (old.rel_sent, old.steps, old.last, &old.queue));
            }
        }
        assert!(releases > 1000, "not vacuous: {releases} releases");
    }

    /// Release indices wrap without ever becoming 0 (= never released), and
    /// the epoch comparison is wrapping.
    #[test]
    fn pin_ledger_index_wraps() {
        assert!(after(5, 4) && !after(4, 4) && !after(3, 4));
        assert!(after(2, u32::MAX - 1), "wrapped index is later");
        let mut g = PinLedger::new();
        g.rel_sent = u32::MAX - 1;
        g.apply_map(0, &map(&[1, 2, 3]), 0);
        g.note_reply(u32::MAX - 1, 3, 3);
        let w = g.step(1, 0, 512);
        assert_eq!(w.len(), 2);
        assert!(g.rel_idx.iter().all(|&i| i == 0 || i == u32::MAX || i == 1), "{:?}", g.rel_idx.iter().filter(|&&i| i != 0).collect::<Vec<_>>());
        // A map at the pre-wrap epoch masks both; at epoch 1 neither.
        g.apply_map(0, &map(&[1, 2, 3]), u32::MAX - 1);
        assert_eq!((1..4).filter(|&e| g.held(0, e)).count(), 1);
        g.apply_map(0, &map(&[1, 2, 3]), 1);
        assert_eq!((1..4).filter(|&e| g.held(0, e)).count(), 3);
    }
}
