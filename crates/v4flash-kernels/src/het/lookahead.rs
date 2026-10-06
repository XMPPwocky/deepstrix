//! PREDICTED-MISS LOOK-AHEAD PREFETCH, hub side
//! (docs/v41/B2_PREDICTED_MISS_PREFETCH_DESIGN.md, rev 4).
//!
//! At layer L's Route the dGPU has already run layer L+1's router on layer
//! L's router input (`V41_LOOKAHEAD_PREFETCH`, b9a2122; `forward_prefill`'s
//! `look_next`, read back with the picks). This module is everything the hub
//! does with those picks under `V41_B2_MISS_PREFETCH`:
//! - the FILTER (design 2.2): keep the box-2-owned, mirror-non-resident,
//!   top-ranked predictions, dedup them, cap them (`classify`, `hint_words`);
//! - the step-tagged hint QUEUE (2.3) a decode submit drains first
//!   (`HintQueue`; the static lives beside `remote_experts::PREFETCH_WORDS`);
//! - the per-step SPECULATIVE BUDGET (2.4) over hints + admissions + restores
//!   (`SpecBudget`; `V41_B2_SPEC_BUDGET`, 0 = today's per-request caps);
//! - the dry-run COUNTERS (section 6), per R in 1..=3, and the live Step 0:
//!   how often a hinted expert is in the router's OWN picks one lane-layer
//!   later (`dry_hits`).
//!
//! SLICE A (design 8.A): hub-only. `dry` launches the look-ahead, filters,
//! queues and counts; no word reaches box 2. SLICE B (8.B): under `k1`/`k2`
//! the queued words go on the wire as `REQ_FLAG_LIKELY` words ahead of
//! admissions and restores -- once box 2 has answered `RESP_FLAG_NURSERY`
//! (`Mode::wire`; `b2_mirror::nursery_supported`), under the per-step budget,
//! the surplus re-queued, the absolute abort bars (`Bars`) turning the rest of
//! a step dry. The only behaviour either adds is the look-ahead launches and
//! host bookkeeping (I1/I2, section 7): nothing here touches a pick, a weight
//! or a kernel input, and the mirror's NURSERY bits never feed the prior.
//!
//! The knobs are read ONCE per decode step (`begin_step`) into a plain `Cfg`
//! that the lane-layers copy -- never on the lane path (KNOB_AUDIT, "getenv on
//! the lane path"; and a live knob flipping between a lane-layer's chain and
//! its route would leave the readback pack and its reader disagreeing on the
//! `look` segment). Everything policy-shaped is a pure function over plain
//! inputs, so it is host-testable without a device; `forward_prefill` and
//! `remote_experts` own the statics they feed.

use crate::config::{N_EXPERT, N_LAYER};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

const NE: usize = N_EXPERT as usize;
const LAYERS: usize = N_LAYER as usize;
const WORDS: usize = NE.div_ceil(64);

/// `V41_B2_MISS_PREFETCH` (design 5). `off`: nothing (the legacy
/// `V41_LOOKAHEAD_PREFETCH=1` path, if set, is untouched). `dry`: the
/// look-ahead launches run, the filter and queue work, the counters count;
/// nothing is sent. `k1` / `k2`: hints live, one / two layers of lead --
/// SLICE B (and box 2's nursery, slice C); until then they behave as `dry`
/// and warn once, so the names are stable for the deploy scripts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Off,
    Dry,
    K1,
    K2,
}

impl Mode {
    /// From the knob's option index (`knobs::B2_MISS_PREFETCH.pick()`).
    pub fn from_knob(i: usize) -> Mode {
        match i {
            1 => Mode::Dry,
            2 => Mode::K1,
            3 => Mode::K2,
            _ => Mode::Off,
        }
    }

    /// The look-ahead launches run and the filter counts.
    pub fn on(self) -> bool {
        self != Mode::Off
    }

    /// Layer L+2's router too (design 2.1: `look_next2` iff `k2`, never the
    /// legacy `lookahead_depth`).
    pub fn depth2(self) -> bool {
        self == Mode::K2
    }

    /// `k1` / `k2`: the hub ASKS for the nursery (`REQ_FLAG_LIKELY` on every
    /// decode request, words or none -- the capability probe).
    pub fn asks(self) -> bool {
        matches!(self, Mode::K1 | Mode::K2)
    }

    /// May the queued words go on the wire? `k1` / `k2` AND the daemon has
    /// answered `RESP_FLAG_NURSERY` (design 3.2, "capability = a RESP flag":
    /// an echoed request bit proves nothing; `b2_mirror::nursery_supported`).
    /// Until then (and under `dry`) the words are counted and dropped.
    pub fn wire(self) -> bool {
        self.asks() && super::b2_mirror::nursery_supported()
    }
}

/// One decode step's knob values, read by `begin_step` and copied by every
/// lane-layer (`PreMoeCarry::mp`) and every decode submit. `step` = the
/// mirror's step clock (`b2_mirror::begin_step`), the tag on queued words.
/// Stored as TWO atomics (`CFG`, `CFG2`), written back to back by
/// `begin_step` and read without a lock on the lane path: a reader between
/// the two stores sees the new mode/rank/cap/budget/step with the old
/// rows/bars/prior/margin -- a tear that is harmless (the second word is
/// per-step policy, a lane-layer late is fine) and lasts one store.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cfg {
    pub mode: Mode,
    /// `V41_B2_MISS_PREFETCH_RANK`: predicted rank cut R (1..=3).
    pub rank: u8,
    /// `V41_B2_MISS_PREFETCH_CAP`: hint words per lane-layer and target layer.
    pub cap: u8,
    /// `V41_B2_SPEC_BUDGET`: speculative words per step; 0 = legacy caps.
    pub spec_budget: u16,
    pub step: u32,
    /// The step's decode rows (`begin_step`).
    pub rows: u16,
    /// The per-step abort bar on hint words, RESOLVED: the knob, or its
    /// rows-aware default `10 + 5 * max(0, rows - 4)` (design 5).
    pub max_words_step: u16,
    /// `V41_B2_MISS_PREFETCH_MAX_PER_LL_X10`: the per-lane-layer bar x10.
    pub per_ll_x10: u8,
    /// `V41_B2_NURSERY_PRIOR`: mark sent hints INCOMING when R >= 2.
    pub nursery_prior: bool,
    /// `V41_B2_MISS_PREFETCH_MARGIN` x1000 (0 = no filter).
    pub margin_x1000: u16,
}

impl Cfg {
    fn pack(self) -> (u64, u64) {
        (
            (self.mode as u64) | (u64::from(self.rank) << 4) | (u64::from(self.cap) << 8) | (u64::from(self.spec_budget) << 16) | (u64::from(self.step) << 32),
            u64::from(self.rows)
                | (u64::from(self.max_words_step) << 16)
                | (u64::from(self.per_ll_x10) << 32)
                | (u64::from(self.nursery_prior) << 40)
                | (u64::from(self.margin_x1000) << 41),
        )
    }

    fn unpack(v: u64, w: u64) -> Cfg {
        Cfg {
            mode: Mode::from_knob((v & 0xf) as usize),
            rank: ((v >> 4) & 0xf) as u8,
            cap: ((v >> 8) & 0xff) as u8,
            spec_budget: ((v >> 16) & 0xffff) as u16,
            step: (v >> 32) as u32,
            rows: (w & 0xffff) as u16,
            max_words_step: ((w >> 16) & 0xffff) as u16,
            per_ll_x10: ((w >> 32) & 0xff) as u8,
            nursery_prior: (w >> 40) & 1 == 1,
            margin_x1000: ((w >> 41) & 0xffff) as u16,
        }
    }

    /// The margin threshold (`V41_B2_MISS_PREFETCH_MARGIN`).
    pub fn margin(self) -> f32 {
        f32::from(self.margin_x1000) / 1000.0
    }
}

/// The rows-aware default of the per-step abort bar (design 5): 10 at `<= 4`
/// rows, +5 per row beyond.
pub fn default_max_words_step(rows: usize) -> u16 {
    (10 + 5 * rows.saturating_sub(4)).min(u16::MAX as usize) as u16
}

/// The current step's `Cfg`, packed (two relaxed loads per read; the two
/// words are written together under `begin_step`, read on the lane path where
/// a torn read between them is harmless -- the second word is per-step
/// policy a lane-layer late is fine with).
static CFG: AtomicU64 = AtomicU64::new(0);
static CFG2: AtomicU64 = AtomicU64::new(0);

/// The step's knob snapshot (`begin_step`); the defaults before any step.
pub fn cfg() -> Cfg {
    Cfg::unpack(CFG.load(Relaxed), CFG2.load(Relaxed))
}

/// Anything to count this step (the `hub_lh2` record is emitted iff so).
pub fn active() -> bool {
    let c = cfg();
    c.mode.on() || c.spec_budget > 0
}

/// A decode step begins (right after `b2_mirror::begin_step`, which advanced
/// `step`) with `rows` decode rows: read the knobs through the framework into
/// this step's `Cfg`, drop the hint queue's earlier-step words, clear its
/// dedup / sent sets, reset the speculative budget and the abort bars.
pub fn begin_step(step: u32, rows: usize) {
    let k = &crate::knobs::B2_MISS_PREFETCH;
    let bar = crate::knobs::B2_MISS_PREFETCH_MAX_WORDS_STEP.get();
    let c = Cfg {
        mode: Mode::from_knob(k.pick()),
        rank: crate::knobs::B2_MISS_PREFETCH_RANK.get() as u8,
        cap: crate::knobs::B2_MISS_PREFETCH_CAP.get() as u8,
        spec_budget: crate::knobs::B2_SPEC_BUDGET.get() as u16,
        step,
        rows: rows.min(u16::MAX as usize) as u16,
        max_words_step: if bar == 0 { default_max_words_step(rows) } else { bar.min(u64::from(u16::MAX)) as u16 },
        per_ll_x10: crate::knobs::B2_MISS_PREFETCH_MAX_PER_LL_X10.get() as u8,
        nursery_prior: crate::knobs::B2_NURSERY_PRIOR.on(),
        margin_x1000: (crate::knobs::B2_MISS_PREFETCH_MARGIN.f64() * 1000.0).round().clamp(0.0, 1000.0) as u16,
    };
    let (v, w) = c.pack();
    CFG.store(v, Relaxed);
    CFG2.store(w, Relaxed);
    let stale = super::remote_experts::hint_queue_begin_step(step);
    bump(Stat::Stale, u64::from(stale));
    BUDGET.lock().unwrap_or_else(|p| p.into_inner()).reset(u32::from(c.spec_budget));
    BARS.lock().unwrap_or_else(|p| p.into_inner()).reset();
}

// ---- the absolute abort bars (design 5) ----

/// Per step: hint words sent, lane-layers routed, and whether a bar tripped
/// (the rest of the step is dry). A bound must not read a runtime estimate:
/// both bars are knobs or fixed defaults (`Cfg::max_words_step`,
/// `Cfg::per_ll_x10`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bars {
    pub sent: u32,
    pub lane_layers: u32,
    pub tripped: bool,
}

/// Lane-layers a step must have routed before the per-lane-layer bar is
/// judged (the first lane-layer alone may send the cap).
pub const PER_LL_MIN_LANE_LAYERS: u32 = 8;

impl Bars {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// A lane-layer routed (one `classify` of layer L+1).
    pub fn note_lane_layer(&mut self) {
        self.lane_layers += 1;
    }

    /// `n` hint words are about to go: how many the bars allow. Tripped (the
    /// step total would pass `max_words_step`, or the per-lane-layer rate
    /// `per_ll_x10 / 10` once `PER_LL_MIN_LANE_LAYERS` have routed), nothing
    /// goes for the rest of the step; returns `(allowed, tripped now)`.
    pub fn allow(&mut self, n: usize, max_words_step: u16, per_ll_x10: u8) -> (usize, bool) {
        if self.tripped || n == 0 {
            return (0, false);
        }
        let total = self.sent + n as u32;
        let over_step = total > u32::from(max_words_step);
        let over_ll = self.lane_layers >= PER_LL_MIN_LANE_LAYERS && total * 10 > u32::from(per_ll_x10) * self.lane_layers;
        if over_step || over_ll {
            self.tripped = true;
            return (0, true);
        }
        self.sent = total;
        (n, false)
    }
}

static BARS: std::sync::Mutex<Bars> = std::sync::Mutex::new(Bars { sent: 0, lane_layers: 0, tripped: false });

/// A lane-layer routed this step (`Bars::note_lane_layer`).
pub fn note_lane_layer() {
    BARS.lock().unwrap_or_else(|p| p.into_inner()).note_lane_layer();
}

/// Has this step tripped a bar? (The submit then takes nothing more.)
pub fn bars_tripped() -> bool {
    BARS.lock().unwrap_or_else(|p| p.into_inner()).tripped
}

/// `Bars::allow` on this step's bars with this step's `Cfg`, called with the
/// words that would GO (after the budget plan); a trip bumps `lh2_bar_trips`.
pub fn bars_allow(n: usize) -> usize {
    let c = cfg();
    let (ok, tripped) = BARS.lock().unwrap_or_else(|p| p.into_inner()).allow(n, c.max_words_step, c.per_ll_x10);
    bump(Stat::BarTrips, u64::from(tripped));
    ok
}

/// Does the look-ahead router of layer L+1 run (`look_next`), and L+2's
/// (`look_next2`)? Design 2.1: `look_next` iff the legacy env var or the knob;
/// `look_next2` iff `k2`, or -- with the knob off -- the legacy var at depth 2.
pub fn look_gates(mode: Mode, legacy_env: bool, legacy_depth: usize) -> (bool, bool) {
    let next = legacy_env || mode.on();
    let next2 = if mode.on() { mode.depth2() } else { legacy_env && legacy_depth >= 2 };
    (next, next2)
}

/// Does the OLD look-ahead word block run (`forward_prefill`, top
/// `V41_LOOKAHEAD_TOPK` ranks of every predicted pick, box-2-owned, no
/// residency test, `push_prefetch_words`; measured a loss 2026-09-21)? Only
/// under the legacy env var with this knob OFF and a driver that allows
/// hints (design 2.1). The knob's own path never calls `push_prefetch_words`.
pub fn legacy_words(mode: Mode, legacy_env: bool, hints_ok: bool) -> bool {
    mode == Mode::Off && legacy_env && hints_ok
}

// ---- the filter (design 2.2) ----

/// One distinct predicted pick of a look-ahead router, box 2's share.
#[derive(Clone, Copy, Debug)]
pub struct Pred {
    pub e: u16,
    /// Best rank over the rows, 1 = the router's first pick.
    pub rank: u8,
    /// Mirror-non-resident at hint time: the `n_pred_miss` predicate
    /// (`!held && !pending && !incoming`, `b2_mirror::lookup`), not
    /// `resident()`, which drops PENDING under live `V41_SUB_PENDING=0`
    /// (review round 1, finding 5). `None` from `lookup` (box 2 has not
    /// reported the layer) counts as resident: nothing is known to be missing.
    pub nonres: bool,
    /// Slice A amendment (10-06): the normalized gate MARGIN of a predicted
    /// rank-1 pick over the row's predicted rank 2, `(w1 - w2) / sum(row)`
    /// from the look-ahead's weights (`classify_w`), the largest over the rows
    /// where it is rank 1. NaN = unknown (no weights packed, or not rank 1):
    /// passes any threshold, counts in no margin bucket. A WEIGHT margin,
    /// while the rank comes from the router's selection on score + bias: an
    /// inversion (rank 1 by bias, lower weight) clamps to 0 and lands in
    /// bucket m0 only -- an empirical proxy, measured against `hits_prot` by
    /// the dry run; if m0 >> m1 at flat precision, the right quantity is the
    /// biased selection score (a `look_score` segment, no pack room under k2).
    pub margin: f32,
}

impl PartialEq for Pred {
    fn eq(&self, o: &Self) -> bool {
        self.e == o.e && self.rank == o.rank && self.nonres == o.nonres && self.margin.to_bits() == o.margin.to_bits()
    }
}

impl Eq for Pred {}

impl Pred {
    /// Does the margin filter pass this prediction? Only rank 1 is judged
    /// (`V41_B2_MISS_PREFETCH_MARGIN`); an unknown margin passes.
    pub fn margin_ok(&self, thr: f32) -> bool {
        self.rank != 1 || thr <= 0.0 || !(self.margin < thr)
    }
}

/// Ranks the filter and the counters care about (R `<= 3`, design 5): a
/// prediction of a lower rank is neither counted nor hinted.
pub const MAX_RANK: u8 = 3;

/// The dry run's margin buckets (`lh2_nonres_m{k}`, `lh2_hits_prot_m{k}`):
/// a rank-1 prediction with margin `>= MARGIN_BUCKETS[k]` counts in bucket k.
pub const MARGIN_BUCKETS: [f32; 4] = [0.0, 0.1, 0.2, 0.3];

/// Per-lane host state of the filter (`BatchDgpuScratch::lh2`), reused across
/// the lane's layers: the Route path allocates nothing after the first use
/// (review: 5-20 us against the 0.03 ms turnaround slack). `best` / `seen` are
/// cleared by walking the same slice that dirtied them, never the whole
/// 384-entry array.
#[derive(Debug, Default)]
pub struct LaneState {
    /// Per expert during `classify`: 0 unseen, 1..=3 best rank, 255 box 1's.
    best: Vec<u8>,
    /// Per expert during `classify_w`: the best margin as a rank-1 pick.
    marg: Vec<f32>,
    /// Per expert during `dry_hits`: best ACTUAL rank in the own picks (0 = absent).
    seen: Vec<u8>,
    /// The last `classify`.
    pub preds: Vec<Pred>,
    /// What this lane predicted for `pending` `(layer, step)`: the non-resident
    /// predictions only (`set_pending`); `None` = nothing pending.
    pub pending: Option<(i32, u32)>,
    pub pending_preds: Vec<Pred>,
    /// The router's `n_used` (the own picks' row width, for the actual rank).
    pub pending_n_used: usize,
    /// The last `dry_hits_demanded`'s PROTECTED hits (own pick at actual rank
    /// `<= protect`), scored again at the layer's reply (`reply_hits`).
    pub prot_hits: Vec<ProtHit>,
    /// The layer's protected box-2 picks the mirror calls non-resident
    /// (`protected_nonres`), hinted or not: the refined objective's denominator.
    pub prot_nonres: Vec<u16>,
}

/// A protected dry hit: the predicted rank and margin of a prediction whose
/// own pick at the target layer sits at a protected actual rank.
#[derive(Clone, Copy, Debug)]
pub struct ProtHit {
    pub e: u16,
    pub rank: u8,
    pub margin: f32,
}

impl PartialEq for ProtHit {
    fn eq(&self, o: &Self) -> bool {
        self.e == o.e && self.rank == o.rank && self.margin.to_bits() == o.margin.to_bits()
    }
}

/// The refined objective (owner, 10-06: "correctly predict protected hits we
/// would otherwise have to BLOCK on paging"), scored at the lane-layer's reply:
/// of the protected dry hits, the ones whose expert the reply PAGED (`paged`,
/// per R and per margin bucket), and of those the ones whose reply the lane
/// STALLED on (`late`); `total` = the request's protected non-resident picks
/// (hinted or not) that were paged AND late -- the reads the step blocked on,
/// the recall denominator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplyHits {
    pub paged: [u32; 3],
    pub paged_m: [u32; 4],
    pub late: [u32; 3],
    pub late_m: [u32; 4],
    pub total: u32,
}

/// `ReplyHits` of `prot_hits` / `prot_nonres` against a reply's PAGED bits
/// (`[u32; RESID_WORDS]`, bit e) and the lane's stall on it.
pub fn reply_hits(prot_hits: &[ProtHit], prot_nonres: &[u16], paged: &[u32], late: bool) -> ReplyHits {
    let bit = |e: u16| paged.get(usize::from(e) / 32).is_some_and(|w| (w >> (e % 32)) & 1 == 1);
    let mut r = ReplyHits::default();
    for h in prot_hits.iter().filter(|h| h.rank >= 1 && h.rank <= MAX_RANK && bit(h.e)) {
        for k in h.rank as usize..=3 {
            r.paged[k - 1] += 1;
            r.late[k - 1] += u32::from(late);
        }
        if h.rank == 1 {
            for (k, &b) in MARGIN_BUCKETS.iter().enumerate() {
                let in_bucket = u32::from(h.margin >= b);
                r.paged_m[k] += in_bucket;
                r.late_m[k] += in_bucket * u32::from(late);
            }
        }
    }
    if late {
        r.total = prot_nonres.iter().filter(|&&e| bit(e)).count() as u32;
    }
    r
}

/// The request's PROTECTED box-2 picks (`sel` as sent to box 2, `[rows][n_used]`
/// in rank order, `NO_PICK` for box 1's; actual rank `<= protect`) that the
/// mirror calls non-resident (`nonres`), distinct, into `out`.
pub fn protected_nonres(sel: &[i32], n_used: usize, protect: u32, nonres: impl Fn(u32) -> bool, out: &mut Vec<u16>) {
    out.clear();
    if n_used == 0 {
        return;
    }
    for (i, &sv) in sel.iter().enumerate() {
        let rank = (i % n_used) as u32 + 1;
        if rank > protect || !(0..N_EXPERT as i32).contains(&sv) {
            continue;
        }
        let e = sv as u16;
        if !out.contains(&e) && nonres(sv as u32) {
            out.push(e);
        }
    }
}

impl LaneState {
    fn ensure(&mut self) {
        if self.best.len() != NE {
            self.best = vec![0; NE];
            self.marg = vec![f32::NAN; NE];
            self.seen = vec![0; NE];
        }
    }

    /// Layer `target`'s predicted picks into `self.preds`: DISTINCT by expert,
    /// box 2's only (`is_box2`), of rank `<= MAX_RANK`, best rank first, then
    /// first appearance (a 3-bucket scan, no sort). `look` is `[rows][n_used]`
    /// in the router's descending selection order (rank r is column r-1), as
    /// `forward_prefill` unpacks `sd.look_sel`; ids outside `0..N_EXPERT`
    /// (NO_PICK, padding) are skipped. Margins unknown (`classify_w`).
    pub fn classify(&mut self, look: &[i32], n_used: usize, is_box2: impl Fn(u32) -> bool, nonres: impl Fn(u32) -> bool) {
        self.classify_w(look, &[], n_used, is_box2, nonres)
    }

    /// `classify` with the look-ahead's gate weights `look_w` (`[rows][n_used]`
    /// like `look`, rank order, normalized to the row's top-k sum; empty = no
    /// weights): each rank-1 prediction's `margin` is the largest `(w1 - w2) /
    /// sum(row)` over the rows where it is rank 1 (slice A amendment).
    pub fn classify_w(&mut self, look: &[i32], look_w: &[f32], n_used: usize, is_box2: impl Fn(u32) -> bool, nonres: impl Fn(u32) -> bool) {
        self.ensure();
        self.preds.clear();
        if n_used == 0 {
            return;
        }
        let valid = |sv: i32| (0..N_EXPERT as i32).contains(&sv);
        let weights = look_w.len() == look.len() && n_used >= 2;
        for (i, &sv) in look.iter().enumerate() {
            if !valid(sv) {
                continue;
            }
            let (e, rank) = (sv as usize, (i % n_used) + 1);
            if self.best[e] == 0 {
                // Box 1's are marked, so the ownership test runs once per expert.
                self.best[e] = if is_box2(e as u32) { rank.min(255) as u8 } else { 255 };
            } else if rank < self.best[e] as usize {
                self.best[e] = rank as u8;
            }
            if weights && rank == 1 {
                let row = &look_w[i..i + n_used];
                let sum: f32 = row.iter().filter(|w| w.is_finite()).sum();
                let m = if sum > 0.0 { ((row[0] - row[1]) / sum).clamp(0.0, 1.0) } else { 0.0 };
                if self.marg[e].is_nan() || m > self.marg[e] {
                    self.marg[e] = m;
                }
            }
        }
        // Bucket by rank; within a bucket, first appearance. Emitting an expert
        // flips its mark to 255, so it comes out once and the slice walk below
        // still finds every dirtied entry.
        for rank in 1..=MAX_RANK {
            for &sv in look {
                if valid(sv) && self.best[sv as usize] == rank {
                    self.best[sv as usize] = 255;
                    let margin = if rank == 1 { self.marg[sv as usize] } else { f32::NAN };
                    self.preds.push(Pred { e: sv as u16, rank, nonres: nonres(sv as u32), margin });
                }
            }
        }
        for &sv in look {
            if valid(sv) {
                self.best[sv as usize] = 0;
                self.marg[sv as usize] = f32::NAN;
            }
        }
    }

    /// Keep the last `classify`'s non-resident predictions as this lane's
    /// prediction for `(layer, step)`.
    pub fn set_pending(&mut self, layer: i32, step: u32) {
        self.pending_preds.clear();
        self.pending_preds.extend(self.preds.iter().filter(|p| p.nonres));
        self.pending = Some((layer, step));
        if self.pending_n_used == 0 {
            self.pending_n_used = crate::config::N_EXPERT_USED;
        }
    }

    /// Layer `layer` of step `step` has routed on this lane: is the pending
    /// prediction its (same lane, next layer, same step)? A mismatch (a new
    /// step at layer 0, a hash-router layer with no look-ahead in between) is
    /// dropped without a count, on purpose. On a match `pending` stays set
    /// for `dry_hits_demanded`'s target layer; the next `set_pending` or
    /// `take_pending` replaces it.
    pub fn take_pending(&mut self, layer: i32, step: u32) -> bool {
        if self.pending == Some((layer, step)) {
            true
        } else {
            self.pending = None;
            false
        }
    }

    /// `dry_hits` of the pending prediction against the own picks `own`.
    pub fn dry_hits(&mut self, own: &[i32]) -> [u32; 3] {
        self.dry_hits_demanded(own, |_| false, 0).hits
    }

    /// The dry match of the pending prediction against the own picks `own`
    /// (`[rows][n_used]`, rank order: column r-1 is actual rank r): per R the
    /// non-resident predictions of rank `<= R` in the own picks at ANY rank
    /// (`hits`) and at ACTUAL rank `<= protect` (`hits_prot`: `V41_SUB_PROTECT`,
    /// the picks the prior cannot swap away -- what a hint can hide; slice A
    /// amendment); the protected hits among the rank-1 predictions per margin
    /// bucket (`hits_prot_m`); and (slice B) the ones SENT as hints this step
    /// (`sent(word)`: `demanded`, `demanded_prot` = at a protected actual rank).
    pub fn dry_hits_demanded(&mut self, own: &[i32], sent: impl Fn(u32) -> bool, protect: u32) -> DryHits {
        self.ensure();
        let valid = |sv: i32| (0..N_EXPERT as i32).contains(&sv);
        let n_used = self.pending_n_used.max(1);
        for (i, &sv) in own.iter().enumerate() {
            if valid(sv) {
                let r = ((i % n_used) + 1).min(255) as u8;
                let s = &mut self.seen[sv as usize];
                if *s == 0 || r < *s {
                    *s = r;
                }
            }
        }
        let mut d = DryHits::default();
        let target = self.pending.map_or(0, |(l, _)| l);
        self.prot_hits.clear();
        for p in self.pending_preds.iter().filter(|p| p.nonres && p.rank >= 1 && p.rank <= MAX_RANK && self.seen[p.e as usize] != 0) {
            let prot = u32::from(self.seen[p.e as usize]) <= protect;
            if prot {
                self.prot_hits.push(ProtHit { e: p.e, rank: p.rank, margin: p.margin });
            }
            for r in p.rank as usize..=3 {
                d.hits[r - 1] += 1;
                d.hits_prot[r - 1] += u32::from(prot);
            }
            if p.rank == 1 && prot {
                for (k, &b) in MARGIN_BUCKETS.iter().enumerate() {
                    d.hits_prot_m[k] += u32::from(p.margin >= b);
                }
            }
            if sent(word(target, p.e)) {
                d.demanded += 1;
                d.demanded_prot += u32::from(prot);
            }
        }
        for &sv in own {
            if valid(sv) {
                self.seen[sv as usize] = 0;
            }
        }
        d
    }
}

/// `LaneState::dry_hits_demanded`'s counts (`lh2_dry_hits_rN`,
/// `lh2_dry_hits_prot_rN`, `lh2_hits_prot_m{k}`, `lh2_demanded`,
/// `lh2_demanded_prot`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DryHits {
    pub hits: [u32; 3],
    pub hits_prot: [u32; 3],
    pub hits_prot_m: [u32; 4],
    pub demanded: u32,
    pub demanded_prot: u32,
}

/// `LaneState::classify` on a fresh state (tests, tools).
pub fn classify(look: &[i32], n_used: usize, is_box2: impl Fn(u32) -> bool, nonres: impl Fn(u32) -> bool) -> Vec<Pred> {
    let mut s = LaneState::default();
    s.classify(look, n_used, is_box2, nonres);
    s.preds
}

/// A hint word: `(layer << 16) | expert`, the `PREFETCH_WORDS` encoding.
pub fn word(layer: i32, e: u16) -> u32 {
    ((layer as u32) << 16) | u32::from(e)
}

/// The words to queue for layer `target` (design 2.2): the non-resident
/// predictions of rank `<= rank`, skipping what `already` says is queued or
/// hinted this step (box 2 reads a word once; the nursery bits take this
/// over in slice C), best ranks first, at most `cap`. Returns the words and
/// how many eligible words the cap dropped (`lh2_dropped_cap`). No admit gate
/// at any rank: a protected or gap-blocked pick is read anyway, and
/// `admit_passes` refuses 22 of 26 candidates/step at r4 (round 1, finding 2).
pub fn hint_words(preds: &[Pred], target: i32, rank: u8, cap: usize, already: impl Fn(u32) -> bool) -> (Vec<u32>, u32) {
    hint_words_m(preds, target, rank, cap, 0.0, already)
}

/// `hint_words` under the MARGIN filter (slice A amendment): a rank-1
/// prediction whose margin is below `margin` is not hinted (not counted as
/// dropped either: it is the filter's job); `margin <= 0` or an unknown margin
/// filters nothing.
pub fn hint_words_m(preds: &[Pred], target: i32, rank: u8, cap: usize, margin: f32, already: impl Fn(u32) -> bool) -> (Vec<u32>, u32) {
    let mut out = Vec::with_capacity(cap.min(16));
    let mut dropped = 0u32;
    for p in preds.iter().filter(|p| p.nonres && p.rank <= rank && p.margin_ok(margin)) {
        let w = word(target, p.e);
        if already(w) {
            continue;
        }
        if out.len() < cap {
            out.push(w);
        } else {
            dropped += 1;
        }
    }
    (out, dropped)
}

/// Per R in 1..=3: the predictions of rank `<= R` (`lh2_cand_rN`) and the
/// non-resident ones among them (`lh2_nonres_rN`).
pub fn count(preds: &[Pred]) -> ([u32; 3], [u32; 3]) {
    let (mut cand, mut nonres) = ([0u32; 3], [0u32; 3]);
    for p in preds {
        for r in (p.rank as usize).max(1)..=3 {
            cand[r - 1] += 1;
            nonres[r - 1] += u32::from(p.nonres);
        }
    }
    (cand, nonres)
}

/// Per margin bucket: the non-resident rank-1 predictions with margin `>=
/// MARGIN_BUCKETS[k]` (`lh2_nonres_m{k}`; an unknown margin counts nowhere).
pub fn count_margin(preds: &[Pred]) -> [u32; 4] {
    let mut m = [0u32; 4];
    for p in preds.iter().filter(|p| p.nonres && p.rank == 1) {
        for (k, &b) in MARGIN_BUCKETS.iter().enumerate() {
            m[k] += u32::from(p.margin >= b);
        }
    }
    m
}

/// The pending layer has routed: per R in 1..=3, how many of the non-resident
/// predictions of rank `<= R` are among the router's OWN picks `own`
/// (`[rows][n_used]`; `sel_orig` under a live cache prior, else `sel_host`),
/// distinct experts. The live Step 0 (design sections 1 and 6): recall on
/// today's exposed misses, not the 09-14 cold tail.
///
/// PER LANE, so a FLOOR on recall: a lane-layer's look-ahead for L+1 is
/// matched against the SAME lane's layer-L+1 picks on its next lane-layer
/// (`LaneState::pending`), whatever the ready-first interleaving or the
/// two-stream ordered cut does in between; a hinted expert that only the OTHER
/// lane's rows demand at L+1 counts as a miss here although box 2 would serve
/// it from the nursery. The 0.71 bar of section 1 is therefore conservative.
pub fn dry_hits(preds: &[Pred], own: &[i32]) -> [u32; 3] {
    let mut s = LaneState { pending_preds: preds.to_vec(), ..LaneState::default() };
    s.dry_hits(own)
}

// ---- the queue (design 2.3) ----

/// Most words waiting at once (a bound, not a policy: `begin_step` drops the
/// earlier step's words and a step queues a few dozen at most).
pub const HINT_QUEUE_MAX: usize = 1024;

/// The step-tagged hint queue (design 2.3): words a decode submit of ANY lane
/// carries, first on the frame. A word is dropped STALE when it is from an
/// earlier step or names a layer `<=` the carrying request's (round 1,
/// finding 9: a word queued late in step s and drained by step s+1's layer-0
/// request is 35 layers stale). `hinted` remembers this step's words, queued
/// or already handed out, for the filter's dedup; `sent` the ones that went
/// on the wire (`lh2_demanded`, `lh2_paged_hinted`).
#[derive(Debug, Default)]
pub struct HintQueue {
    /// `(word, step)`, oldest first.
    words: VecDeque<(u32, u32)>,
    /// Bit `(layer, e)`: hinted this step.
    hinted: Vec<u64>,
    /// Bit `(layer, e)`: on the wire this step (`mark_sent`).
    sent: Vec<u64>,
    step: u32,
}

impl HintQueue {
    pub fn new() -> Self {
        Self { words: VecDeque::new(), hinted: vec![0; LAYERS * WORDS], sent: vec![0; LAYERS * WORDS], step: 0 }
    }

    fn bit(w: u32) -> Option<(usize, u64)> {
        let (l, e) = ((w >> 16) as usize, (w & 0xffff) as usize);
        (l < LAYERS && e < NE).then(|| (l * WORDS + e / 64, 1u64 << (e % 64)))
    }

    /// Queued or handed out this step.
    pub fn hinted(&self, w: u32) -> bool {
        Self::bit(w).is_some_and(|(i, m)| self.hinted[i] & m != 0)
    }

    /// On the wire this step.
    pub fn sent(&self, w: u32) -> bool {
        Self::bit(w).is_some_and(|(i, m)| self.sent[i] & m != 0)
    }

    /// `words` went on the wire.
    pub fn mark_sent(&mut self, words: &[u32]) {
        for &w in words {
            if let Some((i, m)) = Self::bit(w) {
                self.sent[i] |= m;
            }
        }
    }

    /// Words a submit took (`take`) but could not send (the budget or a bar):
    /// back to the FRONT, in order, for the next submit (design 2.4: never
    /// drop the surplus).
    pub fn requeue(&mut self, words: &[u32]) {
        for &w in words.iter().rev() {
            if self.words.len() < HINT_QUEUE_MAX {
                self.words.push_front((w, self.step));
            }
        }
    }

    /// Queue `words` (already filtered and deduped by `hint_words`) tagged
    /// `step`; returns how many fit under `HINT_QUEUE_MAX`.
    pub fn push(&mut self, step: u32, words: &[u32]) -> usize {
        let mut n = 0;
        for &w in words {
            if let Some((i, m)) = Self::bit(w) {
                self.hinted[i] |= m;
            }
            if self.words.len() < HINT_QUEUE_MAX {
                self.words.push_back((w, step));
                n += 1;
            }
        }
        n
    }

    /// A decode request for `layer` at `step` is about to go: the fresh words
    /// it may carry, oldest first, up to `max` (the rest stay queued); stale
    /// words are dropped and counted.
    pub fn take(&mut self, step: u32, layer: i32, max: usize) -> (Vec<u32>, u32) {
        let mut out = Vec::new();
        let stale = self.take_into(step, layer, max, &mut out);
        (out, stale)
    }

    /// `take` into a reused buffer (cleared first); returns the stale count.
    /// In place (`retain`), so a decode submit allocates nothing.
    pub fn take_into(&mut self, step: u32, layer: i32, max: usize, out: &mut Vec<u32>) -> u32 {
        out.clear();
        let mut stale = 0u32;
        self.words.retain(|&(w, s)| {
            if s != step || (w >> 16) as i32 <= layer {
                stale += 1;
                false
            } else if out.len() < max {
                out.push(w);
                false
            } else {
                true
            }
        });
        stale
    }

    /// A new step: drop the earlier step's words (counted stale) and forget
    /// what was hinted.
    pub fn begin_step(&mut self, step: u32) -> u32 {
        let before = self.words.len();
        self.words.retain(|&(_, s)| s == step);
        self.hinted.iter_mut().for_each(|w| *w = 0);
        self.sent.iter_mut().for_each(|w| *w = 0);
        self.step = step;
        (before - self.words.len()) as u32
    }

    /// The decode -> prefill switch (`multistream` beside
    /// `pin_enter_prefill`): nothing queued may ride a prefill chunk, where
    /// `prefetch_words_prefill` would land it prefill-class. Returns how many
    /// were dropped.
    pub fn clear(&mut self) -> usize {
        let n = self.words.len();
        self.words.clear();
        self.hinted.iter_mut().for_each(|w| *w = 0);
        self.sent.iter_mut().for_each(|w| *w = 0);
        n
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }
}

// ---- the per-step speculative budget (design 2.4) ----

/// Restore words a step may always send, whatever hints and admissions took
/// (review round 3, finding 5: insurance against an admissions spike;
/// leftover restore words survive a phase flip via `pin_enter_prefill`, so
/// starvation is impossible either way).
pub const RESTORE_FLOOR: u32 = 16;

/// One request's share of the speculative words (`SpecBudget::plan`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Take {
    pub hints: usize,
    pub admissions: usize,
    pub restores: usize,
    /// Admission / restore words TODAY's rule (admissions up to the frame,
    /// then restores up to `pin_restore_per_request`) would have put on this
    /// frame and the budget held back. Per request: a word held over n
    /// submits counts n times, i.e. this is "words not sent vs legacy", the
    /// quantity the restore refill time depends on.
    pub deferred: u32,
}

/// The per-step speculative budget (design 2.4; review round 2, finding 2).
/// Box 2 can START ~40-110 speculative reads per 84 ms step; today's rule
/// sends up to 16 restores per request = 1,280/step after a phase switch, and
/// what finds no free staging set is dropped at application and lost (6-8
/// per step). One budget instead: hints + admissions + restores `<= budget`
/// per step, hints first, then admissions, then restores at ~1 per request
/// with a floor of `RESTORE_FLOOR` per step. A 1,000-word restore then
/// refills in ~1 s with ~0 loss instead of a burst that lands a fraction.
/// `budget == 0` is never planned: `submit_inner` keeps today's code path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecBudget {
    pub budget: u32,
    pub used: u32,
    pub restores_used: u32,
}

impl SpecBudget {
    pub fn new(budget: u32) -> Self {
        Self { budget, used: 0, restores_used: 0 }
    }

    /// A new step.
    pub fn reset(&mut self, budget: u32) {
        *self = Self::new(budget);
    }

    /// Restores per request: the budget spread over `N_LAYER` requests (one
    /// lane's share of a step; the live step has two or three lanes, i.e.
    /// 80-120 requests), at least 1 -- 1 at 60, so a restore burst paces out
    /// over the step whatever the lane count. With two lanes (80 requests) a
    /// budget `>= 120` gives `pace x 80 > budget`: the step total then binds
    /// first and the step's later requests carry no restore (tested); the
    /// pacing is a smoothing, the total is the cap.
    pub fn restore_pace(&self) -> usize {
        (self.budget as usize / LAYERS).max(1)
    }

    /// One request about to go, with `hints` / `admissions` / `restores`
    /// words of each class waiting and `room` words of frame space: what it
    /// carries. `restores_ok` = pin mode with no release word queued (box 2
    /// applies releases before grants, so a restore must never pass the
    /// release it undoes). `legacy_restore_cap` = `pin_restore_per_request`,
    /// for `deferred`.
    pub fn plan(&self, hints: usize, admissions: usize, restores: usize, room: usize, restores_ok: bool, legacy_restore_cap: usize) -> Take {
        let left = self.budget.saturating_sub(self.used) as usize;
        let h = hints.min(left).min(room);
        let a = admissions.min(left - h).min(room - h);
        let r = if restores_ok {
            // The floor: what the step still owes restores, even past the budget.
            let floor_left = RESTORE_FLOOR.saturating_sub(self.restores_used) as usize;
            restores.min((left - h - a).max(floor_left)).min(self.restore_pace()).min(room - h - a)
        } else {
            0
        };
        // Today's rule, for the difference.
        let la = admissions.min(room);
        let lr = if restores_ok && la < room { restores.min((room - la).min(legacy_restore_cap)) } else { 0 };
        let deferred = (la.saturating_sub(a) + lr.saturating_sub(r)) as u32;
        Take { hints: h, admissions: a, restores: r, deferred }
    }

    /// The request went with `t`.
    pub fn commit(&mut self, t: &Take) {
        self.used += (t.hints + t.admissions + t.restores) as u32;
        self.restores_used += t.restores as u32;
    }
}

static BUDGET: std::sync::Mutex<SpecBudget> = std::sync::Mutex::new(SpecBudget { budget: 0, used: 0, restores_used: 0 });

/// `SpecBudget::plan` on this step's budget (`remote_experts::submit_inner`).
pub fn budget_plan(hints: usize, admissions: usize, restores: usize, room: usize, restores_ok: bool, legacy_restore_cap: usize) -> Take {
    BUDGET.lock().unwrap_or_else(|p| p.into_inner()).plan(hints, admissions, restores, room, restores_ok, legacy_restore_cap)
}

/// `SpecBudget::commit` on this step's budget, and the `lh2_budget_deferred`
/// counter it moves (`lh2_hints_sent` is bumped by the submit for the words
/// that actually went: the bars may cut `t.hints`).
pub fn budget_commit(t: &Take) {
    BUDGET.lock().unwrap_or_else(|p| p.into_inner()).commit(t);
    bump(Stat::BudgetDeferred, u64::from(t.deferred));
}

// ---- counters (design section 6) ----

/// `(hub_lh2 field, ms.stage host stage)` per counter, in `take_stats` order:
/// per R in 1..=3 the box-2-owned predictions of rank `<= R` (distinct per
/// LANE-layer: both lanes predicting one expert for one layer count twice), the
/// mirror-non-resident ones among them, and of THOSE the ones in the same
/// lane's own picks one lane-layer later (a floor, `dry_hits`); then hint words the queue handed a decode
/// submit (slice A: counted, kept off the wire), words on the wire (0 in slice
/// A), words the per-request cap dropped, words dropped stale, and admission /
/// restore words the per-step budget held back vs today's rule; then slice B
/// (`evtrace_kinds::HUB_LH2`): sent hints the same lane demanded at the target
/// layer (and of those the protected-rank ones), paged experts in pin replies
/// that had been hinted, hints deduped against the mirror's NURSERY bits, the
/// `after` placement's late look-aheads (0 under `before`), abort-bar trips.
pub const STATS: [(&str, &str); 46] = [
    ("lh2_cand_r1", "lh2.cand_r1"),
    ("lh2_cand_r2", "lh2.cand_r2"),
    ("lh2_cand_r3", "lh2.cand_r3"),
    ("lh2_nonres_r1", "lh2.nonres_r1"),
    ("lh2_nonres_r2", "lh2.nonres_r2"),
    ("lh2_nonres_r3", "lh2.nonres_r3"),
    ("lh2_dry_hits_r1", "lh2.dry_hits_r1"),
    ("lh2_dry_hits_r2", "lh2.dry_hits_r2"),
    ("lh2_dry_hits_r3", "lh2.dry_hits_r3"),
    ("lh2_dry_words", "lh2.dry_words"),
    ("lh2_hints_sent", "lh2.hints_sent"),
    ("lh2_dropped_cap", "lh2.dropped_cap"),
    ("lh2_stale", "lh2.stale"),
    ("lh2_budget_deferred", "lh2.budget_deferred"),
    ("lh2_demanded", "lh2.demanded"),
    ("lh2_demanded_prot", "lh2.demanded_prot"),
    ("lh2_paged_hinted", "lh2.paged_hinted"),
    ("lh2_nursery_covered", "lh2.nursery_covered"),
    ("lh2_look_late", "lh2.look_late"),
    ("lh2_bar_trips", "lh2.bar_trips"),
    // Slice A amendment (10-06): the protected set and the margin buckets.
    ("lh2_dry_hits_prot_r1", "lh2.dry_hits_prot_r1"),
    ("lh2_dry_hits_prot_r2", "lh2.dry_hits_prot_r2"),
    ("lh2_dry_hits_prot_r3", "lh2.dry_hits_prot_r3"),
    ("lh2_nonres_m0", "lh2.nonres_m0"),
    ("lh2_nonres_m1", "lh2.nonres_m1"),
    ("lh2_nonres_m2", "lh2.nonres_m2"),
    ("lh2_nonres_m3", "lh2.nonres_m3"),
    ("lh2_hits_prot_m0", "lh2.hits_prot_m0"),
    ("lh2_hits_prot_m1", "lh2.hits_prot_m1"),
    ("lh2_hits_prot_m2", "lh2.hits_prot_m2"),
    ("lh2_hits_prot_m3", "lh2.hits_prot_m3"),
    // The refined objective (owner 10-06), scored at the reply (`reply_hits`).
    ("lh2_hits_prot_paged_r1", "lh2.hits_prot_paged_r1"),
    ("lh2_hits_prot_paged_r2", "lh2.hits_prot_paged_r2"),
    ("lh2_hits_prot_paged_r3", "lh2.hits_prot_paged_r3"),
    ("lh2_hits_prot_paged_m0", "lh2.hits_prot_paged_m0"),
    ("lh2_hits_prot_paged_m1", "lh2.hits_prot_paged_m1"),
    ("lh2_hits_prot_paged_m2", "lh2.hits_prot_paged_m2"),
    ("lh2_hits_prot_paged_m3", "lh2.hits_prot_paged_m3"),
    ("lh2_hits_prot_late_r1", "lh2.hits_prot_late_r1"),
    ("lh2_hits_prot_late_r2", "lh2.hits_prot_late_r2"),
    ("lh2_hits_prot_late_r3", "lh2.hits_prot_late_r3"),
    ("lh2_hits_prot_late_m0", "lh2.hits_prot_late_m0"),
    ("lh2_hits_prot_late_m1", "lh2.hits_prot_late_m1"),
    ("lh2_hits_prot_late_m2", "lh2.hits_prot_late_m2"),
    ("lh2_hits_prot_late_m3", "lh2.hits_prot_late_m3"),
    ("lh2_prot_paged_late_total", "lh2.prot_paged_late_total"),
];

/// `STATS` index of the first `lh2_dry_hits_prot_rN` / `lh2_nonres_m{k}` /
/// `lh2_hits_prot_m{k}` / `lh2_hits_prot_paged_*` / `lh2_hits_prot_late_*`
/// entry, and of `lh2_prot_paged_late_total`.
const STAT_DRY_HITS_PROT: usize = 20;
const STAT_NONRES_M: usize = 23;
const STAT_HITS_PROT_M: usize = 27;
const STAT_PROT_PAGED: usize = 31;
const STAT_PROT_LATE: usize = 38;
const STAT_PROT_PAGED_LATE_TOTAL: usize = 45;

/// One lane-layer's reply scored (`reply_hits`): `lh2_hits_prot_paged_*`,
/// `lh2_hits_prot_late_*`, `lh2_prot_paged_late_total`.
pub fn count_reply_hits(r: &ReplyHits) {
    for k in 0..3 {
        COUNTS[STAT_PROT_PAGED + k].fetch_add(u64::from(r.paged[k]), Relaxed);
        COUNTS[STAT_PROT_LATE + k].fetch_add(u64::from(r.late[k]), Relaxed);
    }
    for k in 0..4 {
        COUNTS[STAT_PROT_PAGED + 3 + k].fetch_add(u64::from(r.paged_m[k]), Relaxed);
        COUNTS[STAT_PROT_LATE + 3 + k].fetch_add(u64::from(r.late_m[k]), Relaxed);
    }
    COUNTS[STAT_PROT_PAGED_LATE_TOTAL].fetch_add(u64::from(r.total), Relaxed);
}

static COUNTS: [AtomicU64; STATS.len()] = [const { AtomicU64::new(0) }; STATS.len()];

/// The scalar counters (`bump`); the per-R ones go through `count_preds` /
/// `count_dry_hits`.
#[derive(Clone, Copy, Debug)]
pub enum Stat {
    DryWords = 9,
    HintsSent = 10,
    DroppedCap = 11,
    Stale = 12,
    BudgetDeferred = 13,
    Demanded = 14,
    DemandedProt = 15,
    PagedHinted = 16,
    NurseryCovered = 17,
    LookLate = 18,
    BarTrips = 19,
}

pub fn bump(s: Stat, n: u64) {
    if n != 0 {
        COUNTS[s as usize].fetch_add(n, Relaxed);
    }
}

/// One lane-layer's L+1 predictions classified: `lh2_cand_rN`, `lh2_nonres_rN`,
/// `lh2_nonres_m{k}`.
pub fn count_preds(preds: &[Pred]) {
    let (cand, nonres) = count(preds);
    for r in 0..3 {
        COUNTS[r].fetch_add(u64::from(cand[r]), Relaxed);
        COUNTS[3 + r].fetch_add(u64::from(nonres[r]), Relaxed);
    }
    for (k, m) in count_margin(preds).into_iter().enumerate() {
        COUNTS[STAT_NONRES_M + k].fetch_add(u64::from(m), Relaxed);
    }
}

/// One lane-layer's dry match: `lh2_dry_hits_rN`, `lh2_dry_hits_prot_rN`,
/// `lh2_hits_prot_m{k}`.
pub fn count_dry_hits(d: &DryHits) {
    for r in 0..3 {
        COUNTS[6 + r].fetch_add(u64::from(d.hits[r]), Relaxed);
        COUNTS[STAT_DRY_HITS_PROT + r].fetch_add(u64::from(d.hits_prot[r]), Relaxed);
    }
    for k in 0..4 {
        COUNTS[STAT_HITS_PROT_M + k].fetch_add(u64::from(d.hits_prot_m[k]), Relaxed);
    }
}

/// Read-and-clear every counter, in `STATS` order (once per decode step,
/// `multistream`'s `hub_lh2` record and `ms.stage` rollup).
pub fn take_stats() -> [u64; STATS.len()] {
    let mut out = [0u64; STATS.len()];
    for (o, c) in out.iter_mut().zip(COUNTS.iter()) {
        *o = c.swap(0, Relaxed);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const NU: usize = 6;

    fn pred(e: u16, rank: u8, nonres: bool) -> Pred {
        Pred { e, rank, nonres, margin: f32::NAN }
    }

    #[test]
    fn cfg_packs_and_unpacks() {
        let c = Cfg { mode: Mode::K2, rank: 3, cap: 200, spec_budget: 65535, step: u32::MAX - 7, rows: 300, max_words_step: 1490, per_ll_x10: 25, nursery_prior: true, margin_x1000: 1000 };
        let (v, w) = c.pack(); assert_eq!(Cfg::unpack(v, w), c);
        assert_eq!(Cfg::unpack(0, 0), Cfg::default());
        assert_eq!(Cfg::default().mode, Mode::Off);
    }

    #[test]
    fn the_knobs_parse_and_default_as_the_design_says() {
        use crate::knobs::{B2_MISS_PREFETCH, B2_MISS_PREFETCH_CAP, B2_MISS_PREFETCH_RANK, B2_SPEC_BUDGET};
        let p = |s: &str| B2_MISS_PREFETCH.kind.parse(s).map(|i| Mode::from_knob(i as usize));
        assert_eq!(p("off"), Some(Mode::Off));
        assert_eq!(p("0"), Some(Mode::Off));
        assert_eq!(p("dry"), Some(Mode::Dry));
        assert_eq!(p("DRY"), Some(Mode::Dry));
        assert_eq!(p("k1"), Some(Mode::K1));
        assert_eq!(p("k2"), Some(Mode::K2));
        assert_eq!(p("1"), None);
        assert_eq!(p("nope"), None);
        assert_eq!(B2_MISS_PREFETCH_RANK.kind.parse("0"), Some(1), "R is 1..=3");
        assert_eq!(B2_MISS_PREFETCH_RANK.kind.parse("9"), Some(3));
        assert_eq!(B2_MISS_PREFETCH_CAP.kind.parse("8"), Some(8));
        assert_eq!(B2_SPEC_BUDGET.kind.parse("0"), Some(0), "0 = today's per-request caps");
        assert_eq!(B2_SPEC_BUDGET.kind.parse("99999"), Some(65535), "fits `Cfg`");
        for k in [&B2_MISS_PREFETCH, &B2_MISS_PREFETCH_RANK, &B2_MISS_PREFETCH_CAP, &B2_SPEC_BUDGET] {
            assert!(k.live, "{}: flipped per turn", k.name);
        }
        // Defaults (not `get()`: a test process may carry env/file values).
        assert!(matches!(B2_MISS_PREFETCH.kind, crate::knobs::Kind::Choice { default: 0, .. }));
        assert!(matches!(B2_MISS_PREFETCH_RANK.kind, crate::knobs::Kind::Int { default: 1, .. }));
        assert!(matches!(B2_MISS_PREFETCH_CAP.kind, crate::knobs::Kind::Int { default: 8, .. }));
        assert!(matches!(B2_SPEC_BUDGET.kind, crate::knobs::Kind::Int { default: 0, .. }), "today's rule until the budget is A/B'd live (code review, finding 1)");
        // Slice B + the amendment: the bars, the prior mark, the margin.
        use crate::knobs::{B2_MISS_PREFETCH_MARGIN, B2_MISS_PREFETCH_MAX_PER_LL_X10, B2_MISS_PREFETCH_MAX_WORDS_STEP, B2_NURSERY_PRIOR};
        assert!(matches!(B2_MISS_PREFETCH_MAX_WORDS_STEP.kind, crate::knobs::Kind::Int { default: 0, .. }), "0 = rows-aware default");
        assert!(matches!(B2_MISS_PREFETCH_MAX_PER_LL_X10.kind, crate::knobs::Kind::Int { default: 25, .. }));
        assert!(matches!(B2_NURSERY_PRIOR.kind, crate::knobs::Kind::Flag(false)), "the prior stays hint-blind by default");
        assert!(matches!(B2_MISS_PREFETCH_MARGIN.kind, crate::knobs::Kind::Real { default, .. } if default == 0.0), "no margin filter until the dry run picks one");
        assert_eq!(B2_MISS_PREFETCH_MARGIN.kind.parse("0.15").map(f64::from_bits), Some(0.15));
        for k in [&B2_MISS_PREFETCH_MAX_WORDS_STEP, &B2_MISS_PREFETCH_MAX_PER_LL_X10, &B2_NURSERY_PRIOR, &B2_MISS_PREFETCH_MARGIN] {
            assert!(k.live, "{}: flipped per turn", k.name);
        }
        assert_eq!(STATS.len(), 46);
        assert_eq!(STATS[STAT_DRY_HITS_PROT].0, "lh2_dry_hits_prot_r1");
        assert_eq!(STATS[STAT_PROT_PAGED].0, "lh2_hits_prot_paged_r1");
        assert_eq!(STATS[STAT_PROT_PAGED + 3].0, "lh2_hits_prot_paged_m0");
        assert_eq!(STATS[STAT_PROT_LATE].0, "lh2_hits_prot_late_r1");
        assert_eq!(STATS[STAT_PROT_LATE + 6].0, "lh2_hits_prot_late_m3");
        assert_eq!(STATS[STAT_PROT_PAGED_LATE_TOTAL].1, "lh2.prot_paged_late_total");
        assert_eq!(STATS[STAT_NONRES_M].0, "lh2_nonres_m0");
        assert_eq!(STATS[STAT_HITS_PROT_M + 3].0, "lh2_hits_prot_m3");
        assert_eq!(STATS[Stat::BarTrips as usize].0, "lh2_bar_trips");
        assert_eq!(STATS[Stat::NurseryCovered as usize].0, "lh2_nursery_covered");
        assert_eq!(STATS[Stat::PagedHinted as usize].1, "lh2.paged_hinted");
        assert_eq!(STATS[Stat::Demanded as usize].0, "lh2_demanded");
    }

    /// Design 2.1 (review round 2, finding 1): the knob drives the look-ahead;
    /// `look_next2` only under `k2`; the old word block only under the legacy
    /// var with the knob off -- `dry`/`k1` with the legacy var unset push
    /// nothing from the 9515 block.
    #[test]
    fn knob_gating_of_the_lookahead_and_the_legacy_block() {
        assert_eq!(look_gates(Mode::Off, false, 2), (false, false), "knob off, no legacy var: no launches");
        assert_eq!(look_gates(Mode::Off, true, 2), (true, true), "legacy at depth 2");
        assert_eq!(look_gates(Mode::Off, true, 1), (true, false));
        assert_eq!(look_gates(Mode::Dry, false, 2), (true, false), "dry: L+1 only, depth ignored");
        assert_eq!(look_gates(Mode::K1, true, 2), (true, false), "the knob on: the legacy depth is ignored");
        assert_eq!(look_gates(Mode::K2, false, 1), (true, true));
        for m in [Mode::Dry, Mode::K1, Mode::K2] {
            assert!(!legacy_words(m, false, true), "{m:?}: legacy var unset pushes nothing from 9515");
            assert!(!legacy_words(m, true, true), "{m:?}: the knob replaces the old block even with the legacy var set");
        }
        // Slice B: the wire needs `k1`/`k2` AND box 2's `RESP_FLAG_NURSERY`.
        super::super::b2_mirror::reset_nursery_support();
        for m in [Mode::Off, Mode::Dry, Mode::K1, Mode::K2] {
            assert!(!m.wire(), "{m:?}: nothing on the wire before box 2 answers");
        }
        assert!(Mode::K1.asks() && Mode::K2.asks() && !Mode::Dry.asks() && !Mode::Off.asks());
        super::super::b2_mirror::nursery_reply_seen(true);
        assert!(Mode::K1.wire() && Mode::K2.wire() && !Mode::Dry.wire() && !Mode::Off.wire());
        super::super::b2_mirror::nursery_reply_seen(false);
        assert!(!Mode::K1.wire(), "the capability was lost: dry again");
        super::super::b2_mirror::reset_nursery_support();
        assert!(legacy_words(Mode::Off, true, true), "the old behaviour survives under the legacy var alone");
        assert!(!legacy_words(Mode::Off, true, false), "... where the driver allows hints");
        assert!(!legacy_words(Mode::Off, false, true));
    }

    #[test]
    fn classify_dedups_by_best_rank_and_keeps_box2_only() {
        // Two rows; 7 is rank 3 in row 0 and rank 1 in row 1; 9 is box 1's;
        // -1 / 999 are skipped.
        let look = [5, 9, 7, 11, -1, 999, 7, 11, 13, 5, 9, 9];
        let preds = classify(&look, NU, |e| e != 9, |e| e == 7 || e == 13);
        assert_eq!(preds, vec![pred(5, 1, false), pred(7, 1, true), pred(11, 2, false), pred(13, 3, true)]);
        assert!(classify(&look, 0, |_| true, |_| true).is_empty());
        assert!(classify(&[], NU, |_| true, |_| true).is_empty());
        let (cand, nonres) = count(&preds);
        assert_eq!(cand, [2, 3, 4]);
        assert_eq!(nonres, [1, 1, 2]);
        // Ranks past `MAX_RANK` are neither counted nor hinted.
        assert_eq!(classify(&[1, 2, 3, 4, 5, 6], NU, |_| true, |_| true).len(), 3);
        // The state is reusable: a second classify on the same `LaneState` sees
        // clean marks (an expert of the first call is not stuck at 255 / rank).
        let mut s = LaneState::default();
        s.classify(&look, NU, |e| e != 9, |_| false);
        s.classify(&[9, 7, 1, -1, -1, -1], NU, |_| true, |_| true);
        assert_eq!(s.preds, vec![pred(9, 1, true), pred(7, 2, true), pred(1, 3, true)]);
        s.set_pending(4, 5);
        assert_eq!(s.pending_preds, s.preds.clone());
        assert!(!s.take_pending(4, 6), "another step: dropped");
        assert!(s.pending.is_none());
        s.set_pending(4, 5);
        assert!(s.take_pending(4, 5));
        assert_eq!(s.dry_hits(&[7, 0, 0, 0, 0, 0]), [0, 1, 1]);
        assert_eq!(s.dry_hits(&[7, 0, 0, 0, 0, 0]), [0, 1, 1], "`seen` is cleared between calls");
    }

    #[test]
    fn hint_words_filters_dedups_and_caps() {
        let preds = [pred(1, 1, true), pred(2, 1, false), pred(3, 2, true), pred(4, 2, true), pred(5, 3, true), pred(6, 2, true)];
        // R=1: only the non-resident rank-1 pick.
        assert_eq!(hint_words(&preds, 7, 1, 8, |_| false), (vec![word(7, 1)], 0));
        // R=2: ranks 1-2, non-resident, rank order; 3 is already hinted.
        let (w, dropped) = hint_words(&preds, 7, 2, 8, |w| w == word(7, 3));
        assert_eq!(w, vec![word(7, 1), word(7, 4), word(7, 6)]);
        assert_eq!(dropped, 0);
        // The cap drops the lowest ranks and counts them; a deduped word costs no cap.
        let (w, dropped) = hint_words(&preds, 7, 3, 2, |w| w == word(7, 1));
        assert_eq!(w, vec![word(7, 3), word(7, 4)]);
        assert_eq!(dropped, 2, "6 (rank 2) and 5 (rank 3)");
        assert_eq!(hint_words(&preds, 7, 3, 0, |_| false), (vec![], 5));
        assert_eq!(word(39, 383), (39 << 16) | 383);
    }

    #[test]
    fn dry_hits_count_non_resident_predictions_in_the_own_picks_per_rank() {
        let preds = [pred(1, 1, true), pred(2, 1, false), pred(3, 2, true), pred(4, 3, true), pred(5, 2, true)];
        // Own picks of two rows: 1 and 3 hit (3 twice: distinct), 2 is resident, 4 absent.
        let own = [1, 3, 10, 11, 12, 13, 3, 20, 21, 22, 23, 2];
        assert_eq!(dry_hits(&preds, &own), [1, 2, 2]);
        assert_eq!(dry_hits(&preds, &[]), [0, 0, 0]);
        assert_eq!(dry_hits(&[], &own), [0, 0, 0]);
    }

    #[test]
    fn the_queue_tags_words_by_step_and_drops_stale_ones() {
        let mut q = HintQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.begin_step(5), 0);
        // Lane A at layer 3 hints (4, 10) and (4, 11); lane B at layer 3 would re-hint (4, 10).
        assert_eq!(q.push(5, &[word(4, 10), word(4, 11)]), 2);
        assert!(q.hinted(word(4, 10)) && !q.hinted(word(4, 12)));
        assert_eq!(hint_words(&[pred(10, 1, true), pred(12, 1, true)], 4, 1, 8, |w| q.hinted(w)), (vec![word(4, 12)], 0));
        // Lane A's layer-3 request carries them (layer 3 < 4: fresh), up to `max`.
        assert_eq!(q.take(5, 3, 1), (vec![word(4, 10)], 0));
        assert_eq!(q.len(), 1);
        assert!(q.hinted(word(4, 10)), "handed out still counts for dedup this step");
        // A request for layer 4 (or later) cannot carry a layer-4 word: stale.
        assert_eq!(q.take(5, 4, 8), (vec![], 1));
        // Lane B's request for an EARLIER layer may carry lane A's word (design
        // 2.3: any lane's next decode submit), and a later layer's word rides
        // ahead of an earlier one only by queue order.
        q.push(5, &[word(9, 1), word(6, 2)]);
        assert_eq!(q.take(5, 3, 8), (vec![word(9, 1), word(6, 2)], 0));
        // An earlier step's word is stale too, whatever the layer.
        q.push(5, &[word(9, 1)]);
        assert_eq!(q.take(6, 0, 8), (vec![], 1));
        // `begin_step` drops the old step's words and forgets the dedup set.
        q.push(6, &[word(9, 2)]);
        q.push(5, &[word(9, 3)]);
        assert_eq!(q.begin_step(6), 1);
        assert!(!q.hinted(word(9, 2)) && q.len() == 1);
        // The phase switch empties it.
        assert_eq!(q.clear(), 1);
        assert!(q.is_empty());
        // The bound.
        let many: Vec<u32> = (0..HINT_QUEUE_MAX as u32 + 5).map(|i| word((i % 40) as i32, (i % 384) as u16)).collect();
        assert_eq!(q.push(7, &many), HINT_QUEUE_MAX);
    }

    /// Design 2.4: hints first, then admissions, then restores (~1-2 per
    /// request at the default 60) with a floor of 16 per step; deferred words
    /// stay queued; `deferred` = what today's rule would have sent more.
    #[test]
    fn the_budget_orders_classes_paces_restores_and_keeps_the_floor() {
        let mut b = SpecBudget::new(60);
        assert_eq!(b.restore_pace(), 1, "60 / 40 layers");
        assert_eq!(SpecBudget::new(120).restore_pace(), 3);
        assert_eq!(SpecBudget::new(1).restore_pace(), 1);
        // A quiet request: 2 admissions and a restore burst of 1000 waiting.
        let t = b.plan(0, 2, 1000, 128, true, 16);
        assert_eq!((t.hints, t.admissions, t.restores), (0, 2, 1));
        assert_eq!(t.deferred, 15, "today: 2 admissions + 16 restores");
        b.commit(&t);
        assert_eq!((b.used, b.restores_used), (3, 1));
        // Hints come first and squeeze admissions, then restores.
        let t = b.plan(3, 100, 1000, 128, true, 16);
        assert_eq!((t.hints, t.admissions, t.restores), (3, 54, 1), "60 - 3 used = 57 left: 3 hints, 54 admissions, nothing left for restores ...");
        b.commit(&t);
        assert_eq!((b.used, b.restores_used), (61, 2), "... except the floor, which may overspend the budget by its restores");
        // The floor: restores still get up to 16 this step, paced, with the budget gone.
        let t = b.plan(0, 100, 1000, 128, true, 16);
        assert_eq!((t.hints, t.admissions, t.restores), (0, 0, 1));
        assert_eq!(t.deferred, 100 + 16 - 1);
        for _ in 0..15 {
            b.commit(&b.plan(0, 0, 1000, 128, true, 16));
        }
        assert_eq!(b.restores_used, 16);
        assert_eq!(b.plan(0, 0, 1000, 128, true, 16).restores, 0, "the floor is spent");
        // No restores while a release is queued or outside pin mode.
        assert_eq!(SpecBudget::new(60).plan(0, 0, 1000, 128, false, 16), Take { deferred: 0, ..Take::default() });
        // The frame bounds everything.
        let t = SpecBudget::new(1000).plan(100, 100, 100, 128, true, 16);
        assert_eq!((t.hints, t.admissions, t.restores), (100, 28, 0));
        // A new step resets.
        b.reset(60);
        assert_eq!(b, SpecBudget::new(60));
        // With nothing waiting, nothing is deferred.
        assert_eq!(SpecBudget::new(60).plan(0, 0, 0, 128, true, 16), Take::default());
        // Budget 120, two lanes = 80 requests, a 1000-word restore: the pace is
        // 3 per request, so the step total binds after 40 requests and the
        // later requests carry nothing (review finding 9: the total is the cap).
        let mut b = SpecBudget::new(120);
        let per_req: Vec<usize> = (0..80).map(|_| { let t = b.plan(0, 0, 1000, 128, true, 16); b.commit(&t); t.restores }).collect();
        assert_eq!(per_req.iter().sum::<usize>(), 120);
        assert!(per_req[..40].iter().all(|&r| r == 3) && per_req[40..].iter().all(|&r| r == 0), "{per_req:?}");
    }

    /// Slice A amendment: the margin of a rank-1 prediction from the weights,
    /// the margin buckets, the margin filter, and the PROTECTED set (own pick
    /// at actual rank `<= protect`) per R and per bucket.
    #[test]
    fn margin_filter_and_protected_set_counters() {
        // Two rows, n_used 6. Row 0: 5 (w .5) over 9 (w .25): margin .25 / 1.0 = 0.25.
        // Row 1: 7 (w .4) over 5 (w .35): margin 0.05; 7 is rank 1, 5 rank 2 here.
        let look = [5, 9, 7, 11, 13, 1, 7, 5, 11, 13, 9, 2];
        let w = [0.5f32, 0.25, 0.1, 0.1, 0.03, 0.02, 0.4, 0.35, 0.1, 0.1, 0.03, 0.02];
        let mut s = LaneState::default();
        s.classify_w(&look, &w, NU, |_| true, |_| true);
        let m = |e: u16| s.preds.iter().find(|p| p.e == e).map(|p| p.margin).unwrap();
        assert!((m(5) - 0.25).abs() < 1e-6, "{}", m(5));
        assert!((m(7) - 0.05).abs() < 1e-6, "{}", m(7));
        assert!(m(9).is_nan() && m(11).is_nan(), "ranks 2-3 carry no margin");
        assert_eq!(count_margin(&s.preds), [2, 1, 1, 0], "5 at >= 0 / .1 / .2; 7 only at >= 0");
        // The filter: judged on rank 1 only, unknown passes, 0 filters nothing.
        let (hw, _) = hint_words_m(&s.preds, 4, 1, 8, 0.1, |_| false);
        assert_eq!(hw, vec![word(4, 5)], "7's margin .05 < .1");
        let (hw, dropped) = hint_words_m(&s.preds, 4, 3, 8, 0.1, |_| false);
        assert_eq!((hw, dropped), (vec![word(4, 5), word(4, 9), word(4, 11)], 0), "ranks 2-3 are not judged (13 is rank 4: never a prediction); a filtered word is not a cap drop");
        assert_eq!(hint_words_m(&s.preds, 4, 1, 8, 0.0, |_| false).0.len(), 2);
        assert_eq!(hint_words_m(&s.preds, 4, 1, 8, 0.3, |_| false).0.len(), 0);
        assert!(pred(3, 1, true).margin_ok(0.5), "unknown passes");
        // No weights: margins unknown, nothing counts in a bucket, the filter passes all.
        let mut t = LaneState::default();
        t.classify(&look, NU, |_| true, |_| true);
        assert_eq!(count_margin(&t.preds), [0, 0, 0, 0]);
        assert_eq!(hint_words_m(&t.preds, 4, 1, 8, 0.3, |_| false).0.len(), 2);
        // The protected set: own picks `[rows][6]`; protect 1.
        s.set_pending(4, 9);
        assert!(s.take_pending(4, 9));
        // Row 0 of the own picks: 9 at rank 1, 5 at rank 2; row 1: 7 at rank 1, 11 at rank 4.
        let own = [9, 5, 20, 21, 22, 23, 7, 30, 31, 11, 32, 33];
        let d = s.dry_hits_demanded(&own, |w| w == word(4, 5) || w == word(4, 9), 1);
        assert_eq!(d.hits, [2, 3, 4], "5, 7 (rank 1); + 9 (rank 2); + 11 (rank 3)");
        assert_eq!(d.hits_prot, [1, 2, 2], "7 at actual rank 1; 9 at actual rank 1; 5 is actual rank 2, 11 rank 4");
        assert_eq!(d.hits_prot_m, [1, 0, 0, 0], "7 (margin .05) is the only protected rank-1 hit");
        assert_eq!((d.demanded, d.demanded_prot), (2, 1), "5 and 9 were sent; 9 is protected");
        // protect 2: 5 (actual rank 2) joins.
        s.set_pending(4, 9);
        assert!(s.take_pending(4, 9));
        let d2 = s.dry_hits_demanded(&own, |_| false, 2);
        assert_eq!(d2.hits_prot, [2, 3, 3]);
        assert_eq!(d2.hits_prot_m, [2, 1, 1, 0]);
        assert_eq!(s.dry_hits(&own), [2, 3, 4], "the any-rank diagnostic is unchanged");
        assert_eq!(default_max_words_step(1), 10);
        assert_eq!(default_max_words_step(4), 10);
        assert_eq!(default_max_words_step(8), 30);
    }

    /// The refined objective (owner 10-06): protected dry hits scored at the
    /// reply against its PAGED bits and the lane's stall, per R and per
    /// margin bucket; the denominator counts the request's protected
    /// non-resident picks that were paged AND late, hinted or not.
    #[test]
    fn reply_hits_score_paged_and_late_against_the_protected_set() {
        let hit = |e: u16, rank: u8, margin: f32| ProtHit { e, rank, margin };
        let hits = [hit(5, 1, 0.25), hit(7, 1, 0.05), hit(9, 2, f32::NAN), hit(40, 1, f32::NAN)];
        let mut paged = [0u32; 12];
        paged[0] |= (1 << 5) | (1 << 9); // 5 and 9 paged, 7 served, 40 (word 1) not
        paged[1] |= 1 << 1; // expert 33
        let nonres = [5u16, 7, 33, 70];
        // Not late: paged hits count, nothing late, no denominator.
        let r = reply_hits(&hits, &nonres, &paged, false);
        assert_eq!(r.paged, [1, 2, 2], "5 (rank 1) and 9 (rank 2)");
        assert_eq!(r.paged_m, [1, 1, 1, 0], "5's margin .25");
        assert_eq!((r.late, r.late_m, r.total), ([0; 3], [0; 4], 0));
        // Late: the same hits are late, the denominator = nonres & paged (5, 33).
        let r = reply_hits(&hits, &nonres, &paged, true);
        assert_eq!((r.paged, r.late), ([1, 2, 2], [1, 2, 2]));
        assert_eq!((r.paged_m, r.late_m), ([1, 1, 1, 0], [1, 1, 1, 0]));
        assert_eq!(r.total, 2);
        // Nothing paged: nothing counts, late or not; an unknown margin counts in no bucket.
        let r = reply_hits(&[hit(40, 1, f32::NAN)], &nonres, &[0u32; 12], true);
        assert_eq!(r, ReplyHits::default());
        let mut p40 = [0u32; 12];
        p40[1] |= 1 << 8;
        assert_eq!(reply_hits(&[hit(40, 1, f32::NAN)], &[], &p40, true).paged_m, [0; 4]);
        // The denominator from the sel sent (rank order, NO_PICK = box 1's):
        // protect 1 keeps column 0 only; distinct; non-resident per the mirror.
        let sel = [5, 9, -1, 11, 13, 1, 7, 5, 11, -1, 9, 2, 5, 1, 2, 3, 4, 6];
        let mut out = Vec::new();
        protected_nonres(&sel, 6, 1, |e| e != 7, &mut out);
        assert_eq!(out, vec![5], "7 is resident; 5 once");
        protected_nonres(&sel, 6, 2, |_| true, &mut out);
        assert_eq!(out, vec![5, 9, 7, 1]);
        protected_nonres(&sel, 0, 2, |_| true, &mut out);
        assert!(out.is_empty());
        // `dry_hits_demanded` keeps the protected hits for the reply.
        let mut s = LaneState::default();
        s.classify_w(&[5, 9, 7, 11, 13, 1], &[0.5, 0.25, 0.1, 0.1, 0.03, 0.02], 6, |_| true, |_| true);
        s.set_pending(4, 1);
        assert!(s.take_pending(4, 1));
        let d = s.dry_hits_demanded(&[9, 5, 20, 21, 22, 23], |_| false, 1);
        assert_eq!(d.hits_prot, [0, 1, 1]);
        assert_eq!(s.prot_hits, vec![hit(9, 2, f32::NAN)]);
        let _ = take_stats();
        count_reply_hits(&reply_hits(&s.prot_hits, &[9], &{ let mut p = [0u32; 12]; p[0] = 1 << 9; p }, true));
        let st = take_stats();
        assert_eq!(&st[STAT_PROT_PAGED..STAT_PROT_PAGED + 3], &[0, 1, 1]);
        assert_eq!(&st[STAT_PROT_LATE..STAT_PROT_LATE + 3], &[0, 1, 1]);
        assert_eq!(st[STAT_PROT_PAGED_LATE_TOTAL], 1);
    }

    /// The absolute abort bars (design 5): the step bar, the per-lane-layer
    /// bar once 8 lane-layers routed, and dry for the rest of the step.
    #[test]
    fn bars_trip_and_the_step_goes_dry() {
        let mut b = Bars::default();
        assert_eq!(b.allow(0, 10, 25), (0, false));
        assert_eq!(b.allow(6, 10, 25), (6, false));
        assert_eq!(b.allow(4, 10, 25), (4, false), "exactly the bar is fine");
        assert_eq!(b.allow(1, 10, 25), (0, true), "the 11th word trips");
        assert!(b.tripped);
        assert_eq!(b.allow(1, 100, 255), (0, false), "dry for the rest of the step, counted once");
        b.reset();
        assert_eq!(b, Bars::default());
        // Per lane-layer: 2.5 x lane-layers, judged from 8 lane-layers on.
        for _ in 0..7 {
            b.note_lane_layer();
        }
        assert_eq!(b.allow(30, 1000, 25), (30, false), "7 lane-layers: not judged yet");
        b.note_lane_layer();
        assert_eq!(b.allow(1, 1000, 25), (0, true), "31 > 2.5 x 8");
        let mut c = Bars::default();
        for _ in 0..8 {
            c.note_lane_layer();
        }
        assert_eq!(c.allow(20, 1000, 25), (20, false), "20 = 2.5 x 8");
        assert_eq!(c.allow(1, 1000, 25), (0, true));
    }

    /// The submit path with the wire on, as `submit_inner` sequences it:
    /// take, plan, bars on the words that GO, mark sent, re-queue the
    /// surplus; a tripped step takes nothing more; the next submit picks the
    /// re-queued words first; budget > 0 puts hints first.
    #[test]
    fn submit_path_bars_requeue_and_budget_order() {
        let c = Cfg { spec_budget: 60, max_words_step: 10, per_ll_x10: 255, ..Cfg::default() };
        let mut q = HintQueue::new();
        let mut budget = SpecBudget::new(60);
        let mut bars = Bars::default();
        q.begin_step(1);
        let words: Vec<u32> = (0..30).map(|e| word(5, e)).collect();
        q.push(1, &words);
        // Submit 1 (layer 2): 30 fresh words; the plan takes all 30 (budget
        // 60), the step bar (10) trips on the words that would go -> none go,
        // the budget is charged for none, all 30 are re-queued, `sent` 0.
        let mut scratch = Vec::new();
        let mut submit = |q: &mut HintQueue, budget: &mut SpecBudget, bars: &mut Bars, layer: i32, scratch: &mut Vec<u32>| -> (usize, usize) {
            if bars.tripped {
                return (0, 0);
            }
            q.take_into(1, layer, 128, scratch);
            let mut t = budget.plan(scratch.len(), 5, 100, 128, true, 16);
            let (go, _) = bars.allow(t.hints, c.max_words_step, c.per_ll_x10);
            t.hints = go;
            budget.commit(&t);
            q.mark_sent(&scratch[..go]);
            q.requeue(&scratch[go..]);
            (go, t.admissions)
        };
        assert_eq!(submit(&mut q, &mut budget, &mut bars, 2, &mut scratch), (0, 5), "tripped: no hint went, admissions still go");
        assert!(bars.tripped && bars.sent == 0);
        assert_eq!(budget.used, 6, "charged for the 5 admissions and the one paced restore only");
        assert_eq!(q.len(), 30, "all re-queued");
        assert!(!q.sent(word(5, 0)));
        // Submit 2: a tripped step takes nothing more (the queue is untouched).
        assert_eq!(submit(&mut q, &mut budget, &mut bars, 3, &mut scratch), (0, 0));
        assert_eq!(q.len(), 30);
        // A fresh step with the bar at 100: 30 go (hints first: the plan gives
        // them the budget before admissions), then 40 more fresh words: 24 go
        // (60 - 36 used), 16 are re-queued and come back FIRST next submit.
        let c = Cfg { max_words_step: 100, ..c };
        q.begin_step(2);
        budget.reset(60);
        bars.reset();
        q.push(2, &words);
        let mut submit2 = |q: &mut HintQueue, budget: &mut SpecBudget, bars: &mut Bars, layer: i32, scratch: &mut Vec<u32>| -> (usize, usize) {
            if bars.tripped {
                return (0, 0);
            }
            q.take_into(2, layer, 128, scratch);
            let mut t = budget.plan(scratch.len(), 5, 100, 128, true, 16);
            let (go, _) = bars.allow(t.hints, c.max_words_step, c.per_ll_x10);
            t.hints = go;
            budget.commit(&t);
            q.mark_sent(&scratch[..go]);
            q.requeue(&scratch[go..]);
            (go, t.admissions)
        };
        assert_eq!(submit2(&mut q, &mut budget, &mut bars, 2, &mut scratch), (30, 5));
        assert!(q.sent(word(5, 29)) && q.is_empty());
        let more: Vec<u32> = (100..140).map(|e| word(6, e)).collect();
        q.push(2, &more);
        assert_eq!(submit2(&mut q, &mut budget, &mut bars, 3, &mut scratch), (24, 0), "60 - 36 = 24 hints, nothing left for admissions");
        assert_eq!((bars.sent, q.len()), (54, 16));
        assert_eq!(q.take(2, 3, 128).0, more[24..].to_vec(), "the re-queued surplus comes back first, in order");
    }

    /// The queue's `sent` set and the surplus re-queue (design 2.4: never drop).
    #[test]
    fn queue_marks_sent_and_requeues_the_surplus() {
        let mut q = HintQueue::new();
        q.begin_step(3);
        q.push(3, &[word(5, 1), word(5, 2), word(6, 3)]);
        let (taken, _) = q.take(3, 2, 8);
        assert_eq!(taken, vec![word(5, 1), word(5, 2), word(6, 3)]);
        q.mark_sent(&taken[..1]);
        q.requeue(&taken[1..]);
        assert!(q.sent(word(5, 1)) && !q.sent(word(5, 2)));
        assert_eq!(q.len(), 2);
        assert_eq!(q.take(3, 2, 8).0, vec![word(5, 2), word(6, 3)], "in order, at the front");
        q.push(3, &[word(7, 7)]);
        q.requeue(&[word(5, 2)]);
        assert_eq!(q.take(3, 2, 8).0, vec![word(5, 2), word(7, 7)]);
        assert!(q.hinted(word(5, 2)), "re-queued words stay deduped");
        q.begin_step(4);
        assert!(!q.sent(word(5, 1)), "a new step forgets the sent set");
        q.mark_sent(&[word(5, 1)]);
        q.clear();
        assert!(!q.sent(word(5, 1)));
    }

    #[test]
    fn counters_take_and_clear_in_stats_order() {
        // Process-wide counters: tolerate other tests' bumps by reading twice.
        let _ = take_stats();
        count_preds(&[pred(1, 1, true), pred(2, 2, false)]);
        count_dry_hits(&DryHits { hits: [1, 1, 1], hits_prot: [0, 1, 1], hits_prot_m: [1, 1, 0, 0], ..DryHits::default() });
        bump(Stat::DryWords, 2);
        bump(Stat::Stale, 0);
        bump(Stat::BudgetDeferred, 7);
        let s = take_stats();
        assert_eq!(&s[..9], &[1, 2, 2, 1, 1, 1, 1, 1, 1]);
        assert_eq!(s[Stat::DryWords as usize], 2);
        assert_eq!(s[Stat::HintsSent as usize], 0);
        assert_eq!(s[Stat::Stale as usize], 0);
        assert_eq!(s[Stat::BudgetDeferred as usize], 7);
        assert_eq!(take_stats(), [0; STATS.len()]);
        // The `Stat` discriminants index `STATS` by hand: pin all five.
        assert_eq!(STATS[Stat::DryWords as usize].0, "lh2_dry_words");
        assert_eq!(STATS[Stat::HintsSent as usize].0, "lh2_hints_sent");
        assert_eq!(STATS[Stat::DroppedCap as usize].0, "lh2_dropped_cap");
        assert_eq!(STATS[Stat::Stale as usize].0, "lh2_stale");
        assert_eq!(STATS[Stat::BudgetDeferred as usize].1, "lh2.budget_deferred");
    }
}
