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
//! SLICE A (design 8.A): hub-only. No word reaches box 2: `dry` (and, until
//! slice B, `k1`/`k2`) launch the look-ahead, filter, queue and count. The
//! only behaviour it adds is the look-ahead launches and host bookkeeping
//! (I1/I2, section 7): nothing here touches a pick, a weight or a kernel input.
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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

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

    /// May the queued words go on the wire? SLICE B: `k1` / `k2` AND the
    /// daemon has answered `RESP_FLAG_NURSERY` (design 3.2, "capability = a
    /// RESP flag": an echoed request bit proves nothing). Slice A: never.
    pub fn wire(self) -> bool {
        let _ = self;
        false
    }
}

/// One decode step's knob values, read by `begin_step` and copied by every
/// lane-layer (`PreMoeCarry::mp`) and every decode submit. `step` = the
/// mirror's step clock (`b2_mirror::begin_step`), the tag on queued words.
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
}

impl Cfg {
    fn pack(self) -> u64 {
        (self.mode as u64) | (u64::from(self.rank) << 4) | (u64::from(self.cap) << 8) | (u64::from(self.spec_budget) << 16) | (u64::from(self.step) << 32)
    }

    fn unpack(v: u64) -> Cfg {
        Cfg {
            mode: Mode::from_knob((v & 0xf) as usize),
            rank: ((v >> 4) & 0xf) as u8,
            cap: ((v >> 8) & 0xff) as u8,
            spec_budget: ((v >> 16) & 0xffff) as u16,
            step: (v >> 32) as u32,
        }
    }
}

/// The current step's `Cfg`, packed (one relaxed load per read).
static CFG: AtomicU64 = AtomicU64::new(0);
static WARNED_SLICE_B: AtomicBool = AtomicBool::new(false);

/// The step's knob snapshot (`begin_step`); the defaults before any step.
pub fn cfg() -> Cfg {
    Cfg::unpack(CFG.load(Relaxed))
}

/// Anything to count this step (the `hub_lh2` record is emitted iff so).
pub fn active() -> bool {
    let c = cfg();
    c.mode.on() || c.spec_budget > 0
}

/// A decode step begins (right after `b2_mirror::begin_step`, which advanced
/// `step`): read the four knobs through the framework into this step's
/// `Cfg`, drop the hint queue's earlier-step words, clear its dedup set and
/// reset the speculative budget.
pub fn begin_step(step: u32) {
    let k = &crate::knobs::B2_MISS_PREFETCH;
    let c = Cfg {
        mode: Mode::from_knob(k.pick()),
        rank: crate::knobs::B2_MISS_PREFETCH_RANK.get() as u8,
        cap: crate::knobs::B2_MISS_PREFETCH_CAP.get() as u8,
        spec_budget: crate::knobs::B2_SPEC_BUDGET.get() as u16,
        step,
    };
    CFG.store(c.pack(), Relaxed);
    if matches!(c.mode, Mode::K1 | Mode::K2) && !WARNED_SLICE_B.swap(true, Relaxed) {
        tracing::warn!(knob = k.name, value = %k.show(), "predicted-miss prefetch: slice B (hints on the wire) is not built; behaving as `dry`");
    }
    let stale = super::remote_experts::hint_queue_begin_step(step);
    bump(Stat::Stale, u64::from(stale));
    BUDGET.lock().unwrap_or_else(|p| p.into_inner()).reset(u32::from(c.spec_budget));
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
}

/// Layer `target`'s predicted picks, DISTINCT by expert, box 2's only
/// (`is_box2`), best rank first, then first appearance. `look` is
/// `[rows][n_used]` in the router's descending selection order (rank r is
/// column r-1), as `forward_prefill` unpacks `sd.look_sel`; ids outside
/// `0..N_EXPERT` (NO_PICK, padding) are skipped.
pub fn classify(look: &[i32], n_used: usize, is_box2: impl Fn(u32) -> bool, nonres: impl Fn(u32) -> bool) -> Vec<Pred> {
    if n_used == 0 {
        return Vec::new();
    }
    // Per expert: best rank (0 = unseen) and the index of its first appearance.
    let mut best = [0u8; NE];
    let mut first = [u32::MAX; NE];
    for (i, &sv) in look.iter().enumerate() {
        if !(0..N_EXPERT as i32).contains(&sv) {
            continue;
        }
        let e = sv as usize;
        let rank = ((i % n_used) + 1).min(255) as u8;
        if best[e] == 0 {
            if !is_box2(e as u32) {
                // Mark box 1's so the ownership test runs once per expert.
                best[e] = 255;
                first[e] = u32::MAX;
                continue;
            }
            best[e] = rank;
            first[e] = i as u32;
        } else if rank < best[e] {
            best[e] = rank;
        }
    }
    let mut out: Vec<(u8, u32, u16)> = (0..NE).filter(|&e| best[e] != 0 && first[e] != u32::MAX).map(|e| (best[e], first[e], e as u16)).collect();
    out.sort_unstable();
    out.into_iter().map(|(rank, _, e)| Pred { e, rank, nonres: nonres(u32::from(e)) }).collect()
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
    let mut out = Vec::with_capacity(cap.min(16));
    let mut dropped = 0u32;
    for p in preds.iter().filter(|p| p.nonres && p.rank <= rank) {
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

/// What a lane predicted for its next layer, kept on the lane's scratch
/// (`BatchDgpuScratch::lh2_pending`) until that layer routes. Per lane so the
/// two lanes' predictions never cross: a lane-layer's look-ahead for L+1 is
/// matched against the SAME lane's layer-L+1 picks on its next lane-layer,
/// whatever the ready-first interleaving or the two-stream cut does between.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pending {
    pub layer: i32,
    pub step: u32,
    /// The non-resident predictions only.
    pub preds: Vec<Pred>,
}

/// Layer `pending.layer` has routed: per R in 1..=3, how many of the
/// non-resident predictions of rank `<= R` are among the router's OWN picks
/// `own` (`[rows][n_used]`; `sel_orig` under a live cache prior, else
/// `sel_host`), distinct experts. The live Step 0 (design sections 1 and 6):
/// recall on today's exposed misses, not the 09-14 cold tail.
pub fn dry_hits(preds: &[Pred], own: &[i32]) -> [u32; 3] {
    let mut seen = [false; NE];
    for &sv in own {
        if (0..N_EXPERT as i32).contains(&sv) {
            seen[sv as usize] = true;
        }
    }
    let mut hits = [0u32; 3];
    for p in preds.iter().filter(|p| p.nonres && seen[p.e as usize]) {
        for r in (p.rank as usize).max(1)..=3 {
            hits[r - 1] += 1;
        }
    }
    hits
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
/// or already handed out, for the filter's dedup.
#[derive(Debug, Default)]
pub struct HintQueue {
    /// `(word, step)`, oldest first.
    words: VecDeque<(u32, u32)>,
    /// Bit `(layer, e)`: hinted this step.
    hinted: Vec<u64>,
    step: u32,
}

impl HintQueue {
    pub fn new() -> Self {
        Self { words: VecDeque::new(), hinted: vec![0; LAYERS * WORDS], step: 0 }
    }

    fn bit(w: u32) -> Option<(usize, u64)> {
        let (l, e) = ((w >> 16) as usize, (w & 0xffff) as usize);
        (l < LAYERS && e < NE).then(|| (l * WORDS + e / 64, 1u64 << (e % 64)))
    }

    /// Queued or handed out this step.
    pub fn hinted(&self, w: u32) -> bool {
        Self::bit(w).is_some_and(|(i, m)| self.hinted[i] & m != 0)
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
        let mut stale = 0u32;
        let mut keep = VecDeque::with_capacity(self.words.len());
        for (w, s) in self.words.drain(..) {
            if s != step || (w >> 16) as i32 <= layer {
                stale += 1;
            } else if out.len() < max {
                out.push(w);
            } else {
                keep.push_back((w, s));
            }
        }
        self.words = keep;
        (out, stale)
    }

    /// A new step: drop the earlier step's words (counted stale) and forget
    /// what was hinted.
    pub fn begin_step(&mut self, step: u32) -> u32 {
        let before = self.words.len();
        self.words.retain(|&(_, s)| s == step);
        self.hinted.iter_mut().for_each(|w| *w = 0);
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

    /// Restores per request: the budget spread over ONE lane's requests of a
    /// step (`N_LAYER`), at least 1 -- ~1-2 at the default 60, so a restore
    /// burst paces out over the step even with a single lane running.
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

/// `SpecBudget::commit` on this step's budget, and the `lh2_*` counters it moves.
pub fn budget_commit(t: &Take) {
    BUDGET.lock().unwrap_or_else(|p| p.into_inner()).commit(t);
    bump(Stat::HintsSent, t.hints as u64);
    bump(Stat::BudgetDeferred, u64::from(t.deferred));
}

// ---- counters (design section 6) ----

/// `(hub_lh2 field, ms.stage host stage)` per counter, in `take_stats` order:
/// per R in 1..=3 the distinct box-2-owned predictions of rank `<= R`, the
/// mirror-non-resident ones among them, and of THOSE the ones in the router's
/// own picks one lane-layer later; then hint words the queue handed a decode
/// submit (slice A: counted, kept off the wire), words on the wire (0 in slice
/// A), words the per-request cap dropped, words dropped stale, and admission /
/// restore words the per-step budget held back vs today's rule.
pub const STATS: [(&str, &str); 14] = [
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
];

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
}

pub fn bump(s: Stat, n: u64) {
    if n != 0 {
        COUNTS[s as usize].fetch_add(n, Relaxed);
    }
}

/// One lane-layer's L+1 predictions classified: `lh2_cand_rN`, `lh2_nonres_rN`.
pub fn count_preds(preds: &[Pred]) {
    let (cand, nonres) = count(preds);
    for r in 0..3 {
        COUNTS[r].fetch_add(u64::from(cand[r]), Relaxed);
        COUNTS[3 + r].fetch_add(u64::from(nonres[r]), Relaxed);
    }
}

/// One lane-layer's `dry_hits`: `lh2_dry_hits_rN`.
pub fn count_dry_hits(hits: [u32; 3]) {
    for r in 0..3 {
        COUNTS[6 + r].fetch_add(u64::from(hits[r]), Relaxed);
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
        Pred { e, rank, nonres }
    }

    #[test]
    fn cfg_packs_and_unpacks() {
        let c = Cfg { mode: Mode::K2, rank: 3, cap: 200, spec_budget: 65535, step: u32::MAX - 7 };
        assert_eq!(Cfg::unpack(c.pack()), c);
        assert_eq!(Cfg::unpack(0), Cfg::default());
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
        assert!(matches!(B2_SPEC_BUDGET.kind, crate::knobs::Kind::Int { default: 60, .. }));
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
            assert!(!m.wire(), "{m:?}: slice A puts nothing on the wire");
        }
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
    }

    #[test]
    fn counters_take_and_clear_in_stats_order() {
        // Process-wide counters: tolerate other tests' bumps by reading twice.
        let _ = take_stats();
        count_preds(&[pred(1, 1, true), pred(2, 2, false)]);
        count_dry_hits([1, 1, 1]);
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
        assert_eq!(STATS[Stat::DryWords as usize].0, "lh2_dry_words");
        assert_eq!(STATS[Stat::BudgetDeferred as usize].1, "lh2.budget_deferred");
    }
}
