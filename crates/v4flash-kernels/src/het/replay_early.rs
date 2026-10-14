//! EARLY reads for the CED replay's late picks (`V41_PREFILL_EARLY_REPLAY`, docs/v41/PREFILL_COLD_ASYNC_DESIGN.md
//! section 12): the host-side pieces, none of which touch a GPU (host-tested below).
//!
//! - the replay-pick statistics the plan ranks by (decayed per-(layer, expert) counts of the replay's router picks);
//! - the PLAN: per decoder layer, the highest-count experts that are cold right now (box-2 home, not held by box 2,
//!   not on box 1) -- the experts the replay will most likely give box 1 to read late;
//! - [`EarlyBook`]: what was issued / is reading / arrived / sits landed in a HIDDEN pool slot / was adopted by the
//!   replay's `late_read`, and the per-job counters;
//! - [`ReadQueue`]: the late readers' queue with a demand lane and an early lane (demand first; at most `early_cap`
//!   early reads at once).
//!
//! WHAT IS COMPUTED DOES NOT CHANGE: an early read is invisible to routing until the replay's `late_read` asks for the
//! very pick the knob-off path would read late (same classification, same late set, same late pass); it then serves
//! that read from bytes already landed (or still landing). Its hidden slot is published at the late join, the point
//! where the knob-off path lands the same expert.

use crate::config::{CED_DECODER_START, N_EXPERT, N_LAYER};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};

/// Decoder layers (the replay's) and experts per layer.
const DL: usize = N_LAYER as usize - CED_DECODER_START;
const NE: usize = N_EXPERT as usize;
/// Fixed point of the decayed counts: one pick = 16.
const ONE: u32 = 16;

static COUNTS: [AtomicU32; DL * NE] = [const { AtomicU32::new(0) }; DL * NE];
/// Collecting replay picks: set once the early stages exist (`ExpertPager::start_early_stages`), so a deploy without
/// them runs exactly today's code.
static COLLECT: AtomicBool = AtomicBool::new(false);

pub fn set_collecting(on: bool) {
    COLLECT.store(on, Relaxed);
}

pub fn collecting() -> bool {
    COLLECT.load(Relaxed)
}

/// One replay lane-layer's router picks (`sel`, -1 = none) at decoder `layer`; ignored below the split.
pub fn note_picks(layer: usize, sel: &[i32]) {
    let Some(dl) = layer.checked_sub(CED_DECODER_START).filter(|&d| d < DL) else { return };
    for &e in sel {
        if (0..N_EXPERT as i32).contains(&e) {
            let c = &COUNTS[dl * NE + e as usize];
            c.store(c.load(Relaxed).saturating_add(ONE), Relaxed);
        }
    }
}

/// At each replay's start: every count x 4/5 (a few replays of memory; the plan follows the conversation's drift).
pub fn decay() {
    for c in COUNTS.iter() {
        let v = c.load(Relaxed);
        if v != 0 {
            c.store(decayed(v), Relaxed);
        }
    }
}

#[inline]
fn decayed(v: u32) -> u32 {
    (v as u64 * 4 / 5) as u32
}

/// A copy of the counts, `[decoder layer][expert]`.
pub fn counts_snapshot() -> Vec<u32> {
    COUNTS.iter().map(|c| c.load(Relaxed)).collect()
}

/// THE PLAN from `counts` (`[decoder layer][expert]`, as [`counts_snapshot`]): for each decoder layer in ascending order
/// (the replay needs layer 20 first), the `per_layer` candidates (`cand(layer, e)`) with the highest count > 0, ties
/// by expert id. Returns absolute (layer, expert) in issue order.
pub fn plan_from_counts(counts: &[u32], per_layer: usize, cand: impl Fn(i32, u32) -> bool) -> Vec<(i32, u32)> {
    let mut out = Vec::new();
    if per_layer == 0 || counts.len() < DL * NE {
        return out;
    }
    for dl in 0..DL {
        let layer = (CED_DECODER_START + dl) as i32;
        let row = &counts[dl * NE..(dl + 1) * NE];
        let mut c: Vec<(u32, u32)> = (0..NE as u32).filter(|&e| row[e as usize] > 0 && cand(layer, e)).map(|e| (row[e as usize], e)).collect();
        c.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        out.extend(c.into_iter().take(per_layer).map(|(_, e)| (layer, e)));
    }
    out
}

/// The late readers' queue: DEMAND jobs (late reads a unit waits for) always first; an EARLY job only while fewer than
/// `early_cap` early reads are running, so demand reads keep most readers and the disk. With no early job ever queued
/// it is exactly the old FIFO (`pop` = the demand front).
pub struct ReadQueue<T> {
    pub demand: VecDeque<T>,
    pub early: VecDeque<T>,
    /// Jobs popped and still reading (both kinds) / of them early.
    pub running: usize,
    pub running_early: usize,
    pub early_cap: usize,
}

impl<T> Default for ReadQueue<T> {
    fn default() -> Self {
        ReadQueue { demand: VecDeque::new(), early: VecDeque::new(), running: 0, running_early: 0, early_cap: 0 }
    }
}

impl<T> ReadQueue<T> {
    /// The next job for an idle reader, and whether it is early.
    pub fn pop(&mut self) -> Option<(T, bool)> {
        if let Some(j) = self.demand.pop_front() {
            self.running += 1;
            return Some((j, false));
        }
        if self.running_early < self.early_cap {
            if let Some(j) = self.early.pop_front() {
                self.running += 1;
                self.running_early += 1;
                return Some((j, true));
            }
        }
        None
    }

    /// A popped job finished.
    pub fn finish(&mut self, early: bool) {
        self.running = self.running.saturating_sub(1);
        if early {
            self.running_early = self.running_early.saturating_sub(1);
        }
    }

    /// Demand reads still running (the pass-end cancel waits for these only; early reads may cross pass boundaries).
    pub fn running_demand(&self) -> usize {
        self.running.saturating_sub(self.running_early)
    }
}

/// What `late_read` finds for a pick it is about to read late.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Adopt {
    /// No early read: read it now (today's path).
    No,
    /// Landed in hidden slot `slot`: the join publishes it, no read, no stage.
    Landed(u32),
    /// Still reading: the join waits for it as for a late read of its own.
    Reading,
    /// Read, not landed yet (the result waits in the reader's early-arrived list): the join lands it.
    Arrived,
}

/// What to do with an early read's result when it arrives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arrival {
    /// Adopted while reading: hand it to the join (the late `done` map).
    ToJoin,
    /// Keep it: land it in a hidden slot at the next pump.
    Keep,
    /// Stale (another job / its layer settled / read failed): drop it, the stage goes back.
    Drop,
}

/// Per-job early counters (`prefill_job_replay` line).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EarlyStats {
    /// Early reads queued / read OK / landed in a hidden slot / refused a slot (no eligible victim).
    pub issued: u64,
    pub arrived: u64,
    pub landed: u64,
    pub land_refused: u64,
    /// Late reads the replay served from an early read: landed (no read in the replay) / still reading or not yet
    /// landed (the read started early, the join waits for the rest).
    pub used_landed: u64,
    pub used_inflight: u64,
    /// Early reads never used: dropped results + hidden slots released.
    pub wasted: u64,
    pub failed: u64,
    /// Of `landed`: box-1-home experts landed visible (the read-ahead part; the main pass's `ensure` hits them).
    pub readahead: u64,
}

impl EarlyStats {
    pub fn used(&self) -> u64 {
        self.used_landed + self.used_inflight
    }
}

/// The early bookkeeping of one prefill job (keys are absolute (layer, expert)).
#[derive(Default)]
pub struct EarlyBook {
    pub active: bool,
    pub job: u64,
    /// Bumped by every begin / finish: a result of an older generation is stale.
    pub gen: u32,
    pub plan: VecDeque<(i32, u32)>,
    /// Issued, result not back yet (not adopted).
    reading: HashSet<(i32, u32)>,
    /// Result back, waiting to be landed (not adopted).
    arrived: HashSet<(i32, u32)>,
    /// Adopted by `late_read` while reading: the join takes the result.
    adopted_reading: HashSet<(i32, u32)>,
    /// Landed in a hidden slot (not adopted).
    hidden: HashMap<(i32, u32), u32>,
    /// Adopted by `late_read` after landing: the join publishes the slot.
    adopted_landed: HashMap<(i32, u32), u32>,
    /// Results and hidden slots of layers below this are dropped (the replay has routed them).
    pub settled_below: i32,
    /// Keys landed VISIBLE (box-1 home: the read-ahead set), see `set_visible`.
    visible: HashSet<(i32, u32)>,
    pub stats: EarlyStats,
}

impl EarlyBook {
    /// Start a job's plan. Returns the hidden slots of the previous job to release (its leftovers, counted wasted).
    pub fn begin(&mut self, job: u64, plan: Vec<(i32, u32)>) -> Vec<((i32, u32), u32)> {
        let old = self.finish();
        self.active = true;
        self.job = job;
        self.plan = plan.into();
        self.settled_below = i32::MIN;
        self.stats = EarlyStats::default();
        old
    }

    /// End the job's early reads: everything not adopted is wasted; returns the hidden (and adopted-but-unjoined) slots
    /// to release. Results still reading arrive as stale (`gen` moves on).
    pub fn finish(&mut self) -> Vec<((i32, u32), u32)> {
        let mut rel: Vec<((i32, u32), u32)> = self.hidden.drain().collect();
        rel.extend(self.adopted_landed.drain());
        // Issued and never used: landed (hidden), arrived, or still reading. Unissued plan entries cost nothing.
        self.stats.wasted += rel.len() as u64 + self.arrived.len() as u64 + self.reading.len() as u64;
        self.reading.clear();
        self.arrived.clear();
        self.adopted_reading.clear();
        self.plan.clear();
        self.visible.clear();
        self.gen = self.gen.wrapping_add(1);
        self.active = false;
        self.settled_below = i32::MAX;
        rel.sort();
        rel
    }

    /// The next planned key to read (skipping what is no longer eligible -- resident, in a late read, already early),
    /// registered as reading. LAYER-AHEAD like the prefill read-ahead: only layers below `frontier() + depth` (the
    /// plan is in layer order, so a key past the window waits at the front).
    pub fn next_issue(&mut self, depth: i32, eligible: impl Fn((i32, u32)) -> bool) -> Option<(i32, u32)> {
        let end = self.frontier().saturating_add(depth.max(1));
        while let Some(&k) = self.plan.front() {
            if k.0 >= end {
                return None;
            }
            self.plan.pop_front();
            if k.0 < self.settled_below || self.known(k) || !eligible(k) {
                continue;
            }
            self.reading.insert(k);
            self.stats.issued += 1;
            return Some(k);
        }
        None
    }

    /// The first decoder layer the replay has not routed yet (the encoder: the first decoder layer).
    pub fn frontier(&self) -> i32 {
        self.settled_below.max(CED_DECODER_START as i32)
    }

    /// Mark keys whose landing is VISIBLE (published at once): box-1-HOME experts, which the main pass reads whatever
    /// their residency (the prefill read-ahead's own set); everything else lands hidden.
    pub fn set_visible(&mut self, keys: impl IntoIterator<Item = (i32, u32)>) {
        self.visible.extend(keys);
    }

    pub fn is_visible(&self, k: (i32, u32)) -> bool {
        self.visible.contains(&k)
    }

    /// A visible (read-ahead) key landed and was published at once.
    pub fn landed_visible(&mut self) {
        self.stats.landed += 1;
        self.stats.readahead += 1;
    }

    /// An early read is under way or done for `k` (any state).
    pub fn known(&self, k: (i32, u32)) -> bool {
        self.reading.contains(&k) || self.arrived.contains(&k) || self.adopted_reading.contains(&k) || self.hidden.contains_key(&k)
            || self.adopted_landed.contains_key(&k)
    }

    /// `late_read` is about to read `k` late: take over an early read of it, if any.
    pub fn adopt(&mut self, k: (i32, u32)) -> Adopt {
        if !self.active {
            return Adopt::No;
        }
        if let Some(slot) = self.hidden.remove(&k) {
            self.adopted_landed.insert(k, slot);
            self.stats.used_landed += 1;
            return Adopt::Landed(slot);
        }
        if self.reading.remove(&k) {
            self.adopted_reading.insert(k);
            self.stats.used_inflight += 1;
            return Adopt::Reading;
        }
        if self.arrived.remove(&k) {
            self.stats.used_inflight += 1;
            return Adopt::Arrived;
        }
        Adopt::No
    }

    /// An early result for `k` of generation `gen` arrived (`ok` = read without error).
    pub fn on_arrival(&mut self, k: (i32, u32), gen: u32, ok: bool) -> Arrival {
        if gen != self.gen {
            return Arrival::Drop;
        }
        if self.adopted_reading.remove(&k) {
            return Arrival::ToJoin;
        }
        if !self.reading.remove(&k) {
            return Arrival::Drop;
        }
        if !ok {
            self.stats.failed += 1;
            self.stats.wasted += 1;
            return Arrival::Drop;
        }
        self.stats.arrived += 1;
        if k.0 < self.settled_below {
            self.stats.wasted += 1;
            return Arrival::Drop;
        }
        self.arrived.insert(k);
        Arrival::Keep
    }

    /// An arrived result is about to be landed: still wanted? (`false` -> dropped, counted wasted.)
    pub fn landing(&mut self, k: (i32, u32)) -> bool {
        if !self.arrived.remove(&k) {
            return false;
        }
        if k.0 < self.settled_below {
            self.stats.wasted += 1;
            return false;
        }
        true
    }

    /// An issued early read was taken off the queue before any reader started it (settled / finished): forget it; it
    /// read nothing, so it is not counted issued.
    pub fn drop_unread(&mut self, k: (i32, u32)) {
        if self.reading.remove(&k) {
            self.stats.issued = self.stats.issued.saturating_sub(1);
        }
    }

    /// An arrived result's stage was taken for something else before it landed (its data is gone): forget it.
    pub fn drop_arrived(&mut self, k: (i32, u32)) {
        if self.arrived.remove(&k) {
            self.stats.wasted += 1;
        }
    }

    /// An arrived result was dropped without landing (no victim, or no longer eligible).
    pub fn land_refused(&mut self) {
        self.stats.land_refused += 1;
        self.stats.wasted += 1;
    }

    /// Landed `k` in hidden slot `slot`.
    pub fn landed(&mut self, k: (i32, u32), slot: u32) {
        self.stats.landed += 1;
        self.hidden.insert(k, slot);
    }

    /// The join: the hidden slot of an adopted-landed `k`, to publish.
    pub fn take_adopted_landed(&mut self, k: (i32, u32)) -> Option<u32> {
        self.adopted_landed.remove(&k)
    }

    /// Is `k` adopted while reading (its result goes to the join)?
    pub fn adopted_reading(&self, k: (i32, u32)) -> bool {
        self.adopted_reading.contains(&k)
    }

    /// Forget an adopted-while-reading `k` (its result was taken some other way).
    pub fn forget_adopted(&mut self, k: (i32, u32)) {
        self.adopted_reading.remove(&k);
    }

    /// A late pass ended (`late_cancel_all`): adoptions it never joined (an error path) are void -- adopted-landed slots
    /// to release, adopted-while-reading results drop on arrival. Counted wasted.
    pub fn unjoined(&mut self) -> Vec<((i32, u32), u32)> {
        let mut rel: Vec<((i32, u32), u32)> = self.adopted_landed.drain().collect();
        rel.sort();
        self.stats.wasted += rel.len() as u64 + self.adopted_reading.len() as u64;
        self.adopted_reading.clear();
        rel
    }

    /// The replay has routed every layer below `layer`: hidden slots of those layers are released (wasted); later
    /// results of those layers drop on arrival. Returns the slots to release.
    pub fn settle_below(&mut self, layer: i32) -> Vec<((i32, u32), u32)> {
        if layer <= self.settled_below {
            return Vec::new();
        }
        self.settled_below = layer;
        let mut rel: Vec<((i32, u32), u32)> = self.hidden.iter().filter(|(k, _)| k.0 < layer).map(|(&k, &s)| (k, s)).collect();
        rel.sort();
        for (k, _) in &rel {
            self.hidden.remove(k);
        }
        self.stats.wasted += rel.len() as u64;
        self.plan.retain(|k| k.0 >= layer);
        rel
    }

    /// Early reads issued and not resolved (reading or arrived), for tests and logs.
    pub fn pending(&self) -> usize {
        self.reading.len() + self.arrived.len()
    }

    /// Hidden slots now (landed, adopted or not).
    pub fn hidden_slots(&self) -> usize {
        self.hidden.len() + self.adopted_landed.len()
    }
}

/// The sentinel key a HIDDEN slot holds in the pager's residency maps: never equal to a real `(layer, expert)`, so
/// `is_resident` stays false and no lookup can hit it until the join publishes the slot under its real key.
pub const HIDDEN_LAYER_BASE: i32 = 1 << 16;

pub fn hidden_key(k: (i32, u32)) -> (i32, u32) {
    (k.0 + HIDDEN_LAYER_BASE, k.1)
}

pub fn is_hidden_key(k: (i32, u32)) -> bool {
    k.0 >= HIDDEN_LAYER_BASE
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts_with(entries: &[(usize, u32, u32)]) -> Vec<u32> {
        let mut c = vec![0u32; DL * NE];
        for &(l, e, v) in entries {
            c[(l - CED_DECODER_START) * NE + e as usize] = v;
        }
        c
    }

    #[test]
    fn plan_ranks_per_layer_ascending_and_filters() {
        let s = CED_DECODER_START;
        let c = counts_with(&[(s, 5, 10), (s, 7, 30), (s, 9, 30), (s, 11, 1), (s + 1, 3, 4), (s + 1, 4, 0), (s + 19, 383, 2)]);
        // Candidates: everything but expert 9 at the first decoder layer.
        let p = plan_from_counts(&c, 2, |l, e| !(l == s as i32 && e == 9));
        assert_eq!(p, vec![(s as i32, 7), (s as i32, 5), (s as i32 + 1, 3), (s as i32 + 19, 383)]);
        // Ties by expert id; zero counts never planned.
        let p = plan_from_counts(&c, 8, |_, _| true);
        assert_eq!(&p[..4], &[(s as i32, 7), (s as i32, 9), (s as i32, 5), (s as i32, 11)]);
        assert!(!p.contains(&(s as i32 + 1, 4)));
        assert!(plan_from_counts(&c, 0, |_, _| true).is_empty());
    }

    #[test]
    fn decay_is_four_fifths_and_reaches_zero() {
        assert_eq!(decayed(16), 12);
        assert_eq!(decayed(1), 0);
        let mut v = ONE * 100;
        let mut n = 0;
        while v > 0 {
            v = decayed(v);
            n += 1;
        }
        assert!(n < 40, "a count dies out in {n} replays");
    }

    #[test]
    fn note_picks_counts_decoder_layers_only() {
        // Process-global counts: use a layer/expert no other test touches and compare deltas.
        let (l, e) = (CED_DECODER_START + 7, 301u32);
        let before = counts_snapshot()[7 * NE + e as usize];
        note_picks(l, &[e as i32, -1, e as i32, 400]);
        note_picks(CED_DECODER_START - 1, &[e as i32]);
        let after = counts_snapshot()[7 * NE + e as usize];
        assert_eq!(after - before, 2 * ONE);
    }

    #[test]
    fn queue_demand_first_early_capped() {
        let mut q: ReadQueue<u32> = ReadQueue::default();
        q.early.extend([100, 101, 102]);
        // Cap 0: early never runs (the knob-off FIFO).
        assert_eq!(q.pop(), None);
        q.early_cap = 2;
        q.demand.push_back(1);
        assert_eq!(q.pop(), Some((1, false)));
        assert_eq!(q.pop(), Some((100, true)));
        q.demand.push_back(2);
        assert_eq!(q.pop(), Some((2, false)), "demand jumps the early queue");
        assert_eq!(q.pop(), Some((101, true)));
        assert_eq!(q.pop(), None, "two early reads running = the cap");
        assert_eq!((q.running, q.running_early, q.running_demand()), (4, 2, 2));
        q.finish(true);
        assert_eq!(q.pop(), Some((102, true)));
        q.finish(false);
        q.finish(false);
        assert_eq!(q.running_demand(), 0);
        q.finish(true);
        q.finish(true);
        assert_eq!((q.running, q.running_early), (0, 0));
    }

    #[test]
    fn queue_without_early_is_the_old_fifo() {
        let mut q: ReadQueue<u32> = ReadQueue::default();
        q.demand.extend([1, 2, 3]);
        assert_eq!(q.pop(), Some((1, false)));
        assert_eq!(q.pop(), Some((2, false)));
        q.finish(false);
        assert_eq!(q.running_demand(), 1);
        assert_eq!(q.pop(), Some((3, false)));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn book_lifecycle_adopt_landed_reading_arrived() {
        let mut b = EarlyBook::default();
        assert_eq!(b.adopt((20, 1)), Adopt::No, "inactive book adopts nothing");
        let rel = b.begin(7, vec![(20, 1), (20, 2), (20, 3), (21, 4), (22, 5)]);
        assert!(rel.is_empty());
        let g = b.gen;
        // (20, 2) is resident by now: skipped.
        let k1 = b.next_issue(20, |k| k != (20, 2)).unwrap();
        let k3 = b.next_issue(20, |k| k != (20, 2)).unwrap();
        let k4 = b.next_issue(20, |_| true).unwrap();
        let k5 = b.next_issue(20, |_| true).unwrap();
        assert_eq!((k1, k3, k4, k5), ((20, 1), (20, 3), (21, 4), (22, 5)));
        assert_eq!(b.next_issue(20, |_| true), None);
        assert_eq!(b.stats.issued, 4);
        // (20, 1) arrives and lands hidden; (20, 3) arrives, not landed yet; (21, 4) still reading.
        assert_eq!(b.on_arrival((20, 1), g, true), Arrival::Keep);
        assert!(b.landing((20, 1)));
        b.landed((20, 1), 77);
        assert_eq!(b.on_arrival((20, 3), g, true), Arrival::Keep);
        // late_read adopts all three kinds.
        assert_eq!(b.adopt((20, 1)), Adopt::Landed(77));
        assert_eq!(b.adopt((20, 3)), Adopt::Arrived);
        assert_eq!(b.adopt((21, 4)), Adopt::Reading);
        assert_eq!(b.adopt((20, 9)), Adopt::No, "never planned");
        assert_eq!(b.adopt((20, 1)), Adopt::No, "adopted once");
        // The reading one's result goes to the join.
        assert_eq!(b.on_arrival((21, 4), g, true), Arrival::ToJoin);
        assert_eq!(b.take_adopted_landed((20, 1)), Some(77));
        assert_eq!(b.take_adopted_landed((20, 1)), None);
        // (22, 5): its layer settles before it arrives -> dropped, wasted.
        assert!(b.settle_below(22).is_empty());
        assert_eq!(b.on_arrival((22, 5), g, true), Arrival::Keep, "layer 22 not settled");
        let rel = b.settle_below(23);
        assert!(rel.is_empty(), "arrived, not landed: nothing hidden to release");
        assert!(!b.landing((22, 5)), "settled before landing");
        assert_eq!(b.stats.used(), 3);
        assert_eq!((b.stats.used_landed, b.stats.used_inflight), (1, 2));
        assert_eq!(b.stats.wasted, 1);
        let rel = b.finish();
        assert!(rel.is_empty());
        assert!(!b.active);
        assert_eq!(b.pending(), 0);
    }

    #[test]
    fn book_settle_releases_hidden_and_finish_releases_the_rest() {
        let mut b = EarlyBook::default();
        b.begin(1, vec![(20, 1), (25, 2), (30, 3)]);
        let g = b.gen;
        for _ in 0..3 {
            b.next_issue(20, |_| true).unwrap();
        }
        for (k, s) in [((20, 1), 10u32), ((25, 2), 11), ((30, 3), 12)] {
            assert_eq!(b.on_arrival(k, g, true), Arrival::Keep);
            assert!(b.landing(k));
            b.landed(k, s);
        }
        assert_eq!(b.adopt((30, 3)), Adopt::Landed(12));
        assert_eq!(b.settle_below(26), vec![((20, 1), 10), ((25, 2), 11)]);
        assert_eq!(b.hidden_slots(), 1, "the adopted one stays until its join");
        // An error path never joined it: finish releases it.
        assert_eq!(b.finish(), vec![((30, 3), 12)]);
        assert_eq!(b.stats.wasted, 3);
        assert_eq!(b.hidden_slots(), 0);
    }

    #[test]
    fn book_stale_generation_and_failed_reads_drop() {
        let mut b = EarlyBook::default();
        b.begin(1, vec![(20, 1), (20, 2)]);
        let g = b.gen;
        b.next_issue(20, |_| true).unwrap();
        b.next_issue(20, |_| true).unwrap();
        assert_eq!(b.on_arrival((20, 2), g, false), Arrival::Drop);
        assert_eq!(b.stats.failed, 1);
        // A new job begins: the old job's result is stale.
        b.begin(2, vec![(21, 1)]);
        assert_eq!(b.on_arrival((20, 1), g, true), Arrival::Drop);
        assert_ne!(b.gen, g);
        assert_eq!(b.pending(), 0);
        // A key that was never issued (or already resolved) drops.
        assert_eq!(b.on_arrival((21, 1), b.gen, true), Arrival::Drop);
    }

    #[test]
    fn book_skips_known_and_settled_when_issuing() {
        let mut b = EarlyBook::default();
        b.begin(1, vec![(20, 1), (20, 1), (21, 2), (22, 3)]);
        assert_eq!(b.next_issue(20, |_| true), Some((20, 1)));
        b.settle_below(22);
        assert_eq!(b.next_issue(20, |_| true), Some((22, 3)), "duplicate and settled layers skipped");
        assert_eq!(b.next_issue(20, |_| true), None);
    }

    /// LAYER-AHEAD: in the encoder only layers below 20 + depth are issued; as the replay settles layers the window
    /// slides; visible (box-1-home read-ahead) keys are marked.
    #[test]
    fn issue_window_is_layer_ahead() {
        let s = CED_DECODER_START as i32;
        let mut b = EarlyBook::default();
        b.begin(1, vec![(s, 1), (s + 1, 2), (s + 2, 3), (s + 5, 4)]);
        b.set_visible([(s, 1)]);
        assert!(b.is_visible((s, 1)) && !b.is_visible((s + 1, 2)));
        assert_eq!(b.frontier(), s);
        assert_eq!(b.next_issue(2, |_| true), Some((s, 1)));
        assert_eq!(b.next_issue(2, |_| true), Some((s + 1, 2)));
        assert_eq!(b.next_issue(2, |_| true), None, "layer s+2 is outside the window");
        assert_eq!(b.plan.len(), 2, "it waits at the front");
        b.settle_below(s + 1);
        assert_eq!(b.next_issue(2, |_| true), Some((s + 2, 3)));
        assert_eq!(b.next_issue(2, |_| true), None);
        b.settle_below(s + 4);
        assert_eq!(b.next_issue(2, |_| true), Some((s + 5, 4)));
        b.finish();
        assert!(!b.is_visible((s, 1)), "finish clears the visible set");
    }

    #[test]
    fn hidden_keys_never_collide_with_real_ones() {
        let k = (39, 383);
        let h = hidden_key(k);
        assert!(is_hidden_key(h));
        assert!(!is_hidden_key(k));
        assert_ne!(h, k);
        assert!(h.0 >= N_LAYER);
    }
}
