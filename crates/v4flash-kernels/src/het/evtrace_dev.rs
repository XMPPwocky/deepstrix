//! TIER B device intervals (docs/v41/EVTRACE_REBUILD_PLAN.md 2.4, phase P3).
//!
//! `EventPool::reset()` HANDS its buffer (events, pairs with their stage
//! contexts, streams and host stamps, sync marks) to the Tier B thread
//! (`hand_off`) instead of dropping it, and takes a free one: the scheduler
//! thread no longer waits on a harvest, and every reset site is covered --
//! the drafter, the async ring writes and the prefill units included.
//!
//! The Tier B thread (`tier_b_loop`, also the ring's drain):
//! * CALIBRATES each device every 200 ms (`HipCal`): an anchor marker on its
//!   own high-priority non-blocking stream, bracketed by RAW host stamps (spin
//!   <= 2 ms, else the anchor is discarded: a busy hardware queue), chained to
//!   the previous anchor by `elapsed` -> a device-ns -> RAW-ns line
//!   (`CalMath`: slope fitted over ~13 s, clamped +-200 ppm, every link
//!   self-checked; one `cal` record per anchor in Tier A);
//! * CONVERTS each pair once its end event completed (one `elapsed` per event
//!   against the newest anchor; events after it extrapolate);
//! * on a buffer's completion (or after 2 s: the rest dropped, counted) emits a
//!   Tier B `dev` record per pair and, per (step, device), a Tier A `step_dev`
//!   with the device fields `hub_step` carried before, the causality counts
//!   (rule (a) a stage never starts before the host recorded it; rules (b)/(c)
//!   a stage on a stream ends before a sync of that stream returned,
//!   `EventPool::note_sync`) and Tier B's own lag / cost;
//! * RETURNS the buffer with its decode sums (the `ms.stage` rollup).
//!
//! Nothing here can fail a step: errors drop pairs (counted), a pool with no
//! free buffer records nothing for an epoch (`EventPool::take_skipped`).
//! The calibrator owns its own stream and events and enters its device with a
//! `ScopedDevice` -- never the engine's `set_current_cached`, whose cache is
//! engine-wide and would send the scheduler's kernels to the wrong GPU.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use v4flash_hip::{Device, Event, Stream};

use super::evtrace::{self, CAL, DEV};
use super::evtrace_kinds::STEP_DEV;
use super::trace::{Buf, StageCtx};

/// Calibration anchors per device, this often.
const CAL_EVERY: Duration = Duration::from_millis(200);
/// An anchor whose marker has not completed after this long is discarded.
const SPIN_MAX_NS: f64 = 2e6;
/// Added to every anchor's bracket: the marker's timestamp vs its signal's
/// visibility to the host.
const BIAS_NS: f64 = 2e3;
/// Anchors in the slope fit (~13 s at 200 ms).
const HIST: usize = 64;
/// The device / host clock rates may differ by at most this (and this is the
/// slope's bound until the fit spans 1 s).
const MAX_SLOPE: f64 = 200e-6;
/// The fitted slope's bound never goes below this.
const MIN_SLOPE_Q: f64 = 2e-6;
/// A chain link this far beyond its tolerance is a clock jump: the chain
/// restarts at the new anchor.
const JUMP_NS: f64 = 100e3;
/// How often a buffer with pairs left is looked at again.
const POLL: Duration = Duration::from_micros(500);
/// A buffer is returned after this long even with pairs left (dropped).
const TIMEOUT: Duration = Duration::from_secs(2);
/// A device with no handoff for this long is not calibrated (an idle hub,
/// box 2 with its device records off); the next handoff anchors at once.
const CAL_IDLE: Duration = Duration::from_secs(10);
/// An anchor this long after the previous one starts a new chain (f32 ms of
/// a long link is too coarse; the slope is kept).
const CHAIN_GAP: Duration = Duration::from_secs(2);
/// The ring's drain and the stats line.
const DRAIN_EVERY: Duration = Duration::from_millis(100);
const LOG_EVERY: Duration = Duration::from_secs(60);
/// Handoffs queued to the Tier B thread at most (all pools).
const HANDOFF_CAP: usize = 64;

// ---- the handoff ----

pub(crate) struct Handoff {
    pool: u64,
    device: i32,
    label: &'static str,
    /// Events per spare buffer (`EventPool::set_spare_capacity`).
    spare_cap: usize,
    buf: Box<Buf>,
    ret: SyncSender<Box<Buf>>,
}

static TX: OnceLock<SyncSender<Handoff>> = OnceLock::new();

/// Is a pool reset handing its buffer to Tier B? (`V41_EVTRACE_DEV`, live,
/// with the trace and its Tier B on.)
pub fn offload_on() -> bool {
    TX.get().is_some() && super::evtrace_ring::enabled() && evtrace::enabled() && crate::knobs::EVTRACE_DEV.on()
}

/// Buffers per pool (`V41_EVTRACE_DEV_BUFS`).
pub(crate) fn bufs_per_pool() -> usize {
    crate::knobs::EVTRACE_DEV_BUFS.usize().clamp(2, 16)
}

/// Queue `buf` for the Tier B thread; `Err` gives it back (Tier B behind or
/// gone, counted).
pub(crate) fn hand_off(pool: u64, device: i32, label: &'static str, spare_cap: usize, buf: Box<Buf>, ret: SyncSender<Box<Buf>>) -> Result<(), Box<Buf>> {
    let Some(tx) = TX.get() else { return Err(buf) };
    match tx.try_send(Handoff { pool, device, label, spare_cap, buf, ret }) {
        Ok(()) => Ok(()),
        Err(TrySendError::Full(h)) | Err(TrySendError::Disconnected(h)) => {
            STATS.handoff_full.fetch_add(1, Relaxed);
            Err(h.buf)
        }
    }
}

/// The Tier B thread's end of the handoff (`evtrace_ring::init`, once).
pub(crate) fn channel() -> Option<Receiver<Handoff>> {
    let (tx, rx) = std::sync::mpsc::sync_channel(HANDOFF_CAP);
    TX.set(tx).ok()?;
    Some(rx)
}

// ---- counters (the minute line) ----

#[derive(Default)]
struct Stats {
    jobs: AtomicU64,
    pairs: AtomicU64,
    dropped: AtomicU64,
    deferred: AtomicU64,
    viol_a: AtomicU64,
    viol_b: AtomicU64,
    checked_b: AtomicU64,
    anchors_ok: AtomicU64,
    anchors_failed: AtomicU64,
    resid_bad: AtomicU64,
    handoff_full: AtomicU64,
    busy_ns: AtomicU64,
}

static STATS: Stats = Stats {
    jobs: AtomicU64::new(0),
    pairs: AtomicU64::new(0),
    dropped: AtomicU64::new(0),
    deferred: AtomicU64::new(0),
    viol_a: AtomicU64::new(0),
    viol_b: AtomicU64::new(0),
    checked_b: AtomicU64::new(0),
    anchors_ok: AtomicU64::new(0),
    anchors_failed: AtomicU64::new(0),
    resid_bad: AtomicU64::new(0),
    handoff_full: AtomicU64::new(0),
    busy_ns: AtomicU64::new(0),
};

// ---- events and conversion, generic for the host tests ----

pub(crate) trait DevEvent {
    /// `Some(true)` complete, `Some(false)` pending, `None` an error.
    fn done(&self) -> Option<bool>;
}

impl DevEvent for Event {
    fn done(&self) -> Option<bool> {
        self.query().ok()
    }
}

/// A completed event -> (RAW ns, bound ns).
pub(crate) trait ToRaw<E> {
    fn ready(&self) -> bool;
    fn to_raw(&mut self, e: &E) -> Option<(f64, f64)>;
}

/// The device-to-host line of one device: anchors on one chain (device ns
/// since the chain's first anchor, RAW ns, bound ns), newest last.
pub(crate) struct CalMath {
    hist: VecDeque<(f64, f64, f64)>,
    /// RAW ns per device ns, minus 1.
    slope: f64,
    /// Its bound.
    slope_q: f64,
}

impl CalMath {
    pub(crate) fn new() -> Self {
        Self { hist: VecDeque::new(), slope: 0.0, slope_q: MAX_SLOPE }
    }

    pub(crate) fn ready(&self) -> bool {
        !self.hist.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        self.hist.len()
    }

    /// A new anchor at RAW `raw` +-`q`, `link_ms` after the previous one on
    /// the device (`None`: a new chain). Returns the link's self-check
    /// `(residual, tolerance)` in ns; a residual past tolerance + `JUMP_NS`
    /// restarts the chain at this anchor.
    pub(crate) fn add(&mut self, raw: f64, q: f64, link_ms: Option<f32>) -> Option<(f64, f64)> {
        let (Some(el), Some(&(d0, r0, q0))) = (link_ms, self.hist.back()) else {
            self.restart(raw, q);
            return None;
        };
        let dt = el as f64 * 1e6;
        let resid = raw - (r0 + dt * (1.0 + self.slope));
        let tol = q + q0 + dt.abs() * self.slope_q;
        if resid.abs() > tol + JUMP_NS {
            self.restart(raw, q);
        } else {
            self.hist.push_back((d0 + dt, raw, q));
            if self.hist.len() > HIST {
                self.hist.pop_front();
            }
            self.fit();
        }
        Some((resid, tol))
    }

    /// Keeps the slope (the clocks' rate does not change); its bound
    /// widens until the new chain spans 1 s again.
    fn restart(&mut self, raw: f64, q: f64) {
        self.hist.clear();
        self.hist.push_back((0.0, raw, q));
        self.slope_q = MAX_SLOPE;
    }

    fn fit(&mut self) {
        let n = self.hist.len();
        if n < 3 {
            return;
        }
        let (x0, y0) = (self.hist[0].0, self.hist[0].1);
        let nf = n as f64;
        let (mut mx, mut my) = (0.0, 0.0);
        for &(x, y, _) in &self.hist {
            mx += x - x0;
            my += y - y0;
        }
        mx /= nf;
        my /= nf;
        let (mut sxx, mut sxy) = (0.0, 0.0);
        for &(x, y, _) in &self.hist {
            let (dx, dy) = (x - x0 - mx, y - y0 - my);
            sxx += dx * dx;
            sxy += dx * dy;
        }
        if sxx <= 0.0 {
            return;
        }
        let b = sxy / sxx;
        let mut ss = 0.0;
        for &(x, y, _) in &self.hist {
            let r = (y - y0 - my) - b * (x - x0 - mx);
            ss += r * r;
        }
        let se = (ss / (nf - 2.0).max(1.0) / sxx).sqrt();
        let span = self.hist[n - 1].0 - self.hist[0].0;
        self.slope = (b - 1.0).clamp(-MAX_SLOPE, MAX_SLOPE);
        self.slope_q = if span >= 1e9 { (3.0 * se + MIN_SLOPE_Q).min(MAX_SLOPE) } else { MAX_SLOPE };
    }

    /// An event `el_ms` after the newest anchor -> (RAW ns, bound ns).
    pub(crate) fn to_raw(&self, el_ms: f32) -> Option<(f64, f64)> {
        let &(_, r, q) = self.hist.back()?;
        let dt = el_ms as f64 * 1e6;
        Some((r + dt * (1.0 + self.slope), q + dt.abs() * self.slope_q))
    }

    pub(crate) fn slope(&self) -> (f64, f64) {
        (self.slope, self.slope_q)
    }
}

// ---- one buffer's work ----

/// The pairs of one stream, in end-record order (= completion order).
struct Group {
    idx: Vec<u32>,
    next: usize,
}

pub(crate) struct Job<E> {
    device: i32,
    label: &'static str,
    pub(crate) buf: Box<Buf<E>>,
    groups: Vec<Group>,
    /// Per pair: [start RAW, end RAW, bound ns]; NaN = not converted.
    conv: Vec<[f64; 3]>,
    /// Not complete at the first look.
    deferred: Vec<bool>,
    polls: u32,
    left: usize,
}

impl<E: DevEvent> Job<E> {
    pub(crate) fn new(device: i32, label: &'static str, buf: Box<Buf<E>>) -> Self {
        let n = buf.pairs.len();
        let mut by_stream: Vec<(usize, Group)> = Vec::new();
        for (i, p) in buf.pairs.iter().enumerate() {
            match by_stream.iter_mut().find(|g| g.0 == p.stream) {
                Some(g) => g.1.idx.push(i as u32),
                None => by_stream.push((p.stream, Group { idx: vec![i as u32], next: 0 })),
            }
        }
        Self {
            device,
            label,
            buf,
            groups: by_stream.into_iter().map(|g| g.1).collect(),
            conv: vec![[f64::NAN; 3]; n],
            deferred: vec![false; n],
            polls: 0,
            left: n,
        }
    }

    /// Convert the pairs of every stream whose LAST end completed (a stream
    /// completes in order: then all of its pairs have); `true` once none is
    /// left. Only that one event per stream is ever queried: CLR's
    /// `hipEventQuery` on an incomplete event enqueues a notify marker on its
    /// queue (once per event) -- from this thread, onto the scheduler's
    /// streams -- so a stream still running costs one marker per buffer, not
    /// one per pending pair.
    pub(crate) fn poll(&mut self, cv: &mut impl ToRaw<E>) -> bool {
        let first = self.polls == 0;
        self.polls += 1;
        let Job { buf, groups, conv, deferred, left, .. } = self;
        for g in groups.iter_mut() {
            if g.next == g.idx.len() {
                continue;
            }
            let last = g.idx[g.idx.len() - 1] as usize;
            match buf.events[buf.pairs[last].end_idx].done() {
                Some(true) => {}
                Some(false) => {
                    if first {
                        for &i in &g.idx {
                            deferred[i as usize] = true;
                        }
                    }
                    continue;
                }
                // An error: the stream's pairs stay unconverted (dropped).
                None => {
                    *left -= g.idx.len() - g.next;
                    g.next = g.idx.len();
                    continue;
                }
            }
            for &i in &g.idx[g.next..] {
                let p = &buf.pairs[i as usize];
                if let (Some(s), Some(e)) = (cv.to_raw(&buf.events[p.start_idx]), cv.to_raw(&buf.events[p.end_idx])) {
                    conv[i as usize] = [s.0, e.0, s.1.max(e.1)];
                }
            }
            *left -= g.idx.len() - g.next;
            g.next = g.idx.len();
        }
        *left == 0
    }
}

/// Where a stage's time goes in its step's `step_dev` record.
#[derive(Clone, Copy)]
enum Slot {
    /// A parent stage: busy, and its named field (else `other`).
    Parent { busy: usize, field: usize },
    /// The drafter (`mtp.*`).
    Mtp(usize),
    Skip,
}

/// `STEP_DEV` field indices.
struct Fx {
    idx: HashMap<&'static str, usize>,
}

impl Fx {
    fn get() -> &'static Fx {
        static F: OnceLock<Fx> = OnceLock::new();
        F.get_or_init(|| Fx { idx: STEP_DEV.fields.iter().enumerate().map(|(i, f)| (*f, i)).collect() })
    }
    fn i(&self, f: &str) -> usize {
        self.idx[f]
    }
}

/// String ids and stage slots, cached on the Tier B thread (the interner's
/// lock once per name).
#[derive(Default)]
pub(crate) struct Names {
    stage: HashMap<&'static str, f64>,
    stream: HashMap<usize, f64>,
    slot: HashMap<(&'static str, &'static str), Slot>,
}

impl Names {
    fn stage(&mut self, s: &'static str) -> f64 {
        *self.stage.entry(s).or_insert_with(|| evtrace::intern(s))
    }

    fn stream(&mut self, raw: usize) -> f64 {
        *self.stream.entry(raw).or_insert_with(|| match super::trace::stream_name(raw) {
            Some(n) => evtrace::intern(n),
            None => evtrace::intern(&format!("s{}", evtrace::small_id(raw))),
        })
    }

    fn slot(&mut self, label: &'static str, name: &'static str) -> Slot {
        *self.slot.entry((label, name)).or_insert_with(|| {
            let f = Fx::get();
            let (busy, short, mtp) = match label {
                "dgpu" => ("dgpu_busy_ms", "d_", "d_mtp"),
                "igpu" => ("igpu_busy_ms", "i_", "i_mtp"),
                _ => return Slot::Skip,
            };
            if name.starts_with("mtp.") {
                return Slot::Mtp(f.i(mtp));
            }
            match name.strip_prefix(label).and_then(|r| r.strip_prefix('.')) {
                Some(rest) => {
                    let key = format!("{short}{}", rest.replace('.', "_"));
                    let field = f.idx.get(key.as_str()).copied().unwrap_or_else(|| f.i(if label == "dgpu" { "d_other" } else { "i_other" }));
                    Slot::Parent { busy: f.i(busy), field }
                }
                None => Slot::Skip,
            }
        })
    }
}

/// What a finished buffer produced.
#[derive(Default)]
pub(crate) struct Done {
    pub(crate) dev: Vec<[f64; 11]>,
    pub(crate) step_dev: Vec<Vec<f64>>,
    pub(crate) sums: Vec<(&'static str, f64, u32)>,
    pub(crate) sum_steps: u64,
    pub(crate) pairs: u64,
    pub(crate) dropped: u64,
    pub(crate) deferred: u64,
    pub(crate) viol_a: u64,
    pub(crate) viol_b: u64,
    pub(crate) checked_b: u64,
}

fn opt64(x: u64) -> f64 {
    if x == StageCtx::NO { f64::NAN } else { x as f64 }
}

fn opt32(x: u32) -> f64 {
    if x == StageCtx::NO_IDX { f64::NAN } else { x as f64 }
}

fn add(rec: &mut [f64], i: usize, x: f64) {
    if rec[i].is_nan() {
        rec[i] = 0.0;
    }
    rec[i] += x;
}

/// The records and sums of a finished (or timed-out) buffer.
pub(crate) fn finish<E>(job: &Job<E>, names: &mut Names, lag_ms: f64, cost_us: f64) -> Done {
    let f = Fx::get();
    let buf = &job.buf;
    let dev_id = evtrace::intern(job.label);
    let mut d = Done::default();
    let mut steps: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
    let mut sums: HashMap<&'static str, (f64, u32)> = HashMap::new();
    let mut parent_steps: BTreeSet<u64> = BTreeSet::new();
    let new_rec = |step: u64| {
        let mut r = vec![f64::NAN; STEP_DEV.fields.len()];
        r[f.i("step")] = step as f64;
        r[f.i("device")] = dev_id;
        for k in ["pairs", "deferred", "dropped", "viol_a", "viol_b", "checked_b"] {
            r[f.i(k)] = 0.0;
        }
        r
    };
    for (i, p) in buf.pairs.iter().enumerate() {
        let [s, e, q] = job.conv[i];
        let ok = !s.is_nan() && !e.is_nan();
        d.pairs += 1;
        d.deferred += job.deferred[i] as u64;
        let viol_a = ok && p.t_host.is_finite() && s < p.t_host - q;
        if ok {
            d.dev.push([s, e, names.stage(p.name), dev_id, names.stream(p.stream), opt64(p.ctx.step), opt64(p.ctx.unit), opt32(p.ctx.layer), opt32(p.ctx.lane), p.t_host, q / 1e3]);
            d.viol_a += viol_a as u64;
        } else {
            d.dropped += 1;
        }
        if p.ctx.step == StageCtx::NO {
            continue;
        }
        let rec = steps.entry(p.ctx.step).or_insert_with(|| new_rec(p.ctx.step));
        rec[f.i("pairs")] += 1.0;
        rec[f.i("deferred")] += job.deferred[i] as u64 as f64;
        if !ok {
            rec[f.i("dropped")] += 1.0;
            continue;
        }
        let ts = f.i("t_start");
        if rec[ts].is_nan() || s < rec[ts] {
            rec[ts] = s;
        }
        let qm = f.i("q_us_max");
        if rec[qm].is_nan() || q / 1e3 > rec[qm] {
            rec[qm] = q / 1e3;
        }
        rec[f.i("viol_a")] += viol_a as u64 as f64;
        let ms = (e - s) / 1e6;
        match names.slot(job.label, p.name) {
            Slot::Parent { busy, field } => {
                add(rec, busy, ms);
                add(rec, field, ms);
                parent_steps.insert(p.ctx.step);
            }
            Slot::Mtp(i) => add(rec, i, ms),
            Slot::Skip => {}
        }
        if p.ctx.unit == StageCtx::NO {
            let e = sums.entry(p.name).or_insert((0.0, 0));
            e.0 += ms;
            e.1 += 1;
        }
    }
    // Rules (b) / (c): a stage recorded on a stream before a sync of it
    // returned ended (on the device) before that.
    for m in &buf.syncs {
        for (i, p) in buf.pairs.iter().enumerate().take(m.n_pairs) {
            if p.stream != m.stream {
                continue;
            }
            let [s, e, q] = job.conv[i];
            if s.is_nan() || e.is_nan() {
                continue;
            }
            let viol = e > m.t + q;
            d.checked_b += 1;
            d.viol_b += viol as u64;
            if let Some(rec) = steps.get_mut(&p.ctx.step) {
                rec[f.i("checked_b")] += 1.0;
                rec[f.i("viol_b")] += viol as u64 as f64;
            }
        }
    }
    if let Some((_, rec)) = steps.iter_mut().next_back() {
        rec[f.i("lag_ms")] = lag_ms;
        rec[f.i("tierb_us")] = cost_us;
    }
    d.step_dev = steps.into_values().collect();
    d.sums = sums.into_iter().map(|(n, (ms, c))| (n, ms, c)).collect();
    d.sum_steps = parent_steps.len() as u64;
    d
}

// ---- the HIP calibrator ----

struct HipCal {
    label: &'static str,
    stream: Stream,
    ev: [Event; 3],
    /// The newest valid anchor's slot, and when it was taken.
    valid: Option<usize>,
    valid_t: Instant,
    flip: bool,
    math: CalMath,
    /// The last handoff from this device (`CAL_IDLE`).
    last_use: Instant,
}

impl HipCal {
    fn new(device: i32, label: &'static str) -> color_eyre::eyre::Result<Self> {
        let d = Device::new(device);
        let _g = d.scoped_current()?;
        // The highest priority: its own hardware queue where CLR keeps one
        // per priority, so an anchor does not wait behind the compute queue.
        let prio = d.stream_priority_range().map(|(_, greatest)| greatest).unwrap_or(0);
        let stream = Stream::new_non_blocking_with_priority(device, prio)?;
        let ev = [Event::new()?, Event::new()?, Event::new()?];
        Ok(Self { label, stream, ev, valid: None, valid_t: Instant::now(), flip: false, math: CalMath::new(), last_use: Instant::now() })
    }

    /// Take one anchor (bounded spin), chain it, emit its `cal` record.
    fn anchor(&mut self) {
        // Never the newest valid anchor's slot (S2: it stays until a new one is chained).
        let others: Vec<usize> = (0..3).filter(|&i| Some(i) != self.valid).collect();
        self.flip = !self.flip;
        let slot = others[self.flip as usize % others.len()];
        let t0 = evtrace::now();
        let mut lo = t0;
        let mut hi = f64::NAN;
        if self.ev[slot].record(&self.stream).is_ok() {
            loop {
                let s0 = evtrace::now();
                match self.ev[slot].query() {
                    Ok(true) => {
                        hi = evtrace::now();
                        break;
                    }
                    Ok(false) => lo = s0,
                    Err(_) => break,
                }
                if s0 - t0 > SPIN_MAX_NS {
                    break;
                }
                std::hint::spin_loop();
            }
        }
        let dev = evtrace::intern(self.label);
        if hi.is_nan() {
            STATS.anchors_failed.fetch_add(1, Relaxed);
            let mut v = [f64::NAN; 11];
            v[..4].copy_from_slice(&[t0, dev, 0.0, (evtrace::now() - t0) / 1e3]);
            v[10] = self.math.len() as f64;
            evtrace::emit(&CAL, &v);
            return;
        }
        STATS.anchors_ok.fetch_add(1, Relaxed);
        let raw = (lo + hi) / 2.0;
        let q = (hi - lo) / 2.0 + BIAS_NS;
        let link = self.valid.filter(|_| self.valid_t.elapsed() < CHAIN_GAP).and_then(|v| Event::elapsed_ms(&self.ev[v], &self.ev[slot]).ok());
        let chk = self.math.add(raw, q, link);
        self.valid = Some(slot);
        self.valid_t = Instant::now();
        if chk.is_some_and(|(r, t)| r.abs() > t) {
            STATS.resid_bad.fetch_add(1, Relaxed);
        }
        let (slope, slope_q) = self.math.slope();
        let (resid, tol) = chk.unwrap_or((f64::NAN, f64::NAN));
        evtrace::emit(&CAL, &[
            raw, dev, 1.0, (hi - t0) / 1e3, q / 1e3, resid / 1e3, tol / 1e3, slope * 1e6, slope_q * 1e6,
            link.map(|l| l as f64).unwrap_or(f64::NAN), self.math.len() as f64,
        ]);
    }
}

impl ToRaw<Event> for HipCal {
    fn ready(&self) -> bool {
        self.valid.is_some() && self.math.ready()
    }
    fn to_raw(&mut self, e: &Event) -> Option<(f64, f64)> {
        let v = self.valid?;
        let el = Event::elapsed_ms(&self.ev[v], e).ok()?;
        self.math.to_raw(el)
    }
}

// ---- the thread ----

struct Pending {
    job: Job<Event>,
    ret: SyncSender<Box<Buf>>,
    t0: Instant,
    cost: Duration,
}

#[derive(Default)]
struct TierB {
    jobs: VecDeque<Pending>,
    cals: HashMap<i32, HipCal>,
    no_cal: HashSet<i32>,
    pools: HashSet<u64>,
    names: Names,
    next_cal: Option<Instant>,
}

impl TierB {
    fn accept(&mut self, h: Handoff) {
        let t0 = Instant::now();
        if self.pools.insert(h.pool) {
            // The pool's spare buffers, made here (never on the scheduler
            // thread), at the pool's spare capacity (its own by default; box
            // 2's executors ask for 4096: a few events an epoch).
            let cap = h.spare_cap.max(64);
            let made = (|| -> color_eyre::eyre::Result<Vec<Box<Buf>>> {
                let _g = Device::new(h.device).scoped_current()?;
                (1..bufs_per_pool()).map(|_| Buf::new(cap).map(Box::new)).collect()
            })();
            match made {
                Ok(v) => {
                    for b in v {
                        let _ = h.ret.try_send(b);
                    }
                }
                Err(e) => tracing::warn!(device = h.device, error = %e, "evtrace dev: spare event buffers not made (one buffer: most epochs skip)"),
            }
        }
        if let Some(c) = self.cals.get_mut(&h.device) {
            // Back from idle: anchor now (the old anchor is seconds away).
            if c.last_use.elapsed() >= CAL_IDLE {
                c.anchor();
            }
            c.last_use = Instant::now();
        } else if !self.no_cal.contains(&h.device) {
            match HipCal::new(h.device, h.label) {
                Ok(mut c) => {
                    c.anchor();
                    self.cals.insert(h.device, c);
                    self.next_cal.get_or_insert(Instant::now() + CAL_EVERY);
                }
                Err(e) => {
                    tracing::warn!(device = h.device, error = %e, "evtrace dev: no calibrator (this device's pairs are dropped)");
                    self.no_cal.insert(h.device);
                }
            }
        }
        STATS.jobs.fetch_add(1, Relaxed);
        self.jobs.push_back(Pending { job: Job::new(h.device, h.label, h.buf), ret: h.ret, t0, cost: t0.elapsed() });
    }

    fn calibrate(&mut self) {
        for c in self.cals.values_mut() {
            if c.last_use.elapsed() < CAL_IDLE {
                c.anchor();
            }
        }
    }

    fn poll(&mut self) {
        let mut i = 0;
        while i < self.jobs.len() {
            let w0 = Instant::now();
            let p = &mut self.jobs[i];
            let done = match self.cals.get_mut(&p.job.device) {
                Some(c) if c.ready() => p.job.poll(c),
                _ => false,
            };
            p.cost += w0.elapsed();
            if done || p.t0.elapsed() >= TIMEOUT {
                let p = self.jobs.remove(i).expect("index in range");
                self.finish(p);
            } else {
                i += 1;
            }
        }
    }

    fn finish(&mut self, p: Pending) {
        let w0 = Instant::now();
        let lag_ms = p.t0.elapsed().as_secs_f64() * 1e3;
        let d = finish(&p.job, &mut self.names, lag_ms, (p.cost + w0.elapsed()).as_secs_f64() * 1e6);
        for r in &d.dev {
            super::evtrace_ring::emit_b(&DEV, r);
        }
        for r in &d.step_dev {
            evtrace::emit(&STEP_DEV, r);
        }
        for (c, v) in [
            (&STATS.pairs, d.pairs),
            (&STATS.dropped, d.dropped),
            (&STATS.deferred, d.deferred),
            (&STATS.viol_a, d.viol_a),
            (&STATS.viol_b, d.viol_b),
            (&STATS.checked_b, d.checked_b),
        ] {
            c.fetch_add(v, Relaxed);
        }
        let mut buf = p.job.buf;
        buf.sums = d.sums;
        buf.sum_steps = d.sum_steps;
        STATS.busy_ns.fetch_add(((p.cost + w0.elapsed()).as_nanos()) as u64, Relaxed);
        // The pool is gone: the buffer (and its events) drop here.
        let _ = p.ret.try_send(buf);
    }
}

/// The minute line: deltas of the counters.
fn log_stats(last: &mut [u64; 12]) {
    let s = &STATS;
    let now = [
        &s.jobs, &s.pairs, &s.dropped, &s.deferred, &s.viol_a, &s.viol_b, &s.checked_b, &s.anchors_ok, &s.anchors_failed, &s.resid_bad, &s.handoff_full, &s.busy_ns,
    ]
    .map(|c| c.load(Relaxed));
    let d: Vec<u64> = now.iter().zip(last.iter()).map(|(a, b)| a.saturating_sub(*b)).collect();
    *last = now;
    if d[0] == 0 && d[7] + d[8] == 0 {
        return;
    }
    tracing::info!(
        buffers = d[0], pairs = d[1], dropped = d[2], deferred = d[3], viol_a = d[4], viol_b = d[5], checked_b = d[6],
        anchors_ok = d[7], anchors_failed = d[8], resid_bad = d[9], handoff_full = d[10], tierb_ms = d[11] / 1_000_000,
        "evtrace dev (last minute)"
    );
}

/// The Tier B thread: device buffers, calibration, the ring's drain.
pub(crate) fn tier_b_loop(rx: Option<Receiver<Handoff>>, drain: fn()) {
    let mut st = TierB::default();
    let mut next_drain = Instant::now() + DRAIN_EVERY;
    let mut next_log = Instant::now() + LOG_EVERY;
    let mut last = [0u64; 12];
    loop {
        let now = Instant::now();
        let mut wake = next_drain;
        if let Some(t) = st.next_cal {
            wake = wake.min(t);
        }
        if !st.jobs.is_empty() {
            wake = wake.min(now + POLL);
        }
        let wait = wake.saturating_duration_since(now);
        match rx.as_ref() {
            Some(rx) => match rx.recv_timeout(wait) {
                Ok(h) => st.accept(h),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => std::thread::sleep(wait),
            },
            None => std::thread::sleep(wait),
        }
        if let Some(rx) = rx.as_ref() {
            while let Ok(h) = rx.try_recv() {
                st.accept(h);
            }
        }
        let now = Instant::now();
        if st.next_cal.is_some_and(|t| now >= t) {
            st.calibrate();
            st.next_cal = Some(now + CAL_EVERY);
        }
        st.poll();
        if now >= next_drain {
            drain();
            next_drain = now + DRAIN_EVERY;
        }
        if now >= next_log {
            log_stats(&mut last);
            next_log = now + LOG_EVERY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::trace::{SyncMark, TimingPair};
    use super::*;
    use std::cell::Cell;

    /// A device event: its device time (ns) once complete.
    struct Fake(Cell<Option<f64>>);

    impl DevEvent for Fake {
        fn done(&self) -> Option<bool> {
            Some(self.0.get().is_some())
        }
    }

    /// RAW = 1e12 + device ns (bound 1 us).
    struct Conv;

    impl ToRaw<Fake> for Conv {
        fn ready(&self) -> bool {
            true
        }
        fn to_raw(&mut self, e: &Fake) -> Option<(f64, f64)> {
            e.0.get().map(|t| (1e12 + t, 1e3))
        }
    }

    const RAW0: f64 = 1e12;

    /// A buffer from `(name, stream, step, unit, start ns, end ns)`; host
    /// stamps 10 us before each start.
    fn buf(pairs: &[(&'static str, usize, u64, u64, f64, f64)]) -> Box<Buf<Fake>> {
        let mut events = Vec::new();
        let mut ps = Vec::new();
        for &(name, stream, step, unit, s, e) in pairs {
            let i = events.len();
            events.push(Fake(Cell::new(Some(s))));
            events.push(Fake(Cell::new(Some(e))));
            let ctx = StageCtx { step, unit, ..StageCtx::NONE };
            ps.push(TimingPair { name, start_idx: i, end_idx: i + 1, ctx, stream, t_host: RAW0 + s - 10e3 });
        }
        let mut b = Buf::with_events(events);
        b.pairs = ps;
        Box::new(b)
    }

    fn field(r: &[f64], name: &str) -> f64 {
        r[STEP_DEV.fields.iter().position(|f| *f == name).expect(name)]
    }

    const NO: u64 = StageCtx::NO;

    /// S1: a buffer holding step 7's forward + head and step 8's drafter
    /// sums each pair by ITS step; prefill pairs stay out of every sum.
    #[test]
    fn sums_by_each_pairs_step() {
        let ms = 1e6;
        let b = buf(&[
            ("dgpu.q_chain", 1, 7, NO, 0.0, 2.0 * ms),
            ("dgpu.router", 1, 7, NO, 2.0 * ms, 3.0 * ms),
            ("dgpu.not_named_anywhere", 1, 7, NO, 3.0 * ms, 3.5 * ms),
            ("k.some_kernel", 1, 7, NO, 3.1 * ms, 3.2 * ms),
            ("dgpu.head_batch", 1, 7, NO, 4.0 * ms, 4.3 * ms),
            ("mtp.entry", 2, 8, NO, 5.0 * ms, 9.0 * ms),
            ("dgpu.q_chain", 1, NO, 3, 10.0 * ms, 19.0 * ms),
        ]);
        let mut j = Job::new(0, "dgpu", b);
        assert!(j.poll(&mut Conv));
        let d = finish(&j, &mut Names::default(), 1.5, 20.0);
        assert_eq!(d.pairs, 7);
        assert_eq!(d.dev.len(), 7);
        assert_eq!(d.step_dev.len(), 2);
        let s7 = &d.step_dev[0];
        let s8 = &d.step_dev[1];
        assert_eq!(field(s7, "step"), 7.0);
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        assert!(close(field(s7, "dgpu_busy_ms"), 2.0 + 1.0 + 0.5 + 0.3), "{}", field(s7, "dgpu_busy_ms"));
        assert!(close(field(s7, "d_q_chain"), 2.0));
        assert!(close(field(s7, "d_router"), 1.0));
        assert!(close(field(s7, "d_head_batch"), 0.3));
        assert!(close(field(s7, "d_other"), 0.5));
        assert!(field(s7, "d_mtp").is_nan() && field(s7, "igpu_busy_ms").is_nan());
        assert_eq!(field(s7, "pairs"), 5.0);
        assert_eq!(field(s7, "t_start"), RAW0);
        assert_eq!(field(s8, "step"), 8.0);
        assert!(close(field(s8, "d_mtp"), 4.0));
        assert!(field(s8, "dgpu_busy_ms").is_nan(), "a drafter-only part carries no busy");
        // The job-level numbers ride the largest step's record only.
        assert_eq!(field(s8, "lag_ms"), 1.5);
        assert!(field(s7, "lag_ms").is_nan());
        // Decode sums for `ms.stage`: every decode pair by name; step 7 only
        // has parent stages; the prefill pair (unit 3) is in no sum.
        let q: f64 = d.sums.iter().filter(|s| s.0 == "dgpu.q_chain").map(|s| s.1).sum();
        assert!(close(q, 2.0), "{q}");
        assert!(d.sums.iter().any(|s| s.0 == "mtp.entry" && s.2 == 1));
        assert_eq!(d.sum_steps, 1);
        // The prefill pair still has its dev record, with its unit.
        let pre = d.dev.iter().find(|r| r[6] == 3.0).expect("prefill dev record");
        assert!(pre[5].is_nan());
        assert_eq!(pre[0], RAW0 + 10.0 * ms);
    }

    /// A stream whose last end is pending waits whole (only that event is
    /// queried); the other stream converts; all convert at a later poll.
    #[test]
    fn pending_pairs_are_deferred_then_converted() {
        let b = buf(&[
            ("dgpu.q_chain", 1, 7, NO, 0.0, 1e6),
            ("dgpu.router", 1, 7, NO, 1e6, 2e6),
            ("dgpu.kv_chain", 2, 7, NO, 0.0, 5e5),
        ]);
        b.events[3].0.set(None); // router's end not complete
        let mut j = Job::new(0, "dgpu", b);
        assert!(!j.poll(&mut Conv));
        assert!(j.conv[0][0].is_nan() && j.conv[1][0].is_nan() && !j.conv[2][0].is_nan());
        j.buf.events[3].0.set(Some(2e6));
        assert!(j.poll(&mut Conv));
        let d = finish(&j, &mut Names::default(), 0.0, 0.0);
        assert_eq!((d.deferred, d.dropped), (2, 0));
        assert_eq!(field(&d.step_dev[0], "deferred"), 2.0);
        // Never completed: dropped, counted, no dev record.
        let b = buf(&[("dgpu.q_chain", 1, 9, NO, 0.0, 1e6)]);
        b.events[1].0.set(None);
        let mut j = Job::new(0, "dgpu", b);
        assert!(!j.poll(&mut Conv));
        let d = finish(&j, &mut Names::default(), 2000.0, 0.0);
        assert_eq!((d.dropped, d.dev.len()), (1, 0));
        assert_eq!(field(&d.step_dev[0], "dropped"), 1.0);
    }

    /// Rule (a): a start before its host stamp (beyond the bound) counts;
    /// rule (b): only the pairs recorded on the synced stream before the
    /// sync are checked, and one ending after it counts.
    #[test]
    fn causality_rules() {
        let mut b = buf(&[
            ("dgpu.q_chain", 1, 7, NO, 0.0, 1e6),
            ("dgpu.router", 1, 7, NO, 1e6, 3e6),
            ("dgpu.kv_chain", 2, 7, NO, 0.0, 9e6),
            ("dgpu.head_batch", 1, 7, NO, 4e6, 9e6),
        ]);
        b.pairs[1].t_host = RAW0 + 1e6 + 50e3; // recorded 50 us AFTER it started
        b.syncs.push(SyncMark { stream: 1, t: RAW0 + 2e6, n_pairs: 3 });
        let mut j = Job::new(0, "dgpu", b);
        assert!(j.poll(&mut Conv));
        let d = finish(&j, &mut Names::default(), 0.0, 0.0);
        assert_eq!(d.viol_a, 1);
        // Checked: pairs 0 and 1 (stream 1, before the sync); pair 2 is on
        // another stream, pair 3 after the mark. Pair 1 ends at 3 ms > 2 ms.
        assert_eq!((d.checked_b, d.viol_b), (2, 1));
        assert_eq!(field(&d.step_dev[0], "viol_b"), 1.0);
        assert_eq!(field(&d.step_dev[0], "viol_a"), 1.0);
    }

    /// Deterministic noise in [-1, 1).
    fn noise(i: u64) -> f64 {
        let x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17);
        (x % 2_000_001) as f64 / 1e6 - 1.0
    }

    /// Anchors every 200 ms on a device clock running 30 ppm fast with a
    /// +-2 us bracket: the fit finds the rate, conversions stay inside
    /// their bound, links pass the self-check; a 1 ms step restarts the chain.
    #[test]
    fn calibration_fits_rate_and_catches_jumps() {
        let rate = 1.0 + 30e-6; // device ns per host ns
        let off = 5e11;
        let mut m = CalMath::new();
        let mut prev_dev: Option<f64> = None;
        for k in 0..40u64 {
            let host = off + k as f64 * 200e6 + 1e3 * noise(k);
            let dev = (host - off) * rate;
            let link = prev_dev.map(|p| ((dev - p) / 1e6) as f32);
            let q = 3e3;
            let chk = m.add(host + 2e3 * noise(k + 1000), q, link);
            if let Some((r, t)) = chk {
                assert!(r.abs() <= t, "link {k}: resid {r} tol {t}");
            }
            prev_dev = Some(dev);
        }
        let (slope, slope_q) = m.slope();
        // host per device ns = 1/rate: slope ~ -30 ppm.
        assert!((slope + 30e-6).abs() < 3e-6, "slope {slope}");
        assert!(slope_q < 20e-6, "bound {slope_q}");
        // An event 150 ms after the newest anchor (device time).
        let last_dev = prev_dev.unwrap();
        let ev_dev = last_dev + 150e6;
        let truth = off + ev_dev / rate;
        let (raw, q) = m.to_raw(((ev_dev - last_dev) / 1e6) as f32).unwrap();
        assert!((raw - truth).abs() <= q, "err {} bound {q}", raw - truth);
        // A 1 ms clock step: the link fails its check and the chain restarts.
        let host = off + 40.0 * 200e6 + 1e6;
        let dev = (off + 40.0 * 200e6 - off) * rate;
        let (r, t) = m.add(host, 3e3, Some(((dev - last_dev) / 1e6) as f32)).unwrap();
        assert!(r.abs() > t + JUMP_NS);
        assert_eq!(m.len(), 1);
        assert_eq!(m.slope().1, MAX_SLOPE);
        // A new chain (no link) also restarts.
        m.add(host + 200e6, 3e3, None);
        assert_eq!(m.len(), 1);
    }
}
