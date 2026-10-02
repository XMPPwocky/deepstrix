//! Tracing + per-kernel timing scaffolding for the het orchestrator (M13.2).
//!
//! Two layers:
//!
//! * **`tracing` spans** — per-stage (`debug_span!`) and per-kernel
//!   (`trace_span!`) for log filtering / future Perfetto export.
//! * **[`EventPool`]** — per-device ring of HIP events. Kernel calls
//!   record start+end events with `record(stream)`; the actual host-side
//!   `hipEventElapsedTime` query happens once at token-end, so per-kernel
//!   timing doesn't block the host loop.
//!
//! Use [`EventPool::stage`] to time a kernel-group scope:
//! ```ignore
//! let _t = de.events.stage("attn.q_chain", &de.compute)?;
//! // ... kernel launches ...
//! drop(_t); // or let it go out of scope; end event is recorded then
//! ```
//!
//! At token end, [`EventPool::harvest`] synchronizes on the last event
//! and walks the ring producing `(name, ms)` pairs. The walk is host-
//! side and only fires once per token, so overlap on the device is
//! preserved.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};

use color_eyre::eyre;
use v4flash_hip::{Event, Stream};

// ---- stage context (docs/v41/EVTRACE_REBUILD_PLAN.md 2.4 "Attribution") ----

/// Which decode step / prefill unit / layer / lane a stage belongs to,
/// captured into its pair at `stage()`. `NO` / `NO_IDX` = absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageCtx {
    pub step: u64,
    pub unit: u64,
    pub layer: u32,
    pub lane: u32,
}

impl StageCtx {
    pub const NO: u64 = u64::MAX;
    pub const NO_IDX: u32 = u32::MAX;
    pub const NONE: StageCtx = StageCtx { step: Self::NO, unit: Self::NO, layer: Self::NO_IDX, lane: Self::NO_IDX };
}

thread_local! {
    static CTX: Cell<StageCtx> = const { Cell::new(StageCtx::NONE) };
}

/// Restores the previous context when dropped (guards nest).
#[must_use = "the context lasts as long as the guard"]
pub struct CtxGuard {
    prev: StageCtx,
}

impl Drop for CtxGuard {
    fn drop(&mut self) {
        CTX.with(|c| c.set(self.prev));
    }
}

fn set_ctx(f: impl FnOnce(StageCtx) -> StageCtx) -> CtxGuard {
    CTX.with(|c| {
        let prev = c.get();
        c.set(f(prev));
        CtxGuard { prev }
    })
}

/// A decode step: `(step, -, -, -)` until the guard drops, so the head,
/// compact, upload and drafter stages carry the step and no layer.
pub fn ctx_step(step: u64) -> CtxGuard {
    set_ctx(|_| StageCtx { step, ..StageCtx::NONE })
}

/// A prefill unit (`next_unit`): `(-, unit, -, -)`.
pub fn ctx_unit(unit: u64) -> CtxGuard {
    set_ctx(|_| StageCtx { unit, ..StageCtx::NONE })
}

/// One lane's layer inside the current step / unit.
pub fn ctx_layer(layer: usize, lane: usize) -> CtxGuard {
    set_ctx(|c| StageCtx { layer: layer as u32, lane: lane as u32, ..c })
}

/// This thread's context now.
pub fn ctx() -> StageCtx {
    CTX.with(|c| c.get())
}

/// A process-wide prefill unit id.
pub fn next_unit() -> u64 {
    static U: AtomicU64 = AtomicU64::new(0);
    U.fetch_add(1, Ordering::Relaxed)
}

// ---- stream names (device tracks) ----

static STREAM_NAMES: std::sync::Mutex<Vec<(usize, &'static str)>> = std::sync::Mutex::new(Vec::new());

/// Name `s` for traces (an unnamed stream shows as `s<n>`).
pub fn name_stream(s: &Stream, name: &'static str) {
    let raw = s.raw() as usize;
    let mut v = STREAM_NAMES.lock().unwrap_or_else(|p| p.into_inner());
    v.retain(|e| e.0 != raw);
    v.push((raw, name));
}

pub(crate) fn stream_name(raw: usize) -> Option<&'static str> {
    STREAM_NAMES.lock().unwrap_or_else(|p| p.into_inner()).iter().find(|e| e.0 == raw).map(|e| e.1)
}

// ---- buffers ----

/// One timed stage of a buffer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TimingPair {
    pub(crate) name: &'static str,
    pub(crate) start_idx: usize,
    pub(crate) end_idx: usize,
    pub(crate) ctx: StageCtx,
    /// The stream's raw handle (`name_stream`).
    pub(crate) stream: usize,
    /// CLOCK_MONOTONIC_RAW ns just BEFORE the start event's record (plan N2);
    /// NaN when the buffer is not handed to Tier B.
    pub(crate) t_host: f64,
}

/// `EventPool::note_sync`: the first `n_pairs` pairs on `stream` had all
/// ended (on the device) by RAW `t`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SyncMark {
    pub(crate) stream: usize,
    pub(crate) t: f64,
    pub(crate) n_pairs: usize,
}

/// The events and pairs of one epoch (one decode step / prefill unit). ONE
/// owner at a time: the pool, the handoff channel, the Tier B thread, the
/// return channel (`evtrace_dev`).
pub(crate) struct Buf<E = Event> {
    pub(crate) events: Vec<E>,
    pub(crate) next: usize,
    pub(crate) pairs: Vec<TimingPair>,
    /// How many `pairs` the perfetto exporter has emitted (work recorded
    /// after the per-token export still goes out exactly once).
    pub(crate) exported: usize,
    /// Stages dropped this epoch because the buffer was full.
    pub(crate) dropped: usize,
    pub(crate) syncs: Vec<SyncMark>,
    /// Set by Tier B before it returns the buffer: per stage name, the decode
    /// pairs' device ms and calls, and the decode steps they cover.
    pub(crate) sums: Vec<(&'static str, f64, u32)>,
    pub(crate) sum_steps: u64,
}

// SAFETY: a `hipEvent_t` is a process-wide handle: HIP lets any thread
// query, time or destroy it, and `Event`'s Drop enters the event's own device.
// A `Buf` is never shared, only moved between its single owners (above).
unsafe impl Send for Buf<Event> {}

impl<E> Buf<E> {
    pub(crate) fn with_events(events: Vec<E>) -> Self {
        let n = events.len();
        Self { events, next: 0, pairs: Vec::with_capacity(n / 2), exported: 0, dropped: 0, syncs: Vec::new(), sums: Vec::new(), sum_steps: 0 }
    }

    fn clear(&mut self) {
        self.next = 0;
        self.pairs.clear();
        self.exported = 0;
        self.dropped = 0;
        self.syncs.clear();
        self.sums.clear();
        self.sum_steps = 0;
    }
}

impl Buf<Event> {
    /// `capacity` events on the CURRENT device.
    pub(crate) fn new(capacity: usize) -> eyre::Result<Self> {
        let mut events = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            events.push(Event::new()?);
        }
        Ok(Self::with_events(events))
    }
}

/// Per-device HIP events for kernel-scope timing.
///
/// `enabled` defaults to `false`: in that state, `stage()` returns a
/// no-op guard and recording is skipped entirely. Each suppressed
/// stage saves a pair of `hipEventRecord` calls, which the M20
/// profiling pass found accumulated to ~100 µs per layer transition
/// — roughly all of the dGPU compute-stream gap between consecutive
/// captured graphs.
///
/// Enable by calling [`EventPool::set_enabled`]. The orchestrator's
/// `attach_perfetto` flips both pools on automatically; tests that
/// need the per-kernel INFO summary should also opt in.
///
/// With evtrace's Tier B on (`evtrace_dev::offload_on`), `reset()` HANDS the
/// epoch's buffer to the Tier B thread, which times every pair off this
/// thread, and takes a free one (docs/v41/EVTRACE_REBUILD_PLAN.md 2.4); with no
/// free buffer the epoch records nothing (counted, `take_skipped`), never waits.
pub struct EventPool {
    inner: RefCell<EventPoolInner>,
    label: &'static str,
    enabled: Cell<bool>,
    /// While a stage is being captured into a HIP graph, `stage()` is a no-op:
    /// event records inside a capture become graph nodes whose timings the
    /// pool could not harvest (the parent stage around the graph launch still
    /// times the whole replay).
    capturing: Cell<bool>,
    /// `k.*` kernel sub-stages (one pair per launch inside a parent stage).
    /// The `ms.stage` rollup sums PARENT stages only, so these ~630 pairs per
    /// 8-row step bought nothing there while costing ~8 us of dGPU stream time
    /// each (bench_event_overhead, 2026-09-21). Off unless perfetto is attached
    /// or `V41_PROFILE_KERNEL_STAGES=1`.
    sub: Cell<bool>,
    /// This epoch's buffer goes to Tier B at the next reset (set at every
    /// reset): stages take host stamps, `note_sync` records.
    offload: Cell<bool>,
    device: i32,
    id: u64,
}

struct EventPoolInner {
    /// This epoch's buffer; `None` = Tier B holds every buffer.
    buf: Option<Box<Buf>>,
    /// Buffers back from Tier B (sums folded).
    free: Vec<Box<Buf>>,
    /// Their return channel (made at the first handoff).
    back: Option<(SyncSender<Box<Buf>>, Receiver<Box<Buf>>)>,
    /// Bumped by every reset: a scope that ends in a later epoch drops its pair.
    epoch: u64,
    /// Resets that found no free buffer.
    skipped: u64,
    /// Returned buffers' decode sums, until `take_dev_sums`.
    sums: HashMap<&'static str, (f64, u32)>,
    sum_steps: u64,
}

impl EventPoolInner {
    fn fold(&mut self, b: &mut Buf) {
        for (name, ms, calls) in b.sums.drain(..) {
            let e = self.sums.entry(name).or_insert((0.0, 0));
            e.0 += ms;
            e.1 += calls;
        }
        self.sum_steps += std::mem::take(&mut b.sum_steps);
    }
}

/// Event slots a new stage leaves free: more than the deepest nesting of open
/// stages, so a stage that started can always record its END (an end that
/// found no slot failed its caller through `.end()?`).
const END_RESERVE: usize = 32;

/// Pool-full drops are logged at most this often, process-wide (the reset
/// epochs are one decode step / one prefill unit: ~2 per second).
const DROP_LOG_EVERY_S: u64 = 10;

/// Call with no `EventPool` borrow held: the log line may reach a tracing
/// layer that records pool stages.
fn note_drop(label: &'static str, stage: &'static str, events: usize, dropped: usize) {
    // Seconds since the first drop, on a MONOTONIC base (a wall-clock step
    // backwards must not silence the log); `LAST` = 0 means "never logged".
    static BASE: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = BASE.get_or_init(std::time::Instant::now).elapsed().as_secs() + DROP_LOG_EVERY_S;
    let last = LAST.load(Ordering::Relaxed);
    if now >= last + DROP_LOG_EVERY_S && LAST.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
        tracing::warn!(pool = label, events, stage, dropped_this_epoch = dropped, "event pool full: stages dropped (timing gaps; logged at most every 10 s)");
    }
}

/// One harvested per-kernel timing.
#[derive(Debug, Clone)]
pub struct KernelTiming {
    pub name: &'static str,
    pub ms: f32,
}

impl EventPool {
    /// Create a pool with capacity for `capacity` events (so up to
    /// `capacity / 2` start/end pairs per token). The caller must have
    /// the relevant device already current.
    pub fn new(label: &'static str, capacity: usize) -> eyre::Result<Self> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let buf = Buf::new(capacity)?;
        let device = buf.events.first().map(|e| e.device_id()).unwrap_or(-1);
        Ok(Self {
            inner: RefCell::new(EventPoolInner {
                buf: Some(Box::new(buf)),
                free: Vec::new(),
                back: None,
                epoch: 0,
                skipped: 0,
                sums: HashMap::new(),
                sum_steps: 0,
            }),
            label,
            // DEEPSTRIX_TOKEN_PROFILE=1 turns per-kernel event timing on without
            // attaching a perfetto exporter. Until 2026-09-13 the ONLY switch was
            // `attach_perfetto`, so every `het.token.summary` ever logged by the
            // server (and every one on the paged decode path) read
            // `dgpu_busy_us=0 igpu_busy_us=0` — the decode chain had never been
            // profiled end to end. Costs a pair of hipEventRecord per stage
            // (~100 us/layer, M20), so it stays opt-in.
            // Either profile turns recording on: DEEPSTRIX_TOKEN_PROFILE for the
            // decode breakdown, DEEPSTRIX_PREFILL_PROFILE for the prefill aggregate.
            enabled: Cell::new(token_profile() || prefill_profile::enabled()),
            capturing: Cell::new(false),
            sub: Cell::new(kernel_stages()),
            offload: Cell::new(false),
            device,
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        })
    }

    /// Record `k.*` kernel sub-stages too (perfetto wants them; the rollup does not).
    pub fn set_kernel_stages(&self, on: bool) {
        self.sub.set(on);
    }

    /// Turn recording on or off. When off, `stage()` returns a no-op
    /// guard — no HIP events are recorded and `harvest()` returns
    /// empty.
    pub fn set_enabled(&self, on: bool) {
        self.enabled.set(on);
    }

    pub fn set_capturing(&self, on: bool) {
        self.capturing.set(on);
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.get()
    }

    /// Back to the construction-time recording flags (env profiles), e.g.
    /// after a live perfetto trace detaches.
    pub fn restore_defaults(&self) {
        self.enabled.set(token_profile() || prefill_profile::enabled());
        self.sub.set(kernel_stages());
    }

    /// Start the next epoch (token / step / prefill unit). With Tier B on, a
    /// buffer holding pairs goes to the Tier B thread and a free one is
    /// taken (its sums folded for `take_dev_sums`); otherwise the pairs are
    /// dropped. Never waits.
    pub fn reset(&self) {
        let off = super::evtrace_dev::offload_on();
        let mut inner = self.inner.borrow_mut();
        inner.epoch += 1;
        let mut back = Vec::new();
        if let Some((_, rx)) = inner.back.as_ref() {
            while let Ok(b) = rx.try_recv() {
                back.push(b);
            }
        }
        for mut b in back {
            inner.fold(&mut b);
            inner.free.push(b);
        }
        // `V41_EVTRACE_DEV` flipped (live): sums folded across the change
        // belong to no `ms.stage` window.
        if off != self.offload.get() {
            inner.sums.clear();
            inner.sum_steps = 0;
        }
        if off && inner.buf.as_ref().is_some_and(|b| !b.pairs.is_empty()) {
            let b = inner.buf.take().expect("checked");
            let ret = inner.back.get_or_insert_with(|| std::sync::mpsc::sync_channel(super::evtrace_dev::bufs_per_pool())).0.clone();
            if let Err(b) = super::evtrace_dev::hand_off(self.id, self.device, self.label, b, ret) {
                inner.buf = Some(b); // Tier B behind or gone: this epoch's pairs are lost
            }
        }
        if inner.buf.is_none() {
            inner.buf = inner.free.pop();
            if inner.buf.is_none() {
                inner.skipped += 1;
            }
        }
        if let Some(b) = inner.buf.as_mut() {
            b.clear();
        }
        drop(inner);
        self.offload.set(off);
    }

    /// Is this epoch's buffer going to Tier B (its sums come back through
    /// `take_dev_sums`; `harvest` would block on it for nothing)?
    pub fn offloading(&self) -> bool {
        self.offload.get()
    }

    /// The device sums of the decode pairs (step set, no prefill unit) of the
    /// buffers Tier B returned since the last call, by stage name, and the
    /// decode steps they cover (parent stages present).
    pub fn take_dev_sums(&self) -> (Vec<(&'static str, f32, u32)>, u64) {
        let mut inner = self.inner.borrow_mut();
        let v = inner.sums.drain().map(|(n, (ms, c))| (n, ms as f32, c)).collect();
        (v, std::mem::take(&mut inner.sum_steps))
    }

    /// Resets since the last call that found no free buffer (their epochs
    /// recorded nothing).
    pub fn take_skipped(&self) -> u64 {
        std::mem::take(&mut self.inner.borrow_mut().skipped)
    }

    /// The calling thread just returned from a synchronization of `stream`
    /// (`synchronize()`, a blocking readback): every stage recorded on it
    /// so far has ended. Tier B checks that against the device clock (plan 2.4
    /// rules (b) / (c)). Returns the RAW stamp, NaN when not offloading.
    pub fn note_sync(&self, stream: &Stream) -> f64 {
        if !self.offload.get() {
            return f64::NAN;
        }
        let t = super::evtrace::now();
        let mut inner = self.inner.borrow_mut();
        if let Some(b) = inner.buf.as_mut() {
            let n_pairs = b.pairs.len();
            b.syncs.push(SyncMark { stream: stream.raw() as usize, t, n_pairs });
        }
        t
    }

    /// Treat every pair recorded so far as exported: a perfetto exporter
    /// attached now has no anchor for them (their slices would land at the
    /// anchor with garbage durations).
    pub fn mark_exported(&self) {
        let mut inner = self.inner.borrow_mut();
        if let Some(b) = inner.buf.as_mut() {
            b.exported = b.pairs.len();
        }
    }

    /// Open a timing scope on `stream` named `name`. Records a start event
    /// immediately. Returns a guard that records the end event on drop.
    ///
    /// When the pool is disabled, returns a no-op guard that records
    /// nothing — neither the start nor end HIP event is issued. This
    /// is the hot-path fast exit for bench/production runs.
    pub fn stage<'a>(
        &'a self,
        name: &'static str,
        stream: &'a Stream,
    ) -> eyre::Result<StageScope<'a>> {
        let open = self.open(name, stream)?;
        Ok(StageScope { pool: self, stream, open })
    }

    /// `stage` without the guard: the start is recorded now, the end by
    /// `close` on the same stream (for code that holds the pool's owner
    /// mutably in between). `None` = not recorded (disabled, capturing, no
    /// buffer, full); a token never closed leaves a gap, nothing else.
    pub fn open(&self, name: &'static str, stream: &Stream) -> eyre::Result<Option<OpenStage>> {
        if !self.enabled.get() || self.capturing.get() || (!self.sub.get() && name.starts_with("k.")) {
            return Ok(None);
        }
        let mut inner = self.inner.borrow_mut();
        let epoch = inner.epoch;
        // No free buffer this epoch (counted at the reset).
        let Some(buf) = inner.buf.as_mut() else { return Ok(None) };
        let idx = buf.next;
        if idx + END_RESERVE >= buf.events.len() {
            // FULL: drop the stage (a timing gap) rather than fail the
            // caller -- a decode step (every live stream) or a prefill job.
            // Reachable from a live perfetto trace (`V41_PERFETTO_KERNELS`).
            // `END_RESERVE` slots stay free for the ends of open stages.
            buf.dropped += 1;
            let (n, len) = (buf.dropped, buf.events.len());
            drop(inner);
            note_drop(self.label, name, len, n);
            return Ok(None);
        }
        buf.next += 1;
        // Causality rule (a): stamped BEFORE the record (plan N2).
        let t_host = if self.offload.get() { super::evtrace::now() } else { f64::NAN };
        buf.events[idx].record(stream)?;
        drop(inner);
        Ok(Some(OpenStage { name, start_idx: idx, epoch, ctx: ctx(), t_host, pool: self.id, stream: stream.raw() as usize }))
    }

    /// Record the end of an `open` stage on `stream` (its start's stream).
    pub fn close(&self, open: Option<OpenStage>, stream: &Stream) -> eyre::Result<()> {
        let Some(o) = open else { return Ok(()) };
        // Another pool's token, or another stream's (its start would not be
        // covered by this stream's completion): no pair.
        debug_assert!(o.pool == self.id && o.stream == stream.raw() as usize, "stage {} closed on another pool / stream", o.name);
        if o.pool != self.id || o.stream != stream.raw() as usize {
            note_drop(self.label, o.name, 0, 0);
            return Ok(());
        }
        let mut inner = self.inner.borrow_mut();
        // Open across a reset (resets are top-level, so not today): the start
        // event is in a buffer Tier B now holds. Drop the pair.
        if inner.epoch != o.epoch {
            drop(inner);
            note_drop(self.label, o.name, 0, 0);
            return Ok(());
        }
        let Some(buf) = inner.buf.as_mut() else { return Ok(()) };
        let end_idx = buf.next;
        if end_idx >= buf.events.len() {
            // Backstop (`END_RESERVE` should make this unreachable): no pair,
            // a timing gap, never an error for the caller.
            buf.dropped += 1;
            let (n, len) = (buf.dropped, buf.events.len());
            drop(inner);
            note_drop(self.label, o.name, len, n);
            return Ok(());
        }
        buf.next += 1;
        buf.events[end_idx].record(stream)?;
        buf.pairs.push(TimingPair { name: o.name, start_idx: o.start_idx, end_idx, ctx: o.ctx, stream: stream.raw() as usize, t_host: o.t_host });
        Ok(())
    }

    /// Synchronize on the last event in the ring then walk the pairs
    /// computing elapsed milliseconds. Returns one entry per pair in
    /// recording order.
    pub fn harvest(&self) -> eyre::Result<Vec<KernelTiming>> {
        let inner = self.inner.borrow();
        let Some(b) = inner.buf.as_ref() else { return Ok(Vec::new()) };
        let Some(last) = b.pairs.last() else { return Ok(Vec::new()) };
        b.events[last.end_idx].synchronize()?;
        let mut out = Vec::with_capacity(b.pairs.len());
        for p in &b.pairs {
            let ms = Event::elapsed_ms(&b.events[p.start_idx], &b.events[p.end_idx])?;
            out.push(KernelTiming { name: p.name, ms });
        }
        Ok(out)
    }

    /// Label (e.g. "dgpu", "igpu") used for trace fields.
    pub fn label(&self) -> &'static str {
        self.label
    }

    /// Synchronize on the last event in the ring, then invoke `f` once
    /// per recorded pair with `(name, &start_event, &end_event)`. Used
    /// by the perfetto device-time exporter to emit per-stream tracks
    /// without copying event metadata out of the pool.
    pub fn for_each_pair<F>(&self, mut f: F) -> eyre::Result<()>
    where
        F: FnMut(&'static str, &Event, &Event) -> eyre::Result<()>,
    {
        let inner = self.inner.borrow();
        let Some(b) = inner.buf.as_ref() else { return Ok(()) };
        if let Some(last) = b.pairs.last() {
            b.events[last.end_idx].synchronize()?;
        }
        for p in &b.pairs {
            f(p.name, &b.events[p.start_idx], &b.events[p.end_idx])?;
        }
        Ok(())
    }

    /// Like [`Self::for_each_pair`] but only over pairs recorded SINCE the last
    /// call, and it advances the watermark.
    ///
    /// Needed because work can be recorded into this pool AFTER the per-token
    /// export runs. The DSpark drafter is exactly that: `forward_token_impl`
    /// exports at its end, then the accept loop runs `dspark_draft`, whose stages
    /// land here and are then discarded by the NEXT token's `reset()`. The result
    /// was that the drafter -- ~26 ms/step, both GPUs -- could never appear on a
    /// perfetto trace at all, showing up only as an unexplained gap.
    pub fn for_each_pair_new<F>(&self, mut f: F) -> eyre::Result<()>
    where
        F: FnMut(&'static str, &Event, &Event) -> eyre::Result<()>,
    {
        let mut inner = self.inner.borrow_mut();
        let Some(b) = inner.buf.as_mut() else { return Ok(()) };
        let from = b.exported.min(b.pairs.len());
        if from >= b.pairs.len() {
            return Ok(());
        }
        if let Some(last) = b.pairs.last() {
            b.events[last.end_idx].synchronize()?;
        }
        for i in from..b.pairs.len() {
            let p = &b.pairs[i];
            f(p.name, &b.events[p.start_idx], &b.events[p.end_idx])?;
        }
        b.exported = b.pairs.len();
        Ok(())
    }
}

/// A stage `EventPool::open` recorded the start of (`close` ends it).
#[derive(Clone, Copy, Debug)]
pub struct OpenStage {
    name: &'static str,
    start_idx: usize,
    /// The pool's epoch at the start: a later close drops the pair.
    epoch: u64,
    ctx: StageCtx,
    t_host: f64,
    /// The pool (`EventPool::id`) and stream it must be closed on.
    pool: u64,
    stream: usize,
}

/// RAII guard that records its end event on drop.
pub struct StageScope<'a> {
    pool: &'a EventPool,
    stream: &'a Stream,
    open: Option<OpenStage>,
}

impl<'a> StageScope<'a> {
    /// Explicit end (for early termination before the natural drop point).
    pub fn end(mut self) -> eyre::Result<()> {
        self.pool.close(self.open.take(), self.stream)
    }
}

impl<'a> Drop for StageScope<'a> {
    fn drop(&mut self) {
        // Drop-time errors are reported via tracing and swallowed; the
        // alternative (panic) is worse for orchestrator code.
        let name = self.open.map(|o| o.name).unwrap_or("?");
        if let Err(e) = self.pool.close(self.open.take(), self.stream) {
            tracing::warn!(stage = name, label = self.pool.label(), error = %e, "EventPool stage end failed");
        }
    }
}

/// `V41_PROFILE_KERNEL_STAGES=1`: also record the `k.*` per-launch sub-stages.
pub fn kernel_stages() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("V41_PROFILE_KERNEL_STAGES").as_deref() == Ok("1")
    });
    *ON
}

/// `DEEPSTRIX_TOKEN_PROFILE=1`: enable HIP event timing on every EventPool and
/// emit the per-stage rollup + host phase breakdown at INFO each token.
pub fn token_profile() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("DEEPSTRIX_TOKEN_PROFILE").map(|v| v != "0" && !v.is_empty()).unwrap_or(false)
    });
    *ON
}

/// Host-side phase accumulators for the PAGED decode path.
///
/// The device EventPools cover kernels; these cover the host work the paged path
/// interposes between them — the per-layer `synchronize()` + `d_selected`
/// readback, and `ExpertPager::ensure` (LRU bookkeeping + miss service + the
/// synchronous `remap_dev` H2D). Reset at token start, read at token end.
pub mod phase {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub static SEL_SYNC_NS: AtomicU64 = AtomicU64::new(0);
    pub static ENSURE_NS: AtomicU64 = AtomicU64::new(0);
    pub static ENGRAM_STAGE_NS: AtomicU64 = AtomicU64::new(0);
    /// Host wall BLOCKED in `wait()` for the partial: the EXPOSED part of the
    /// round trip. Was misnamed `REMOTE_RTT_NS` until 2026-09-21 and quoted
    /// as link cost; it collapses to ~0 whenever box 1 is the slower side.
    pub static REMOTE_WAIT_NS: AtomicU64 = AtomicU64::new(0);
    /// True submit -> reply-landed round trip (`RemotePartial::rtt_us`), the
    /// whole of it, hidden or not. `rtt - srv` is the link + hub-side handoffs.
    pub static REMOTE_RTT_NS: AtomicU64 = AtomicU64::new(0);
    /// Box 2's OWN reported service time for the same exchanges, so the
    /// token summary can split the wait into "box 2 working" and "everything
    /// else" (wire, wake-up, protocol). Without the split, `remote_rtt_us`
    /// only says box 1 waited, not what it waited ON.
    pub static REMOTE_SRV_NS: AtomicU64 = AtomicU64::new(0);
    /// Box 2's own accounting from each reply: page (disk) time, compute
    /// time and miss count, summed over the layers of one step.
    pub static REMOTE_PAGE_NS: AtomicU64 = AtomicU64::new(0);
    pub static REMOTE_COMPUTE_NS: AtomicU64 = AtomicU64::new(0);
    pub static REMOTE_MISSES: AtomicU64 = AtomicU64::new(0);
    /// Replies whose page time was non-zero: a demand read OR a parked wait
    /// for a read to land (`knobs::park`, which the miss count does not see).
    /// `page / paged` = cost per paging reply.
    pub static REMOTE_PAGED: AtomicU64 = AtomicU64::new(0);

    /// Host phases OUTSIDE `forward_token_impl`'s `token_start..sync` bracket
    /// but ON the decode loop's critical path (engine_worker.rs
    /// `finish_decode` / `forward_one!`): the caller adds to these between one
    /// forward's return and the next forward's entry. Deliberately NOT cleared
    /// by `reset()` — that runs at forward entry, i.e. AFTER the caller has
    /// already added to them — the token summary drains them with `take()`,
    /// so each `het.token.summary` line reports the glue that ran since the
    /// previous line (attributed to the token whose forward follows it).
    ///
    /// Why: the lever-1 A/B logs (2026-09-14, l1_1.log) put the decode loop's
    /// wall at 60.8 ms/token against `total_us` 57.6 ms — 3.3 ms/token that no
    /// counter covered. (The rest of that A/B's "88 ms/token" was the 36-token
    /// prompt's 6.9 s CED-replay prefill amortised over 256 tokens by a
    /// curl-wall harness — see `decode.loop.summary` / `request.summary`.)
    pub static CALLER_ENGRAM_NS: AtomicU64 = AtomicU64::new(0);
    pub static CALLER_EMBED_NS: AtomicU64 = AtomicU64::new(0);
    pub static CALLER_SAMPLE_NS: AtomicU64 = AtomicU64::new(0);
    pub static CALLER_STREAM_NS: AtomicU64 = AtomicU64::new(0);

    pub fn reset() {
        SEL_SYNC_NS.store(0, Relaxed);
        ENSURE_NS.store(0, Relaxed);
        ENGRAM_STAGE_NS.store(0, Relaxed);
        REMOTE_WAIT_NS.store(0, Relaxed);
        REMOTE_RTT_NS.store(0, Relaxed);
        REMOTE_SRV_NS.store(0, Relaxed);
        REMOTE_PAGE_NS.store(0, Relaxed);
        REMOTE_COMPUTE_NS.store(0, Relaxed);
        REMOTE_MISSES.store(0, Relaxed);
        REMOTE_PAGED.store(0, Relaxed);
    }
    pub fn add(c: &AtomicU64, ns: u64) {
        c.fetch_add(ns, Relaxed);
    }
    pub fn get(c: &AtomicU64) -> u64 {
        c.load(Relaxed)
    }
    /// Read-and-clear, for the `CALLER_*` counters (see above).
    pub fn take(c: &AtomicU64) -> u64 {
        c.swap(0, Relaxed)
    }
}

/// Process-wide monotonic clock (ns since first use) for gaps that span two
/// calls, e.g. `TokenTiming::gap_us` between consecutive `forward_token_impl`s.
pub fn epoch_ns() -> u64 {
    static EPOCH: std::sync::LazyLock<std::time::Instant> =
        std::sync::LazyLock::new(std::time::Instant::now);
    EPOCH.elapsed().as_nanos() as u64
}

/// Per-token timing summary, emitted at INFO once the token's events
/// have been harvested.
#[derive(Debug, Default, Clone)]
pub struct TokenTiming {
    pub token_pos: u32,
    pub total_us: u64,
    pub dgpu_busy_us: u64,
    pub igpu_busy_us: u64,
    pub dgpu_idle_us: u64,
    pub igpu_idle_us: u64,
    pub peer_bytes: u64,
    /// Host wall in the token loop before the final sync, and the sync itself.
    pub host_us: u64,
    pub sync_us: u64,
    /// Paged-path host phases (0 on the resident path).
    pub sel_sync_us: u64,
    pub pager_ensure_us: u64,
    /// Of `pager_ensure_us`: the miss path's host read and its H2D copies.
    pub pager_read_us: u64,
    pub pager_h2d_us: u64,
    pub pager_misses: u64,
    /// Two-box split: host wall from remote submit to the partial landing.
    /// Overlapped work, so this is NOT additive with the rest — compare it
    /// against `total_us` to see whether the round trip is hidden.
    pub remote_rtt_us: u64,
    pub remote_wait_us: u64,
    pub remote_srv_us: u64,
    /// `stage_engram_rows` H2D inside the bracket (V4.1 Engram layers 1, 14).
    pub engram_stage_us: u64,
    /// Bracket edges inside `forward_token_impl`, NOT part of `total_us`:
    /// `pre_us` = fn entry -> `token_start` (residual H2D + per-token scalar
    /// writes); `post_us` = final sync -> summary (perfetto export + event
    /// harvest).
    pub pre_us: u64,
    pub post_us: u64,
    /// Wall from the previous `forward_token_impl`'s return to this one's
    /// entry: EVERYTHING the caller did between two tokens. On a request's
    /// first token this spans the whole prefill — exclude it from averages.
    pub gap_us: u64,
    /// Caller phases drained from `phase::CALLER_*` (a subset of `gap_us`):
    /// Engram SSD gather (`EngramCtx::rows_for`), embedding lookup, sampler
    /// (kernels + sync + 4 B D2H), detokenise + chunk send.
    /// `gap_us - (engram_us + embed_us + sample_us + stream_us)` is the
    /// unattributed caller glue.
    pub engram_us: u64,
    pub embed_us: u64,
    pub sample_us: u64,
    pub stream_us: u64,
}

impl TokenTiming {
    pub fn emit(&self) {
        tracing::info!(
            token_pos = self.token_pos,
            total_us = self.total_us,
            dgpu_busy_us = self.dgpu_busy_us,
            igpu_busy_us = self.igpu_busy_us,
            dgpu_idle_us = self.dgpu_idle_us,
            igpu_idle_us = self.igpu_idle_us,
            peer_bytes = self.peer_bytes,
            host_us = self.host_us,
            sync_us = self.sync_us,
            remote_rtt_us = self.remote_rtt_us,
            remote_wait_us = self.remote_wait_us,
            remote_srv_us = self.remote_srv_us,
            remote_link_us = self.remote_rtt_us.saturating_sub(self.remote_srv_us),
            sel_sync_us = self.sel_sync_us,
            pager_ensure_us = self.pager_ensure_us,
            pager_read_us = self.pager_read_us,
            pager_h2d_us = self.pager_h2d_us,
            pager_misses = self.pager_misses,
            engram_stage_us = self.engram_stage_us,
            pre_us = self.pre_us,
            post_us = self.post_us,
            gap_us = self.gap_us,
            engram_us = self.engram_us,
            embed_us = self.embed_us,
            sample_us = self.sample_us,
            stream_us = self.stream_us,
            "het.token.summary"
        );
    }
}

/// Cumulative per-stage totals across a whole PREFILL (all chunks), emitted once
/// at the end of the request.
///
/// Prefill records stages via `events.stage(..)` but has never harvested them —
/// the ONLY `het.stage` emit site in the tree is `forward_token_impl`, the DECODE
/// path. That gap caused a real misreading on 2026-09-14: a one-token_pos dump
/// from a `max_tokens=1` run was read as "the last prefill token" when it was the
/// only DECODE token, and a strategy was built on it. This closes the gap.
///
/// Gated by `DEEPSTRIX_PREFILL_PROFILE=1` because harvesting SYNCHRONIZES, which
/// serialises the two prefill lanes. Per-stage GPU busy time survives that; the
/// `.wait` stages and the wall DO NOT. Read the busy times, not the wall.
pub mod prefill_profile {
    use std::sync::{LazyLock, Mutex};

    static ON: LazyLock<bool> = LazyLock::new(|| {
        std::env::var("DEEPSTRIX_PREFILL_PROFILE").as_deref() == Ok("1")
    });
    #[allow(clippy::type_complexity)]
    static ACC: LazyLock<Mutex<Vec<(&'static str, &'static str, f64, u32)>>> =
        LazyLock::new(|| Mutex::new(Vec::new()));

    pub fn enabled() -> bool {
        *ON
    }

    /// Fold one chunk's rollup into the request accumulator.
    pub fn add(device: &'static str, rolled: &[(&'static str, f32, u32)]) {
        if !enabled() {
            return;
        }
        let Ok(mut acc) = ACC.lock() else { return };
        for &(name, ms, calls) in rolled {
            match acc.iter_mut().find(|e| e.0 == device && e.1 == name) {
                Some(e) => {
                    e.2 += ms as f64;
                    e.3 += calls;
                }
                None => acc.push((device, name, ms as f64, calls)),
            }
        }
    }

    /// Emit and clear. Call once at the end of a prefill.
    pub fn emit_and_clear(prompt_tokens: usize) {
        if !enabled() {
            return;
        }
        let Ok(mut acc) = ACC.lock() else { return };
        acc.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        let total: f64 = acc.iter().map(|e| e.2).sum();
        for (device, name, ms, calls) in acc.iter() {
            tracing::info!(
                prompt_tokens,
                device = *device,
                stage = *name,
                total_ms = format!("{ms:.1}"),
                calls = *calls,
                pct = format!("{:.1}", if total > 0.0 { 100.0 * ms / total } else { 0.0 }),
                "prefill.stage"
            );
        }
        tracing::info!(prompt_tokens, total_ms = format!("{total:.1}"), "prefill.stage.total");
        acc.clear();
    }
}

/// Aggregate harvested timings into a single `(name, total_ms, calls)`
/// rollup, useful for the per-token DEBUG dump.
pub fn rollup_by_name(timings: &[KernelTiming]) -> Vec<(&'static str, f32, u32)> {
    use std::collections::HashMap;
    let mut by: HashMap<&'static str, (f32, u32)> = HashMap::new();
    for t in timings {
        let entry = by.entry(t.name).or_insert((0.0, 0));
        entry.0 += t.ms;
        entry.1 += 1;
    }
    let mut out: Vec<_> = by.into_iter().map(|(k, (s, c))| (k, s, c)).collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}
