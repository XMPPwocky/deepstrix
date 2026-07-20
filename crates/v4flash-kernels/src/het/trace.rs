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

use std::cell::RefCell;

use color_eyre::eyre;
use v4flash_hip::{Event, Stream};

/// Per-device ring of HIP events for kernel-scope timing.
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
pub struct EventPool {
    inner: RefCell<EventPoolInner>,
    label: &'static str,
    enabled: std::cell::Cell<bool>,
}

struct EventPoolInner {
    events: Vec<Event>,
    next: usize,
    pairs: Vec<TimingPair>,
}

struct TimingPair {
    name: &'static str,
    start_idx: usize,
    end_idx: usize,
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
        let mut events = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            events.push(Event::new()?);
        }
        Ok(Self {
            inner: RefCell::new(EventPoolInner {
                events,
                next: 0,
                pairs: Vec::with_capacity(capacity / 2),
            }),
            label,
            enabled: std::cell::Cell::new(false),
        })
    }

    /// Turn recording on or off. When off, `stage()` returns a no-op
    /// guard — no HIP events are recorded and `harvest()` returns
    /// empty.
    pub fn set_enabled(&self, on: bool) {
        self.enabled.set(on);
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.get()
    }

    /// Reset for the next token. Drops all timing pairs.
    pub fn reset(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.next = 0;
        inner.pairs.clear();
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
        if !self.enabled.get() {
            return Ok(StageScope {
                pool: self,
                stream,
                name,
                start_idx: usize::MAX,
                done: true,           // skip record_end
            });
        }
        let start_idx = {
            let mut inner = self.inner.borrow_mut();
            let idx = inner.next;
            if idx >= inner.events.len() {
                return Err(color_eyre::eyre::eyre!(
                    "EventPool[{}] exhausted at {} events",
                    self.label,
                    inner.events.len()
                ));
            }
            inner.next += 1;
            inner.events[idx].record(stream)?;
            idx
        };
        Ok(StageScope {
            pool: self,
            stream,
            name,
            start_idx,
            done: false,
        })
    }

    /// Synchronize on the last event in the ring then walk the pairs
    /// computing elapsed milliseconds. Returns one entry per pair in
    /// recording order.
    pub fn harvest(&self) -> eyre::Result<Vec<KernelTiming>> {
        let inner = self.inner.borrow();
        if inner.pairs.is_empty() {
            return Ok(Vec::new());
        }
        let last_end_idx = inner.pairs.last().unwrap().end_idx;
        inner.events[last_end_idx].synchronize()?;
        let mut out = Vec::with_capacity(inner.pairs.len());
        for p in &inner.pairs {
            let ms = Event::elapsed_ms(&inner.events[p.start_idx], &inner.events[p.end_idx])?;
            out.push(KernelTiming { name: p.name, ms });
        }
        Ok(out)
    }

    /// Label (e.g. "dgpu", "igpu") used for trace fields.
    pub fn label(&self) -> &'static str {
        self.label
    }

    /// Harvest per-pair intervals as (name, start_ms, end_ms) relative to
    /// the FIRST recorded start event. Real GPU HW timestamps (not host
    /// stage spans) — usable for computing device-busy via interval union.
    pub fn harvest_intervals(&self) -> eyre::Result<Vec<(&'static str, f32, f32)>> {
        let inner = self.inner.borrow();
        if inner.pairs.is_empty() {
            return Ok(Vec::new());
        }
        let last_end_idx = inner.pairs.last().unwrap().end_idx;
        inner.events[last_end_idx].synchronize()?;
        let ref_idx = inner.pairs[0].start_idx;
        let mut out = Vec::with_capacity(inner.pairs.len());
        for p in &inner.pairs {
            let s = Event::elapsed_ms(&inner.events[ref_idx], &inner.events[p.start_idx])?;
            let e = Event::elapsed_ms(&inner.events[ref_idx], &inner.events[p.end_idx])?;
            out.push((p.name, s, e));
        }
        Ok(out)
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
        if let Some(last) = inner.pairs.last() {
            inner.events[last.end_idx].synchronize()?;
        }
        for p in &inner.pairs {
            f(p.name, &inner.events[p.start_idx], &inner.events[p.end_idx])?;
        }
        Ok(())
    }
}

/// RAII guard that records its end event on drop.
pub struct StageScope<'a> {
    pool: &'a EventPool,
    stream: &'a Stream,
    name: &'static str,
    start_idx: usize,
    done: bool,
}

impl<'a> StageScope<'a> {
    /// Explicit end (for early termination before the natural drop point).
    pub fn end(mut self) -> eyre::Result<()> {
        self.record_end()
    }

    fn record_end(&mut self) -> eyre::Result<()> {
        if self.done {
            return Ok(());
        }
        let mut inner = self.pool.inner.borrow_mut();
        let end_idx = inner.next;
        if end_idx >= inner.events.len() {
            return Err(color_eyre::eyre::eyre!(
                "EventPool[{}] exhausted on stage `{}` end",
                self.pool.label,
                self.name
            ));
        }
        inner.next += 1;
        inner.events[end_idx].record(self.stream)?;
        inner.pairs.push(TimingPair {
            name: self.name,
            start_idx: self.start_idx,
            end_idx,
        });
        self.done = true;
        Ok(())
    }
}

impl<'a> Drop for StageScope<'a> {
    fn drop(&mut self) {
        // Drop-time errors are reported via tracing and swallowed; the
        // alternative (panic) is worse for orchestrator code.
        if let Err(e) = self.record_end() {
            tracing::warn!(stage = self.name, label = self.pool.label(), error = %e, "EventPool stage end failed");
        }
    }
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
            "het.token.summary"
        );
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
