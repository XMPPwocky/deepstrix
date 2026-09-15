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
            // DEEPSTRIX_TOKEN_PROFILE=1 turns per-kernel event timing on without
            // attaching a perfetto exporter. Until 2026-09-13 the ONLY switch was
            // `attach_perfetto`, so every `het.token.summary` ever logged by the
            // server (and every one on the paged decode path) read
            // `dgpu_busy_us=0 igpu_busy_us=0` — the decode chain had never been
            // profiled end to end. Costs a pair of hipEventRecord per stage
            // (~100 us/layer, M20), so it stays opt-in.
            // Either profile turns recording on: DEEPSTRIX_TOKEN_PROFILE for the
            // decode breakdown, DEEPSTRIX_PREFILL_PROFILE for the prefill aggregate.
            enabled: std::cell::Cell::new(token_profile() || prefill_profile::enabled()),
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
    /// Wall time from remote submit to the partial landing. Aggregate only —
    /// the OVERLAP question needs the `remote.submit`/`remote.wait` perfetto
    /// host tracks, because this counter looks identical whether the round
    /// trip hid under local compute or serialised in front of it.
    pub static REMOTE_RTT_NS: AtomicU64 = AtomicU64::new(0);
    /// Box 2's OWN reported service time for the same exchanges, so the
    /// token summary can split the wait into "box 2 working" and "everything
    /// else" (wire, wake-up, protocol). Without the split, `remote_rtt_us`
    /// only says box 1 waited, not what it waited ON.
    pub static REMOTE_SRV_NS: AtomicU64 = AtomicU64::new(0);

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
        REMOTE_RTT_NS.store(0, Relaxed);
        REMOTE_SRV_NS.store(0, Relaxed);
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
