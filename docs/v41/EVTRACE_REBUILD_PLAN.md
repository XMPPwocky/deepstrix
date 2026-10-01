# Tracing rebuilt around evtrace

Status: PLAN rev 2 (2026-10-01), architect review round 2. Owner's call, 10-01:
keep building on evtrace rather than go back to emitting perfetto directly,
with our own tracing layer for the CPU side.

Rev 2 answers review round 1: device clocks from a calibrator, not the harvest's
sync stamp (R1.1); a separate Tier B path with batched device records converted
off the critical path (R1.2, R1.4); harvest coverage of the drafter, ring writes
and prefill (R1.3); spans that cost what the budget says (R1.5); and the
format, naming, trigger, placement, sequencing and rollback items (R1.6-R1.22).

## 1. Why

- **Three trace writers, three formats, three clocks.** evtrace (both boxes,
  binary records, CLOCK_MONOTONIC_RAW, always on); the hub's
  `DeviceTimingExporter` (perfetto protobuf, CLOCK_REALTIME, opt-in); box 2's
  `ExpertdTracer`/`TrackExporter` (perfetto protobuf, box-2 realtime,
  `expertd --trace`). `tracing-perfetto` survives in one test.
- **They drift.** The hub exporter was drained only on the serial decode path;
  on the arena path it recorded nothing for weeks (fixed 10-01, 0773ad1..464146e,
  not deployed).
- **evtrace is where the analysis lives**: request joins across the boxes by
  seq, the clock-offset fit, percentiles over 40-field records. It has no
  strings, no CPU spans, no device intervals (device time only as per-step sums
  by stage, `hub_step` `d_*`/`i_*`; host phases only as per-step `lh_*` sums).
- **The arena path has almost no `tracing` spans** (23 of 28 span sites are on
  the legacy serial path, forward_layer.rs).

Goal: ONE recording pipeline (evtrace, both boxes, one clock) carrying CPU spans
and device stage intervals; perfetto as a VIEW (`scripts/evt2perfetto.py`); the
protobuf exporters retire.

## 2. Target

### 2.1 Two tiers, two paths

| tier | what | rate (hub, measured / estimated) | path |
| --- | --- | --- | --- |
| A, always on | today's kinds + `str`, `site`, `knob` | ~1-3K rec/s | today's channel + writer, to disk, rotation unchanged |
| B, flight recorder | device stage intervals, CPU spans, WARN/ERROR instants | ~1840 dev pairs per two-lane step (live `ms.stage`, 23:01 UTC) ≈ 18K/s + spans ≈ 1.5-2.5 MB/s serialized | its OWN channel and thread into an in-memory ring; disk only on a dump |

**Tier B never touches Tier A.** Producers send Tier B data to a separate
bounded channel (`try_send`, drop + count on full, `meta.dropped_b`); the Tier B
thread serializes into the ring. A stall or a dump on Tier B cannot drop a
`hub_req`.

**Ring.** Serialized records in sealed, record-aligned blocks (64 KB,
`Arc<[u8]>` once full) in a deque bounded by bytes. Sizes are stated in
seconds: hub 96 MB ≈ 40-60 s, box 2 64 MB ≈ 2 min (box 2's rate is ~0.5 MB/s,
estimate, measured in P1); knobs `V41_EVTRACE_RING_MB` (static), 0 = Tier B
off.

**Dumps.** A dedicated dump thread: a dump clones the deque of sealed block
`Arc`s (O(blocks), no copy, no lock held while writing) plus the current open
block's bytes, and writes `<dir>/ring/<role>-<utc>-<pid>.evt` -- its own
directory and retention (`V41_EVTRACE_RING_KEEP`, default 8), so Tier A pruning
and readers never see dumps. Default `<dir>/ring` is under `/dev/shm` on both
boxes (box 2's paging is disk-bound and E100 writes are a measured source of
step variance; ~/logs on box 2 sits on the E100, to be confirmed in P1);
`trace_now.sh` moves a dump off `/dev/shm` after pulling it. Back-to-back dumps
keep history (copies, not swaps).

**Trigger.** A one-shot request file `<dir>/ring/dump-request` (contents: seconds,
default all) polled by the dump thread every 250 ms and deleted when acted on:
no edge-trigger traps, nothing left to fire at the next start, no race with the
A/B scripts that rewrite the knob file. `trace_now.sh --dump N` writes it on both
boxes. A dump selects records by their own time with a margin (device records
arrive after their step; box-2 background reads land seconds later).

### 2.2 Format: still `EVT1`, `format_rev: 2`

- Magic stays `EVT1` (evtrace.py, evt2perfetto.py and an out-of-repo forensics
  script assert it). The header gains `"format_rev": 2` and `"strings"` (the
  table at file open); new kinds are additive -- unknown kinds read as `kindNN`.
  P0 acceptance: today's readers, unchanged, read the new files.
- New kinds (explicit ids; 0, 1, 10-12, 20-23 are taken):
  - `str` (3): `[id, len, bytes packed 8 per f64]` -- interned strings defined
    after the header.
  - `site` (4): `[id, name, target, level, nfields, field name ids x4]` -- a
    span callsite.
  - `knob` (5, Tier A and B): `t (RAW), name, value, source` (value = string id).
  - `span` (30, B): `t_start, t_end, site, span_no, parent_no, os_tid, thread,
    f0..f3` -- `span_no` a process-local counter (Registry ids are sharded-slab
    keys above 2^53, not exact in f64); `os_tid` numeric (tokio workers share a
    name; per-step helper threads are unnamed); `thread` = interned name only
    when the thread HAS one, never a formatted tid.
  - `dev` (31, B): `t_start, t_end, device, stream, stage, step, unit, layer,
    lane, quality_us` (stage = string id; device 0 dGPU / 1 iGPU / 2 box-2 iGPU;
    `unit` = prefill unit / job id; `quality_us` = the calibration bracket).
  - `step_dev` (32, A): `step, d_*, i_*` -- the per-step stage sums, emitted by
    the Tier B thread (2.4).
  - `instant` (33, B): `t, site, level` -- WARN/ERROR events, rate-limited.
- **Strings never lost**: a producer interns (lock, assign id, push bytes to a
  pending list, bump an atomic pending count) BEFORE sending the record that
  uses the id; each writer (A and B) checks the atomic count and drains pending
  strings into its output before the record. Every new Tier A file and every dump
  writes the full table into its header.
- **`cut` keeps strings**: `cut` collects every `str` / `site` record up to `--to`
  and writes them into the cut's header (records without a time are otherwise
  dropped by the window filter). P0 test: cut -> trace keeps every name.
- **Table bounded**: interned strings are names, never formatted values (box 2's
  `moe L{} B={}` becomes numeric fields); a hard cap (64K strings, then a
  `<overflow>` id) and the header reader reads the table, not a fixed 4 MB.

### 2.3 CPU spans: coarse via `tracing`, fine via a direct guard

- **Subscriber.** The hub's `fmt().with_env_filter(..)` (main.rs:119-124)
  becomes a `Registry` with per-layer filters: `fmt` keeps today's filter AND
  excludes the target `evt` (so it never formats those spans); `EvtLayer` takes
  target `evt` (and only it) through a `Targets` filter whose reload handle is
  driven by a knob HOOK (2.6), never by reading knobs inside the layer.
- **Coarse spans via `tracing`** (target `evt`, level `info`): scheduler tick,
  decode step, prefill unit, drafter, head + sampling, accept + ring writes,
  knobs watcher pass, box-2 request phases -- ~10-30 per step, ~0.5-1 us each:
  <= 30 us per step.
- **Fine spans via `evtrace::Span`** (an RAII guard, no `tracing`): two RAW
  reads and a push into a thread-local Tier B buffer flushed once per step
  (~50-100 ns): per lane x layer x phase (pre-MoE, remote submit, pager ensure,
  post-MoE wait / combine) ≈ 400-500 per step ≈ 40-50 us. Same `span` kind,
  `site` registered once per static name.
- **EvtLayer internals**: per-thread enter stamps in a preallocated thread-local
  stack (`try_with`: no TLS during thread teardown), no Registry extensions on
  the hot path, `span_no` from an atomic counter, numeric fields only (up to 4,
  recorded at span creation).
- **Box 2** gets a subscriber for the first time: `fmt` at `warn` (the kernels'
  `info!` lines do not suddenly flood its stderr; the knobs module stops echoing
  what fmt now prints) + `EvtLayer`. `tracing-subscriber` becomes a dependency of
  `deepstrix-expertd` (workspace dep already; owner sign-off per the deps rule).

### 2.4 Device stage intervals

**Clock calibration (R1.1).** A calibrator per device, off the scheduler
thread: every 200 ms it records an anchor event on a dedicated idle calibration
stream of that device and spins on `query()` with a RAW stamp before and after
each poll; the bracket width is the anchor's error bound (expect a few us). It
keeps a ring of the last ~16 anchors (events stay alive). An event's RAW time =
interpolation between the two nearest anchors (`t_a + elapsed(anchor, e)`),
drift clamped to +-200 ppm like `Offsets`; `quality_us` = the bracket width.
All streams of one device share the timestamp counter, so one calibration
stream per device serves every stream.

**Pools double-buffered and harvested off the critical path (R1.2, R1.4).**
Each device gets a small set of event pools (3). At every RESET point -- decode
step start, prefill unit start, tick start -- the current pool is HANDED to the
Tier B thread with its stage contexts and the step / unit ids, and a free pool
is taken. The Tier B thread waits for the pool's last event (`query()` polling,
never an error), measures each event against the calibration anchors (one
elapsed call per event), emits `dev` records and the per-step `step_dev` sums,
logs the `ms.stage` rollup, and returns the pool. If no pool is free (Tier B
behind), the step records nothing (stages no-op, counted) -- never a wait.
This moves today's synchronous harvest (~0.4 ms per two-lane step, ~1840 pairs)
OFF the scheduler thread: the critical path gets cheaper, not dearer.

**Coverage (R1.3).** Handing over at every reset point covers what the
current harvest misses: the drafter (recorded between the leftovers export and
the reset), the async ring writes (after the harvest), and prefill units (which
reset without harvesting). A pair whose end event is not complete when the pool
is processed is deferred, never an error; `harvest()?` failing a decode step
(multistream.rs ~1883) goes away.

**Attribution.** A stage context `(step, unit, layer, lane)` set through an RAII
guard that restores the previous value (head / compact / upload / drafter
stages get `(step, -, -, -)`, not the last layer's), captured into `TimingPair`
at `stage()` along with the STREAM (today the track is guessed from a name
substring, engine.rs ~934). The ready-first driver sets it per phase call.

**Analyses.** `hub_step` keeps its non-device fields; `d_*` / `i_*` move to
`step_dev` (joined by step). `evtrace.py`, `evt2perfetto.py` and tonight's
scripts read either.

**Box 2 (R1.12).** Its per-request GPU event pair exists only under `--trace`
today (+2.7 us host / +5.5 us stream per pair on box 2's iGPU, the bottleneck;
~800 requests/s ≈ 0.2-0.4% of its GPU time). Behind a knob, default OFF; when on,
the same calibrator and Tier B path, `dev` records keyed by `seq` so a box-2
request's kernels join its `b2_req`.

### 2.5 The converter

`evt2perfetto.py` writes **perfetto protobuf directly** (a ~100-line encoder,
interned names via `InternedData`), streaming: Tier B volumes (a 30 s dump ≈
600K dev + 30K span records) would be > 1 GB of Python dicts and a
multi-hundred-MB JSON. Tracks: CPU spans per thread (nested), device tracks per
device / stream / lane, flows hub request -> box-2 request -> box-2 device work
(by `seq`), knob instants, WARN/ERROR instants, the existing request / paging /
read tracks. Default dump window for conversion 10 s; memory bounded per
window. **Causality check** (R1.1 acceptance): no device interval may end after
the host span that waited for it; violations counted and reported.

### 2.6 Re-entrancy and lock order (R1.7)

- The layer and the `Span` guard never call `Knob` accessors (`resolve_first`
  holds `RESOLVE` across `tracing::warn!`, knobs.rs ~286-305); knob changes reach
  the layer through a HOOK that writes an atomic / the reload handle.
- Lock order: `CHANGES -> INTERNER` (the knobs watcher feeds `knob` records
  while holding `CHANGES`, knobs.rs ~656-670). Nothing holding `INTERNER` calls
  `knobs::snapshot()` / `changes_since` / any tracing macro; the header builder
  copies the table, drops the lock, then gathers knobs.
- The layer takes no `EventPool` borrow, no exporter mutex, no lock a caller
  may hold except `INTERNER` (leaf).
- Host tests with a timeout for each: a first-hit callsite under `RESOLVE`; a
  span inside `step_now` (`WARNED`) while the hook reloads the filter; a
  `knob` record under `CHANGES`; a dump racing a rotation.

### 2.7 Knobs in every trace (R1.14)

Every Tier A file and every dump snapshots the CURRENT knob table into its
header (not the startup table reused by rotations, evtrace.rs ~222/270);
`knob` change records go to Tier A and B. `knob.t` is RAW (the change log's
`t_ns` is CLOCK_REALTIME today, knobs.rs ~511/543/657: the watcher stamps both).

## 3. Phases

Each phase: code + host tests + persistent-reviewer rounds. All tracing code
infallible (a failure counts and logs, never fails a step). Costs measured
DIRECTLY (self-timed Tier B / harvest work per step into `step_dev`,
`meta.dropped` per tier, dump thread wall time and bytes), not by an
end-to-end A/B that cannot resolve sub-millisecond effects.

| phase | content | box 2? | acceptance |
| --- | --- | --- | --- |
| P0 | `format_rev 2`: interner, `str`/`site`/`knob` kinds, header table, readers + `cut` keep strings, knob snapshot per file, `knob.t` RAW | yes (b2 files) | unchanged readers read new files; cut -> trace keeps names; dropped records cannot orphan a string |
| P1 | Tier B channel + thread + ring + dump thread + request file, `/dev/shm` dumps, `trace_now.sh --dump` | yes | synthetic ring tests; dump cost measured on the hub; Tier A `dropped` unchanged under dumps |
| P3 | calibrator, double-buffered pools handed off at every reset, `dev` + `step_dev`, stage context + stream in `TimingPair`, box-2 `dev` behind a knob | yes | causality check clean; scheduler-thread time per step DROPS by the old harvest (~0.4 ms); parity vs the 10-01 exporter in a GPU gate window (box 2 attached) |
| P2 | Registry + `EvtLayer` + `evtrace::Span`, coverage pass (2.3), box-2 subscriber | yes (sign-off: dep) | span cost per step measured (<= 80 us total); re-entrancy tests pass |
| P4 | converter v2 (protobuf, flows, CPU / device tracks); retire `DeviceTimingExporter`, `ExpertdTracer` / `TrackExporter`, the tracing-perfetto test | yes | one `trace_now.sh --dump` yields both boxes' CPU + device + request + paging tracks for a DSpark window |
| P5 | auto-dump on anomalies (a step > 3x the rolling median, a stream abort, a box-2 reconnect), rate-limited | yes | -- |

P3 goes before P2: device intervals are what the DSpark request needed, and
the layer carries the most re-entrancy risk.

**Deploys.** Box-2 parts of P0-P3 ride ONE box-2 restart (cold pool, two-box
order); hub parts can go hub-only in between. P4 keeps `expertd --trace` and
`V41_PERFETTO_*` ACCEPTED as warn + no-op for one release (an unknown argument
makes expertd exit today, main.rs ~136). **Rollback** of the subscriber
restructure and the format change is by binary only (the backup binaries per
deploy, as today).

## 4. Costs and risks

- **Critical path.** Net change on the scheduler thread: - the synchronous
  harvest (~0.4 ms) + coarse spans (<= 30 us) + fine spans (~40-50 us) + pool
  handoff (~us). Measured per step in P2/P3.
- **Memory.** Hub ring 96 MB + 3 pools per device (events only) on ~3-5 GB free.
- **Disk.** Tier A unchanged; dumps to `/dev/shm` (RAM), moved off by the
  script.
- **Clock.** Device intervals carry their calibration bracket (`quality_us`);
  the converter's causality check catches a bad calibration.
- **Box 2.** Device records off by default (its iGPU is the bottleneck); a
  subscriber at `warn`; one new dependency (sign-off).
- **What we give up.** Perfetto-native data we do not emit (sampled stacks,
  kernel scheduler tracks); perfetto's `traced` can add those next to our output
  if ever wanted.

## 5. Owner decisions

1. Ring sizes: hub 96 MB (~40-60 s), box 2 64 MB (~2 min).
2. Tier B on in production (recommended: device data reuses the profile already
   on, and the critical path gets cheaper) or opt-in.
3. `tracing-subscriber` as a dependency of `deepstrix-expertd` (box 2's first
   subscriber, `fmt` at `warn`).
4. Box-2 device records default off (yes / no).
5. P5 auto-dump wanted?
