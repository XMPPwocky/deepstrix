# Tracing rebuilt around evtrace

Status: PLAN rev 3, APPROVED by the architect review (round 3, 2026-10-01);
**PARTLY BUILT (docs audit 2026-10-04): P0 (2871b69), P1 (849dd90) and P3 (ee1fe74) built;
P2, P4 and P5 not built.** No `EvtLayer`, `evtrace::Span`, `span` or `instant` kind exists;
`evt2perfetto.py` still writes Chrome JSON; `DeviceTimingExporter`, `ExpertdTracer` and
`TrackExporter` are still in the tree; `deepstrix-expertd` has no `tracing-subscriber`
dependency. The kinds, ring and dump mechanics as built differ from section 2: see section 8.
Production runs with `V41_EVTRACE_DEV=0` (Tier B device timing off) — section 8.
The round-3 notes are in section 6. Owner's call, 10-01:
keep building on evtrace rather than go back to emitting perfetto directly,
with our own tracing layer for the CPU side.

Rev 2 answers review round 1: device clocks from a calibrator, not the harvest's
sync stamp (R1.1); a separate Tier B path with batched device records converted
off the critical path (R1.2, R1.4); harvest coverage of the drafter, ring writes
and prefill (R1.3); spans that cost what the budget says (R1.5); and the
format, naming, trigger, placement, sequencing and rollback items (R1.6-R1.22).
Rev 3 answers round 2: one string drainer (R2.1); a numeric calibration history
on the Tier B thread (R2.2-R2.3); `EventPool::reset()` as the handoff, with
epochs, deferral timeouts and 6 pools (R2.4-R2.5); device sums still folded into
`ms.stage` on the scheduler thread and `step_dev` merged back into `hub_step` by
the readers (R2.6); P3 causality rules on host stamps (R2.7); and R2.8-R2.13.

## 1. Why

- **Three trace writers, three formats, three clocks.** evtrace (both boxes,
  binary records, CLOCK_MONOTONIC_RAW, always on); the hub's
  `DeviceTimingExporter` (perfetto protobuf, CLOCK_REALTIME, opt-in); box 2's
  `ExpertdTracer`/`TrackExporter` (perfetto protobuf, box-2 realtime,
  `expertd --trace`). `tracing-perfetto` survives in one test.
- **They drift.** The hub exporter was drained only on the serial decode path;
  on the arena path it recorded nothing for weeks (fixed 10-01, 0773ad1..464146e,
  not deployed; both on main since — note 2026-10-04).
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
seconds: 128 MB on both boxes (owner, 10-01) ≈ 50-85 s on the hub, ~4 min on box 2 (box 2's rate is ~0.5 MB/s,
estimate, measured in P1); knobs `V41_EVTRACE_RING_MB` (static), 0 = Tier B
off.

**Dumps.** A dedicated dump thread: a dump clones the deque of sealed block
`Arc`s (O(blocks), no copy, no lock held while writing) plus the current open
block's bytes, and writes `<ring_dir>/<role>-<utc>-<pid>.evt`, where
`V41_EVTRACE_RING_DIR` is its own directory with its own retention -- Tier A
pruning, `ls hub-*.evt` and `evtrace.py <dir>` never see dumps. Placement:
the HUB writes dumps to disk (`~/logs/evtrace-ring`; its disk is not the
bottleneck the way box 2's is); BOX 2 writes to `/dev/shm/evtrace-ring` (its
paging is disk-bound and E100 writes are a measured source of step variance)
and `trace_now.sh` DELETES a box-2 dump after pulling it. Both capped by bytes
(`V41_EVTRACE_RING_KEEP_MB`, default 256: tmpfs pages are pinned RAM). While a
dump writes, its cloned `Arc`s keep up to one extra ring alive (counted in the
memory budget). Back-to-back dumps keep history (copies, not swaps).

**Trigger.** A one-shot request file `<ring_dir>/dump-request` (contents: seconds,
default all; written temp-then-rename, so a half-written file is never read as
"all") polled by the dump thread every 250 ms and deleted when acted on:
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
- **Strings never lost**: a producer interns (lock, assign id, append to an
  append-only table, bump an atomic length) BEFORE sending the record that uses
  the id. ONLY the Tier A writer emits `str` / `site` records: it keeps a
  cursor into the table and, before writing any record it dequeued, writes the
  entries from its cursor to the table's current length. Every new Tier A file
  and every dump writes the full table into its header; the Tier B ring holds no
  `str` records (two drainers of one list would each take strings the other
  needed -- review R2.1).
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
  `Targets` `evt=info` plus a default of `warn` (WARN/ERROR events from any
  target become `instant` records), ignores non-`evt` spans in `on_new_span`,
  and its reload handle is driven by a knob HOOK (2.6), never by reading knobs
  inside the layer (R2.11).
- **Coarse spans via `tracing`** (target `evt`, level `info`): scheduler tick,
  decode step, prefill unit, drafter, head + sampling, accept + ring writes,
  knobs watcher pass -- ~10-30 per step, ~0.5-1 us each: <= 30 us per step. A
  coarse span is created and entered on the same thread (its fields live in the
  thread-local stack; asserted in debug builds).
- **Fine spans via `evtrace::Span`** (an RAII guard, no `tracing`): two RAW
  reads and a push into a thread-local Tier B buffer (~50-100 ns): per lane x
  layer x phase on the hub (pre-MoE, remote submit, pager ensure, post-MoE wait /
  combine) ≈ 400-500 per step ≈ 40-50 us; on box 2, every request's phases
  (dequeue, merge, run_path, ensure, D2H, reply) -- box 2's compute loop is the
  bottleneck and never goes through `tracing` per request (R2.9). Same `span`
  kind, `site` registered once per static name. Flushing (R2.10): the scheduler
  thread flushes its buffer at the step end; other threads (pager, remote reader /
  writer, the per-step Engram helper, box 2's readers) flush when the buffer
  fills and on thread exit (a TLS destructor reached through `try_with`); spans
  lost to a dying thread are counted.
- **EvtLayer internals**: per-thread enter stamps in a preallocated thread-local
  stack (`try_with`: no TLS during thread teardown), no Registry extensions on
  the hot path, `span_no` from an atomic counter, numeric fields only (up to 4,
  recorded at span creation).
- **Box 2** gets a subscriber for the first time: `fmt` at `warn` plus
  `v4flash_kernels::knobs=info` (knob changes stay on its stderr; the knobs
  module's own echo goes, so nothing prints twice -- R2.12) + `EvtLayer`. The
  kernels' other `info!` lines do not suddenly flood its stderr. `tracing-subscriber` becomes a dependency of
  `deepstrix-expertd` (workspace dep already; owner sign-off per the deps rule).

### 2.4 Device stage intervals

**Clock calibration (R1.1, R2.2, R2.3).** Runs ON THE TIER B THREAD (one owner
of every calibration event: no re-record races). Every 200 ms, per device: record
an anchor event `A_k` on that device's own calibration stream (created by the
Tier B thread under a `DeviceGuard`; it never calls `HeterogeneousEngine`
helpers -- `set_current_cached` caches a per-thread `hipSetDevice` in an
engine-wide atomic, and one call from another thread would send the scheduler's
kernels to the wrong GPU) and spin on `query()` with a RAW stamp before and
after each poll, the spin BOUNDED at 2 ms (a HIP stream can share a hardware
queue with a busy stream: then discard the anchor and retry later). The bracket
width plus the CLR start/end bias of `elapsed` (start side = the event's start,
host observes its completion: ~1-3 us) is the anchor's error, `quality_us`.

The HISTORY is numbers, not events: when `A_k` is taken, `elapsed(A_{k-1}, A_k)`
is measured while both are alive, giving every anchor a coordinate on one
continuous device timeline plus its RAW stamp -- minutes of history from 2-4
live anchor events (generation-tagged; a generation mismatch drops the
conversion, counted). An event converts with ONE call: `elapsed(A_live, e)`
against the newest live anchor, then the piecewise-linear device-timeline -> RAW
map; events after the newest anchor EXTRAPOLATE (never wait for a bracketing
anchor, which would hold pools), slope clamped to +-200 ppm. Precision: f32 ms
from `elapsed` is ~15 ns at 200 ms, ~2 us at 20 s (prefill units). The chain is
a free self-check every 200 ms: `elapsed(A_{k-1}, A_k)` vs the RAW delta must
agree within the two brackets plus 200 ms x drift; the residual is logged
(catches counter jumps, clock-translation steps, runtime-PM resets). All streams
of one device share the timestamp counter, so one calibration stream per device
serves every stream.

**Pools handed off at every reset, harvested off the critical path (R1.2,
R1.4, R2.4, R2.5).** `EventPool::reset()` ITSELF becomes the handoff, so every
reset site is covered (multistream.rs ~1715; forward_prefill.rs ~1170, ~1343,
~1456, ~2658, ~3150, ~3343; engine.rs ~1022; the single-layer bench): if the
buffer holds pairs, it goes to the Tier B thread with its stage contexts, its
step / unit ids and its perfetto watermark (the watermark lives IN the buffer),
and a free buffer is taken; an empty buffer is just cleared (tick-start resets
do not burn buffers). Legacy callers that `harvest()` synchronously before their
reset keep working on the same buffer. While the perfetto exporter lives (until
P4) a reset point exports first, then hands off. Buffers per device: 6 (events
are cheap; the ROCr signal budget for 6 x 16384 x 2 devices is verified in P3,
fallback 4). The buffer type gets a justified `unsafe impl Send` (a raw
`hipEvent_t`, the `Graph` pattern); the handoff and return channels have
capacity = the buffer count, so no `try_send` ever drops a buffer.

Every `StageScope` carries the buffer's EPOCH; a scope that ends after a handoff
(not possible today -- resets are top-level -- but checked) drops its pair,
counted. The step id is allocated at the TOP of `decode_rows` (today `EV_STEP`
is incremented after the drafter runs, so the drafter would carry the previous
step's id).

The Tier B thread processes a buffer's pairs AS THEIR END EVENTS COMPLETE
(sleep-poll `query()` every ~0.5 ms, never a spin, never an error), converts
each event (one `elapsed`), emits `dev` records and the buffer's `step_dev`,
and returns the buffer with its device SUMS (2.4.1). After a 2 s timeout the
pairs still incomplete are dropped (counted) and the buffer returns: a wedged
stream cannot hold buffers forever. If no buffer is free, the step records
nothing (stages no-op, counted in `step_dev`) -- never a wait. The lag is NOT
random: the async ring writes exist only in lone-stream DSpark steps, so their
buffers return last; 6 buffers and the 2 s timeout keep that from biasing
against DSpark steps, and `step_dev` carries Tier B's lag and the per-step
deferred / dropped counts so a bias would show.

Prefill units (R2.13): an LM window can exceed 8K pairs per unit; with kernel
sub-stages off the expected drop rate is measured in P3, and prefill units
get their own larger buffers if it is not ~0.

This moves today's synchronous harvest (~0.4 ms per two-lane step, ~1840 pairs)
OFF the scheduler thread. The Tier B thread now makes ~2-4K HIP calls per step
(`query` / `elapsed`) concurrently with the scheduler's launches; CLR takes
per-queue locks on some of those paths, so P3 measures the scheduler thread's
LAUNCH wall with Tier B busy vs idle, not only the harvest it removed.

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

**2.4.1 Consumers (R2.6).** `hub_step` stays on the scheduler thread, emitted
at once (routing it through Tier B would let a Tier B drop lose a Tier A
record), with its non-device fields and `profiled`; its `d_*` / `i_*` become
NaN. The per-step device sums go to `step_dev` (Tier A, from the Tier B thread)
with `t_start` = the step's start (a time field: windowed reads and `cut` keep
it) and are joined back by `(pid, step)` -- `EV_STEP` is process-local.
`evtrace.py load()` MERGES `step_dev` into `hub_step` by `(pid, step)`, so every
existing expression (`report`'s `dgpu_busy_ms` / `igpu_busy_ms`, the `r(field,
fwd_ms)` correlates, `hist` / `corr` / `csv -k hub_step -e d_...`, which today
turn a missing field into NaN silently) keeps working; `evt2perfetto.py` does
the same for its step args. The `ms.stage` rollup stays ONE block from ONE
thread: a returned buffer carries its device sums and the scheduler folds them
into `profile_acc` when it takes a free buffer (1-2 steps late, harmless for
20-step means), with an explicit `steps_dev` count next to `steps`, so
`windows.sh` / `audit_ab.sh` (which parse `ms.stage.total` as a block reset)
are unaffected.

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
| P3 | calibrator on the Tier B thread, `reset()` as the handoff, `dev` + `step_dev` + readers' merge, stage context + stream + a host stamp in `TimingPair`, box-2 `dev` behind a knob | yes | the causality rules below hold (violations counted, ~0); scheduler-thread time per step drops by the old harvest (~0.4 ms) with the LAUNCH wall unchanged with Tier B busy vs idle; calibration residual within its bound; parity vs the 10-01 exporter in a GPU gate window (box 2 attached) |
| P2 | Registry + `EvtLayer` + `evtrace::Span`, coverage pass (2.3), box-2 subscriber | yes (sign-off: dep) | span cost per step measured (<= 80 us total); re-entrancy tests pass |
| P4 | converter v2 (protobuf, flows, CPU / device tracks); retire `DeviceTimingExporter`, `ExpertdTracer` / `TrackExporter`, the tracing-perfetto test | yes | one `trace_now.sh --dump` yields both boxes' CPU + device + request + paging tracks for a DSpark window |
| P5 | auto-dump on anomalies (a step > 3x the rolling median, a stream abort, a box-2 reconnect), rate-limited | yes | -- |

P3 goes before P2: device intervals are what the DSpark request needed, and
the layer carries the most re-entrancy risk.

**P3 causality rules (R2.7)**, on data P3 has without P2's spans:
(a) `stage()` takes a RAW host stamp when it records the start event (one read,
~25 ns, ~45 us per two-lane step; every 4th stage if measured too costly):
`dev.t_start >= t_host_record` -- a marker cannot run before it was enqueued;
(b) every dGPU compute-stream forward stage of step N ends before the
ready-first driver's `compute.synchronize()` returns (forward_prefill.rs ~4226;
a RAW stamp at its return goes into `hub_step` as `t_fwd_sync`);
(c) head stages end before the head readback returns (stamped likewise).

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
- **Memory.** Hub ring 128 MB (+ up to one cloned ring while a dump writes) +
  6 buffers per device (events only) on ~3-5 GB free; box-2 dumps in tmpfs
  capped at 256 MB.
- **Disk.** Tier A unchanged; hub dumps to its disk; box-2 dumps to tmpfs,
  deleted after the pull.
- **Clock.** Device intervals carry their calibration bracket (`quality_us`);
  the converter's causality check catches a bad calibration.
- **Box 2.** Device records off by default (its iGPU is the bottleneck); a
  subscriber at `warn`; one new dependency (sign-off).
- **What we give up.** Perfetto-native data we do not emit (sampled stacks,
  kernel scheduler tracks); perfetto's `traced` can add those next to our output
  if ever wanted.

## 6. Round-3 notes, binding on the implementation

- **S1 step attribution.** A buffer handed off at step N+1's reset holds step
  N's forward, head and ring writes AND step N+1's drafter. Tier B sums by EACH
  PAIR's stage-context step and emits one `step_dev` per (step, device) present
  in the buffer; the readers' merge SUMS partial `step_dev` records per
  `(pid, step)`. `d_*` / `dgpu_busy` keep today's parent-prefix rule (`dgpu.*` /
  `igpu.*` names only, multistream.rs ~1938-1947), so the newly covered `mtp.*`
  drafter and ring-write stages do not change their meaning (they get their own
  `ms.stage` rows). P3 test: synthetic steps with distinct per-step durations,
  checked through the join.
- **S2 calibrator under load.** `quality_us` grows with extrapolation distance
  (|t - t_anchor| x 200 ppm), not just the bracket; the newest VALID anchor is
  never retired before a new one is chained; P3 acceptance includes the anchor
  success rate under full two-lane decode and the p99 extrapolation distance; if
  anchors fail under load, try a high-priority calibration stream (verify CLR's
  per-priority HW-queue pools on gfx1201 / gfx1151 first). The self-check
  tolerance is both anchors' `quality_us` (bracket + the 1-3 us start/end bias,
  which every chain link carries).
- **S3 quiet threads.** Each thread's fine-span buffer is registered in a global
  list (per-buffer mutex, uncontended on push, or an SPSC ring); the Tier B
  thread drains them every ~100 ms and before every dump, and a dump includes
  each thread's OPEN spans (its enter stack) as `open` records -- what every
  thread was doing at the moment of an anomaly.
- **N1** causality rule (b) applies to all four arena drivers (each ends in
  `dgpu.compute.synchronize()`, forward_prefill.rs ~3672, ~3856, ~3982, ~4226):
  `t_fwd_sync` is stamped at whichever returned.
- **N2** the host stamp of rule (a) is taken BEFORE `hipEventRecord`, tolerance
  `quality_us`.
- **N3** `ms.stage` device rows and `ms.stage.total`'s `dgpu_busy_ms` /
  `igpu_busy_ms` divide by `steps_dev`, not `steps`.
- **N4** a step that got no buffer is counted scheduler-side in `hub_step`
  (`dev_skipped`), so readers tell "skipped" from "profile off".
- **N5** the scheduler thread flushes its fine-span buffer at every TICK end
  (prefill-only ticks have no step end).
- **N6** an `evtrace::Span` site id is cached per call site (a static in the
  macro): the guard never takes the interner lock after first use.

## 5. Owner decisions

1. Ring sizes: **128 MB on both boxes** (owner, 2026-10-01).
2. `tracing-subscriber` as a dependency of `deepstrix-expertd`: **approved**
   (owner, 2026-10-01).
3. Open, defaults until decided: Tier B ON in production (device data reuses
   the profile already on; the critical path gets cheaper); box-2 device records
   OFF by default; P5 auto-dump after P4.

## 7. P3 as built (2026-10-02)

Code: `het/trace.rs` (stage context, buffers, the handoff in `reset()`),
`het/evtrace_dev.rs` (calibrator, conversion, `step_dev` / `dev` / `cal`),
`scripts/evtrace.py` / `evt2perfetto.py` (merge, device tracks, causality).
Where it differs from sections 2.4 / 6:

- **Conversion bias.** CLR 7.2 `hipEventElapsedTime` subtracts END timestamps of
  both events and `hipEventRecord` always enqueues a NEW marker, so chain links
  and conversions carry no start/end bias; the anchor bound is half its bracket
  + 2 us.
- **Rules (b) / (c) are one mechanism.** `EventPool::note_sync(stream)` after a
  sync returns stores (stream, RAW, pairs so far): every pair already ENDED on
  that stream must end on the device by then. Called after the four arena
  drivers' final dGPU sync (also `hub_step.t_fwd_sync`) and after the head's
  readback sync. Tier B counts per step (`step_dev.viol_a` / `viol_b` /
  `checked_b`); the converter re-checks (a) and (b) from the records.
- **`step_dev.t_start`** = the step's first stage START on that device
  (converted), not the host's step start.
- **Spare buffers** are created on the Tier B thread at a pool's first handoff
  (its first epoch with pairs is skipped: counted); default 6 per pool
  (`V41_EVTRACE_DEV_BUFS`); returned buffers are drained at every reset, so
  their sums reach `ms.stage` at the next step.
- **`hub_step.dev_skipped`** = resets since the previous step (prefill units
  included) that found no free buffer.
- **Kill switch**: `V41_EVTRACE_DEV=0` (live) = the synchronous harvest again.
- **Polling (review 27).** CLR's `hipEventQuery` on an incomplete event
  enqueues a notify marker on its queue (once per event). Tier B queries only
  each stream's LAST end event of a buffer and converts the stream whole once
  that one completed: at most one such marker per still-running stream per
  buffer, from the Tier B thread.
- **Calibration follows traffic (review 28).** A device with no handoff for
  10 s is not anchored (an idle hub, box 2 with its records off); the next
  handoff anchors at once; an anchor more than 2 s after the previous one
  starts a new chain (the slope is kept, its bound widens until 1 s of span).
- **Spare buffers** have the pool's capacity unless it asks otherwise
  (`EventPool::set_spare_capacity`: box 2's executors 4096; the hub keeps
  16384).
- **Sync marks only while recording** (review 29): `note_sync` is a no-op
  unless the pool is enabled AND offloading (a disabled pool is never reset:
  box 2's knob off would pile one mark per request up forever).
- **Used events renewed before a buffer goes back** (2026-10-03, hypothesis,
  not yet measured). A recorded hipEvent keeps its marker, and through it an
  HSA interrupt signal (one KFD event slot), until it is recorded again (ROCm
  CLR/ROCr source). Knob off, a pool re-records one buffer from slot 0: ~one
  epoch of events live. Knob on, it rotates through `V41_EVTRACE_DEV_BUFS`
  buffers: ~6 epochs live, across both pools past the per-process ~4096 KFD
  signal events at multi-lane steps, after which waits poll (decode / prefill
  slower with more lanes). Tier B now destroys `events[..next]` of a finished
  or timed-out buffer and creates fresh ones on its device before returning
  it (`Buf::renew_used`; a failed create keeps the old event, counted as
  `renew_failed` in the minute line). A/B: record-call latency in
  `tests/evtrace_dev_gpu.rs` at `EVTRACE_GPU_PAIRS=3000`,
  `V41_EVTRACE_DEV_BUFS=2` vs `6`, before and after.
- **Box-2 `dev` records** (`V41_B2_EVTRACE_DEV`, by that full name, live,
  default off; recorded only when its Tier B takes the buffers): each
  request's GPU span `b2.run` through the executor's engine pool
  (`EventPool::open` / `close`: a guard-free token that knows its pool and
  stream), `dev.unit` = the request's seq, rule (b) after `ev_done`; the pool is
  handed to Tier B every 100 ms after the replies, on idle ticks, and at an off
  flip (`MoeExecutor::dev_epoch`; the parking executor `exec2` too, its streams
  named `park.*`). Drawn per stream on box 2's process (`trace_now.sh
  B2DUMP=1`).
- GPU smoke test: `tests/evtrace_dev_gpu.rs` (`--ignored`, server down, ~10 s;
  `EVTRACE_GPU_STEPS=3000` long, `EVTRACE_GPU_PAIRS` empty stages a step,
  default 600 ~ the hub's): conversion vs `elapsed` twins, causality, anchors,
  every record call's latency early vs late -- run with
  `V41_EVTRACE_DEV_BUFS=2` and `=6` to compare -- and the anchors' bracket bound.

## 8. As built vs sections 2-3 (docs audit 2026-10-04)

Checked against `het/evtrace.rs`, `het/evtrace_kinds.rs`, `het/evtrace_ring.rs`,
`het/evtrace_dev.rs`, `scripts/trace_now.sh` and `scripts/evt_dump_now.sh`.

**Kinds** (`evtrace::kinds()`: `meta` 0, `sys` 1, `str` 3, `site` 4, `knob` 5, `dev` 6, `cal` 7,
plus `evtrace_kinds::ALL`: `hub_req` 10, `hub_step` 11, `hub_phase` 12, `step_dev` 13,
`b2_req` 20, `b2_read` 21, `b2_ensure` 22, `b2_write` 23):
- `str` (3) packs **six** bytes per f64, as an exact integer (byte i at bit 8i), not eight —
  raw bit patterns could form NaNs a reader might not preserve.
- `dev` is id **6**, not 31. Fields: `t_start, t_end, name, device, stream, step, unit, layer,
  lane, t_host, q_us`. `name` / `device` / `stream` are string ids (no numeric 0/1/2 device code);
  `t_host` is the host's RAW stamp before it recorded the start; `q_us` replaces `quality_us`.
- `step_dev` is id **13**, not 32.
- `cal` (7) is new: one device-clock calibration anchor per record (`t, device, ok, spin_us,
  q_us, resid_us, tol_us, slope_ppm, slope_q_ppm, link_ms, anchors`).
- `span` (30) and `instant` (33) do not exist (P2 not built). `site` (4) is defined but nothing
  registers span callsites yet.

**P1 ring and dumps** (`evtrace_ring.rs` module doc):
- No Tier B channel. A producer serializes into its own thread's buffer (`emit_b`); a full buffer
  is copied into the ring's open block if the ring lock is free; the Tier B thread drains every
  registered buffer every 100 ms and is the only thread that seals or evicts. Records sit in batch
  order, not time order; readers sort.
- Dump file name: `<ring dir>/<role>-dump-<utc>-<pid>-<n>.evt`, streamed block by block in synced
  4 MB pieces; pruning (`V41_EVTRACE_RING_KEEP_MB`, default 256) touches only that role's
  `-dump-` files.
- Ring dir: `V41_EVTRACE_RING_DIR`, default `/dev/shm/evtrace-ring` on box 2 (role `b2`) and
  `<V41_EVTRACE_DIR>-ring` elsewhere; it may never be the Tier A dir. The hub's
  `V41_EVTRACE_DIR` is `~/logs/evtrace` in production (hub env 2026-10-04), so its ring dir is
  `~/logs/evtrace-ring`.
- Request file as planned (`<ring dir>/dump-request`, seconds or `all`, temp-then-rename, polled
  every 250 ms); `scripts/evt_dump_now.sh RING_DIR ROLE SINCE_RAW_NS` writes it and waits.
- `trace_now.sh` has no `--dump` flag. The hub dump is on by default (`DUMP=1`; `DUMP=0` skips it,
  `RING=` overrides the dir, windows older than `DUMP_MAX_S`, default 150 s, get none). The box-2
  dump is opt-in (`B2DUMP=1`, `B2DUMP_MAX_S`). A box-2 dump is **not** deleted after the pull; only
  the script's temp dirs are removed.

**Production state.** Code default `V41_EVTRACE_DEV=1` (`knobs::EVTRACE_DEV`, live). Production
sets `V41_EVTRACE_DEV=0` (hub env 2026-10-04): since 2026-10-02 03:34 UTC, after Tier B device
timing measurably regressed decode (two-lane DSpark about +100 ms); cause not isolated.
The section 5 default "Tier B ON in production" is therefore not what runs, and a launch without
that setting turns the regression back on. The 2026-10-03 event-renewal change in section 7
(`Buf::renew_used`) is one hypothesis for it and is still unmeasured. Box-2 device records
(`V41_B2_EVTRACE_DEV`) default off.
