# Tracing rebuilt around evtrace

Status: PLAN (2026-10-01), for architect review. Owner's call, 10-01: keep
building on evtrace rather than go back to emitting perfetto directly, with our
own tracing layer for the CPU side.

## 1. Why

Tonight's DSpark trace request showed the state of things:

- **Three trace writers, three formats, three clocks.** evtrace (both boxes,
  binary records, CLOCK_MONOTONIC_RAW, always on); the hub's
  `DeviceTimingExporter` (perfetto protobuf, CLOCK_REALTIME, opt-in at
  startup); box 2's `ExpertdTracer`/`TrackExporter` (perfetto protobuf, box-2
  realtime, `expertd --trace` at launch). `tracing-perfetto` survives in one
  test.
- **They drift.** The hub exporter was only drained on the serial decode path;
  when decode moved to the arena it silently recorded nothing for weeks
  (fixed tonight, 0773ad1..464146e, not deployed).
- **evtrace is where the analysis lives**: request joins across the boxes by
  seq, the clock-offset fit, percentiles over 40-field records. Nearly every
  measured result of the last two weeks came from it. But it has no strings, no
  CPU spans, and no device intervals: device time only as per-step sums by
  stage (`hub_step` `d_*`/`i_*`), and host phases only as per-step aggregates
  (`lh_*`).
- **The arena path has almost no `tracing` spans** (23 of the 28 span sites are
  on the legacy serial path in forward_layer.rs).

Goal: ONE recording pipeline (evtrace, both boxes, one clock), carrying CPU
spans and device stage intervals too; perfetto becomes a VIEW generated from it
(`scripts/evt2perfetto.py`); the protobuf exporters retire.

## 2. Target

### 2.1 Two tiers in one recorder

| tier | what | rate | where |
| --- | --- | --- | --- |
| A, always on | today's kinds (`hub_step`, `hub_req`, `hub_phase`, `b2_*`, `sys`, `meta`) + low-rate new ones (`knob` changes, string definitions) | ~1-10K rec/s | disk, as today (rotation, keep N) |
| B, flight recorder | CPU spans, device stage intervals (hub dGPU/iGPU, box-2 iGPU), optionally `k.*` kernel slices | ~20-40K rec/s, ~1 MB/s | in-memory ring per process; written to disk only when DUMPED |

A dump writes the ring's last N seconds (default: all of it) to
`<dir>/<role>-<utc>-<pid>-ring.evt` with the same header, string table and
record format, so every reader handles both. The ring lives in the evtrace
writer's domain: producers never block on it.

### 2.2 Format EVT2

- `b"EVT2"`, header JSON as today plus `"strings": [...]` (the table at file
  open) and `"tiers"`.
- New reserved kinds:
  - `str` (id 2): `[id, len, bytes packed 8 per f64 slot...]` — a string
    defined after the header. Strings are interned process-wide (`u32` ids).
  - `span` (Tier B): `t_start, t_end, name, target, thread, span_id, parent_id,
    f0..f3` (name/target/thread are string ids; `f*` = up to 4 numeric span
    fields, named in the span's `str` metadata).
  - `dev` (Tier B): `t_start, t_end, device, stream, stage, layer, lane, step`
    (stage = string id; device 0 dGPU / 1 iGPU / 2 box-2 iGPU).
  - `knob` (Tier A): `t, name, value, source` (value as string id).
- **Strings are never lost to a full queue**: producers add to the interner
  (lock, assign id, push the bytes into a pending list) BEFORE sending the record
  that uses the id; the writer drains the pending list before writing any record
  it dequeues, and writes the whole table into every new file's header (rotation)
  and every dump. A dropped record can no longer orphan a name.
- Readers: `scripts/evtrace.py` and `evt2perfetto.py` accept EVT1 and EVT2
  (EVT1 = no strings, no Tier B).

### 2.3 CPU spans: `EvtLayer` (a `tracing_subscriber::Layer`)

- Records span enter/exit per thread into Tier B: `on_new_span` interns name +
  target (+ field names) and stores up to 4 numeric fields in the span's
  extensions; `on_enter` stamps RAW ns in the extension (per thread: a span can
  be entered on several threads); `on_exit` emits one `span` record. Events
  (`tracing::info!` etc.) are NOT recorded (the log is the log).
- Thread ids: `std::thread::current().name()` interned once per thread
  (thread-local cache), else the OS tid.
- Filtering: a per-layer `Targets` filter with a reload handle, driven by a live
  knob `V41_EVTRACE_SPANS` (`off` / `info` / `debug`, default `info`), so a
  disabled span costs the callsite check only.
- **Re-entrancy rules** (review round 22): the layer must not take any lock or
  `RefCell` borrow that its callers may hold — knob `RESOLVE`/`WARNED`, the
  event pools, exporter mutexes, the evtrace interner — except the interner's own
  lock, never held while calling out. Every path inside the layer is
  allocation-free after a span's first use (interned ids, fixed-size record).
- Composition: hub = `fmt` (as today) + `EvtLayer`. Box 2 gets a subscriber for
  the first time: `fmt` to stderr at `warn` (so the kernels' existing `info!`
  lines do not suddenly flood its log) + `EvtLayer`; its `eprintln!` lines stay.
- Span coverage pass on the arena path (P2): scheduler tick, prefill unit
  (chunk / LM window / finish), decode step and its phases (Engram hash +
  gather + join, drafter, forward per lane per layer, head + sampling, accept +
  ring writes), box-1 pager `ensure` / `drain_prefetched`, remote submit / wait,
  the knobs watcher pass. Box 2: request dequeue, merge, run_path, ensure, D2H,
  reply write, reader threads.

### 2.4 Device stage intervals

- Source: the event pools the profile already fills every step
  (`V41_MS_PROFILE=1` in production). Today `harvest()` calls
  `hipEventElapsedTime(start, end)` per pair (2 events per pair, ~1600-3000
  pairs per two-lane step).
- Change: `harvest` measures every event against ONE base event per pool
  (`elapsed(base, e)`, the same number of HIP calls), giving start/end offsets;
  absolute RAW time = `t_sync - elapsed(e, last)` where `t_sync` is the RAW
  stamp taken right after `harvest` synchronizes the last end event (it already
  does). No extra syncs, no anchors, no per-step re-anchor (tonight's perfetto
  path syncs four streams per step for that).
- Attribution: a thread-local stage context `(step, layer, lane)` set by the
  arena drivers at each layer / lane phase entry, captured into `TimingPair` at
  `stage()`. The ready-first driver interleaves lanes on one thread, so the
  context is set per phase call, not per layer loop.
- Rollup unchanged: `ms.stage` and `hub_step` `d_*`/`i_*` come from the same
  harvest.
- Emission: one `dev` record per pair into Tier B, from the scheduler thread
  after the harvest — cost per step to be measured (P3 acceptance: < 0.5 ms per
  step on the scheduler thread; if not, hand the pair list to the evtrace writer
  thread and convert there).
- Box 2: the per-request GPU work `ExpertdTracer::gpu_slice` times today
  (`moe L{layer} B={b}`, remote_experts.rs ~7444) becomes `dev` records
  (device 2) with `seq`, so a box-2 request's kernels join its `b2_req`.

### 2.5 Dumps and triggers

- `V41_EVTRACE_RING_MB` (static; default hub 64, box 2 256 — the hub has ~3-5
  GB free) sizes the ring; 0 = Tier B off.
- `V41_EVTRACE_DUMP` (live knob, edge-triggered on change like
  `V41_PERFETTO_STEPS`): `N` = dump the last N seconds now. The writer thread
  swaps the ring for a fresh one under a short lock and writes the snapshot
  on its own thread; producers never wait on the disk.
- Both boxes in one go: `scripts/trace_now.sh --dump N` writes the knob on the
  hub and on box 2 (its knob file), waits for both dumps, pulls box 2's over the
  link rate-limited (as today's cut), converts.
- P5 (optional): auto-dump on an anomaly (a decode step > 3x the rolling median,
  a stream aborted, a box-2 reconnect), rate-limited, so the moment after
  something odd is captured without anyone watching.

### 2.6 The converter is the only perfetto writer

`evt2perfetto.py` grows: CPU span tracks per thread (nested by time), device
tracks per device / stream / lane (stage + layer in the name), flow arrows from
each hub request to its box-2 request (`seq`) and on to its box-2 device work,
knob instants from `knob` records, the existing request / paging / read tracks.
It keeps bounded memory (streams, window-only dicts, 2 GB cap in trace_now).

## 3. Phases

Each phase: code + host tests + persistent-reviewer rounds; deploys batched with
the owner's OK (hub-only where possible; box-2 parts ride a box-2 restart).

| phase | content | acceptance |
| --- | --- | --- |
| P0 | EVT2: string interner + `str` kind + header table; readers accept EVT1/EVT2; `knob` kind fed by the knobs watcher | round-trip tests; a dropped-record test cannot orphan a string; existing analyses unchanged on EVT1 files |
| P1 | Tier B ring + `V41_EVTRACE_RING_MB` + `V41_EVTRACE_DUMP` on both boxes; `trace_now.sh --dump` | dump of a synthetic ring (test); on the hub: dump under load does not move decode p50/p99 (A/B 10 min) |
| P2 | `EvtLayer` (hub + box 2 subscriber) + span coverage pass (2.3) | span overhead per decode step measured (target < 50 us); nesting correct on the converter's CPU tracks |
| P3 | device intervals from the harvest (base-relative elapsed, stage context) + box-2 `dev` records | parity against tonight's perfetto exporter on the same steps: same stages, start/end within 50 us; harvest cost per step measured |
| P4 | converter v2 (CPU / device tracks, flows, knobs); retire `DeviceTimingExporter`, `ExpertdTracer`/`TrackExporter`, `V41_PERFETTO_*`, the tracing-perfetto test | one `trace_now.sh` run yields both boxes' CPU + device + request + paging tracks for a DSpark window |
| P5 | auto-dump triggers | — |

## 4. Costs and risks

- **Hot-path cost.** Tier B adds span enter/exit (a layer callback, a RAW
  clock read, a ring write) and per-pair `dev` records. Budgets in P2/P3; spans
  only at phase granularity, never per kernel launch.
- **Memory.** Ring sizes are static and small on the hub; Tier B off with
  `V41_EVTRACE_RING_MB=0`.
- **Disk.** Tier A unchanged (~1 GB/h hub). Dumps are 10-100 MB, written by the
  writer thread; the hub's disk also serves Engram and box-1 expert reads, so a
  dump writes in bounded chunks with a small sleep between them (ionice does
  nothing on these NVMe drives: scheduler `none`).
- **Clock.** Device times inherit the harvest's sync stamp: accurate to the
  wake-up latency after the sync (tens of us), which is also the precision of
  tonight's anchors.
- **Box 2 logging.** A subscriber on box 2 turns on the kernels' `tracing`
  lines there: `warn` level by default, raised only by env.
- **Format churn.** EVT1 files keep working; `evtrace.py` is the reference
  reader; the EVT2 change is additive.
- **What we give up.** Perfetto-native features we do not emit ourselves
  (sampling profiles, counters from the kernel). If wanted later, perfetto's
  `traced` can record system tracks next to our converter output.

## 5. Owner decisions

1. Ring defaults (hub 64 MB ~ 60 s of Tier B; box 2 256 MB).
2. Tier B on in production (recommended: the device intervals reuse the profile
   already on; spans at phase granularity) or opt-in.
3. Box 2's log level once it has a subscriber (`warn` proposed).
4. Whether P5 auto-dump is wanted.
