# Box 2: a read-ahead lane that actually runs (design, rev 2, 2026-10-11)

rev 2 = review r1 (REVISE): readahead keeps the demand-read chunk pause and gets its OWN reader threads (MAJOR-1);
drain-poll landing gated on completed reads, pin set built once per run_path (MINOR-2); park scope stated (MINOR-3);
2.1 implemented as sorted + binary search (8815255f; no N_EXPERT-sized bitset, NIT-4); memory and throughput numbers
made consistent (NIT-5/6).

Owner 10-11: "build that + last-unit-only", "ensure you read straight into destination slot".

## 1. Problem (measured 10-11)

* A cold first window re-reads ~40 GB on box 2 (decode evicts the encoder set between turns; a partition cannot fix it:
  LRU simulation, 4,700-slot pool, both working sets larger than the pool). Box 2 reads it at ~4.4 GB/s vs ~10-12 GB/s
  the drives deliver: reads happen only while a request for that layer is served, nothing reads layer L+1 while L
  computes (drives idle ~half the window). Floor with overlap: ~4.5-5 s vs ~9 s.
* The layer-ahead streaming that should overlap it does not: of 161,875 words the hub sent in a 14-min slot, box 2 ran
  ~14K more speculative reads than without (~1 in 10) -- the rest were dropped for want of a staging set; the
  per-layer first-request stall did not move (210 vs 222 ms). And in multi-window jobs its landings evicted the next
  window's pages (steady windows 0 -> 1.8 misses/request) -- fixed on the hub: streaming only in a job's LAST window
  (`V41_LM_PREFETCH_LAST_ONLY`, 970c16a8).
* Why the words drop (code map, remote_experts.rs = RE):
  1. A background read holds its staging set ~46 ms: read ~10 ms, then ~28 ms until the compute thread lands it
     (`admit_prefetched` runs only at the next request's `ensure` RE:6445 or in the park loop RE:8894 -- not while
     the thread waits for a frame RE:8483, not in `run_path`'s GPU-drain poll RE:7766), then ~6 ms landing.
  2. The ~6 ms landing is the VICTIM SCAN (measured: `victim_scan_ns` p50 5.7 ms vs `repack_ns` 0.3 ms; demand
     claims scan in 0.02 ms): `pick_victim_ranked` tests `extra.contains(&(l, e))` per occupied slot (RE:4145) over a
     linear, non-deduplicated `pinned + parked_pins` list (`push_parked_pins` RE:2010 pushes every row's picks: up to
     ~6K entries for a 1024-row parked prefill request).
  3. Speculative words share the 16 general sets with certain/park reads, are refused when `free <= 2*reserve`
     (`pick_set` RE:2045), and under `route=split` pop only when no certain read runs (`pop_mode` RE:1859), yield up
     to 20 ms behind demand reads (RE:5366-5383) and pause per chunk (RE:1955).

"Read straight into the destination slot": not literally possible -- a read produces the HF layout and the slot holds
the repacked ggml layout; even demand reads pread into pinned staging and the GPU permute (`repack_in_place`) writes
the slot (RE:5150-5164, 6980-7044). What the owner's ask buys -- no slow landing, no wait -- this design gets by making
the landing as cheap as a demand landing (~0.3 ms, the repack) and doing it as soon as the read finishes.

## 2. Mechanism

### 2.1 O(1) pin test in the victim scan

IMPLEMENTED (8815255f): `ExtraPins` is built by `extra_pins` (pinned ++ parked_pins, sorted, deduplicated) and the
scan tests membership by binary search (~10-13 compares instead of up to ~6K). Same exclusion set, same victim; only
the cost changes (unit test vs the linear oracle; a debug_assert checks sortedness at every scan). Expected landing
~6 ms -> ~0.4 ms.

### 2.2 Prompt landing on the compute thread (no new thread, no locks)

IMPLEMENTED (with 2.3): `land_completed` (bounded `admit_prefetched_core`, leaves the `ensure` bookkeeping alone),
gated by `prefetch_done_ready` (an atomic count readers raise before each send). The drain poll lands ONE read per
call (the poll re-checks the GPU between calls, so a finished pass's reply waits for at most one landing); the idle
wait lands up to `land_max_per_poll` per 1 ms slice while reads are outstanding (keep-warm keeps its own period).

Call `admit_landed` (RE:5549, today only in the park loop) at two more points, bounded per call (e.g. <= 4 landings or
<= 1 ms) so a reply is never delayed:
* in `run_path`'s GPU-drain poll (RE:7761-7769, the serve loop's `overlap` closure with park == false), with
  `shard.pinned = cur_pins` around it exactly as the early-page path does (RE:8785-8790; `parked_pins` already covers
  enclosing parks) -- the current pass's picks are excluded as victims, as today. The poll runs every ~20 us: the
  landing call is GATED on a completed read being available (`rx_done` non-empty, checked with a cheap peek/counter)
  and the pin set (`extra_pins`) is built once per `run_path` call, not per poll (review r1 MINOR-2);
* in the idle wait for the next frame (RE:8483): a short `recv_timeout` (e.g. 1 ms) that lands what finished while no
  request is in flight (nothing to pin).
Safety argument unchanged: victims exclude want / pins / parked pins / hub pins / nursery; the repack runs on
`repack_stream` and synchronizes, as park landings already do concurrently with a running pass.
Scope (review r1 MINOR-3): `serve_interleaved` runs `run_path` with a no-op overlap (RE:9445), so requests served
INSIDE a park are covered only by the park loop's existing `admit_landed` after each served frame; the "<= ~1 ms to the
next landing point" estimate does not hold inside a park with a long nested pass.

### 2.3 A read-ahead class with reserved capacity

IMPLEMENTED per review r2 MINOR-A as ONE queue with reader ROLES (not a second queue): `PfJob::readahead`, queued at
the back; `PfQueue::pop_role` -- general readers skip not-certain readahead jobs, readahead readers take only
readahead jobs (certain or not), ungated; `promote` / `urgent` / `pending` are shared, so a promoted readahead job is
certain and either role runs it. Chunks pause with the LIKELY token (demand + certain reads). `pick_set_ra`: own sets
or dropped. Push notifies all readers (two roles on one condvar).

Hub prefill read-ahead words (`prefetch_words_prefill`, `prefill && !certain`) become a READAHEAD class:
* own free list of `V41_B2_READAHEAD_SETS` staging sets (default 8; allocated at load beside the LIKELY sets,
  RE:5171-5179; `release_set` RE:2086 and `n_general` ranges RE:5447 extended);
* `pick_set`: a readahead word takes only a readahead set (never a general one, so park reads are untouched);
* OWN READER THREADS (review r1 MAJOR-1b): `V41_B2_READAHEAD_PAR` (default 2) dedicated "b2-readahead" threads
  serve a separate readahead queue; the general prefetch readers (`V41_B2_PREFETCH_PAR`, 4) never take readahead
  jobs and readahead readers never take certain/park/LIKELY ones -- so a running or paused readahead read can never
  hold the reader a certain (park) read needs, and no `spec_cap` arithmetic is shared;
* THROTTLE (review r1 MAJOR-1a): no certain-read pop gate and no 20 ms pre-yield (the two things that kept
  speculative reads from starting), but the per-chunk pause for DEMAND reads stays (the io_throttle speculative pause
  token, RE:1951-1971: 1 MB chunks, while `DEMAND_READS > 0` a chunk waits <= 20 ms): readahead fills the drives'
  idle time between demand reads and never lengthens the per-layer first-request stall (layer L's demand reads, the
  critical path) -- the same rule LIKELY reads follow with `likely_pause_for_certain` (on by default);
* landing: unchanged path (prefill-class, rank at the pass's current layer; with the last-unit flag the victims are
  the job's passed layers -- `last_sweep_rank`).
Expected (review r1 NIT-6): throughput is bounded by the readers -- each holds one read, ~18.8 MB in ~4-10 ms when
the drives are free (paused while demand reads run) -> 2 readers ~4-9 GB/s of otherwise idle drive time; sets only
need to cover readers + reads awaiting landing: with prompt landing (<= ~1 ms) a set is held ~read + 1 ms, so
2 readers need ~3-4 sets; default 4 sets (headroom for landing delays inside parks). The cold window's ~40 GB then
moves at drive speed in the gaps instead of only on demand.

### 2.4 Knobs (box 2, live where read per word)

`readahead_sets` (startup only: pinned allocation, default 4 -- each set ~32 MB pinned (role0 is coalesce-sized), so
~128 MB more pinned host memory on box 2, ~6.6 GB available; 0 = today's behaviour, readahead words fall back to the
general path), `readahead_par` (startup: reader threads, default 2), `land_poll` (live flag: the new landing points,
default on), `land_max_per_poll` (live, default 4). Old hub: no readahead words (fine). The hub side ships with
`V41_LM_PREFETCH_AHEAD=0` until the A/B.

## 3. Measurement

A/B on the hub's `V41_LM_PREFETCH_AHEAD` 2 vs 0 (live), `V41_LM_PREFETCH_LAST_ONLY=1`, box 2 with 2.1-2.3 on:
first-window time (stratified by job shape), per-layer first-request stall, box-2 `victim_scan_ns`, readahead drops,
set hold time, decode tok/s guard. Separately before/after the redeploy: background `victim_scan_ns` (2.1) and
park-read set hold (2.2) on the same metrics with AHEAD=0.

## 4. Tests

Host: `pick_victim_ranked` with the bitset == the old linear exclusion on randomized pools (oracle); `pick_set` /
`pop_mode` readahead class (no general sets taken, no certain gate); landing-point bound. GPU: the existing
`remote_experts_pin_loopback` / `remote_experts_nursery_loopback` gates + a loopback that streams readahead words
during a parked prefill pass (no pinned/wanted slot ever a victim).
