# Profiling / tracing audit (2026-09-21)

Scope: everything that measures the two-box decode path at 8 rows — the hub's
`V41_MS_PROFILE` stage rollup, the LH_* host counters, the perfetto exporters
(hub and daemon), the daemon's page stats and per-request records, the client's
link/hop counters, the analysis scripts, and the external tools available on the
boxes. Read against the question we are actually asking: *where do the ~70 ms of
box-2 idle per 8-row step come from?*

## 1. What exists

| Layer | Switch | Output | Cost |
|---|---|---|---|
| `EventPool::stage` (HIP event pair per stage, both devices) | `V41_MS_PROFILE=1` (ON in the production launch line), `DEEPSTRIX_TOKEN_PROFILE`, `DEEPSTRIX_PREFILL_PROFILE`, or perfetto attached | `ms.stage` rollup every 20 steps (means only) | see §2 |
| `LH_*` host counters (`LayerHostTimer`) | forced on by `V41_MS_PROFILE` | `lh.*` lines in the same rollup, BOTH-lane sums | ~0 |
| `trace::phase` atomics (`REMOTE_RTT/PAGE/COMPUTE/MISSES`, `SEL_SYNC`, `ENSURE`, `REMOTE_SRV`, `ENGRAM_STAGE`) | always | `host.*` / `box2.*` lines | ~0 |
| `ms.step` line per step (wall, fwd, engram, sample) | always | log | ~0 |
| Hub perfetto (`DeviceTimingExporter`) | `V41_PERFETTO_OUT=<path>` | device tracks + `remote.expert (host)` + pager tracks | large, and BROKEN on the multistream path (§3) |
| tracing-perfetto host-span layer | never installed in the server (tests only) | — | — |
| Daemon perfetto (`ExpertdTracer`, `--trace`) | not in `restart_expertd_b2.sh` | request span, one GPU slice per request, SSD reads AT LOAD only | unused |
| Daemon `RequestRecord` (read/queue/h2d/gpu/d2h/write µs) | always, 100k ring | `summarize` p50/p90/p99 ONLY at connection close (= every hub restart) | ~0 |
| Daemon page stats | always | one stderr line per 2000 requests (means) | ~0 |
| Client link stats (`link_stats`, `HOP_*`, `ClockSync`) | always | reported only by the LEGACY decode loop (`decode.loop.summary`); never on the multistream path | ~0 |
| `~/scratch-ms/windows.sh` | — | per-20-step rows/wall/dgpu/rwait/b2page/b2comp/miss/"b1 ensure" | — |
| `scripts/analyze_trace.py`, `~/scripts/analyze_pftrace_gaps.py`, `scripts/kt_agg.py` | — | gap analysis of a pftrace; rocprofv3 CSV rollup | — |

## 2. Overhead of what is ON in production

Measured with `tests/bench_event_overhead.rs` (timing-enabled `hipEventRecord`
pair around a tiny kernel, 2000 iterations, both GPUs, server live):

| | dGPU gfx1201 | iGPU gfx1151 |
|---|---|---|
| kernel only, host / stream wall | 0.77 / 2.98 µs | 0.97 / 6.07 µs |
| kernel + event pair, host / stream wall | 2.11 / 11.16 µs | 3.66 / 11.54 µs |
| **pair adds** host / stream | **+1.3 / +8.2 µs** | **+2.7 / +5.5 µs** |
| `hipEventElapsedTime` harvest | 0.22 µs/pair | 0.23 µs/pair |

Pairs per 8-row two-lane step (from `calls` in the `ms.stage` block of
16:08 UTC): **1020 dGPU + 280 iGPU**. So `V41_MS_PROFILE=1` costs per step:

- ~8.4 ms of dGPU stream time and ~1.5 ms of iGPU stream time (of 92 / 114 ms busy),
- ~2.1 ms of hub host-thread time inside the forward (on the critical path of
  the lane turnaround), plus ~0.3 ms harvest after it (`ms.step` `other` = 2.1 ms
  total incl. embed and finish, so the harvest is not the problem).

62% of the dGPU pairs are `k.*` kernel sub-stages that `ms.stage.total` never
sums (it takes `dgpu.*` / `igpu.*` parents only); they exist for perfetto,
which is not attached. Gating `k.*` behind "perfetto attached" removes ~630
pairs/step (~5 ms dGPU, ~1 ms host) at no loss to the rollup. Whether the
remaining 8 ms shows up in tok/s needs a `V41_MS_PROFILE=0` A/B (one hub
restart, 5-minute warm rule); it was NOT run today because the server had live
user streams (45K-token prompts admitted 16:37-16:39).

Everything else that is always on (`V41_MISS_HIST`, `link_stats`, daemon
records, `ms.step` line) is atomics and one log line; negligible.

## 3. Broken or misleading

1. **`V41_PERFETTO_OUT` is unusable on the multistream path.** The arena step
   never exports or re-anchors the pools (`for_each_pair*` is called only from
   `forward_token_impl` and the prefill chunk loop). With `V41_MS_PROFILE=1`
   the step's `reset()` discards every pair before export, so the trace has
   no GPU tracks at all; with `V41_MS_PROFILE=0` nothing resets the pool and
   `EventPool exhausted` errors the step after ~6 steps (16384 / 2600 records).
   The only slices that would land are the host `remote.expert` submit/wait
   pairs and the `remote.pager` slices.
2. **`box2.compute_ms` includes box 2's paging.** The daemon's `t_compute_us`
   brackets all of `run_path`, which calls `ensure_layer` internally. In the
   16:08 block: `box2.compute_ms` 232 = page 132 + ~100 of real service.
   Same for the daemon's own `gpu` column in `summarize` (`timing.gpu =
   t1.elapsed()` spans paging), which is why its p99 is 14 ms at B=3.
3. **`host.remote_rtt` is the exposed wait, not an RTT.** It is fed
   `t_wait_end - t_wait` (identical to `lh.remote_wait`, 161.21 vs 161.17).
   The submit→reply RTT (`partial.rtt_us`) is not accumulated anywhere on this
   path. Per the "exposed wait is not work" rule this column must never be
   quoted as link cost.
4. **Dead columns on the multistream path**, all reported as 0.00 every
   block: `host.sel_sync`, `host.ensure`, `host.engram_stage`,
   `host.remote_srv`, `lh.pre_moe`, `lh.post_moe`, `lh.engram`, `lh.owns`,
   `lh.audit`, `lh.work_items_sync`, `prefetch.*` (when B1 prefetch has nothing
   to do). `windows.sh`'s "b1 ensure" column reads `host.ensure` and is
   therefore always 0; the live number is `lh.ensure` (7.1 ms/step).
   `host.remote_srv` is the one that matters: box 2's `t_server_us` is in
   every reply, so the link-vs-service split (`remote_link_us`) is a two-line
   fix and we currently cannot say what the wire + wake-up costs per step at 8
   rows.
5. **Daemon page-stats `pread` is polluted.** `pread 20.98 ms/miss` against
   `read 4.34`: the miss chunk differences the process-wide
   `expert_read_profile()` counter, which the prefetch reader pool also
   increments, so concurrent prefetch preads are attributed to the demand
   misses. Ignore that bracket until it is per-thread.
6. **`lh.pager_block` has ~31 ms/step unattributed.** 100.2 = sel_sync 50.3 +
   remote_submit 7.2 + ensure 7.1 + remap_h2d 3.6 + excl 0.7 + **31.3 ms of
   nothing timed**. Inside the block and untimed: the `V41_PAGER_SYNC_IGPU`
   full iGPU drain (a cross-lane sync — lane A waits for lane B's MoE), the
   three D2H copies after `sel_sync` (sel/look/look2, outside the `_t_sync`
   scope), the host pick partition / hash loops, and the remote-submit
   readback (`de.compute.synchronize()` + xq/ew D2H inside `lh.remote_submit`,
   so that 7.2 ms is mostly a dGPU sync, not the send).
7. **Stage brackets on one stream can include host stalls.** Known and
   fixed for `ffn_combine` (split into `.local` / `.remote`); the rule is
   documented at the site but nothing enforces it. `dgpu.router` ends before
   `sel_sync`, so it is clean; any future bracket around a `synchronize()`
   silently inflates "busy".
8. **Perfetto timestamps are CLOCK_REALTIME** (`now_ns` = `SystemTime`). Box
   1's clock stepped ~8.5 h forward today; a trace spanning a step like that
   is garbage, and `emit_host_slice` errors on the inversion. Perfetto has a
   monotonic builtin clock; use it with a clock snapshot.
9. **Attaching perfetto perturbs the thing it measures.** `Anchor::new` does
   `record + stream.synchronize()` on all four streams at every re-anchor (per
   token / chunk), serialising the overlap the trace exists to show. Anchor
   once and re-anchor rarely (drift is ppm-level).
10. `analyze_pftrace_gaps.py` infers token count from `dgpu.mhc_pre_attn`
    slices per `N_LAYER`; with two lanes that is 2x wrong.

## 4. Not instrumented (in the order the 8-row question needs them)

1. **No timeline, only sums.** Every hub number is a per-step mean of a
   both-lane total. The box-2 idle question is a phase question: when does
   lane B's request land relative to lane A's reply. Nothing today records
   (lane, layer, t_submit, t_reply, t_wait_enter) — the hub thread's own
   timeline. The `HOP_*` counters were built for exactly this
   (blocked-vs-slack split per wait) and are drained only by the legacy loop.
2. **Box 2 has no busy/idle/queue-depth metric in production.** `queue_us`
   (time a frame waited behind the previous request) exists per request and
   is only printed at connection close: last connection p50 0.9 ms / p90 5.9
   ms / p99 15 ms at B=3, i.e. the queue is usually empty (lockstep) and
   sometimes deep. Nothing records the gap between requests (idle) or the
   number of frames pending at arrival. Both are one line in the page-stats
   print.
3. **No per-lane attribution.** `LH_*` are both-lane totals by design; with
   lanes at different phases (A waiting on box 2 while B computes) the sum
   cannot say which lane paid.
4. **No distributions.** Every stage is a mean over 20 steps; the daemon's
   percentiles show 10-20x p99/p50 spreads (`gpu` 676 µs → 13.9 ms) that the
   hub's means flatten. "A mean hides a cliff."
5. **The link is treated as a constant.** `ss -tin` on the live socket after
   ~9 minutes of use: hub→daemon `retrans 112` (all 112 DSACK'd = spurious),
   `reord_seen 13226`, `rcv_ooopack 4051`; daemon→hub `retrans 621`,
   `bytes_retrans 40 MB`, `dsack_dups 238`, `rwnd_limited 2.7 s (1.5%)`,
   daemon-side `srtt 6.3 ms` (hub delayed-ACK, `ato 40`) and `rto 207 ms`. No
   instrumentation sees loss-recovery stalls, reordering holds, or receive-
   window stalls; they land in `remote_wait` as unexplained tail. Zero-cost
   to capture: `ss -tin` deltas around every burst.
6. **Prefetch readers (box 2) have no timing** — `waited` is a count, not a
   time; the `PfDone` record carries no timestamps; the daemon tracer's
   `ssd_read` spans fire only in `load_traced`, never for misses or prefetch.
7. **The untimed syncs listed in §3.6** — `V41_PAGER_SYNC_IGPU` drain, the
   post-`sel_sync` D2H copies, the submit-side dGPU sync.
8. **GPU clock state.** The dGPU idles at 41 MHz; nothing samples sclk during
   the ~1 ms lane gaps to see whether the lockstep pays a ramp penalty
   (`pp_dpm_sclk` / `gpu_busy_percent` are readable without root but only at
   ~10 Hz; an A/B with `power_dpm_force_performance_level=high` needs root).
9. **No HTTP-layer latency**: queue wait before prefill (`queued` exists,
   never logged), TTFT, per-request p50/p99 — the user-visible numbers.

## 5. Tools we should be using

| Tool | Status | What it gives us |
|---|---|---|
| `rocprofv3 --kernel-trace --hip-trace --output-format pftrace` | available from the flake-pinned nixpkgs (memory `rocprofv3-kernel-trace`; the ambient one aborts on gflags) | THE missing timeline with zero code changes: true kernel start/end on every stream and every blocking HIP call on the host thread (`hipEventSynchronize`, `hipMemcpy`, `hipStreamSynchronize`) — exactly the untimed syncs of §3.6 — on both boxes, loadable in the perfetto UI. Use for a 20-step window, not a run. |
| `perf record -g -p <hub pid>` / `perf report` | `nix-shell -p linuxPackages.perf` works (perf 7.2.2); `perf_event_paranoid=2` allows own-process user-space sampling without root | host-thread flame graph of the hub and the daemon: where the ~31 ms/step of untimed host time and the "92% software" link cost go. `perf sched` / off-CPU needs root. |
| `bpftrace` (`biolatency`, `biosnoop`, `tcpretrans`, `offcputime`) | needs root (user runs) | per-read NVMe latency distribution under the real workload, TCP retransmit timestamps, off-CPU stacks of the hub thread. |
| `ss -tin dst 10.99.0.2` deltas, `nstat -d` | available, no root | link loss/reorder/rwnd accounting per burst (see §4.5). |
| `/proc/diskstats` sampler (field 9 in-flight, field 10 io_ticks) | available, no root | drive utilisation and queue depth per second on box 2 during a burst; replaces the synthetic QD bench for the "are we saturating the drives" question. |
| `ethtool -c/-S thunderbolt0` | available | coalescing (none exposed on tb-net) and driver drop counters. |
| perfetto trace_processor (python) | pip in a nix-shell | merge hub + daemon traces with `ClockSync::perfetto_shift_ns` — designed for, no script exists. |

## 6. Recommended order

1. Make the production profile cheap: gate `k.*` sub-stages on perfetto being
   attached; run the `V41_MS_PROFILE=0` A/B once in a quiet window.
2. Fix the labels and dead columns (§3.2-3.5): `box2.service_ms` =
   compute+page, add `box2.compute_ms` = service − page, feed
   `REMOTE_SRV_NS` and `RTT` so `remote_link` exists, drop the dead `host.*`
   columns, point `windows.sh` at `lh.ensure`.
3. Take one rocprofv3 `--hip-trace --kernel-trace` window of the hub at 8
   rows (and one of the daemon) — this answers §3.6 and §4.1 without new
   code.
4. Add the cheap production counters: daemon idle-gap + pending-depth in the
   page-stats line; hub `HOP_*` drain into `ms.stage`; `ss -tin` delta in the
   burst script.
5. Only then decide between the multistream perfetto export (per-lane host
   spans on a monotonic clock) and continuing on rocprofv3 windows.

## 7. USE table (Gregg: Utilization / Saturation / Errors per resource)

8-row two-lane step, 298 ms wall, from the 16:08 UTC `ms.stage` block and the
daemon / socket counters read at 16:50. "—" = nothing measures it today.

| Resource | Utilization | Saturation | Errors |
|---|---|---|---|
| dGPU (9070 XT) | 92 ms / 298 = **31%** (event-pair sum; ~8 ms of it is the profiling itself) | — (single compute stream; "idle because the host has not enqueued" is the number and nothing records it) | none logged |
| Box-1 iGPU | 114 / 298 = **38%** | — | none logged |
| Box-2 iGPU | ~100 / 298 = **34%** real compute (the 232 ms "busy" is service INCLUDING 132 ms of paging) | daemon queue time p50 0.9 / p90 5.9 / p99 15 ms at B=3 — printed only at connection close; no per-window depth | none logged |
| Hub host thread | on-CPU ≈ 298 − 161 wait − 50 sel_sync − ≤31 untimed syncs = **56-87 ms, 19-29%**; blocked the rest | it is the serialiser: box 2 idle while the hub has not yet submitted = the ~70 ms we are hunting; no run-queue or off-CPU data | 0 errors since 16:30 |
| Daemon compute thread | 232 / 298 = **78%** in service, but most of the paging part is waiting on reads (join / `ev_done` poll) | `pending` queue depth on arrival — not recorded | 26 lines in the whole log, all connection resets from hub restarts |
| Box-2 NVMe x2 | 24 misses x 18.8 MB / 0.298 s = **1.5 GB/s of ~10 GB/s** aggregate (15%) | latency-bound, not bandwidth-bound: QD1 4.1 / 3.6 ms, linear in concurrency; prefetch readers contend with demand (3.25 → 4.85 ms/miss); in-flight depth under the real workload never sampled (`/proc/diskstats` field 9) | none (SMART needs root) |
| Box-1 NVMe | 0.25 misses/step warm ≈ **0** | — | — |
| Thunderbolt link | ~80 req x (110 KB + 61 KB) ≈ 14 MB/step → **~46 MB/s of ~785 MB/s** (6%) | `rwnd_limited` 1.5%, reordering holds (`reord_seen` 13226, `rcv_ooopack` 4051), cwnd 10-15 x 65 KB segments | **retrans 621 (daemon→hub, 40 MB) + 112 (hub→daemon, all spurious)** — the only non-zero E in the system and nothing of ours measures it |
| dGPU↔iGPU peer copies | `peer_push` stages ~7 ms/step | xfer-stream depth — | none |
| Host RAM / GTT | box 1: 78 GB pool of 93; box 2: 116 of 128 | box 1 has ~1.5 GB of page cache, so every file read is a disk read | no OOM since the launch fix |
| CPU cores (both boxes) | load avg 1.3 (box 1) / 0.3 (box 2) on 32 threads | none | none |

Reading: no device is above ~40% real utilization, so the step is not a
utilization problem on any single resource. It is a saturation problem on the
hub host thread (the one serial resource every lane passes through), and the
only resource with non-zero errors is the link, which no instrumentation of ours
watches. The USE-driven measurements to add, cheapest first: daemon queue depth
+ idle gap per page-stats window; `/proc/diskstats` in-flight sampler on box 2
during bursts; `ss -tin` deltas per burst; `perf record -g` on the hub thread
for its on-CPU share; rocprofv3 `--hip-trace` for its blocked share.

## 8. Results of the step-1 fixes (2026-09-21, commits 8c551be / later)

**Profile cost in tok/s: not measurable.** Same seeds (70/80/90), hub restarted
between phases, box 2 untouched, 8-row 400-token bursts:

| phase | rows=8 steps | mean step | tok/s |
|---|---|---|---|
| profile on (k.* gated) | 122 | 325.1 ms | 24.6 |
| profile off | 77 | 329.1 ms | 24.3 |
| profile on again | 157 | 321.8 ms | 24.9 |

Within the 8% noise floor; the ~8 ms of dGPU stream time hides under the
box-2 pole and the ~2 ms of host time is <1% of a step. Keep `V41_MS_PROFILE=1`.
(These bursts read 24-25 tok/s where the 16:08 burst read 27.2: different
prompt seed, 25-31 box-2 misses/step vs 23-25. Seed-to-seed spread exceeds
the noise floor; always A/B on the same seed.)

**First honest 8-row breakdown** (phase 1, first six windows, ms/step, 80
box-2 waits per step):

    wall ~300 | dgpu 83 igpu 121 | box2 service 245 = page 140 + compute 105
    hub blocked in wait 170 on 70-76% of waits | rtt 465 (5.8 ms/req) srv 361 (4.5) link 104 (1.3 ms/req)
    lh.pager_sync_igpu 31-39 (54 at 7 rows) | lh.sel_sync 44-49 | lh.remote_sync 7 | lh.ensure 7-9

The pager block is now fully attributed: the previously untimed 31 ms was the
`V41_PAGER_SYNC_IGPU` cross-lane iGPU drain (grows to 54 ms/step at 7 rows,
where lane B is thinner and lane A waits for it more often), plus 2.4 ms of
D2H after the router readback. `lh.remote_submit` (7 ms) is entirely the
dGPU drain + xq/ew D2H, not the send.

Daemon percentiles for these runs (B=4, per connection close): queue p50 1 µs
/ p90 3.1 ms / p99 11 ms; service p50 1.44 ms / p90 6.4 / p99 14.3. More than
half the requests arrive to an EMPTY daemon queue and take 1.4 ms: the median
request is a hit-only lockstep exchange; the mean is paging tails.

**The link is not a 205 µs constant.** Per 8-row 400-token burst (~34k
requests), from the new `ss -tin` deltas in `e2e_div.sh`:

    hub<-box2: retrans +3761, ALL DSACK'd (spurious), bytes_retrans +246 MB, reord_seen +21643
    hub socket: rcv_ooopack +24355;   box2 socket: rcv_ooopack +0

Every reply is two segments (header write + body write) and on the hub side
they arrive out of order tens of thousands of times per burst; box 2 then
retransmits the whole ~65 KB reply spuriously ~11% of the time. The
asymmetry (box 2 never receives out of order) points at the hub's receive
path (SO_BUSY_POLL 500 µs on the client socket) rather than the wire.
`rtt - srv` = 1.3 ms/request at 8 rows is where this lands. A/B in progress
via `V41_REMOTE_BUSY_POLL_US` / `V41_REMOTE_QUICKACK` (`~/scratch-ms/link_ab.sh`).

**Link A/B (19:13-19:22, one 8-row 150-token burst per hub restart, ~12k
requests each):**

| hub variant | hub rcv_ooopack | box2 spurious retrans | link ms/step @8 rows |
|---|---|---|---|
| default (busy_poll 500) | +11519 | +1379 (90 MB) | 92 |
| `V41_REMOTE_BUSY_POLL_US=0` | +10215 | +1285 (84 MB) | (7 rows) 52-67 |
| `V41_REMOTE_QUICKACK=1` | +12220 | +1633 (107 MB) | 92-95 |
| default again | +10838 | +1392 (91 MB) | 96 |

Neither application knob moves it: ~95% of replies reach the hub out of
order and ~11% are retransmitted spuriously regardless. The reply is ONE
`write_all` on box 2 (single 61 KB TCP segment at MSS 65468), yet the hub
sees it as ~3 segments (`rcvmss 20560`): the thunderbolt-net TSO path on box 2
splits it and the hub's GRO reassembles the pieces out of order. Box 2 never
receives out of order because the hub's 110 KB requests take the same path
in reverse... and do not get reordered, so the asymmetry is between the two
boxes' driver/GRO state, not the protocol. Next A/Bs need root (user):
`ethtool -K thunderbolt0 gro off` on the hub, `ethtool -K thunderbolt0 tso off`
on box 2, then `tcpdump` on both ends if neither helps. Expected prize: the
1.2 ms/request `link` term, ~90 ms/step at 8 rows, though most of it is
currently hidden under box 2's paging.
