# Hot-split simulator: where should the routed experts live? (2026-10-03)

> **Status (docs audit 2026-10-04):** the side finding's fix landed right after this doc as
> `V41_REMOTE_PARTIAL_ASYNC` (fe90910; live knob, code default off, production on per the hub env
> 2026-10-04). The hot-set parameters used below (top-103, hysteresis 100, <= 3 newcomers) are the
> production env (`V41_B1_HOT_PER_LAYER=103`, `V41_B1_HOT_HYST=100`, `V41_B1_HOT_MAX_CHANGE=3`);
> the code defaults in `expert_pager::hot_set` are 90 / 40 / 0 (unlimited), and `simlib.HotSet`
> hardcodes the production values. Re-running needs `data/final/params.json`, which no step of
> `run_all.sh` writes and which is not committed; `calibrate.py` implements `--grid`, not the
> `--fit` its usage line names. The box-2 link of the post-move run is USB4 / thunderbolt0.

Offline replay simulator for single-stream DSpark decode steps under different
policies for splitting the routed experts between box 1 (hub iGPU) and box 2.
Code: `scripts/split_sim/` (stdlib Python, streams every input). Data: the
post-move run of 2026-10-03 18:52 (`picks-sub-20261003-1852.trace`,
`hub-20261003-185254-7270-000.evt`), 19,081 decode steps, of which 16,253 are
warm single-stream steps (after the first hot-set refresh).

## TL;DR

* **Balancing the head is worth ~12% of forward time, not 20-25%.** Best policy:
  **top-10 per layer replicated on both boxes, the rest interleaved so box 1
  owns ~60% of the non-replicated mass** (`d_rep10_0.60`): traffic-weighted
  held-out fwd **89.4 -> 78.1 ms (-12.6%)**; plain interleave with no
  replication (`c_share0.55`) **-10.3%**. Two-lane steps at 4-6 rows gain 17-21%
  (2L r4 93.3 -> 77.2 ms, 2L r6 114.0 -> 89.6); 1-3 row steps gain 1-7%.
* **The swap (box 2 owns the head) loses** (+3.2%): box 2 becomes the pole
  (busy 70 ms/step, every lane waits on it).
* The dGPU floor is not reached because each lane's own per-layer cycle
  (chain -> route -> max(iGPU, box 2) -> post -> next chain) plus a host/dGPU
  coupling (next point) bounds a 2-lane step, not the devices' busy sums.
* **Side finding, policy-independent:** box 2's partial is uploaded with a
  synchronous `hipMemcpy` (`forward_prefill.rs:10904`, `copy_from_host`; **[2026-10-04: now
  the blocking arm of `V41_REMOTE_PARTIAL_ASYNC`]**) on the
  null stream, and `de.compute` is a blocking stream (`engine.rs:217`), so every
  post with a box-2 reply drains the dGPU queue -- including the posting lane's
  `wait(moe_arrived)` -- with the single host thread blocked. The model only
  calibrates across row counts WITH this coupling (without it the 2-lane errors
  are row-dependent, -7..+12%, for any host cost). Removing it (async copy) is predicted at
  **-10%** for today's split (89.4 -> 80.9) and composes with the split
  (**70.8 ms, -21% vs today**). Cheaper to try than the split; A/B it first.
* Confidence: **medium**. Calibration is within +-2% per (lanes, rows) cell on
  16k steps; the policy ranking survives every perturbation tried; the gain's
  size moves with link latency and box-2 per-expert cost (range -6.6% .. -16%).

## 1. Data and alignment

**Pick trace** (writer: `forward_prefill.rs` `pre_moe_route` ~8560-8800):

| line | meaning |
|---|---|
| `P <layer> <b> <6 ids>` | one ROW of a lane-layer batch; the picks that RAN (after the cache prior `V41_SUB=3`, before any mode-2 substitution). **b is the LANE's rows** (`lane_rows(rows, lanes)`: 3 -> 2+1, 4 -> 2+2, 5 -> 3+2, 6 -> 3+3), not the step's. b >= 21 are prefill chunks. |
| `O <layer> <b> <row> <6 ids>` | the ROUTER's own picks for that row, only when the prior changed it; follows the batch's P lines |
| `C\|c <layer> <b> <row> <from> <to>` | one cache-prior swap (c = dry run) |
| `S/s`, `A`, `D` | mode-2 swaps, alternatives, legacy decode path: absent from this trace |

A step is `lanes x 40` batches in emission order; the lanes interleave per
layer and lane A can run up to ~4 layers ahead of lane B (ready-first; B never
passes A). `build_cache.py` walks the hub_step records in step order and consumes
the matching batches: **all 19,081 records aligned**, 40 foreign batches (one
unrecorded 6+6-row sequence) skipped, 13,280 prefill batches set aside.

**Ownership replica** (`simlib.HotSet` = `hot_set::refresh`: top-103 by decayed
router-pick counts, hysteresis 100 ranks, <= 3 newcomers/layer/refresh, counts
halved; refresh times from the server log, 39 refreshes): per lane-layer, the
box-2 pick count AND distinct count the replica predicts equal hub_req's
`n_picks`/`n_distinct` exactly for **91.9%** of 1.37M warm lane-layers
(differences are symmetric +-1, tie-breaking at the rank-103 boundary).
Before the first refresh the hash split uses the **0.42** default, not
`V41_PARTITION_BOX1_SHARE=0.15` (99.7% match vs 3.4%): `set_partition_share`
is never called on this path.

## 2. Model

**Costs** (`fit_costs.py`, all warm single-stream steps):

| leg | fit | quality |
|---|---|---|
| box-1 iGPU MoE per lane-layer (incl. push back) | 0.100 + 0.0774 d1 + 0.0046 n1 + 0.0397 b ms (d1 distinct local experts, n1 local picks, b rows) | per-step R^2 0.992, n 15.6k; cells within 3% |
| box-2 service per request | 116 + 99.9 d2 + 15.6 b us; +3.28 ms per paged expert | 986k requests; medians linear d = 1..20 |
| link (rtt - srv) | 58/87/118/276/295/318 us for b = 1..6 | non-paging requests |
| dGPU per lane-layer | chain 0.534 + 0.0153 b (router event at its end); shared expert + mix_late 0.106 + 0.0038 b; push 0.05 ms (xfer stream); combine 0.017 | R^2 0.88-0.98 |

Box 2's iGPU costs ~100 us per distinct expert vs ~82 us on box 1 (both ~200
GB/s of 18.8 MB experts); box 2 is the slower place for an expert.

**Step DES** (`des.py`), mirroring `forward_step_arena` (1 lane) and
`forward_step_arena_ready_first` (2 lanes): one host thread polling the lanes in
order; in-order queues for the dGPU compute stream (both lanes' chains and
combines, a `wait(moe_arrived)` blocks everything queued behind it), dGPU xfer,
iGPU, a FIFO box-2 server with parked paging, per-request link latency. Route
runs when the lane's router event fired, Post when its box-2 reply is in (not
when its MoE is done); lane B enters layer l only after lane A. Post with a
reply blocks the host until `de.compute` drains (the synchronous H2D above).

Host costs (ms per lane-layer; fitted on the first 7k steps, validated on all
19k): route 0.02 + 0.035, post 0.03 + 0.05 (copy) + 0.03, plus 0.09 per route
and per post in ready-first mode (polling/locking); step setup + final sync
T0 = 1.05 (1 lane) / 1.77 ms (2 lanes).

**Pools.** Box 1 owns a set (always resident; today's live policy pays a sync
read for newly owned ids). Box 2 = the policy's pinned ids + an LRU over the
rest of its 106 slots/layer. The LRU over-counts today's demand misses 3.2x
(5.89 vs 1.82 per step: the pin ledger, background admits and the prior steer
around them), so pool-model misses are thinned to 31% (`--miss-keep 0.31`,
calibrated on today, applied to every policy). Replays use the RAN picks;
picks the prior swapped in count as hits.

## 3. Calibration (today's policy replayed, 16,253 warm steps)

Measured misses fed in (timing model alone; `data/final/calibration_measured_misses.txt`):

| cell | n | fwd meas | sim | err | iGPU meas/sim | dGPU meas/sim | box-2 rtt meas/sim | exposed wait meas/sim | period meas/sim |
|---|---|---|---|---|---|---|---|---|---|
| 1L r1 | 1218 | 52.9 | 53.3 | +0.9% | 22.3/22.4 | 29.8/30.8 | 8.8/8.6 | 7.3/7.0 | 1.22/1.30 |
| 1L r2 | 130 | 70.8 | 69.6 | -1.7% | 36.1/35.0 | 31.1/31.7 | 16.3/15.5 | 14.4/13.3 | 1.62/1.62 |
| 1L r3 | 150 | 82.4 | 81.7 | -0.9% | 48.0/46.5 | 31.8/32.6 | 20.9/20.1 | 18.7/17.9 | 1.91/1.93 |
| 1L r4 | 623 | 94.2 | 93.6 | -0.7% | 57.9/57.3 | 32.5/33.4 | 30.5/30.9 | 28.0/28.4 | 2.20/2.24 |
| 1L r5 | 426 | 105.7 | 105.1 | -0.5% | 67.2/66.9 | 33.3/34.3 | 36.9/36.9 | 34.1/34.2 | 2.46/2.49 |
| 1L r6 | 637 | 115.6 | 114.4 | -1.0% | 71.8/71.1 | 34.8/35.2 | 50.8/49.1 | 48.1/46.4 | 2.66/2.70 |
| 2L r2 | 843 | 66.6 | 67.4 | +1.2% | 44.6/44.7 | 60.4/60.5 | 18.5/18.2 | 0.0/0.0 | 1.57/1.59 |
| 2L r3 | 5330 | 79.7 | 80.1 | +0.6% | 57.3/57.2 | 61.2/61.4 | 24.3/24.1 | 0.1/0.0 | 1.85/1.89 |
| 2L r4 | 3719 | 92.4 | 92.5 | +0.1% | 70.4/70.1 | 62.2/62.3 | 31.5/31.3 | 0.1/0.0 | 2.10/2.04 |
| 2L r5 | 1465 | 105.3 | 103.7 | -1.5% | 82.7/82.9 | 62.7/63.1 | 37.9/37.3 | 0.1/0.0 | 2.39/2.31 |
| 2L r6 | 1712 | 114.3 | 112.3 | -1.7% | 88.1/88.7 | 63.3/64.0 | 57.8/56.5 | 0.1/0.0 | 2.50/2.48 |

Medians (ms); period = submit-to-submit of one lane, from hub_req. Traffic-weighted
median fwd 88.5 meas / 88.2 sim; within-cell per-step correlation 0.943.
With the thinned pool model instead of measured misses (what every policy
uses): every cell within -1.6..+2.1%, misses 1.83 vs 1.82 per step, traffic-
weighted +0.8% (`calibration_pool_model.txt`).

## 4. Policies (learned on steps 0-9,539, evaluated on 9,540-19,080)

Box 1 holds 133 owned ids/layer (135 slots less the hot set's 2 slack) except
in (a); box 2 pins at most 90 (>= 16 LRU slots). Shares are of held-out picks.

| policy | fwd mean | vs a | share box 1 | box-2 / box-1 misses per step | busy iGPU / box 2 / host (ms/step) |
|---|---|---|---|---|---|
| measured (held-out half) | 89.9 | | | 1.8 / | |
| a today (live hot set K=103) | 89.4 | 0 | 0.84 | 1.78 / 0.02 | 62.9 / 20.5 / 22.3 |
| a static top-103 | 89.8 | +0.4% | 0.85 | 1.98 / 0 | 63.5 / 19.7 / 22.2 |
| a static top-133 | 88.3 | -1.2% | 0.91 | 0.68 / 0 | 67.3 / 13.3 / 21.6 |
| b swap (box 2 pins top-90, box 1 next 133) | 92.2 | +3.2% | 0.17 | 0.71 / 0 | 24.4 / 69.7 / 23.3 |
| c interleave, box-1 target 0.35 | 85.0 | -4.9% | 0.37 | 0.70 / 0 | 35.1 / 56.8 / 23.3 |
| c 0.45 | 82.2 | -8.0% | 0.45 | 0.66 / 0 | 40.3 / 50.6 / 23.3 |
| c 0.50 | 80.9 | -9.5% | 0.51 | 0.61 / 0 | 43.8 / 46.3 / 23.3 |
| **c 0.55** | **80.2** | **-10.3%** | 0.56 | 0.61 / 0 | 46.7 / 42.8 / 23.3 |
| c 0.60 | 80.5 | -9.9% | 0.61 | 0.62 / 0 | 50.2 / 38.5 / 23.2 |
| c 0.65 | 81.7 | -8.6% | 0.66 | 0.63 / 0 | 53.5 / 34.3 / 23.2 |
| c 0.70 | 83.7 | -6.4% | 0.72 | 0.63 / 0 | 57.0 / 30.0 / 23.1 |
| d rep10, 0.50 | 78.4 | -12.3% | 0.57 | 0.93 / 0 | 47.2 / 42.4 / 23.3 |
| **d rep10, 0.60** | **78.1** | **-12.6%** | 0.59 | 0.90 / 0 | 49.0 / 40.1 / 23.3 |
| d rep10, 0.70 | 78.7 | -11.9% | 0.60 | 0.92 / 0 | 50.6 / 38.0 / 23.3 |
| d rep25, 0.60 | 78.9 | -11.7% | 0.59 | 1.53 / 0 | 48.8 / 40.3 / 23.3 |
| d rep40, 0.60 | 80.8 | -9.6% | 0.59 | 2.50 / 0 | 48.8 / 40.3 / 23.3 |

95% block-bootstrap intervals of the change are +-0.3 points (sampling only).
(c): head ranks dealt box 1 / box 2 alternately (2:1 or 3:1 to box 2 below a
0.5 target), box 1 fills the rest of its slots with the ranks after the head,
box 2 pins its head share, LRU for the rest; target = box 1's share of the
(non-replicated) first-half mass. (d): top-K resident on both; per lane-layer
each replicated pick goes to the leg (iGPU vs link + box 2) that finishes
first. Replication beyond ~10 costs box-2 slots and buys misses.

Per cell (median fwd ms, today -> d rep10 0.60): 1L r1 53.7 -> 50.6, 1L r4 95.1 ->
81.2, 1L r6 116.9 -> 99.3, 2L r2 67.6 -> 66.8, 2L r3 80.7 -> 75.4, 2L r4 93.3 ->
77.2, 2L r5 104.7 -> 86.5, 2L r6 114.0 -> 89.6. Full tables: `data/final/compare.txt`.

**Why not -20-25%.** At 2L r4, today's lanes lose (per step, summed over
lane-layers) 22 ms with the host busy elsewhere when their router fires and 29
ms with the host blocked in the post drain; d rep10 0.60 cuts these to 8 and 10
ms, but each lane still runs chain (0.56) -> route -> max(MoE, box 2) (~0.6) ->
post -> combine serially per layer, and 2 lanes do not hide one lane's cycle
under the other enough to reach the 60 ms dGPU busy (sim 77 ms). Box 2 also
gets ~20% slower per expert than box 1, which is why the optimum gives box 1
55-60% of the picks, not 50%.

## 5. Sensitivity (first 7,031 steps; same split; `data/sens_v1.txt`)

| perturbation | today | c 0.55 | d rep10 0.60 |
|---|---|---|---|
| base | 90.5 | -10.6% | -13.2% |
| link x2 | 90.6 | -7.7% | -10.7% |
| link +150 us per request | 90.2 | -6.4% | -9.6% |
| link +400 us per request | 94.9 | -2.7% | -6.6% |
| box-2 service x1.25 / x0.8 | 90.7 / 90.8 | -6.1% / -13.6% | -10.6% / -15.9% |
| box-2 per-expert cost +25% | 90.8 | -7.0% | -11.2% |
| box-1 per-expert cost +15% | 96.5 | -13.9% | -16.5% |
| ready-first host cost 0.06 / 0.12 (T0 refit) | 89.9 / 89.8 | -11.5% / -9.7% | -14.0% / -12.2% |
| raw LRU misses (no thinning) | 98.4 | -15.1% | -16.1% |
| page 4.4 ms (single drive) | 92.4 | -11.7% | -13.9% |
| prefill flushes box-2 LRU | 90.6 | -10.0% | -12.2% |
| router picks, no cache prior | 102.2 | -12.0% | -12.1% |
| post only after the MoE event too | 88.7 | -8.1% | -9.8% |
| async H2D (no drain) | 81.8 | -10.4% | -13.0% |
| placements learned on the pre-move trace (~1 day stale) | 90.5 | -9.7% | -11.6% |

The ranking (d rep10 > c 0.55-0.60 > today > swap) holds in every row. What
erodes the gain is a slower box-2 leg: +400 us per request halves it.
Counter-intuitive model output: speeding up box 2 alone does not help today's
split (90.8 at x0.8): earlier replies trigger earlier posts, whose drain then
blocks the host while the posting lane's MoE finishes.

## 6. Held-out and drift

Within today's run a static top-K learned on the first half keeps almost all
of its coverage (top-103: 0.817 of second-half picks vs 0.826 in-sample; top-25
0.479 vs 0.487). Across the pre-move trace (40 h, 301 segments of ~8 min) the
loss grows with distance: top-103 0.770 one segment later, 0.688 two hours later,
0.675 four hours later (in-sample ~0.80); learned on the whole pre-move trace it
covers 0.724 of today's picks. The interleaved policies are robust to this (-11.6%
with the stale placement vs -12.6% fresh) because a stale head rank lands on one
box or the other either way; a static top-103 is not (+2.5% vs live). Deploy the
split LIVE (re-derive it at each hot-set refresh from the same decayed counts)
and keep a change cap; this run cannot price the refresh churn.

## 7. Recommendation

1. A/B the async partial upload first (`hipMemcpyAsync` from pinned memory on
   `de.compute`, or a non-blocking stream + event): predicted -10% with no
   placement change, and it confirms or kills the drain mechanism the model
   depends on.
2. Then replace "box 1 owns the top-103" with the interleave: box 1 owns
   alternating head ranks + mid-tail to ~55-60% of the mass (133 ids/layer), box
   2 pins its head share. Expected -10% fwd (traffic-weighted), -17% at 2L r4-r6.
3. Add top-10 replication with per-lane-layer leg choice for a further ~2.3
   points (-12.6% total). Not 25 or 40: box-2 slots buy misses.
Expected decode gain with today's step mix: ~10-13% forward time from the split,
~20% with the async copy as well. Confidence medium: the calibration is tight,
the extrapolation (box-1 share 0.84 -> 0.59) rests on per-expert costs that are
linear over the measured ranges (box 2 measured up to 20 distinct per request),
and the result is sensitive to the box-2 leg's latency.

## 8. What the model cannot capture

* The cache prior's interaction with a new placement: replays use the picks that
  ran under today's residency; misses are an LRU thinned by a constant 0.31
  calibrated on today, not a model of the pin ledger, background admits or box
  2's speculative reads.
* DSpark K and lane choice adapting to faster steps (today's mix is held fixed);
  multi-stream steps (live >= 2) are excluded.
* Box-2 request merging/OOO beyond a FIFO server with parked paging; link
  contention between the lanes' requests; CPU/iGPU memory-bandwidth contention
  on box 1.
* The host costs are fitted constants (0.09 ms per ready-first action is the
  least-identified number; +-0.03 moves the gain about +-1 point).
* Box-1 capacity: 133 owned/layer assumes the 5,425-slot pool's decode LRU holds
  them (not verified against `hot_set::per_layer`'s clamp at runtime).
* Refresh churn of a live interleave, prefill phase displacement on box 2, the
  Engram join (taken as each step's measured `lh_engram_join`), box-1 sync reads
  under the static policies (assumed 0).
* Pre-move steps were not used to validate the timing model (different hub
  machine, link and box-2 pool).

## 9. Re-run

    scripts/split_sim/run_all.sh [HUB_EVT] [PICK_TRACE] [SERVER_LOG]

or step by step (all under `nice -n 19`, `ulimit -v 4000000`, from `scripts/split_sim`):

    python3 extract_evt.py HUB.evt data/final
    sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | grep "hot set refreshed" | awk '{print $1,$9,$10,$11}' > data/final/hot_refresh.txt
    python3 build_cache.py TRACE data/final/STEM.hub_step.tsv data/final/cache.pkl
    python3 check_ownership.py data/final/cache.pkl data/final/STEM.hub_req.tsv data/final/hot_refresh.txt HUB.evt 420
    python3 fit_costs.py data/final/cache.pkl data/final/STEM.hub_req.tsv data/final/hot_refresh.txt HUB.evt data/final/costs.json
    python3 calibrate.py data/final/cache.pkl data/final/STEM.hub_req.tsv data/final/hot_refresh.txt HUB.evt --costs data/final/costs.json --params data/final/params.json [--pool-model --miss-keep 0.31] [--grid "h_poll=0.07:0.09:0.11;h_copy=0.05"]
    python3 compare.py data/final/cache.pkl data/final/hot_refresh.txt HUB.evt --costs data/final/costs.json --params data/final/params.json --miss-keep 0.31 [--set h2d_drain=0] [--counts-from data/counts_premove_segs.pkl]
    python3 sensitivity.py CACHE REFRESH HUB.evt --costs COSTS --params PARAMS
    python3 count_trace.py TRACE OUT.pkl; python3 drift.py data/counts_today_segs.pkl data/counts_premove_segs.pkl

`data/` (~425 MB of regenerable TSVs/pickles) is not meant to be committed.
