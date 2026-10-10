# iGPU prefill MoE: what is left after the int8-WMMA arm (plan, 2026-10-10)

Status (updated same day): the f16 kernel is BUILT and measured in the B_moe_prefill harness on the iGPU,
beside the live hub (owner: "slot beside live traffic"; ~2.7 GB, seconds per run, no profiler).
**Results are at the end of this file.** WIRED INTO DISPATCH later on 10-10 (see "Driver integration (10-10)" at the end):
`V41_MOE_WMMA_F16` live knob, default off.

CORRECTION: the first compile table below ("3 tiles 233 VGPR, free") was measured on a generated file whose
3-tile dispatch was dead (it only existed under `WM_NTMAX >= 4`, so NTMAX=3 silently ran the 1-tile path).
The real numbers are in the results section.

## Where we are

The int8-WMMA MXFP4 arm already ships (`mxfp4_moe_wmma.hip`, `V41_MOE_WMMA_GATEUP` / `_DOWN`, default ON,
sweep ledger #27/#28: gate+up 1.56x at B=1024, down 2.7x). Measured iGPU matrix peak is ~55-59 TOPS
(`project_igpu_moe_wmma_f16`; dot4 and WMMA share it). The shipped arm realises about 12 TOPS-equivalent
at B=1024 (gate+up 7.3 ms / 96 experts, down 3.4 ms), i.e. ~20% of peak; the prefill roofline needs ~35.

## Why it stops at ~20% (from the kernel + the ledger's PMC table)

1. Weight-stationary over <= 32 members. `WM_MAX_CHUNK = 32`, one work item per (expert, 32-member chunk).
   An expert with 64 members is read from DRAM twice, with 1024 members 32 times. At 18.8 MB per
   chunk of 32 members the DRAM roof is 32 x 70.8 MF / 18.8 MB x 230 GB/s = **27.7 TOPS** even with
   perfect compute; layer-major windows (N/64 members per expert on average) make this the typical case.
2. Per-(block, tile) float epilogue. `acc += (sc*yd) * (float)C` is ~24 VALU per 2 WMMA (64 clk) per tile per
   block, linear in tiles, and WMMA occupies the VALU (nothing overlaps; ISA 7.9, measured in the f16 work).
   Plus the nibble unpack: 84 VALU per 32-weight block per matrix.
   Cycle model, fraction of the WMMA-pipe peak: 2 tiles 48%, 3 tiles 55%, 4 tiles 58%, infinite 73%.

## Compile-only data (gfx1151, this tree, `cand_wide.hip` = `mxfp4_moe_wmma.hip` with `-DWM_NTMAX=n`)

| tiles (members/work item) | gate+up VGPR / LDS / scratch | down VGPR |
|---|---|---|
| 2 (32, shipped) | 230 / 10.1 KB / 0 | 133 |
| 3 (48) | 233 / 15.2 KB / 0 | 128 |
| 4 (64) | 256 / 20.2 KB / 500 B (spills) | 231 |

So 48 members per work item is free on registers; 64 spills slightly on gate+up. Wider chunks alone buy the
DRAM side (27.7 -> 41 TOPS roof at 48 members) but not the VALU side (55% of peak).

## Proposal: f16 WMMA with the E8M0 scale folded into the dequant LUT (not built)

Replace the int8 dot + per-block float epilogue with an **f16 x f16 -> f32 WMMA chain over the whole 256-K
super-block**, so there is no per-block epilogue at all:

* Weights as f16 `kv * 2^(e8-128)` (kv = {0,1,2,3,4,6,8,12} x sign). Every e2m1 magnitude has f16 low byte 0, so
  the pattern is one hi byte: `(exp<<2 | mant?2:0)`. Scaling by 2^s is `+4*s` on that byte. Build the 8-byte
  perm LUT per (row, block) with **2 VALU** (add `4*(s+14)*0x01010101`, mask the zero entry), then per 4 weights:
  and, perm (LUT select), and+shift-or (sign), 2 spread perms = ~6 VALU. Block cost ~56 VALU vs 84+epilogue.
* Activations as exact f16 `q` (int8 values are exact in f16); `yd` is applied once per super-block per tile
  (`acc += yd * t`, 8 fma per 8 blocks). No `-12*bsum` offset trick (signed weights). Staging converts int8 -> f16
  once per WG (shared by 8 waves; ~2% of loop cost), LDS x2 (528 B/member row).
* Cycle model, per block per matrix: 64*NT clk WMMA + ~58 VALU -> NT=2 69%, NT=3 77%, NT=4 82% of the WMMA peak
  (vs 48/55/58% now): **1.4-1.7x in the compute-bound regime**, on top of the member-chunk widening.
* Numerics: products q*w are exact in f32 (<= 11 x 4 significant bits), the sum order differs from the int path
  only by f32 re-association inside WMMA/accumulation: the same class as the shipped arm (rel_rmse ~5e-7),
  so it goes through the same combined golden/KLD gate. Not bit-exact with kwide.

### De-risk evidence (all CPU, done today)

* **Scale range is narrow.** Read-only scan of the HF checkpoint (shards 05 and 10 = layers 2 and 7, first 1 MB of
  each of 40/60 expert scale tensors, 14.7M + 22.1M scales): e8 in 119..123, 99.99% in 120..121 (2^-8, 2^-7).
  f16 range needs e8-128 in [-14, 9]; huge margin. (`scan_scales.py` is in the job dir, not committed; rerun on
  more layers before the real build. Layers 20+ and the DSpark/MTP experts are unscanned.)
* **LUT math is exact.** `f16lut_emul.py`: emulates v_perm_b32, builds the LUT for every e8 in 100..139 (valid
  subset), all 16 nibbles x 4 lanes, compares with `kv*2^(e8-128)`: 1536 values, 0 mismatches, lane order correct.

## Build plan -- REVISED 2026-10-10 (owner: "batch across subchunks first")

Why this order. Today a prefill unit is `B_MAX = 1024` rows in TWO pipeline lanes of 512 rows, and each lane's routed
MoE is its own call (~8 members/expert, `batch_scratch.rs:59`, `forward_prefill.rs` `pipelined_range_impl`).
Measured, same-process interleaved A/B, shipped int8 arm vs the f16 arm (medians; p10 in the RESULTS section):

| rows per MoE call | members/expert | gate+up shipped -> f16 | down shipped -> f16 |
|---|---|---|---|
| 512 (today) | 8 | 12.6 -> 8.7 ms | 4.6 -> 3.9 ms |
| 1536 | 24 | 8.1 -> 5.2 ms | 3.2 -> 2.6 ms |
| 3072 (3 tiles, chunk 48) | 48 | 11.2 -> 7.6 ms | ~tie (p10 4.0 -> 3.4) |

(slices of 96/48/48 experts; absolute ms are not comparable across rows, the hub is live.) At today's 512-row calls
the f16 arm saves ~15% (p10) to 26% (median) of MoE kernel time, which is ~0-5% end to end because the wall is box 2's
round trip (RTT 47 ms/layer-unit = wire 28 + box-2 run 18, plus 30 ms turnaround; hub MoE units are ~28% of wall
and mostly hidden). The f16 kernel only turns into tok/s once the MoE call is >= ~1.5-3K rows AND the round trip is
not the wall. Routed-expert compute alone at 3072 rows: ~1.3K tok/s shipped vs ~1.8K f16 on ONE iGPU (x2 with both
iGPUs sharing), against a ~430-600 tok/s e2e today.

Constraints found while scoping:
* **dGPU VRAM is 97% full AT REST** (16.55 of 17.1 GB used, checked 10-10) -- but prefill never runs concurrently with
  decode, so decode-only memory is idle during a prefill phase and can be LENT to it. Source: the 10-09 dGPU-bundle
  session's census + phase-swap report (subagent `ae6e21916f26c4a0e`, transcript
  `~/.claude/projects/-home-claude-code-deepstrix--claude-worktrees-dgpu-fusion/749c61d3-d78c-4aa5-be8f-4f524d26dbb7.jsonl`
  line 7666, 2026-10-09 19:11 UTC; its verdict on hot experts is line 7524; "dGPU phase-pool VRAM work" is parked in
  the 10-10 01:41 summary). Findings that matter here:
  * KvArena (decode KV, 3.33 + 0.26 GB): prefill only reserves rows and fills them at admission, so the UNRESERVED tail
    is idle in prefill: **~2.5-2.8 GB typical**, one contiguous run after `compact_stores`; up to 3.6 GB if live KV
    (~0.8 GB, ~0.14 s each way at 6 GB/s) is swapped to host. The lent tail must be returned before any `reserve` /
    `grow` / compaction / `fill_reserved` (job start, admission, parked `try_admit` retries; those run every tick today
    and would need phase gating). Phases: prefill burst 120 s, decode burst 30 s (`multistream.rs:1112-1146`).
  * Decode-only twins of uncaptured prefill scratch (`attn_scores` 176, `attn_active_comp_kv` 235, `r3` 96,
    `candidate_block_score` 165, comp snapshots 34, `indexer_scores` un-carved from r1 ~600 MB; ~1.2 GB total) can be
    swapped per phase without touching graphs; two dead buffers (`verify_scores`, `DgpuScratch.attn_scores`, 190 MB).
  * Precedents: `dgpu_loan.rs` (embed phase lends views over existing allocations, no device alloc/free);
    `lm_store` lazy alloc with halve-on-OOM (`forward_prefill.rs:1121-1165`). No general VRAM budget, no HIP VMM.
  * Consequence for this plan: attention-side lanes CAN grow in prefill (the two-lane dGPU set is ~885 MiB at 512-row
    lanes per `batch_scratch.rs`; it scales ~linearly, so 1024-row lanes are ~1.8 GB and 2048-row lanes ~3.5 GB), so both "bigger lanes" and MoE-side staging are
    open. (iGPU GTT has room too: ~20 GB, but system RAM is tight, ~6 GB available.)
* **Wire (measured 2026-10-10; this CORRECTS the RTT breakdown above).** Bigger requests do not cut wire BYTES (f16
  replies are 10 KB/row/layer whatever the request size); they cut per-request latency, hub turnaround and box-2
  kernel time. Facts: `thunderbolt0` is a `usb4` link trained at **2 lanes x 10 Gb/s = 20 Gb/s on BOTH ends** (sysfs
  `rx_speed`/`tx_speed`; owner says the cable should be Gen3 = 2 x 20; the kernel cannot see the cable; no typec class).
  iperf3 over the Thunderbolt addresses, 3 s each: **10.1 Gb/s (1.26 GB/s) box1->box2, 9.1 Gb/s (1.13 GB/s) reverse, 0
  retransmits; 4 parallel streams 9.7 Gb/s total** (no gain), i.e. ~half the Gen2 line rate, typical of the
  `thunderbolt-net` software path. Live decode traffic is ~10-50 MB/s with 0 rx misses/s; the cumulative 840K rx
  "missed" on box 1 (0.24% of packets; box 2: 51) happens in bursts under bulk load; the current hub connection has had
  1 retransmit in 94 GB. THE STALE PART: the "wire 28 ms = 10 out + 18 back" of the 10-01 time split is 3.0 MB / 10 ms
  and 5.24 MB / 18 ms = 293 MB/s = the **2.5 GbE** link of that day (the hub moved to Thunderbolt on 10-03). At the
  measured 1.1-1.3 GB/s the pure transfer is ~2.5 + 4.4 = ~7 ms per round trip, so the 78 ms unit-layer breakdown is
  OUT OF DATE (no post-move breakdown exists; pull evtrace for one). Consequences: a Gen3 link (if it lifts the
  software path ~2x as line-rate scaling suggests; unverified) saves only ~3-4 ms of the round trip, not ~14; the wire
  is a small part of the wall today; at 2K tok/s the bulk traffic is ~640 MB/s = ~50% of the measured 1.2 GB/s cap, so
  it only becomes binding above ~2.5K tok/s.
* **Latency.** Prefill units interleave with decode bursts (scheduler); a longer non-preemptible MoE batch delays
  decode streams. Keep the batch <= ~2048 rows (~1 unit-equivalent in time) and A/B decode p90 with it.

### POST-MOVE BREAKDOWN (evtrace, measured 2026-10-10; supersedes the 10-01 numbers above)

Source: read-only analysis by a subagent over hub `hub-20261010-162836-3651418-000.evt` joined (seq,t2) with box-2
`b2-20261010-*-07{3,4}.evt`; main job = a 192K-token cold prefill, 10-10 16:44-16:52 UTC (47 windows x 4096 rows, 160
requests/window = 20 layers x 4 sub-chunks x 2 lanes x 512 rows; 7,560 requests, all joined). Scripts/extracts in the
job tmp `pf2/` (`join_1644.txt`). Its 10-01 re-run reproduced the old note (8.77 s, 467 rows/s), so the method is sound.

* **Steady window (<=100 hub-visible box-2 misses, 19 windows): 8.02 s = 511 rows/s** (10-01: 8.77 s, 467). Hub blocked on
  box 2 53%, hub busy 22%, Engram/G0 prep gaps 16% (1.25 s/window, 3 x 0.42 s), other group-boundary gaps 8% (0.63 s).
  Mid (100-800 misses) 470 rows/s; heavy paging (first window / right after a decode burst) 304; the 11:31 job on the
  previous binary 587 (cause of the job-to-job spread unknown).
* **Per 512-row request (steady, p50, ms):** wire out 2.7 (3.01 MB, 1.13 GB/s) + wire back 4.4 (5.24 MB, 1.18 GB/s) = **7.1
  total (was 28)**; **box-2 queue behind the other lane 8.6 (lane 1 ~28-32)**; **box-2 run 28.4 (floor 18.4 with 0 missing
  experts, rises with `n_missing`)**; D2H 0.35; RTT 56.8 (was ~47-50); hub reply->next-submit **10.4 (was 25-30)**.
* **Box-2 run by experts not yet landed (`n_missing`): 0-2: 19.6 ms (29% of requests) | 3-6: 26.1 (34%) | 7-12: 47.5
  (22%) | 13+: 79.6 (15%).** Run time = GPU waiting on page reads (corr 0.93). Mean n_missing 7.1 now vs 2.2 on 10-01;
  requests with 0 missing 7% vs 38%; **box-2 pool 4,740 slots now (`--experts L0-L39:265-383`) vs 6,160** (the 96 GB
  machine; -23%), pinned 136 vs 1,584. The pool shrink is the INFERRED cause of the extra paging.
* **Box 2 serves the two lanes strictly serially**: `served_under` is set on 0 of 5,968 prefill requests (decode 159 of
  3,018), yet 5,599 prefill requests park >1 ms -> while one lane's GPU waits on pages, the other lane queues and the GPU
  idles. Floor period (<=2 missing) = 40.3 ms = 2 x 19.4 ms of box-2 GPU back to back (executor ~96% busy there); the hub's
  21 ms/unit-layer and the 7 ms wire are hidden UNDER box 2. Unit-layer period: p50 66.5 / mean 69.3 steady (10-01: 78 /
  80); by total missing: 40 ms (<=2), 47 (3-6), 62 (7-12), 82 (13-20), 108 (21-40), 181 (41+).
* Pipeline depth (steady): 0 in flight 24.8%, 1: 27.4%, 2: 47.7%, >=3 never. Box-2 executor busy 72% (GPU compute ~37%);
  box-2 disk 2.8 GB/s (io-util 22-35%, NOT saturated; 68% of the time no read in flight, 17.5% >=4 in flight: page-ins
  are requested late and in bursts). Wire: 14% busy.
* **Verdict: the wall is BOX 2 (kernel floor + paging + serial service), no longer the wire or the hub.** Inferred
  ceiling if every unit-layer ran at the <=2-missing period (43 ms): 80 x 43 ms = 3.5 s + 1.9 s gaps = ~5.4 s (-33%).
* Not determinable from the trace (`V41_EVTRACE_DEV=0` on both boxes): dGPU chain vs iGPU MoE inside the hub's 10.4 ms; true
  box-2 GPU busy; the composition of the 0.42 s G0 prep; why the 11:31 job was faster.

**What this does to the plan (re-ranked):**
1. **Box-2 paging** (up to ~-33% window): restore pool slots (memory budget on the 96 GB box), prefill-mode resident pool /
   prefetch (`V41_LM_PREFETCH`, group prefetch), and above all **let the other lane run while one is parked** (serve
   prefill requests under park like decode; today 0 of 5,968). Overlapping park with the other lane's compute attacks
   the ~30 ms/unit-layer of GPU idle directly.
2. **Box-2 MoE floor 19.4 ms per 512-row request (x2 lanes serial = the 40 ms floor)**: THIS is where the MoE batching and
   the f16 arm pay. Merge the two lanes into ONE box-2 pass (existing `V41_B2_MERGE`/`REQ_FLAG_PARTNER`, needs both lanes'
   requests submitted together; the hub's staggered pipeline is mostly hidden under box 2 anyway, so losing the stagger is
   cheap) or 1024-row lanes: one read of each expert's weights serves 1024 members. f16 arm: -15..26% of kernel time. Both
   boxes' kernels are the same file, so the hub side gets it too.
3. **Hub-only gaps 23% of the window** (Engram/G0 prep 1.25 s + boundaries 0.63 s) while box 2 idles: Engram overlap and
   cross-group overlap; ~-1 s of 8.
4. Wire (Gen3 cable): ~3-4 ms of 57 ms, last. Hub-side kernel work (hub 21 ms/unit-layer) only matters once box 2 is
   faster than ~40 ms/unit-layer (it then becomes the floor).
5. Instrumentation: enable `V41_EVTRACE_DEV` (Tier B regressed decode 10-02: use a short prefill-only window) for the
   dGPU/iGPU split.
Revised estimate of what steps 1-3 buy on a 8.0 s steady window: kill paging -> ~5.4 s; box-2 floor 19 -> ~13 ms ->
~4.3 s; gaps 1.9 -> ~0.8 s -> ~3.2-3.5 s, i.e. **~1.2K rows/s** from box-2-side work alone. 2K needs the hub and box 2 to
share the experts more evenly (the hub is ~50% idle at 21 of 40 ms) on top of that.

### WHY box 2 pages in a pure prefill: its working set exceeds the prefill budget (owner prompt 10-10: "just fetch every expert, many layers ahead"; data analysis `pf2/rereads2.py` on the 16:44 job's 51,509 box-2 reads)

* Only **30% of box-2's reads are experts in its static range (265-383)**; 70% are experts outside it (hot-split interleave /
  mirror routing: `V41_B1_HOT_POLICY=interleave` live since 10-09 19:16 UTC, `IL_PER_LAYER` 115, plus `mirror_frac` 0.70).
  Distinct experts box 2 touched per encoder layer in the job: L0 266, L1 264, L8 262, L9 264, mid layers ~190-230 (119 range +
  ~115 interleaved = ~234, consistent). Over the 20 encoder layers that is **~4,000-4,700 experts = ~the whole 4,740-slot pool and
  at/above what the pool can give a prefill**. (CORRECTION, same day: `prefill_budget` is NOT a cap on prefill-resident slots; it
  is the number of DECODE-class victims one prefill phase may take before it evicts its own oldest pages, 3,500 default, and it
  applies to jobs <= `V41_LONG_PREFILL_TOKENS` = 16,384. The 192K job is a LONG job: `prefill_budget_long` = unlimited, own pages
  evicted farthest-next-use first (`own_sweep_rank`). So the limit that bites is the pool itself, 4,740 slots minus whatever is
  pinned or otherwise protected, vs a long job's own set: the code's own 10-03 measurement says 4,304 distinct experts for a 58K job
  and 5,161 for a 229K job.) With farthest-next-use eviction a set larger than the usable capacity costs ~(set - capacity) re-reads
  per sweep: observed **1,146 reads (~21 GB) per 4096-row window** => usable capacity ~3.5K of a ~4.7K set. Victims are 95% ENCODER-layer experts (self-eviction), only 5%
  decoder layers; 2,022 experts read once, 1,606 twice, a tail of ~100 experts read 10+ times.
* So the user's idea is right, and the reason it does not already work is CAPACITY, not prediction: whole-layer prefetch many
  layers ahead needs no routing knowledge (layer-major + long context touch ~every expert), but prefetching a set larger than the
  budget just evicts what the next window needs. Make the set fit first, then prefetch it in layer order:
  - total encoder experts = 20 x 384 = 7,680; hub pool 5,425 + box-2 pool 4,740 = 10,165 slots, so a STATIC expert->box
    partition during prefill (each encoder expert resident in exactly one pool) fits with ~2.5K slack and needs NO steady-state
    paging; today both boxes hold overlapping hot sets (mirror) and box 2 also serves 70% non-owned picks.
  - (RETRACTED: raising `prefill_budget` would not help long jobs, see the correction above; for long jobs the unlimited
    `prefill_budget_long` already applies.) PINS ARE NOT THE CAUSE (checked): `pins 3789/4232` is the decode-time pinned/budget
    figure (hub-requested "never evict" slots, REQ_FLAG_PIN/RELEASE); on every prefill entry the hub releases the coldest pins
    to `budget - V41_B2_PIN_PREFILL_BAND` and the hub runs BAND=4096, so ~136 stay pinned (matches the trace) and a long job gets
    ~4,600 of the 4,740 slots. So the long job's real TOUCHED set is ~5.2-5.8K (the code's own 10-03 measurement: 5,161 distinct
    experts for a 229K job; reads only reveal the missed ones, hence my earlier 4.0-4.7K undercount): ~1,146 re-reads/window =
    set - capacity. First fix: a prefill-time routing mode that does not send mirrored/interleaved picks to box 2 (shrink its set by
    ~1.2K = ~60 experts/layer; the pool cannot grow on the 96 GB box). Third (renumbered below): background whole-layer prefetch (`V41_LM_PREFETCH=1` is already ON
    in the hub env and hints only the last window's demand reads; extend to "all box-2-owned experts of layer L+k").
* **Streaming-budget arithmetic (owner follow-up 10-10: "with streaming the budget should be tiny").** Resident slots needed =
  lookahead layers x per-layer set (read ~1.5 s vs ~0.4 s compute per layer -> ~3-4 layers ahead -> ~4 x 234 ~ 1,000 slots).
  Two caveats: (1) the loop is GROUP-major, so a group (layers 2-7) re-runs its six layers per sub-chunk and must keep 6 x 234 ~ 1,400
  resident across the 4 sub-chunks; strict layer-outer order (macro-lanes) removes that. (2) Pure streaming re-reads the whole box-2
  encoder set every window: ~4,700 experts x 18.8 MB ~ 88 GB per pass = ~11 GB/s at a 4096-row/8 s window vs 2.8-4 GB/s measured
  (8.5 GB/s theoretical, 2 drives). Bytes per pass are constant, so it only fits with LONGER windows (16K rows ~32 s -> ~2.8 GB/s;
  lm_store ~1.3 GB from the lent arena tail) or a smaller set. Shrinking box-2's prefill set to its static ~119/layer (2,400 experts,
  45 GB) fits resident under the 3,500 budget with zero steady-state reads; streaming then only refills after decode bursts.
* Partial compute already exists: box 2 runs `V41_B2_HITS_FIRST=1` (resident experts first, misses read meanwhile, a second pass for
  the landed ones), so a request is not blocked by one missing expert; the reply simply cannot complete until the LAST miss lands
  and its pass runs. `n_missing` in the trace is only the count of non-resident experts when the pass starts. The measured
  run-time growth (19.6 -> 26.1 -> 47.5 -> 79.6 ms) is the tail of those reads (3 ms/miss at p50 demand, background reads 7.4 ms p50
  under queueing) outlasting the ~15 ms of hits compute.
* Open question the 10-10 01:41 summary left: "check whether interleave slowed prefill". This is circumstantial evidence it did
  (box-2 set 119 -> ~234/layer). A pre-interleave long-prefill trace (before 10-09 19:16 UTC) would confirm.

### ROOT CAUSE: the expert->box partition is tuned for DECODE and applied unchanged to prefill (read-only code trace, 2026-10-10)

* `forward_prefill.rs` ~9610-9630: under `V41_T2_PARTITION=1` a pick's box is `home2 = expert_pager::partition_box2(layer, e)`
  (the decode-pick-mass holder fallback `hot_set::serve_on_box2` is for ARENA/decode rows only: `holder_rows`); `owns_eff` =
  `extra_remote[e] || ...` is then built from that. `partition_box2` (expert_pager.rs ~571): once the hot set is warm, **box 1 owns
  each layer's hottest ids by measured DECODE pick mass (`V41_B1_HOT_PER_LAYER`=103, interleave `V41_B1_HOT_IL_PER_LAYER`=115, target
  0.60) and EVERYTHING ELSE is box 2's** (~270 of 384 per layer). `mirror_frac` is NOT ownership (it is the disk-read mirror fraction).
  Decode picks are Zipfian, so that split is right for decode; prefill touches nearly every expert, so it hands box 2 ~270/layer x 20
  encoder layers = **~5,400 experts vs ~4,600 usable slots**, which is the ~1,146 re-reads/window, the 70% of reads outside box-2's
  advertised range, and the 266 distinct experts read at L0. It also loads box 2's kernel with ~70% of the experts while the hub
  holds only ~30% (hub busy 22% of the window, its disk idle).
* **Fix (hub-only, no box-2 change, no new protocol): a PREFILL partition for the encoder layers (<= `CED_DECODER_START`), long jobs
  only.** Balance by capacity instead of decode mass: box 1 (pool 5,425) takes its resident hot head PLUS the next-ranked ids up to
  ~190/layer, box 2 the other ~194/layer (20 x 194 = ~3,900 <= 4,600 usable, ~85%); decoder layers (replay) keep the decode rule.
  Deterministic (rank by the existing `hot_set` counts, hash fallback when cold). Knob e.g. `V41_PREFILL_B1_PER_LAYER` (default 0 =
  today); gate on the job being LONG (> `V41_LONG_PREFILL_TOKENS`) so short prefills do not pay a cold fill: box 1 pages its own half
  from its idle disk (`V41_B1_PAGE_MISSES=1`, `V41_PAGER_MISS_PAR=8`; ~3.9K experts = 73 GB ~ 17 s once at 4.2 GB/s, then resident).
  Where: a `partition_box2_prefill(layer, e)` consulted at the `home2` site when the rows are prefill-shaped and layer <= 20.
* Risks: (1) hub pager residency: its prefill set (~3.9K) plus the decode hot head of the 20 decoder layers (~2.3K) exceeds 5,425
  slots, so a prefill displaces ~700 decode-hot slots on the hub (re-warmed from its own disk after, ~3 s); watch hub `ensure` stalls
  in the first long window. (2) f32 association changes (which box sums which expert): same class as every partition change; needs the
  golden/KLD gate, and `lm_prefill_bitexact`'s reference assumes today's partition. (3) The group-prefetch hints (`V41_LM_PREFETCH`)
  follow box-2 reads, so they follow the new split automatically; verify. (4) `verify_routing_exactly_once` must still pass (owns_eff
  is built from the same `extra_remote`).
* Expected (INFERRED, not measured): box 2's steady paging -> ~0 after the first window; its run floor ~19 -> ~14 ms (fewer
  experts per request); hub work per lane-layer ~11 -> ~16 ms (its share 30% -> ~50%), now balanced against box 2 at ~30 ms per
  unit-layer: window ~80 x 32 ms + 1.9 s gaps = ~4.5 s = **~900 rows/s (+75% over 511)**, before any gap/overlap work. Needs a hub
  restart (static env knob) and one long-prefill A/B with the same evtrace analysis.

### Box-2 "serve the other lane under park": scoping (read-only, 2026-10-10)

What exists. Box 2 (`remote_experts.rs`, expertd compute loop ~8340-8840) already PARKS a request that must page
(`park=1` in `~/expertd-knobs.txt`; the hub sets `REQ_FLAG_OOO` on every request, 10116): its misses go to the prefetch
readers (`prefetch_words_cls`, up to `V41_B2_PREFETCH_SETS`=16 concurrent) and, until they land or `PARK_MAX` = 50 ms,
box 2 serves OTHER queued frames on a second executor `exec2`. The limit that makes it a no-op for prefill:
`servable()` accepts only `REQ_FLAG_OOO && b <= rows2` with `rows2 = PARK_MAX_ROWS` = **16** (the pin-reserve proof counts on
it), and the design comment says outright "a prefill pass is never served inside a park" (~3123). A 512-row lane is
never servable -> `served_under` 0 of 5,968, matching the trace.

What it would take (box-2-only change, hub binary unchanged; replies already match by seq, OOO is supported):
1. `servable()`: also accept a prefill-shaped frame (b <= 512) **only if its experts are already resident / landed**
   (cheap predicate with the existing `prefetch_pending_any(&words)` + residency check on its `early_page`d words). If it
   has misses, serving it inside the park would just read them synchronously on the same drives (`serve_interleaved`
   comment ~9254) and gain nothing.
2. `exec2` sized `exec_rows` (512) instead of 16 for this case: ~0.1-0.15 GB iGPU on box 2 (partials 123 KB/row + mid 55
   KB/row ...); box 2 has ~15 GB available.
3. Pin reserve: today `PIN_RESERVE_MIN` = 384 + 16*6 = 480 assumes W (<= 384) + P (parked, <= 96). With a parked PREFILL
   P (<= 384) plus a served-inside PREFILL W (<= 384) the no-deadlock bound becomes R (+STAGE) >= 768
   (`V41_B2_PIN_RESERVE`; prod runs `V41_B2_PREFILL_STAGE=0`; only 136 experts are pinned now, budget room is fine).
4. Tests: the loopback tests (`remote_experts_*_loopback.rs`) need a prefill-under-park case; the order of RELEASE/pin
   words ("served order = arrival order") must still hold.
Expected gain (inferred from the trace timeline, not measured): lane 1 arrives ~21-25 ms after lane 0, i.e. while lane 0 is
parked, and its early-page reads then run concurrently with lane 0's; if they land by the time lane 0 starts its kernel,
lane 1's 18 ms kernel overlaps lane 0's (same GPU, so < 2x) instead of queueing 27 ms behind it. Rough period 66 -> 50-55 ms
(-20%); the -33% ceiling needs the paging itself removed. Size: ~100-200 lines + tests, medium risk (pin-reserve proof).

**Cheaper first experiment (live, reversible in ~1 s, NEEDS OWNER GO because it changes production box 2):** the park cost
fits ~3 ms per missing expert (run 19.6 / 26.1 / 47.5 / 79.6 ms at 0-2 / 3-6 / 7-12 / 13+ missing) -- the signature of
SERIALISED reads. `miss_par` is 1 in `~/expertd-knobs.txt` and the code comment says "MEASURED 2026-09-20 no gain at 4 ...
kept for a faster drive" -- that was the old drive set; since the 10-03 move box 2 reads from SN5000 + 990 Pro, and the
disk is only 22-35% busy in steady windows. The daemon is started with `V41_B2_MISS_PAR=4` (staging sets sized for 4) and
the knob file can only LOWER it, so `miss_par=4` in the file (watched every second; or `kill -USR2`) raises demand-read
concurrency 1 -> 4 with no restart. A/B it on one steady long prefill with the same evtrace analysis (n_missing vs run
ms, window wall) and watch decode p90 (miss_par also applies to decode ensure). Revert = write `miss_par=1`.
(`V41_B2_PREFETCH_SETS`=16 already governs the park/early-page reads, so this tests only the post-park demand path; if no
change, the stall is read LATENCY/issue timing, and prefetch lead time is the next lever.)

Steps (each gated; nothing GPU beyond the harness until a window is agreed). NOTE: the step order below was written before
the breakdown above; use the re-ranked list above for priority (step 1 here, MoE batching, is item 2 there):

1. **MoE-side batching across lanes/sub-chunks.** Design + code-site map: see "Cross-sub-chunk MoE batching: design"
   below. Knob `V41_PREFILL_MOE_ROWS` (default OFF/0 = today), target 2048. Gate: bit-compatible routing (same experts
   per row), f32-association-only differences, `lm_prefill_bitexact`-style test extended with a batched arm.
2. **f16 arm into dispatch** (`V41_MOE_WMMA_F16`, default OFF): chunk 48 into the work-item builder, Rust wrappers for
   `cand_f16.hip` (NTMAX=3), `build.rs` registration, box 2 gets it for free (same kernels). Bench with the harness at
   the batched row counts first.
3. Kernel overlap work (double-buffered activation staging; PMC one counter per pass, never ATT on the iGPU): the f16 arm
   is at ~25% of the ~57 TOPS peak with ALL global loads removed -> ~28%; target ~40%.
4. (GPU, server down) golden/KLD gate for the f16 arm (and, still owed, for f16 box-2 replies), then e2e prefill A/B at
   long context with layer-major windows; decode p90 check.
5. Rest of the E8M0 range scan (only layers 2 and 7 sampled).

## Risks / unknowns

* The model assumes WMMA and VALU fully serialize and DRAM at 230 GB/s; measured kernels sit at 73-79% (gate+up)
  and 82-90% (down) of that, so the realistic ceiling is ~35 TOPS, not 48.
* 2 mats x 3 tiles at 233 VGPR is already at the occupancy edge (earlier pipelining attempts spilled badly:
  ledger #74/#75/#76).
* Box 2's iGPU is identical, but it runs the same kernels through the same dispatch; the xq rows it receives
  already carry the data this needs (int8 q + yd).
* NPU / dGPU are not touched by this.

## RESULTS (2026-10-10, gfx1151 iGPU, harness `scratch/kernel_ideas/B_moe_prefill`, beside live hub traffic)

Files: `mid_f16.inc` (the kernel arm) + `cand_wide.hip` (shared scaffold) -> `cand_f16.hip` via
`mkf16.py` (`build_f16abl.sh` calls it and builds the timing-only ablations). `cand_f16.hip` is committed too.
Same entry names/args as the shipped arm, so the harness swaps it in with `KB_CAND`. Harness additions: `KB_CHUNK`
(candidate work-item chunk), `abx` mode (shipped arm vs two candidates interleaved in one process), `run_f16.sh`.

**Resources (hipcc gfx1151, no spills anywhere):**

| | gate+up VGPR / LDS | down VGPR |
|---|---|---|
| shipped int8 arm (2 tiles) | 230 / 10.1 KB | 133 |
| f16, 2 tiles (32 members) | 179 / 17.3 KB | 123 |
| f16, 3 tiles (48 members) | 224 / 25.9 KB | 159 |

Static mix per super-block (`isa_mix.py`): VALU per WMMA 51.2 -> 29.6 (gate+up), 49.1 -> 28.5 (down) at 2 tiles.

**Correctness** (`cmp`, vs production kwide/kwide2, real harness inputs): gate+up rel_rmse 3.9e-7, down 1.1e-7
at 2 tiles B=512, and the same at 3 tiles / chunk 48 / B=3072 (max_group 68, so chunking is exercised). The shipped
int8 arm is 5.4e-7 / 1.7e-7, so the f16 arm is at least as close. The baseline itself matches the CPU reference
to 4e-6 / 1.7e-6. Not bit-exact (f32 reassociation, same class as the shipped arm): still needs the golden/KLD gate.

**Speed, interleaved in one process** (`abx`, B=3072, 48 experts = 48.5 members/expert, 12 rounds; the hub is
live so absolute numbers move 9-16 ms between runs for the SAME kernel; only same-process ratios are usable):

| comparison (vs shipped int8 arm) | gate+up med (p10) | down med (p10) |
|---|---|---|
| f16, 2 tiles, chunk 32 | 9.42 vs 15.99 ms = **1.70x** (9.05 vs 12.82 = 1.42x) | 4.66 vs 5.39 = 1.16x (4.56 vs 5.02) |
| f16, 3 tiles, chunk 48 | 7.57 vs 11.22 ms = **1.48x** (7.28 vs 10.57 = 1.45x) | 5.57 vs 5.44 = tie (3.43 vs 4.02 = 1.17x) |
| (f16 3 tiles vs f16 2 tiles, from the two runs) | ~1.24x | ~1.0x |

So the scale-in-LUT f16 arm is a real **~1.4-1.7x on gate+up** over what ships, and ~1.15-1.2x on down (down was
already near its DRAM roof). Wider chunks add a further ~1.2x on gate+up. At B=1024 / 16.5 members per expert the
two arms tie on gate+up (memory/latency bound there), as predicted: this lever only matters at big windows.

**Ablation (f16, 2 tiles, gate+up, B=3072):** removing the activation global loads: 8.9 ms (vs 9.2); weight loads:
8.1 (p10 6.8); both: 8.7 (p10 6.8). So even with ALL global loads removed the kernel sits at ~6.8 ms = 16 TFLOP/s =
28% of the ~57 TOPS peak: the WMMA pipe is ~1/3 busy, and it is NOT instruction count (VALU/WMMA fell 42% for ~1.4x)
nor memory. Suspects, untested: two `__syncthreads` per super-block with stage->barrier->compute serialisation
(no overlap inside a WG), LDS fragment reads (4 ds_read_b128 per tile-block), only 4 independent accumulator chains
per wave. Next experiments: double-buffered staging (stage sb+1 while computing sb), PMC (`SQ_BUSY_CYCLES`,
`SQ_INSTS_LDS`, ONE counter per pass, iGPU PMC only via the flake-pinned rocprofiler, NEVER ATT on the iGPU).

**Not yet done:** driver integration (a `V41_MOE_WMMA_F16` knob in `het::dispatch`, chunk 48 into the work-item builder,
rust wrappers, `build.rs` registration, expected default OFF), the golden/KLD gate (needs the hub down), e2e prefill
A/B, box 2 (same binary), and a scan of the remaining layers' E8M0 range (only layers 2 and 7 sampled; the kernel
clamps to e8 in [114,137], a weight outside that would silently flush/saturate).

## Cross-sub-chunk MoE batching: design (read-only code scoping, 2026-10-10; line numbers in `crates/v4flash-kernels/src/het/`)

Scoped by a read-only agent from the code (nothing built or run); per-row byte sizes are its arithmetic, not measurements.

**Where the per-lane MoE happens.** `pipelined_range_impl` (`forward_prefill.rs:2486`) runs `forward_layer_pre_moe_v2` (4813 =
`pre_moe_chain` 4848 -> `pre_moe_route` 8972 -> `pre_moe_prep` 10123 -> `pre_moe_launch` 10668) once per LANE per layer,
staggered by half a layer (steady state: `post_A(L)`, `pre_A(L+1)`, `post_B(L)`, `pre_B(L+1)`; 2740-2872): the dGPU chain of
one lane overlaps the iGPU MoE + box-2 round trip of the other. The box-2 request is built at `pre_moe_route` (submit at 10044:
`b x 5840 B` xq, `b x 6` picks, `b x 6` weights, f16 reply by default, ticket stashed per lane, nothing waits there); the iGPU
MoE builds expert member lists (`moe_group_builder_hetsplit`, `(b<<16)|slot`, atomicAdd order nondeterministic), then
gate+up / down WMMA arms, then the peer-push back; `forward_layer_post_moe_v2` (11575) adds moe, then shared, then remote
(that ORDER must be kept for a bit-compatible batch), then `hc_post`.

**There is no hub-side lane merge.** The only merge is box 2's same-layer coalescing (`V41_B2_MERGE`, `REQ_FLAG_PARTNER`,
`remote_experts.rs:2213, 8435-8520`, waits `V41_B2_MERGE_WAIT_US`=400 us for a partner). `partner_follows` is true only in
arena decode (4271), false in `pipelined_range_impl`; and lane B's request is submitted a whole chain time after A's, so
the daemon never sees a partner. (Inferred from code order, not traced.) Even when it fires it shares only box 2's pass.

**Dependency check (verified in code).** Unit k+1's layer-L attention needs layer L-1 output of its own rows and layer L's
KV / compressor / SWA ring after unit k's layer-L attention and eviction; it needs NOT unit k's layer-L MoE output. So
attention for several units at layer L, then ONE MoE, is legal. Positional state (KV/ring, compressor, index keys, Engram
rows) is safe layer-outer. What spans the MoE per lane: `after_attn_hc` (80 KB/row), `split`+`hc_pre_carry` (~0.2 KB/row),
`ffn_shared` (20 KB/row), the residuals (`lm_store` already homes them), `indexer_sel_saved` (2 KB/row, only inside a group).
Not row-independent host state: box-2 ownership split, pager `ensure`, `b2_mirror` substitution, cache-prior `V41_SUB`
(time-ordered by request order); open question whether `lm_prefill_bitexact` runs with them off.

**Memory for an M-row batch.** iGPU (GTT): ~0.35-0.45 GB at M=2048, ~0.7-0.9 GB at 4096 (`d_mid_cat` 55 KB/row,
`q2k_partials` 123 KB/row, `d_midq_cat` 15.8, `d_mid16` 27.6, xq 5.8+5.8, `ffn_moe` 20). Box 2 (expertd executor): ~0.2 GB at
1024 rows, ~0.4 GB at 2048. Host: request xq 12 MB / reply 21 MB (f16) at M=2048.

**Designs.**
* **A. Macro-lanes**: attention stays per 512-row sub-unit; the MoE runs per K staged sub-units; keep TWO macro-lanes so the
  dGPU still overlaps (2 x 2048 = exactly one default 4096-row window). dGPU footprint ~unchanged (+82 MB `ffn_shared`
  stash +8 MB index store). New driver replacing the lane loop, per-unit stashes, one consolidated box-2 request, `si`/`bi`
  sized for M. ~600-900 lines, HIGH risk (host sync ordering, `drain_before_ensure`, pager, `V41_SUB`, PREFETCH words,
  `partner_follows`). Unit latency 2-4x.
* **B. Merge the two existing lanes' MoE (1024 rows)**: serialises chain A, chain B, MoE(A+B), loses the half-layer overlap;
  ~150-250 lines. Reaches only 1024.
  * **B-hybrid (few lines, untested)**: set `partner_follows` on lane A's request in `pipelined_range_impl` so box 2 merges
    A+B (1024 rows) via the existing path; needs lane B's route/submit to happen inside the 400 us wait, i.e. reorder
    route/submit before the iGPU leg. Box-2 side only.
* **B'. Bigger lanes carved from the lent KV-arena tail** (see the VRAM bullet above): 1024-row lanes +885 MiB (fits the
  lent 2.5-2.8 GB with margin), 2048-row lanes +2.6 GiB (+ `lm_store` min window 2 x chunk = 8192 rows ~672 MB: no margin).
  Mostly constants/sizes (`B_MAX`, `KV_CACHE_ROWS`, lane alloc, `chunk_rows`, expertd `--max-batch`, which currently
  REJECTS > `B_MAX`=1024 at `deepstrix-expertd/src/main.rs:147` and `remote_experts.rs:9966`) plus the lend/return hooks
  (lend after `compact_stores` at prefill job start; return before any reserve/grow/compaction/`fill_reserved`/parked
  `try_admit`; fallback = halve the lane as `lm_store` does). Keeps the pipeline overlap. Unit latency 2x/4x. The reference
  for bit-identity moves (chunked prefill at the same lane size). UNKNOWN: whether the arena tail can be freed and
  hipMalloc'd again, or only lent as views (`dgpu_loan.rs` precedent = views via `LoanAlloc::take`, would need an allocator
  threaded through `BatchDgpuScratch::alloc_rows_ctx`).

**Box 2.** Same MoE kernels. The WMMA work item is <= 32 members (`WM_MAX_CHUNK`), so at 64 members/expert an expert's weights
are streamed ~twice; the gain from bigger batches is mainly FILLING the 32-member tile (~8 of 32 used today), which is
why the f16 arm's 48-member chunk (step 2) multiplies the batching gain.

**Recommendation (staged, cheapest information first).**
1. **Measure sensitivity before building**: June's 256 -> 512 rows/call gave +9.4% e2e, and the wall now is the box-2 round
   trip. Cheapest probe = B' at lane 1024 behind a knob, on a quiet window; compare e2e prefill, TTFT and decode p90.
   It needs only ~0.9 GB of the lent tail, and a 1024-row lane request is within box 2's current cap (`--max-batch`
   <= `B_MAX` = 1024), though a 2048-row chunk (two such lanes) changes what `B_MAX` and `KV_CACHE_ROWS` mean -- audit
   the other `B_MAX` uses first.
2. If lane 1024 moves e2e: add the f16 arm with chunk 48 (steps 2 above) and consider A (macro-lanes, K=2 over 1024-row
   lanes = 2048-row MoE) for the remaining 2x. If it does not move e2e, the wall is the round trip / wire and the right
   next work is the link (Gen3), not the MoE.
3. B-hybrid is a cheap side experiment on box 2 only; do not rely on it.

**Unverified (do not assume):** why the merge never fires (code-order inference); whether `hc_post`/`mhc` launches take
store views (avoids ~4 D2D copies of ~40 MB per unit-layer); `lm_prefill_bitexact` with `V41_SUB` / cache-prior off;
whether the arena tail can be freed and reallocated; the wire width of `b` in the box-2 header; whether the WMMA kernels
are tile-composition-independent per row (reduces to f32 association only if so).

## Driver integration (10-10, evening)

* `kernels/mxfp4_moe_wmma_f16.hip` = scratch `cand_f16.hip` (WM_NTMAX 2 = the int8 arm's 32-member chunk, so the work-item
  builder is unchanged). Same entry names/args as `mxfp4_moe_wmma.hip`; loaded as a SECOND module on gfx1151
  (`Mxfp4PairMatvec::wmma_f16`, `Mxfp4Matvec::wmma_f16`); `launch_fused_swiglu_wmma_arm` / `launch_by_expert_wmma_arm`
  take `f16: bool` (the old `_ex` entry points = int8).
* `het::dispatch`: a pass that takes the WMMA arm (`V41_MOE_WMMA_GATEUP` >= 256 rows / `_DOWN` >= 128) runs the f16 kernels
  when `V41_MOE_WMMA_F16` is on (LIVE knob, default off, read per launch). Trace label `igpu.pair_wmma_f16`. Box 2 runs the
  same dispatch, so it gets the arm with an expertd redeploy + `V41_MOE_WMMA_F16=1` in its knob file.
* **E8M0 range: SAFE.** `scratch/kernel_ideas/B_moe_prefill/e8_scan.py` read every routed-expert scale tensor (47,232
  tensors, 17.4 GB, 364 s): global e8 min 115, max 128; 0 bytes outside the LUT's exact [114, 137].
* **GPU tests (tests/mxfp4_moe_sweep.rs, gfx1151, beside the live hub):** `wmma_arm_matches_kwide_within_reassociation`
  now also runs the f16 arm and the dispatch with the knob on at B = 37/128/256/512/300: worst rel_rmse vs kwide gate+up
  1.7e-7, down 9.0e-8 (int8 arm 3.0e-7 / 1.7e-7); dispatch routes to the f16 arm exactly at the WMMA thresholds.
  `wmma_f16_arm_exact_over_its_whole_scale_range`: f16 vs int8 arm (both vs kwide) at uniform e8 114 (1.8e-7 vs 3.4e-7),
  uniform 137 (1.2e-5 vs 1.9e-5: the gate outputs run far past the SwiGLU clamp, both arms), and the model's own range
  115..128 mixed inside every super-block (4.3e-7 vs 4.8e-7; down 1.5e-7 vs 1.7e-7). Report-only: the 24-octave mix
  [114,137] is 1.6e-5 vs 9.1e-6 -- the f16 arm accumulates a super-block's 8 blocks in one f32 chain; that spread does
  not occur in this model.
* **What it buys today: ~nothing until MoE calls get bigger.** Prefill MoE calls are one 512-row lane unit (~15 members
  per expert on box 1, ~5 on box 2): the regime where the harness measured the two arms TIED. The 1.4-1.7x gate+up win
  needs ~48 members/expert (B ~3072). Next lever: batch the MoE across a layer-outer window's units (4096 rows/window ->
  ~100+ members/expert on box 1) -- the cross-sub-chunk batching design below, now natural under layer-outer.
* Still owed: the golden/KLD gate (server down) before it goes default-on; an e2e A/B (live knob, no restart).
* **REAL WEIGHTS (10-10 22:0x UTC, `wmma_f16_arm_on_real_weights`, 32 s beside the live hub):** 8 real experts from each of
  layers 0/2/5/9/13/17/19/24/31/39 (e8 117..125 in the loaded experts), B = 512 and 3072, f16 and int8 arms vs kwide/kwide2:
  f16 worst rel_rmse gate+up 1.90e-7 / down 9.1e-8, int8 3.73e-7 / 1.68e-7 -- the f16 arm has ~HALF the int8 arm's error
  on every layer (max_abs 2-3x smaller). The int8 arm passed the end-to-end golden gate (09-26), so the f16 arm is gated by
  dominance at kernel level.
* **Live A/B (2.5K-token prompt, i8/i8/f16/f16/i8 under layer-outer + split):** no power -- the int8 A/A null is already
  0.19-0.25 nats (no two runs share a partition: the hot set refreshes ~every minute under traffic); f16 vs int8 0.08-0.36,
  inside the null. Records recovered from the busy dump by top-5 cluster + per-arm record counts.
* **ON in production 10-10 ~22:20 UTC (hub only):** `V41_MOE_WMMA_F16=1` in the hub knob file + env file (owner, after
  "that's a long time" for the ~35-min golden gate). Revert: delete the line (live, next MoE launch). Box 2 still runs
  the int8 arm (needs an expertd redeploy + `V41_MOE_WMMA_F16=1` in ~/expertd-knobs.txt). The end-to-end golden gate
  (`tests/v41_golden_gate.rs`, server down, ~30-40 min for off/on in one process) can ride the next restart.
