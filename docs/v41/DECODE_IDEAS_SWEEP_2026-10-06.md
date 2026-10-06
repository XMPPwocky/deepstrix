# Decode speedup sweep, 2026-10-06

Owner asks, in order: "how do we speed up multistream or DSpark decode now? ideas, rank em, workflow";
a second, disjoint sweep; "what are our blind spots?"; "build pipeline models and think about dependency
graphs much more aggressively". Standing rule set during the sweep: **no lossy requant of the routed
experts** (IQ3/IQ2 hot head or cold tail, rank-6 drops). Dense Q8 -> Q6 is allowed in principle but the
dGPU chain is launch/latency-bound, so bytes there buy nothing.

Method: three Workflow runs (33 agents; every idea priced on one traffic-weighted model and then checked
by two independent skeptics: numbers/physics and project history), plus one calibrated discrete-event
pipeline simulator of the lane-layer loop (`decode_ideas_2026-10-06/sim.py`, ±4% on 9 cells). Data: the
live hub log 2026-10-05T23:38..10-06T04:33 UTC (hub 3eccfdf9 = 5d3ce67; two-stream DSpark live from
01:39), the host-side evtrace of the same window, pick traces, code at b5def35. Read-only: no GPU work.

Related: `MS_DSPARK_STREAMS_DESIGN.md` (what went live), `HOT_SPLIT_SIM.md` (lever #1's simulator),
`DSPARK_SINGLE_STREAM_PERF.md` (the 10-01 budget this supersedes), `KV_PREFIX_STORE_DESIGN.md` (C4).

## 1. Where the time goes (measured)

Decode step wall by regime (POST window 01:39-02:36 UTC, n=16,719 steps, 1739 s):

| regime | share of step wall | step/block | tokens | tok/s |
|---|---|---|---|---|
| lone stream, DSpark | **0.483** | 104.3 ms/block | 3.59 | 34.4 |
| two live, both speculate | 0.096 | 155.8 ms/step | 6.05 | 38.8 agg / 19.4 per stream |
| two live, one speculates | 0.027 | 119.5 | | 33.8 |
| plain multistream 3-8 live (no speculation) | **0.376** | 81 (M3) .. 150 (M8) | | 37.0 .. 53.2 agg |

- Lone two-lane verify: step = 45.0 + 9.95 ms/row (rows 2-6). Drafts 11.5 ms per stream, **serial** (23.4 for
  two), fully exposed. K mean 3.48 lone (K=5 in 36.5%, fully accepted 24%), 2.69 paired (27% K=1). Accept
  0.746, rising with K (0.54 at K=1 .. 0.83 at K=5). The depth cap binds far less than the 10-03 note said.
- Lone block = draft 11.5 + host 0.9 + **iGPU MoE busy 72.2 (the pole)** + Engram join 3.2 (exposed) +
  pager_block/sel_sync/remote waits 5.1 + other iGPU idle ~9 + head/sample 1.2.
- dGPU chain stage sum **56-61 ms in every cell** (plain r3 .. plain r8, lone r3 .. two-stream r10): per
  lane-layer (80/step) output_proj 159 us, q_chain 115, shared_expert 84, attn 75, peer push 46, mhc_pre_attn 34,
  router 32, kv_chain 26, mhc_mix 22, kv_append 16, eight stages at 8-12 us; indexer 16 x 160.
- iGPU MoE per lane-layer: pair_kwide 546/580/618 us and q2k_down 244/258/277 at lone r3/r4/r6.
- Prefill bursts parked 1776 stream-seconds of live decode in that hour (> the whole decode step wall).

What the user feels (agent turns, 23:38-02:36, n=111): turn mean 73.5 s / p50 40.2 / p90 147; TTFT p50
10.5 s. Shares of turn-seconds: active decode 59%, **parked behind UTILITY prefills 13.4%** (summaries /
compaction: a restored 10K system prefix + ~4.5K fresh tokens each), own prefill 25% (of which the 128-row
decoder replay is **3.1 s every turn** = 7.7% of the p50 turn), embed stalls 2.4%, queued 2.2%. Zero KV-room
parks all night (the 09-26 parking problem is gone); the system-prefix snapshot cache is already live.

## 2. The dependency graph and the pipeline model

Both lanes share ONE dGPU compute FIFO (`de.compute`: chains, shared expert, combine, head) and one
`de.xfer` (peer push); the iGPU has `ie.compute` (MoE) + `ie.xfer` (push-back). One host thread runs the
Chain/Route/Post state machine round-robin (forward_prefill.rs:4548-4668). `Route(l)` waits on the router
readback event; **`Post(l)` is gated only by the box-2 reply**, not by the MoE (the combine's wait on
`moe_arrived` is device-side).

```
HOST  prologue 0.55 ms (embed, StepRows, compact)   draft 11.5 (lone) / 23.3 (two streams) BEFORE t0
      Engram gather thread spawned at t0 ... joined inside chain!(i,1) -> both lanes stall ~3.2 ms
per lane i, layer l:
 DC   combine(i,l-1) 17 us [device-waits moe_arrived(i,l-1) + partial H2D]
      -> chain(i,l) ~500 us: mhc_pre_attn > q_chain > kv_chain > kv_append > attn (+indexer on 8 layers,
         +engram on L1,L14) > output_proj > mhc_post > mhc_pre_ffn > router > rb_pack -> selected_ready
 HOST Route: poll + unpack + substitution/pins + box-2 submit ~0.13 ms      (box 2: rtt med 0.47 / p90 0.85 @r4)
 DC   shared_expert + mhc_mix 106 us (after the submit)      DX peer_push 46 us -> selected_pushed
 HOST launch ~60 us: group builder / work items / q8k / pair_kwide / q2k_down
 IC   MoE 0.84-0.87 ms @r4 (xq_recv 6 > group 8 > work_items 7 > q8k 9 > pair_kwide 580 > q2k_down 258)
 IX   push-back 13 us -> moe_arrived
 HOST Post (when the box-2 reply is in): accounting + partial H2D + combine enqueue ~20 us -> chain(i,l+1)
epilogue: L39 push-back + combine + sync + head_batch 1.17 ms + sample -> 2.8-3.1 ms after the last MoE
```

Steady state at lone r4: lane turnaround (combine -> chain -> readback -> host -> launch) **0.81 ms vs the
other lane's MoE 0.84 ms: 0.03 ms of slack**. The median fits; the p90 of that segment (1.65 ms, 2.2x the
median) does not, and every jitter source (indexer layers +160 us, pager spikes, late box-2 reply, DPM
clocks) lands on the iGPU as idle. iGPU idle per step (means): prologue 1.1-1.3, second lane's L0 start
1.1-7.2, **box-2 reply after the MoE 5.2-7.5**, chain+handoff tail 12.5-19, epilogue 2.6-4.2.

Host-stamp evidence (evtrace, 277-588 steps per cell): per lane-layer the box-2 reply **median** RTT
(0.42-1.0 ms) is below the MoE in every cell, but the **p90 (0.78-2.7 ms) exceeds it; 6-10 lane-layers per
step get their reply after the MoE ends.** The `remote_wait` rollup only measures the host's final wait, so
the step rollups hid this.

### Simulator
`decode_ideas_2026-10-06/sim.py` (DES: 4 in-order streams with the real dependencies, one host thread
running the exact state machine with polling, box 2 as a parallel server drawing the cell's empirical RTT,
ordered lanes for spec cells, indexer/Engram extras, Engram join and staging stalls). Device durations =
stage medians (`stage_digest.md`); ONE global set of host constants for every cell.

| cell | measured fwd | sim | err |
|---|---|---|---|
| plain M3 / M4 / M5 / M8 | 75.5 / 87.3 / 102.6 / 144.6 | 74.3 / 83.9 / 99.7 / 143.1 | -1.6 / -3.9 / -2.9 / -1.1% |
| lone r3 / r4 / r5 / r6 | 73.9 / 81.2 / 92.9 / 100.8 | 75.4 / 80.7 / 92.3 / 97.9 | +2.1 / -0.7 / -0.6 / -2.9% |
| two-stream r7 | 119.7 | 119.7 | -0.1% |

Known gap: ~5 ms/step less in-loop idle than measured (production jitter), so chain/handoff levers may be
worth up to ~1.4x the sim's numbers; MoE-byte levers are unaffected.

### What-ifs (delta fwd ms/step, weighted by tonight's regime shares)

| what-if | M3 | r3 | r4 | r6 | M8 | two r7 | weighted ms | % |
|---|---|---|---|---|---|---|---|---|
| dGPU launches -30% (latency floor x0.7) | 1.8 | 2.0 | 0.7 | 0.8 | 1.1 | 0.7 | +1.0 | 1.1 |
| dGPU bytes -50% (merged two-lane dense GEMV) | 3.2 | 4.3 | 1.8 | 1.3 | 1.7 | 1.2 | +1.9 | 2.2 |
| handoff -50% (host route/prep/launch/post/poll) | 2.5 | 4.0 | 1.3 | 1.3 | 1.5 | 1.2 | +1.7 | 1.9 |
| all three dGPU levers | 6.7 | 8.7 | 3.4 | 3.0 | 4.2 | 3.5 | +4.2 | **4.9** |
| **iGPU MoE bytes -45% (hot split)** | 1.0 | 1.6 | 5.2 | 14.3 | 37.4 | 24.2 | +12.1 | **11.3** |
| **box-2 leg fully hidden** (reply never after the MoE) | 9.2 | 11.0 | 8.0 | 8.8 | 10.8 | 11.5 | +9.0 | **9.9** |
| Engram join hidden | 3.4 | 2.7 | 3.1 | 2.6 | 3.6 | 4.7 | +3.2 | 3.5 |
| Engram staging copy async (L1, L14) | 0.3 | 0.3 | 0.5 | 0.2 | 0.3 | 0.3 | +0.3 | 0.4 |
| sel_sync hidden | 1.6 | 1.0 | 0.7 | 0.4 | 0.5 | 0.4 | +0.7 | 0.8 |
| pager_block hidden | 1.2 | 2.0 | 0.6 | 0.5 | 1.0 | 0.5 | +0.8 | 0.9 |
| 3 lanes | - | - | -8.2 | -2.0 | 2.4 | 0.7 | -2.9 | **-3.6** |
| combine on its own dGPU stream | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hot split + handoff -50% + Engram | 7.8 | 6.4 | 9.1 | 20.0 | 46.2 | 32.6 | +18.0 | 17.7 |

iGPU byte floor (every non-MoE term hidden): 12-28% headroom, 18% at the modal lone r4.

### Answers the model gives
- **Is the dGPU chain launch-bound?** One third. Per lane-layer 728 us = byte floor 41% (q 51 MB, wo 80 MB,
  shared 38 MB, mHC/router/kv ~10 MB at ~600 GB/s) + attention compute 10% + indexer/Engram/head 10% +
  peer-push DMA 6% + **latency floor 33%** (q_chain 115 vs 86 floor, output_proj 159 vs 134, router 32 vs
  6.5, kv_chain 26 vs ~5, eight stages at 8-16 us).
- **Is it on the critical path?** Only through its tail. Shrinking its MEAN buys 1-2% per lever, 4.9% for
  all three. It becomes **the** pole once the hot split halves iGPU bytes.
- **Both lanes read the dense dGPU weights every layer** (~2x dense bytes per step, ~16 ms of dGPU read
  time); the 09-26 sweep skipped the merged two-lane GEMV as "serializes lanes".
- **3 lanes are negative**: 3 chains + 3 shared experts per layer = 1.8 ms dGPU vs 1.76 ms iGPU.

## 3. Other measured facts worth keeping
- iGPU MoE kernels at lone 4-6 rows move 10.1-12.7 GB in 66-81 ms = **151-157 GB/s**, vs 200-205 the same
  kernels reached at 8 rows (09-29) and 214 achievable: a 19-22 ms/block in-busy shortfall at b=2-3 per lane.
  UNVERIFIED as a lever (the 09-26 small-b knobs were refuted); microbench at b=2-3 with realistic expert
  counts first.
- Agent requests carry tools, so reasoning IS re-sent in history (prompt_v41.rs:564 `effective_drop =
  !any_tools`) -> turn-end KV tails (prefix-store M4) are usable: previous completion = 62% of the warm
  suffix, 3.1 s/turn mean on 102/111 turns.
- `DEFAULT_TEMPERATURE` 1.0 / `top_p` 0.95 (handler.rs:48-65), clients send none. **T=0 acceptance is
  LOWER** (0.604 vs 0.735; position-matched 0.650 vs 0.700): a lower default T is not a lever.
- iGPU DPM is `auto` again (the 09-26 pin was lost at the 10-03 reboot) but runs 2655-2714 MHz under decode
  load -> the pin is worth +0.5..2%, not the 09-26 -6.7% igpu_busy.
- `GPU_MAX_HW_QUEUES` default 4 = exactly full on both GPUs (compute, xfer, two lane rb_streams); any 5th
  stream shares a queue (09-25: 247 -> 205 ms with 8). Likely the Tier-B device-timing regression's cause.
- dGPU VRAM census: 15.6 GB itemized vs 16.3 total; 0.37-0.69 GB free; dGPU-hot-experts value slope is
  ~0.64 x bytes removed (HOT_SPLIT_SIM sensitivity and tonight's regression), not 1.0.
- E2M1 codes carry 3.89 of 4 bits: entropy coding is dead at GPU speed. The scale plane (UE8M0 byte -> 3-bit
  + per-expert base) is -3.7% expert bytes, bit-exact: +2.4% weighted, M build (touches the expert path).
- The hot-split simulator says a tensor-parallel hot head (half of every hot expert on each box) buys 0..-3%
  over the policy-only interleave + top-10 replication: same balance, same next bottleneck.

## 4. Ranked levers

Traffic-weighted aggregate tok/s unless marked capacity/turn-time. Gains are after both skeptic passes.

| # | lever | gain | cost | notes |
|---|---|---|---|---|
| 1 | **Hot split** across boxes (interleave head ranks, box 1 ~55-60% of mass, replicated top-10 per layer) | **+10..15%** | L 4-5 d, two-box restart | sim +12.1 ms weighted; ~0 at 3 rows, 37 ms at M8; the only lever that moves the floor |
| 2 | **Box-2 reply tail** (p90 RTT > MoE; 6-10 late lane-layers/step; reply gates Post -> next chain) | **~10% ceiling** | M; decompose first | paging tail vs link vs server from `hub_req` stamps; levers: paging policy/pins, fewer marginal box-2 experts, f16 decode reply (KL gate), pool +~270 slots |
| 3 | **Engram join hidden** (spawn before the draft / join off the lane thread) + join each table at its own layer (L14's gather sits on L1's path today) + async staging copy | **+3.5%** | S-M 1.5-2 d; threads sweep 0 d | one host thread joins at L1 -> both lanes stall |
| 4 | **dGPU bundle**: capture the 4 remaining direct launches / fuse small stages (-30% latency floor), merged two-lane dense GEMV (-50% dGPU bytes), handoff -50% (route/prep/launch/post) | +4.9% together (<= 1.4x) | M-L | each alone 1-2%; becomes the pole after #1 -- build as #1's companion |
| 5 | Small-b MoE kernel bandwidth (151-157 GB/s at lone 4-6 rows vs 200+ at 8 rows) | up to +20% lone IF real | microbench first (one GPU window) | unverified |
| 6 | Scale-plane 3-bit packing (4.25 -> 4.10 b/w, bit-exact) | +2.4% | M 3-5 d | lossless but touches the expert path: owner veto |
| 7 | dGPU-resident hot experts in the 0.5-0.75 GB free today (1.5 GB after a VRAM diet) | +1.5-2.5% (+4% at 1.5 GB) | L 5 d | MXFP4 MoE on gfx1201; `DEEPSTRIX_ALLOC_TRACE=1` at the next restart first |
| 8 | DSpark host-tail bundle: batched drafter (P2 per stream +3-4%), pipelined drafts, exit early-stop, device-side p/q + pre-committed uniforms, skip-second-draft | ~+2-3% | S-M each | |
| 9 | Env knobs: `GPU_MAX_HW_QUEUES=8` (enabler) + `HIP_FORCE_DEV_KERNARG=1` | 0..+1% | one env line each + hub restart | A/B both in one restart |
| 10 | iGPU DPM pin `high`, persisted in the flake (dGPU stays auto) | +0.5..2% | root, 0 code | |
| 11 | getenv -> LazyLock on the lane path; profile sampling 1-in-N; one peer push per lane-layer | +0.3..1% each | S | ride the same restart |
| C1 | Embed phase: persist the 2 x 107 MB pinned buffers (alloc per phase ~0.8 s stall) | capacity +2.5-3.5% | S | |
| C2 | 128-row decoder replay: 3.1 s every agent turn | turn time -7.7% at p50 | M | context-independent fixed cost (24 ms/row x layers 21-39) |
| C3 | Utility prefills parking agent decode (10 s/turn mean, 25% of the top-decile turn) | turn time | scheduling | |
| C4 | KV prefix store M2/M3 (1.7 s/turn) + M4 turn-end tails (3.1 s/turn) | turn time | L | M1 merged; shadow `kv.turn_lcp` decides M4 |
| C5 | Box-2 eviction for long cold prefills (each cold expert re-read 3-4x) | cold prefill -25..35% | M | capacity |

## 5. Refuted or closed this round (do not re-propose without new evidence)
expert requant of any kind (owner) · gate-mass top-k / rank-6 drop (09-24: NLL +0.012, final turn +0.041,
"fallback-only") · trees/siblings and depth > 5 (<= +1.7% weighted even for a perfect depth-7 drafter;
conditional acceptance is flat by depth) · 3 lanes (-3.6%) · tensor-parallel hot head over the interleave
(0..-3%) · FP4 entropy coding (3.89 bits/code) · persistent iGPU MoE engine (the idle is data dependencies
and box-2 paging, not launches) · lower default temperature (T=0 accepts less) · pre-drafting rejection
branches (every root needs the head) · sibling at the rejected position (a row costs 9.9 ms, worth <= 0.15
tok) · stale-p mix for the next block (ceiling +3%, realistic < 1%) · drafter requant (break-even alpha loss
< 0.009) · n-gram hybrid (<= +0.3%) · B1 top-K 80 (negative: moves ranks to box 2's LRU) · MALL-aware order
(lane A ends on down-proj bytes, lane B starts on gate/up) · K-cost-cell convex projection (reverses an owner
decision) · root lane during the draft (iGPU busy with the drafter) · decode inside prefill waits (multi-week,
~+3%) · free-running lanes (+0.6-1.4% after the shipped tail trim) · bigram-only drafts for 3-8 live ·
embedding weights resident between phases (embed design constraint 2) · "deploy the fast chain" (live on
box 2 since dd83657) · combine on its own dGPU stream (0).

## 6. Workflow
1. **No code, this week:** env knobs (#9) + DPM pin (#10) + Engram threads sweep (#3) in ONE hub restart with
   `DEEPSTRIX_ALLOC_TRACE=1` (#7); decompose the box-2 tail from tonight's evtrace (#2, CPU only); small-b
   kernel microbench at b=2-3 (#5, ~30 min GPU window).
2. **Week 1:** Engram join work (#3) + embed pinned buffers (C1) + the S items in #11 -> one hub restart,
   step-cell A/B. Start the hot-split build (#1) on `scripts/split_sim` policy c/d.
3. **Weeks 2-3:** hot split gates -> two-box restart -> A/B. Design the dGPU bundle (#4) in parallel: it is
   what the split exposes.
4. **Weeks 3-4:** box-2 tail levers per the decomposition (#2); dGPU bundle (#4); #6 only if the veto lifts.
5. **Later:** C2-C4 (they move turn time more than any step lever after #1), C5.

Judge every item but #1/#2/#5 by step-cell A/Bs (regime, rows, lanes): all sit inside the 8% E2E noise floor.

## 7. Files
- `decode_ideas_2026-10-06/sim.py` simulator, `calib.py` calibration, `whatif.py` the table above,
  `extract.py` / `analyze.py` the evtrace host-stamp reader and critical-path decomposition,
  `stage_digest.md` the per-cell ms.stage medians (input to the sim).
- Sweep outputs (agent reports, per-idea JSON) in the session scratchpad
  `~/.claude/jobs/749c61d3/tmp/` (wf1_result.json, wf2_result.json, wf3_result.json, wf2_digest.txt, wf3_digest.txt).
- Memory: `project_v41_decode_ideas_sweep_2026-10-06.md`, `feedback_no_expert_requant.md`.
