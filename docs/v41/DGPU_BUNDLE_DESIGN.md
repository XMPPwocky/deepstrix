# dGPU bundle: cut the per-lane-layer dGPU critical path (design, 2026-10-10)

Status: DRAFT rev 1, branch `worktree-dgpu-bundle` (from main 1a95637 = production source). Owner 10-10:
"Let's go dGPU bundle. Remember, btw -- the dGPU is always on the critical path, because all attention runs
there, blocking later layers!"

Sources: the 10-06 sweep (`DECODE_IDEAS_SWEEP_2026-10-06.md` lever #4), the 10-10 code map of one decode
lane-layer's dGPU work and the 10-10 critical-path measurement on interleave traffic (job scratch
`~/.claude/jobs/749c61d3/tmp/dgpu/measure/`; numbers below are from it unless cited).

## 0. Decisions

1. **Target: per-lane-layer dGPU WALL latency** (enqueue -> readback ready, and the next chain's start), not
   busy time or bytes alone. Each lane's cycle is serial: chain -> route -> MoE -> post -> next chain.
2. **Bit-identical first.** Slices 1-3 change no arithmetic (same kernels, same order of every reduction);
   each is gated by `multistream_step` G5 bit-exact vs the deployed binary (G-top style). Slice 4 changes a
   GEMV's row grouping and is judged by KL.
3. **Every slice behind a live knob** (default ON once gated, so a rollback is one knob line), and every slice
   measurable: the sampled profile (`V41_MS_PROFILE_SAMPLE`, bundle 0, committed a81d16d) brings `ms.stage`
   back at ~1/20 of the cost.
4. **Hub only.** No box-2 change.

## 1. The measured picture (interleave traffic, 10-09/10)

- dGPU device chain ~510-560 us per lane-layer, unchanged by the hot split; the iGPU MoE leg shrank 10-17%.
- At <= 4 rows per lane the dGPU side is the LARGEST leg: plain r3 ~912 us (HOL + device + tail) vs iGPU
  719 vs box 2 684; lone r4 882 / 815 / 702. dGPU share of the cycle 41% weighted, ~50% at low rows.
- Inside the chain (lone r4, us/lane-layer): q_chain 116 (byte floor 86), output_proj 160 (134), shared
  expert 87 (63), router 51 (THREE router matvecs since k1: +18), kv_chain 27 (5), attention 35 + indexer 34
  amortized, mhc_pre_attn 20 + Engram 15 amortized, mix_late 22, combine 19, kv_append 18, mhc post/pre_ffn
  10 each, rb_pack 9; peer push 43 on `de.xfer` (gates the iGPU MoE).
- Latency floor 36-41% of `de.compute` work (224-293 us per lane-layer), up from 10-06's 33%.
- Host: chain enqueue (`lh.pre_moe`) 82-102 us per lane-layer for ~520 us of device work; route block 19-35;
  the one-lane driver blocks in `sel_sync` for the whole chain (~3% of traffic); the Engram join blocks the
  single host thread 1.5-3.2 ms per step.
- A **slow-xfer mode** on 25-48% of steps (peer push 43 -> ~155 us, shared expert ~90 -> ~230 us, +3.9
  ms/step matched) -- suspected PCIe LCLK DPM; a live pin-vs-auto A/B runs 10-10 02:47-06:47 UTC
  (`~/scratch-ms/ab_pcie_pin.py`). Not code; listed for completeness (section 6).
- DES (recalibrated, -4%): handoff -50% +3.4%, launches -30% +2.2%, dense bytes -50% +4.1%, all +7.7%.

## 2. Slice 1: host handoff (bit-identical)

Per lane-layer host work that does not need to exist or can leave the lane path:

| item | where | change | gain |
|---|---|---|---|
| ~20 uncached `getenv` per lane-layer | FP 6810, 7526, 8972, 9151, 9490, 9755, 9781, 9802, 9911, 10087, 10127, 10132, 10171, 10214, 10409, 10675, 10894, 11209, 11652, 11713; `prefill_f32_matvec()` 492 | `LazyLock` / knobs (static ones stay static; none is meant live) | ~20 x ~0.3-1 us |
| `verify_routing_exactly_once` (prep audit) | FP ~10242 | keep the cheap per-pick ownership count; the O(N_EXPERT) audit only under `V41_ROUTE_AUDIT=1` (debug) | O(384) scan per lane-layer |
| `Vec` allocations in route (`seen`, `ids`, masks, `sel_wants`, ...) | FP route | per-lane scratch reused across layers | allocator churn |
| cache-prior `d_prior` H2D (384 f32 from a per-layer pinned slot) | FP 8396-8424 | the prior is exactly `boost x held`: pass a 48-byte held bitmask + the boost scalar as kernel ARGUMENTS of `router_topk` (it is uncaptured) | 1 H2D + 384-float host fill per lane-layer, off the router path |
| `expert_sel_count` (stats kernel + mutex) | FP 8687 / engine.rs 1801 | skip on decode rows (`DEEPSTRIX_SEL_STATS` stays for prefill) | 1 launch + 1 mutex |
| post `vec_add` (moe + shared) then `hc_post_add` | FP 11519 / 11797 | one 3-input combine kernel keeping the `(moe + shared) + remote` order (bit-identical by construction: same f32 adds, same order) | 1 launch |
| Engram rows: blocking null-stream `hipMemcpy` (waits for ALL queued dGPU work incl. the other lane's chain) | FP 3939 | pinned staging + `hipMemcpyAsync` on `de.compute` + an event | removes a device-wide drain twice per step |
| Engram join on the lane path (`LazyEngramRows::get`, 12352) | FP 4494/4512 | join on a helper / before the step's first chain enqueue (the gather started at sampling, 10-06 sweep #3) | 1.5-3.2 ms/step host block |
| one-lane driver blocks in `sel_sync` | FP 3968 / 9012 | poll `selected_ready` like ready-first, doing the next chain's host prep meanwhile | ~3% of traffic |

Knob: `V41_DGPU_HANDOFF` (default on after gate; off = today's code paths kept verbatim for the A/B).
Gate: G5a-h + G6 bit-exact, and the deployed-source comparator bit-identical (as G-top).

## 3. Slice 2: launch folding (bit-identical)

- **Router block:** the router matvec for l and the k1 look-ahead's for l+1 (and l+2 under k2) are three
  launches reading three weight matrices with the same input; one kernel with a layer loop (or a grid over
  layers) does the same arithmetic per output. Then top-k (+ prior bitmask, slice 1), look-ahead top-k, xq
  quantize and `rb_pack`: with `n_protect` / `dry` moved into device memory (the code comment at FP ~8445
  says so) the whole block is capturable as ONE graph per (rows, prior on/off, look-ahead on/off, layer 39).
- **q_chain + kv_chain:** adjacent graphs with no host code between (FP 5338 / 5606): one graph. Also carry
  the context entry on the previous layer's `hc_post_add` so `mhc_pre_attn` becomes its first node (layer 0
  keeps today's path).
- **Attention block:** `attn_meta` (FP 6844) per lane-layer -> a once-per-step device table indexed by
  (layer class, row); then kv_append, the optional gather, `attn_dec`, output_proj, hc_post and mhc_pre_ffn
  become one graph per topology class (window-only / dense compressed / gathered top-k). KV-source layers and
  layers where the indexer fires stay direct (host-decided boundary rows and top-k grids).
- Folding, not single captures: a one-node graph replay costs +6.7 us GPU vs a direct launch
  (GRAPH_KEYS_DESIGN 2.11).
Knobs: `V41_DGPU_FOLD_ROUTER`, `V41_DGPU_FOLD_QKV`, `V41_DGPU_FOLD_ATTN` (live; graphs re-key on change).

## 4. Slice 3: peer push -> zero-copy

`rb_pack` already lands xq + selections + weights in a pinned host buffer (batch_scratch 1300). The iGPU MoE
reads it directly (waiting on `selected_ready`) instead of three `hipMemcpyPeerAsync` on `de.xfer` (43 us,
~155 us in slow-xfer mode, on the iGPU's start). Overwrite safety by stream order (the next `rb_pack` comes
after the device waits on `moe_arrived`). Unverified: iGPU read bandwidth/latency on that pinned memory and
coherence -- a microbench gates it. Knob `V41_DGPU_ZC_PUSH`.

## 5. Slice 4: shared expert merged across lanes (KL-judged)

The only workable merged-GEMV form (full lockstep measured worse 09-22): hold lane A's shared expert at layer
l until lane B's router(l), run one pass over [A | B] rows (one 38 MB weight read instead of two), each lane's
combine reads its slice. Needs a joint row buffer, a graph keyed by b_A + b_B, and changes nothing else.
Changes the GEMV's row grouping only (each row's dot products are the same per-row reductions in the dp4a /
WMMA kernels -- verify; if per-row results are bit-identical it is a bit-exact slice). Knob
`V41_DGPU_SHARED_MERGE`.

## 6. Not code

- PCIe LCLK pin (the slow-xfer mode): the live A/B decides; persisting needs a NixOS module (udev/oneshot).
- `GPU_MAX_HW_QUEUES` 4 is full; a side stream for shared+mix (sim +0.5%) needs a queue: not in this bundle.

## 7. Gates and rollout

1. Slices built in order, each with host tests; one hub restart window (owner go) gating all built slices:
   comparator G5 (deployed source) vs the new binary with all bit-identical slices ON = bit-identical; G5a-h
   + G6 per slice knob off/on; slice 4 under KL bars.
2. Live: per-slice knob A/B per turn where the effect is per step (no placement state): the per-turn
   ab_knob.py works here, unlike the hot split.
