# dGPU bundle: cut the per-lane-layer dGPU critical path (design, 2026-10-10)

Status: DRAFT rev 2 (review round 1 folded in, section 9), branch `worktree-dgpu-bundle` (from main 1a95637 =
production source). Owner 10-10: "Let's go dGPU bundle. Remember, btw -- the dGPU is always on the critical
path, because all attention runs there, blocking later layers!"

## 0. Decisions

1. **Target: per-lane-layer dGPU WALL latency** (each lane's cycle is serial: chain -> route -> MoE -> post ->
   next chain), not busy time or bytes alone.
2. **Bit-identical first**: slices 1-3 change no arithmetic and are gated bit-exact against the deployed source
   (comparator G5, as the hot split's G-top). A slice that changes picks (not arithmetic) is called out.
3. **Every change behind a live knob** where it has a runtime alternative; rollback = one knob line.
4. **Hub only.**
5. **Measure before folding**: no new graph capture until Step 0c prices multi-node replay overhead.

## 1. Baseline -- what we know after review round 1

- The 10-10 measurement ran with `V41_MS_PROFILE=1`: ~28 `hipEventRecord` per dGPU lane-layer on every step.
  **Measured tax** (production, interleave; profile on 10-09 23:40 -> 10-10 02:23 UTC vs off since the 02:24
  restart, PCIe-pin A/B `auto` blocks only): lone r3 70.4 -> 64.3 ms (-8.7%), r4 76.7 -> 69.9 (-8.9%), r5 85.5
  -> 81.7 (-4.4%), r6 ~0. Banked by turning it off; `V41_MS_PROFILE_SAMPLE` (bundle 0, a81d16d) brings
  `ms.stage` back at 1/20 of it. Section 1 of rev 1 (chain 510-560 us, host enqueue 82-102 us, floor 36-41%,
  DES +7.7%) is inflated by that tax; **re-baseline (Step 0a)** at the bundle's restart: profiled vs
  unprofiled steps at matched rows (the tax per lane-layer) and `V41_LAYER_HOST_TIMING=1` without events.
- Unaffected by the tax (relative facts): the hot split cut the iGPU MoE leg 10-17%; at <= 4 rows per lane the
  dGPU side is the largest leg.
- The **slow-xfer mode** (25-48% of steps): only the shared expert (excess over its byte floor 22 -> ~140 us)
  and the SDMA peer push (43 -> ~155 us) stretch; q/out/kv and `rb_pack` (kernel stores over PCIe) do not. The
  shared expert runs on `de.compute` exactly while the push runs on `de.xfer` -- the 09-25 copy-beside-compute
  signature (rb_stream copies beside the shared expert slowed it 2.4x; the kernel pack fixed it). The PCIe LCLK
  pin A/B (running 10-10 02:47-06:47 UTC) tests the other explanation.

## 2. Slice 1: cheap host work (bit-identical)

| item | change | note |
|---|---|---|
| getenv on the lane path | read once (`lane_env`) for names nobody flips in-process | DONE 17018ca / 1361670 |
| cache-prior fill loop (FP ~8396-8423: 384 x `partition_box2` + `hot_set::moving` + mirror lookup (5 atomics) + SipHash `pg.is_resident`, ~15-25 us/lane-layer) | per-layer residency BITSETS kept incrementally (pager slot map changes, mirror map updates, hot-set refresh) and the mask built with word ops; the device prior stays a float array (`held ? boost : 0`), filled from the mask | same values, same kernel; top-k stays DIRECT (its live scalars) |
| post: `vec_add` (moe + shared) then `hc_post_add` (+ remote) | one 3-input combine with the `(moe + shared) + remote` order where `fuse_remote_add` holds; a 2-input `+shared` twin where no remote partial | bit-identical by construction (`hc_post.hip:104-136`); the moe+shared add moves behind box 2's `wait()` (harmless) |
| `expert_sel_count` (separate kernel + mutex; feeds M62 placement + `REQ_TOUCHED_PF`) | fold the count into `rb_pack` (it already reads `d_selected` / `d_orig_sel`) | consumers unchanged |
| Engram: both tables joined at layer 1 on the lane path (1.5-3.2 ms/step host block); a blocking null-stream `hipMemcpy` (a device-wide drain) | per-table handle (layer-14 table joins at layer 14); ready-first's Chain(1)/Chain(14) poll `is_finished()` and serve other lanes meanwhile; gather threads write pinned per-(lane, layer) buffers -> `hipMemcpyAsync` on `de.compute` (step-end sync covers lifetime). NEVER enqueue chain(1) before the copy is recorded (an event waited before it is recorded does not wait) | gate with `V41_MS_CTX_CHECK=1` + race probes (the null-stream drain was an implicit sync) |
| step prologue: per-row blocking null-stream `input_hcs` uploads + pageable `dev.upload` / `pos_per` copies | one pinned batch upload | |

Dropped from rev 1: the route-audit downgrade (it is O(picks), < 1 us, and guards exactly-once routing),
one-lane `sel_sync` polling (no host work to overlap in that driver).

## 3. Slice 2 (ranked up): zero-copy peer push

The iGPU MoE reads xq / selections / weights straight from the pinned `rb_pack` buffer (or its existing
`xq_recv -> d_xq_q8k` copy reads from it), waiting on `selected_ready`, instead of three `hipMemcpyPeerAsync`
on `de.xfer` -- which removes the copy that runs beside the shared expert (the slow-xfer suspect) and 43-155 us
off the iGPU's start. Per lane-layer predicate with the peer-push FALLBACK when: `b2_mirror::mode() == 2` (route
rewrites `d_selected` / `d_ew` on the host after the pack), the pack is absent (`rb_pack_on && b <= max rows &&
pager union`), `xq` is not in the pack (`remote_split_on && rb_u8`), or the f32 `ain` path (WMMA MoE /
`!xq_pushed`). Overwrite safety by stream order (rb_pack(l+1) follows post(l)'s wait on moe_arrived(l)).
**Microbench first (window)**: the iGPU has the `hipHostMalloc(0)` allocation mapped, reads are coherent across
layers (no stale L2), and its read latency/bandwidth on it. Knob `V41_DGPU_ZC_PUSH`.

## 4. Slice 3: kernel fusion + graph merge (bit-identical)

- **Step 0c microbench (window)**: N direct launches vs an N-node replay of the real stage bodies at b = 1/4/8,
  and a `V41_MS_GRAPHS=0` window arm. Decides whether multi-node graph replay has per-replay overhead.
- q_chain + kv_chain into one graph (one replay fewer; a pure win if overhead is per replay).
- Router: the main router matvec and the k1 look-ahead's (2 under k1) in ONE kernel with a layer dimension
  (each output's reduction unchanged; needs a second logits buffer: the look-ahead overwrites
  `sd.router_logits` today); main + look-ahead top-k in one launch. Top-k stays direct.
- Router-block / attention-block CAPTURE only if 0c and the context-slot budget allow (16 of 32 ArenaCtx
  slots free; a router block needs ~19-20, an attention block 5-6 more; the attention graph must key
  `eff_n_total_max` / dec_fused eligibility / indexer firing -- the dense class changes every token).

## 5. Slice 4 (last or cut): shared expert merged across lanes

Bit-identical only when b_A + b_B <= 8 (dp4a tB arms; above 8 the WMMA f16x arm changes numerics); couples the
lanes (A's combine behind B's chain enqueue). ~30-40 us per lane-layer when it applies. A/B it last.

## 6. Missing lever (review #11): host head-of-line in ready-first

The single host thread cannot route/post the other lane while it enqueues a chain (~90 us under the profile).
Measure `selected_ready -> route start` and `remote ready -> post`; if material, split `chain!` in two phases
(through `mhc_pre_ffn`, then router onward) and poll the other lanes between them.

## 7. Not code

- PCIe LCLK pin: the live A/B decides; persisting needs a NixOS module.
- `GPU_MAX_HW_QUEUES` is full; a shared-expert side stream needs a queue: not in this bundle.

## 8. Gates and rollout

One hub restart window (owner go): Step 0a (profile tax re-baseline: sampled profile live), 0c (graph replay
microbench) and the zero-copy microbench; comparator G5 (deployed source vs new binary, all bit-identical
slices ON) bit-identical; G5a-h + G6 with each slice knob off/on; `V41_MS_CTX_CHECK=1` arm; a kernel unit test
for every fused kernel (old vs new on random inputs, exact). Live: per-turn knob A/B (`ab_knob.py`) per slice.

## 9. Review round 1 (2026-10-10, NEEDS REWORK) -> rev 2

1 audit downgrade dropped. 2 baseline re-measured (profile tax, section 1) + Step 0a. 3 zero-copy ranked up
(section 3). 4 zero-copy fallbacks + microbench. 5 router-block capture conflicts with the prior bitmask arg ->
top-k stays direct, fusion instead. 6 slot budget stated. 7 slice 4 capped / last. 8 attention-graph keys. 9
Step 0c before captures. 10 Engram per-table join + poll + pinned async. 11 head-of-line lever (section 6). 12
incremental residency bitsets. 13 sel_count folded into rb_pack. 14 combine scope + 2-input twin. 15 unit tests
for fused kernels (gate coverage without box 2). 16 one-lane polling dropped. 17 step prologue batch. 18 k1 = 2
router matvecs. 19 IQ2/Q2K variants back to per call.

## 10. Deployed (2026-10-10 04:20 UTC, hub 04739a1f = 5809521) and first result

Window (owner "ok"): hub down 03:45 -> 04:20 UTC. G5 from the deployed source a37a799 (now with
`V41_PUSH_XQ=1`, as production: no earlier window had set it, so production's xq-push path was never
gated) == the new source with slices 1a-1e ON and zero-copy OFF bit for bit (244 logits files, 132,989
pick lines); == zero-copy ON bit for bit (34,240 lane-layers through the pull); `V41_MS_CTX_CHECK=1`
PASS. Env: `V41_MS_PROFILE=1` + `V41_MS_PROFILE_SAMPLE=20` + `V41_DGPU_ZC_PUSH=1` (backup
`.pre-dgpu-20261010`).

First 2.5 h of traffic vs the profile-OFF baseline on hub fd655378 (10-10 02:53-03:44 UTC), ms.step p50:
lone DSpark r3-r6 -7.6..-8.2% (r4 69.4 -> 64.9, r6 90.1 -> 82.7), lone plain r1 -6.9%, two-stream r4-r12
-3..-8%, plain3 r3 +1.5% (n 4173 vs 6084: watch -- suspect zero-copy at 1-row lanes); traffic-weighted
0.950 (understates: the sampled profile, ~0.5%, is back). Lone DSpark 40.9 -> 48.2 tok/s (acceptance
also differs). The PCIe LCLK pin A/B (12 blocks before the window) was null (auto/pin 1.005): the
slow-xfer mode is copy-beside-compute, which zero-copy removes. Next: per-turn A/B of `V41_DGPU_ZC_PUSH`
(attribution + plain3 r3), then slice 3 after Step 0c.

## 11. Zero-copy per-turn A/B (2026-10-10 09:00-12:00 UTC, 176 turns) and the queued fix

`V41_DGPU_ZC_PUSH` 1 vs 0, ms.step p50: lone DSpark r3-r6 0.982-0.996 (faster), plain 3/4/5 streams
0.978-0.990 (faster; plain3 r3 0.990 -- so the deploy's plain3 r3 +1.5% is not zero-copy), two-stream DSpark
r4-r12 1.003-1.030 (slower; 5 cells above the 1.01 bar). Net ~ -0.5%. Reading: zero-copy puts THREE copy
operations on the iGPU's in-order `ie.compute` (the peer push left one, `xq_recv -> d_xq_q8k`); two-stream
DSpark is the iGPU-bound regime (both streams' drafters + the larger MoE), where queue operations on the
iGPU cost the step; in the dGPU-bound regimes removing the dGPU's SDMA copies wins.

**Owner 10-10: keep zero-copy ON everywhere; queue the fix as part of slice 3:** one copy per lane-layer --
lay sel / ew / xq out contiguously in the readback pack AND in the iGPU landing scratch (one contiguous
allocation with typed views), so a single host->device copy fills all three; then re-A/B, two-stream cells
first.

## 12. Step 0c result (2026-10-10, tests/bench_graph_replay.rs, dGPU beside the live hub)

Seven small kernels (vec_add over b x N_EMBD), paired random order, 1500 rounds, 95% CIs < +-0.05 us:

| b = 4 | device (queued) | device (cold) | host submit |
|---|---|---|---|
| 7 direct launches | 28.7 us | 25.8 | 4.7 |
| one 7-node graph (q+kv merged) | 31.0 | 28.1 | 2.5 |
| 5-node + 2-node graphs (today) | 37.5 | 34.7 | 4.4 |
| seven 1-node graphs | 68.6 | 68.4 | 12.1 |

(b = 1 the same within 1.5 us.) A graph replay costs ~6.5 us of FIXED device time; a direct launch < 1 us
of host. The `V41_MS_GRAPHS` default's premise ("~40 kernels per lane-layer at 20-40 us each") no longer
holds. Production replays four stage graphs per lane-layer (q_chain, kv_chain, output_proj, shared): if
the real stages behave like this, graphs cost ~26 us of dGPU time per lane-layer (~2 ms/step) on the
critical path. **Plan:** `V41_MS_GRAPHS` made LIVE (decided at each stage's start) so the next deploy
A/Bs graphs off vs on per turn; the q+kv merge (-6.5 us) only matters if graphs stay on.

## 13. The zero-copy fix: regime gate instead of one contiguous copy

A true single copy needs either fixed-offset padding in the readback pack (its segments are dense) or
re-pointing ~8 destructure sites in `pre_moe_launch` at typed views of one landing buffer -- and it only
goes from 3 queue operations to 2 (`xq` lands in the shared `si.d_xq_q8k`, `sel` / `ew` in per-lane
`bi`). The penalty it targets is ~0.4% overall (two-stream DSpark only). Option 3's other variant does it
fully: zero-copy stays OFF in steps where >= 2 streams speculate (`STEP_MULTI_SPEC`, set by the scheduler
after drafting; `V41_DGPU_ZC_MULTI_SPEC=1` re-enables it there). Behaviour of lone / plain steps unchanged.

## 14. Slice 3a/3b deployed; why a graph replay costs device time; kernargs in VRAM (2026-10-10)

**Deploys.** 3a (4efe41f, hub 85fc7275, 16:28 UTC): `V41_MS_GRAPHS` live + the ZC regime gate. 3b
(b00eadd, hub c18af233, 19:58 UTC): the fused router (`V41_DGPU_ROUTER_X2`, default on: layer l's gate
matvec and the k1 look-ahead's in one `f16_matvec_batched_h20_x2` launch, direct path only; dGPU 12.2 ->
8.0 us at b = 1, 27.5 -> 24.4 at b = 8) + env `DEBUG_CLR_KERNARG_HDP_FLUSH_WA=1`. Every gate bit-identical
to the deployed source (244 logits files, 132,989 picks); proofs `ZC: zc_pulls`, `RX2: router_x2`, and
CLR's `Using dev kernel arg wa = 3` on both GPUs. `AMD_LOG_MASK` is parsed as DECIMAL: `0x800` prints
nothing, `2048` is LOG_INIT. A gate that turns graphs off must not assert G6's capture counts.

**Graphs A/B on 85fc7275 (147 turns).** Lone DSpark: graphs ON +1.7..2.1% step (r3-r5, thousands of
steps per cell); plain 3-7 streams ~neutral (plain3 r3 +1.7..5%); two-stream DSpark: ON 2-3% faster
(small n). Device-bound cost model: 4 replays x 40 layers x ~6.5 us = ~1 ms per stream per step
(lone measured +0.9..1.2 ms, plain3 +3.4); graphs pay only where the host enqueue is the bottleneck.

**Why a replay costs device time (CLR rocm-7.2.3 source).** `hipGraphLaunch` on a linear kernel-only
graph (`EnqueueGraphWithSingleList`, packets pre-built at instantiate under the default
`DEBUG_CLR_GRAPH_PACKET_CAPTURE=1`) adds TWO barrier-AND packets per replay: the accumulate command and an
unconditional callback marker (`hip_graph_internal.cpp:747-815, 1141-1150`), each with a host-memory
interrupt completion signal; scopes NONE (no cache flush). The CP drains the queue at each (~3 us).
Bench (`bench_graph_replay`, beside the live hub, paired): interrupts off / queue ring in VRAM change
nothing; `DEBUG_CLR_GRAPH_PACKET_CAPTURE=0` removes one barrier (-4.6 us/replay) but sends nodes down the
per-node host path (not adopted). No instantiate / stream flag removes the marker: fewer replays (merge
the 4 stage graphs) is the only graph-side lever.

**Kernargs.** Direct launches read their kernargs from HOST memory over PCIe: on a PCIe dGPU,
`HIP_FORCE_DEV_KERNARG=1` alone is a no-op (`rocsettings.cpp:239-267`: device kernargs need the HDP-flush
workaround, `DEBUG_CLR_KERNARG_HDP_FLUSH_WA`, default false). With both: mode 3 (DeviceKernelArgsHDP),
direct launches -0.66 us/kernel on the dGPU (7 kernels 28.4 -> 23.8 us); graphs unchanged (their kernargs
already live in a VRAM pool). Graphs re-A/B'd on c18af233 from 20:00 UTC (direct launches are now cheaper).

**CORRECTION (2026-10-10 20:34 UTC): VRAM kernargs REVERTED.** The flag is process-wide, and on the iGPU
(gfx1151, kernargs in the DRAM carve-out + an HDP flush per launch) mode 3 roughly DOUBLES every kernel's
device time (`BENCH_ARCH=gfx1151 bench_graph_replay`, two runs: 7 direct 20.6 -> 40-41 us, 5+2 graphs
30.5 -> 54-58), ~+2.8 us/kernel on the MoE's critical path, vs the dGPU's ~-1.2. Env line removed (hub
c18af233 unchanged otherwise). Do NOT set `DEBUG_CLR_KERNARG_HDP_FLUSH_WA` until CLR decides it per device
(planned: patch `Settings::setKernelArgImpl`, which runs per device with the device's `isa`, in the nix
flake; gate on the ISA, not `apuSystem_`, which is set only for full-profile agents).
