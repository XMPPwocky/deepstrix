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
