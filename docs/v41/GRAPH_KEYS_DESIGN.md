# Arena stage graphs keyed by (stage, rows) only

Status: DESIGN rev 2, 2026-10-04 (review round 1: NEEDS REWORK; dispositions in section 7).
Branch `worktree-ms-dspark2` (on eb84ebb).

## 0. Problem

The arena decode step captures its per-layer dGPU stages as HIP graphs (f71a9d5: the 1-8-row
chain was host-launch-bound, ~40 kernels x 20-40 us per lane-layer; a graph replays each stage as
one launch). The cache key is `layer | b << 8 | (bd.residual address >> 8) << 24`
(forward_prefill.rs ~1921): one graph per (stage, LAYER, rows, LANE BUFFER ADDRESS), never freed.
Default knobs hold ~2240 graphs (8 stages x 40 layers x 7 (lane, rows) pairs), ~2560 with a lone
DSpark stream; every new (rows, lane) shape costs ~320 more executables of device memory for good.
Two speculating streams brought new shapes on 2026-10-04 and took the dGPU to 99.5% VRAM (the
layer-major prefill window halved 8 times; a capture whose instantiate runs out of memory fails
its step). eb84ebb's memory reserve is a safety net, not a fix.

Owner, 2026-10-04: "row count b? sure. keying off that is fine. keying off a buffer address seems
obviously bad, as does keying off layer." Goal: graphs keyed by **(stage, b)** (plus a bounded
TOPOLOGY class, 2.6) -- one graph per stage and row count, replayed for every layer and lane --
replay cost unchanged (about one launch per stage), numerics bit-identical.

## 1. Facts (code map + review round 1, eb84ebb)

- Stages: 8 on the default path (`g.mhc_pre_attn`, `g.q_chain`, `g.kv_chain`, `g.output_proj`,
  `g.mhc_pre_ffn`, `g.router_matvec`, `g.shared_expert`, `g.mhc_mix_ffn_late`); 4 more in mHC split
  mode (de.hc). On the default path every capture and every operand read is on `de.compute`.
- Captured stages of a lane-layer are inside one `pre_moe_chain` call EXCEPT `g.shared_expert`
  under `V41_PREFILL_PRESUBMIT=1` + remote split, which is issued from `pre_moe_prep` (~9677) --
  after another lane's chain in the pipelined and ready-first drivers (review finding 1). So the
  context cannot be written per chain; it is written per capture (2.2).
- Layer-uniform topology on the default V4.1 path: no capture-region branch reads the layer; per
  layer only the `dlw.*` weight pointers and the rope scalars (two classes) differ; grids depend
  on b and constants. Arms that are NOT indirect-able exist and are chosen inside the stage body:
  D2D memcpy nodes on per-lane buffers (`hc_pre_carry := split`, ~4963 / 7791 / 8264, mhc
  fast/fused off), per-row `bd.residual` slices (pre-scaled arm ~4880-4901), the b > 8 WMMA arms,
  a pointer-alignment-chosen f16 variant (f16.rs:64-71, b > 64), and per-layer dtype arms (GGUF
  loads may mix Q5_K/Q6_K per layer). Hence 2.5's taint guard.
- Per lane: fixed `bd.*` buffers and `bd.residual` (swaps after each layer, before the next
  chain; current when a lane-layer's captures run). No captured stage reads `RowTablesDev`.
  Shared `sd.*` and engine pointers are one per process and stay baked (2.7).
- Every launch goes through `launch_kernel!` -> `hipModuleLaunchKernel`; arguments are raw
  pointers and scalars; any `Copy` struct can be passed by value (precedent: `AmfArgs`).

## 2. Design

### 2.0 Step 0: measure before building (GPU micro-benchmark, short window)

Extend tests/bench_launch_overhead.rs (dGPU, hub down for a few minutes) and measure:
1. Host enqueue time and the event-timed GPU gap between two dependent kernels on one stream,
   with nothing / a `hipMemcpyAsync` H2D of 256 B from pinned memory / a 1-WG `ctx_store`
   kernel taking a 256-B struct by value / `hipStreamWriteValue32` in between.
2. 80 back-to-back `hipGraphLaunch` of ONE executable queued behind a long kernel (does ROCm block
   the host or serialize an executable that is still in flight?), vs 80 distinct executables.
3. An `_ind` vs direct twin of one gemv (`q8_0_gemv_bpack_tB{b}`) at b = 1, 4, 8: time and
   bit-exactness.
Go / no-go: the context write adds <= ~5 us GPU and <= the cost of one graph launch on the host,
re-launching one executable 80x per step neither blocks nor serializes, and the `_ind` twin is
bit-exact and within noise. Otherwise this design stops here and the doc is revised.

### 2.1 Operands: `Arg` and a pointer-only context

Wrappers take operands as `Arg::Dev(&buf)` (direct, today) or `Arg::Ctx(slot, &buf)` (read the
pointer from context slot `slot` at run time). The real buffer is passed in both cases, so the
wrappers' byte_len / len checks stay. The context is POINTER-ONLY: `struct ArenaCtx { const void*
p[N_SLOTS]; u64 seq; }` (~24 slots + a sequence number for the canary, 2.8). The per-layer rope
scalars live in a static device array `rope_dev[layer]` built at load; the context carries a
POINTER to the layer's entry, so rope kernels read their 6 floats through one slot.

### 2.2 The context write is tied to the capture (`ensure_ctx`)

In indirect mode `stage_cap_on` receives the lane-layer's entry (a host `ArenaCtx` built once per
lane-layer from `dlw`, `bd` and `rope_dev`, carried with the lane's chain state so `pre_moe_prep`
has it too) and calls `ensure_ctx(entry)` BEFORE it replays or begins a capture: if the entry
differs from a host shadow of the last enqueued entry (whole-entry compare, residual pointer
included), it enqueues a write of the entry into the single device slot `ctx_dev` on the stage's
stream and updates the shadow. Stream order makes the stage read that entry. This holds for every
capture site and driver, including the presubmit shared expert, at ~1 write per lane-layer (2 with
presubmit). The shadow resets at step start and on any error.

The write is a 1-WG `ctx_store` kernel taking the entry by value (256 B, well under the kernel
argument limit): same HW queue as the stages, no pinned ring, nothing to keep alive. (Step 0 checks
it against `hipStreamWriteValue32` + a static table and the H2D copy.)

### 2.3 Indirect kernel twins

Each default-path kernel family with an L / P / R operand gets an `_ind` twin generated by a macro
around its existing `__device__ __forceinline__` body: the twin takes `const ArenaCtx* __restrict__
ctx`, the usual operand pointers, and a `uint32_t ind_mask`; its prologue resolves, before any
store, `p_i = (ind_mask >> i) & 1 ? ctx->p[slot_i] : p_i` (thread-uniform, scalar loads) and calls
the body. Direct kernels are untouched (prefill, single-token decode, uncaptured paths). Families:
`mhc_fast_batched`, `q8_0_gemv_bpack_tB{b}`, `q8_0_grouped_gemv_bpack[_tB{b}]`,
`rms_quant_q8_1280_batched`, `rope_tail_batched_copy`, `kv_rms_rope_fp8`, `rope_inv_quant_q8`,
`f16_matvec_batched_h20`, `shared_gateup_swiglu_q8_tB{b}_r1`, `q8_0_quantize_f32_wave`, and the b
6-8 shared-expert gate/up/swiglu chain.

### 2.4 The key and the mode

`stage_cap_on` keys indirect graphs `(stage, b, topo)` (2.6). The mode comes from
`V41_MS_GRAPH_KEYS` (`legacy` | `stage_b`), read ONCE per step and carried down; `StageCap::src()`
returns the `ArgSrc` the stage body must use (the body never chooses it). Indirect mode needs the
v41 build, the arena layout (`cap_ok`), mHC split off, and the startup uniformity check (2.6).
Legacy graphs keep today's key under distinct names; when a step moves to `stage_b` the legacy
entries may be dropped after a device sync (`GraphCache::retain`).

### 2.5 Taint guard: a stage that is not indirect-able cannot be cached as (stage, b)

During an indirect capture a thread-local flag is set. A launch is VETTED only if it goes through
a converted wrapper whose L / P / R operands are all `Arg::Ctx` (a `Dev` operand whose address is
in the registered per-layer / per-lane buffer set -- every `dlw.*` tensor and every `bd.*`
buffer, registered at load -- does not count). Any unvetted launch, and any `copy_*_async` on the
captured stream, TAINTS the capture. At `StageCap::end` a tainted capture is instantiated and
launched ONCE (its baked pointers are this lane-layer's, so this call is correct), NOT inserted,
and `(stage, b, topo)` is marked legacy in a small map: later calls capture it under the legacy key
(or run uncaptured under the memory reserve). A predicate drift therefore costs performance and a
warning, never correctness.

### 2.6 Topology classes and the startup check

At load, derive per layer, from the slot table, the topology-relevant facts of every operand a
captured kernel reads: dtype, byte_len, the lengths of the scale / base / norm vectors, 16-B
alignment. Layers with identical facts share a `topo` class id (expected: one class on V4.1); the
key carries the class id -- bounded, not a layer or address key -- so a mixed-dtype GGUF still
gets indirect graphs per class instead of falling back wholesale.

### 2.7 Baked process-static operands

`sd.*` and engine scratch stay direct (one per process, engine_worker.rs:1136). The first
`stage_b` capture records a fingerprint of the `sd` buffer addresses; a later capture with a
different fingerprint clears the `stage_b` entries first (never replays foreign pointers). One host
thread submits to `ctx_dev` (documented invariant).

### 2.8 Device canary (tests and `V41_MS_CTX_CHECK=1`)

`ensure_ctx` stamps each entry with an increasing `seq`; every `_ind` kernel (thread 0 of block 0)
appends `ctx->seq` and its stage id to a per-step device log; after the step the host checks that
every indirect launch read the seq the host enqueued for its lane-layer. (A readback of the slot
proves memory contents, not what the kernels read.)

### 2.9 Graph count, memory, cost

Indirect: 8 stages x distinct b (1..8; rows 9-16 per lane fall back to legacy) x topo classes (1) =
~64 graphs per process, whatever the stream mix. Cost per lane-layer: one `ctx_store` launch
(skipped when the entry is unchanged) plus one uniform load per operand per kernel; Step 0 gates
both. The eb84ebb reserve stays.

### 2.10 Alternatives considered

- (A) `hipGraphExecKernelNodeSetParams` per node before each replay: ~40 host calls per
  lane-layer. Rejected.
- (B) Per-lane device counter advanced inside the first stage's graph: zero extra calls, keyed per
  lane, and a silent wrong-layer hazard whenever the first stage runs captured / replayed /
  uncaptured differently (reserve, `V41_MS_GRAPHS=0`, taint). Rejected in favour of 2.2.
- (C) `hipStreamWriteValue32` index into a static per-(lane, layer) table: one call, a second
  indirection, a static table to keep in sync. Measured in Step 0 as the fallback if `ctx_store`
  is slow.
- (D) Weights at a constant per-layer stride: weight re-layout plus a device layer index. Rejected.

## 3. Correctness and gates

- Per kernel (host GPU tests): every `_ind` twin vs its direct kernel bit-exact (extend
  q8_0_tb_bitexact, mhc_arena_bitexact, decode_fusion_bitexact) and timed.
- multistream_step, with `stage_b`: G5a-h; PLUS a `stage_b` vs `legacy` vs `V41_MS_GRAPHS=0` arm on
  the same schedule, logits and final state bit for bit (a consistent indirection bug passes the
  self-consistent G5 arms); counters asserting the mode was active (each (stage, b) captured once,
  replayed for >= 2 layers and >= 2 lanes; no silent startup disable); a 3-lane case; every b in
  1..8; `has_room()` forced false mid-run (direct fallback interleaved with indirect replays); a
  knob flip between steps; the canary (2.8) clean.
- Presubmit: an arm with `V41_PREFILL_PRESUBMIT=1 V41_REMOTE_SPLIT=1` if it runs with box 2, else a
  unit test that interleaves `chain(B)` between `chain(A)` and `prep(A)`.
- `V41_SLACK_PROBE` armed sets `cap_ok = false` (its alternating mode would be baked).

## 4. Rollout

Step 0 (window) -> build -> host tests + GPU gates (window) -> deploy with `V41_MS_GRAPH_KEYS=
legacy` (no change) -> per-turn A/B legacy vs stage_b (VRAM, `ms.step` p50, pick-trace exactness)
-> flip the default -> resume the two-stream A/B.

## 5. Out of scope

The single-token decode path's `(name, layer)` graphs (forward_layer.rs); fusing stages; the iGPU
routed-MoE graphs.

## 7. Review round 1 dispositions (reviewer: NEEDS REWORK)

1. Presubmit shared expert outside the chain: ACCEPTED -- `ensure_ctx` per capture (2.2).
2. Live knob mid lane-layer: ACCEPTED -- read once per step; `ensure_ctx` safe by construction.
3. Fallback decided before the body: ACCEPTED -- `StageCap::src()` + taint guard (2.5).
4. Perf unknowns: ACCEPTED -- Step 0 measurements gate the build; `ctx_store` default (2.2).
5. Pinned ring on error paths: moot (no ring).
6. Field ids by role: ACCEPTED -- `Arg::{Dev, Ctx}` + `ind_mask`; pointer-only context, rope
   scalars via `rope_dev[layer]` (2.1, 2.3).
7. `sd` baked: ACCEPTED -- fingerprint + clear (2.7).
8. Uniformity check: ACCEPTED -- derived from the slot table, v41-only, topo class in the key (2.6).
9. Indirect-load cost: ACCEPTED -- `__restrict__`, prologue resolution, per-kernel timing (2.3, 3).
10. Gate gaps: ACCEPTED (3).
11. NITs: ACCEPTED (2.9 rows 9-16 fall back; 3 slack probe).
12. Rollout default legacy: ACCEPTED (4).
