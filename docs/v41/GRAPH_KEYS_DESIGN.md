# Arena stage graphs keyed by (stage, rows) only

Status: DESIGN rev 3.1, 2026-10-04 -- APPROVED (review round 1 NEEDS REWORK, round 2 APPROVE
WITH CHANGES, round 3 APPROVE; dispositions in section 7). Next: Step 0 (2.0) on the GPU.
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

A dGPU bench (tests/bench_graph_keys.rs; hub down for a few minutes). Repo evidence already
favours the by-value write: 4eb4b69 measured tiny `hipMemcpyAsync` uploads at ~4.1 us of device
timeline each (pinned memory no faster), and the by-value `attn_meta_fill` (attn_meta.rs:18-81)
that replaced 3-5 of them saved 13-20 us per lane-layer (tests/bench_decode_latency_ab.rs).
1. The context write, AMORTIZED: N x [kernel; write] vs N x [kernel] inside one event pair, the
   queue pre-filled behind a long `slack_probe_spin` (events cost ~10 us themselves): none / H2D
   from pinned / `ctx_store` by value / `hipStreamWriteValue32`; host enqueue time per write.
2. 80 `hipGraphLaunch` of ONE executable vs 80 distinct executables, behind a ~50 ms spin.
   BLOCKS = the host enqueue time of the 80 launches grows with the spin; SERIALIZES = GPU time per
   launch > 10% above the distinct-executable case.
3. COHERENCE: `ctx_store` then a graph whose first node reads the slot (scalar loads), alternating
   entries, 10k rounds, every read checked.
4. `_ind` twins vs direct: `q8_0_gemv_bpack_tB{b}` at b = 1, 4, 8 (tB8 = the VGPR worst case) and
   `mhc_fast_batched` (13 pointer operands, short per-WG work: the prologue worst case), each
   bit-exact and timed, with the canary log compiled in (null log pointer), VGPR/SGPR counts
   compared.
5. The mechanism: one captured `_ind` graph replayed after storing entry A, then B, reproduces the
   direct kernel on A, then on B; and a capture of N launches has exactly N nodes (2.5's vetted
   count assumes one kernel node per launch -- if ROCm added nodes, every capture would taint and
   the design would silently run all-legacy).
Go / no-go: per step, the added GPU time plus the host critical-path time <= 1% of `ms.step` p50
(~80-160 writes per step at <= ~5 us GPU each); neither BLOCKS nor SERIALIZES; coherence clean;
each twin bit-exact and within max(2%, 0.3 us) of its direct kernel. Otherwise the design stops
and is revised.

### 2.1 Operands: `Arg` and a pointer-only context

Wrappers take operands as `Arg::Dev(&buf)` (direct, today) or `Arg::Ctx(slot, &buf)` (read the
pointer from context slot `slot` at run time). The real buffer is passed in both cases, so the
wrappers' byte_len / len checks stay. The context is POINTER-ONLY: `struct ArenaCtx { const void*
p[N_SLOTS]; u64 seq; }` (~24 slots + a sequence number for the canary, 2.8). The per-layer rope
values live in a static device array `rope_dev[layer]` built at load holding EXACTLY the kernel
arguments the host derives today, `RopeTail::device_args(params, N_ROT)` (rope.rs:269-282:
theta_scale, freq_scale, ext_factor, mscale_eff, corr_low, corr_high) -- never `RopeParams`' raw
floats (recomputing powf / ln / corr_dims on the device is not bit-exact). One entry per layer
suffices because all three rope kernels use n_rot 64 (asserted at load). The context carries a
POINTER to the layer's entry.

### 2.2 The context write is tied to the capture (`ensure_ctx`)

In indirect mode `stage_cap_on` receives the lane-layer's entry (a host `ArenaCtx` built once per
lane-layer from `dlw`, `bd` and `rope_dev`, carried with the lane's chain state so `pre_moe_prep`
has it too) and calls `ensure_ctx(entry)` BEFORE it replays or begins a capture: if the entry
differs from a host shadow of the last enqueued entry (whole-entry compare, residual pointer
included), it enqueues a write of the entry into the single device slot `ctx_dev` and, once the enqueue
returned Ok, updates the shadow (compare the payload, not `seq`). Enforced, not described: the
stream must be `de.compute` (asserted: two streams would race on one slot), no capture may be open
on it (asserted; 2.5's node count would also catch a write captured as a node), and one host thread
submits (debug-asserted thread id). Stream order makes the stage read that entry. This holds for
every capture site and driver, including the presubmit shared expert, at ~1 write per lane-layer (2
with presubmit). The shadow resets at step start and in `StageCap::drop` on an abnormal end.

The write is a 1-WG `ctx_store` kernel taking the entry by value (264 B, well under the kernel
argument limit) -- the production precedent is `attn_meta_fill` (attn_meta.rs:18-81, a 256-B
`#[repr(C)]` struct by value, stream-ordered): same HW queue as the stages, nothing to keep alive.

### 2.3 Indirect kernel twins

Each default-path kernel family with an L / P / R operand gets an `_ind` twin generated by a macro
around its existing `__device__ __forceinline__` body: the twin takes the usual operand pointers and a
`uint32_t ind_mask`; for an indirect operand the wrapper passes, in that operand's own pointer
argument, the address of its context slot (`ctx_dev + 8 x slot`, process-static, safe to bake),
and the prologue dereferences it, before any store: `p_i = (ind_mask >> i) & 1 ? *(T* const*)p_i :
p_i` (thread-uniform, scalar loads). No slot table or slot ids in HIP. The canary's `seq` and log
pointer (2.8) come the same way, read in the same prologue load sequence. Direct kernels are untouched (prefill, single-token decode, uncaptured paths). Families:
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

A WHITELIST, enforced at one choke point. During an indirect capture:
- every `Arg::Dev` pointer a converted wrapper passes must lie INSIDE a registered process-static
  range (`sd.*` and engine scratch -- the set 2.7 fingerprints; containment check, so interior
  slices count); a `Dev` operand outside it taints the capture (a wrapper cannot know an operand's
  role, and a blacklist of `dlw` / `bd` addresses would miss buffers allocated later and new
  fields);
- converted wrappers set a thread-local token consumed by `Function::launch_raw` (module.rs:105),
  which counts VETTED launches;
- after `end_capture`, the graph's node count (`Graph::nodes()`, graph.rs:60) must equal the
  vetted count: an unconverted wrapper, a memcpy, a memset (`fill_zero_async`), a
  `write_value32`, a peer copy -- any node not vetted -- taints, without listing APIs.
Converted wrappers, two rules. (a) Handed `Arg::Ctx` on an arm with no `_ind` twin (b > 8, the
runtime `q8_0_gemv_bpack_warp8` under `V41_GEMV_TB=0`, `q8_0_quantize_f32` under
`V41_Q8_QUANT_WAVE=0`), the wrapper launches the DIRECT kernel on the real buffer the `Arg`
carries, unvetted: the capture taints and runs once, correctly (never `Err`, which would fail the
step; never the slot address to a direct kernel, which would compute on garbage). (b) A converted
wrapper whose operands are all whitelisted `Dev` vets and launches its direct kernel (the q_chain
and output_proj quantizes on `sd`, forward_prefill.rs:5013, 7576).
A tainted capture is instantiated and launched ONCE (its `_ind` nodes read the entry written
before `begin_capture`, its direct nodes baked this lane-layer's pointers: this call is correct),
pushed onto a per-step RETIRE list (dropping a `GraphExec` destroys it at once, graph.rs:164-168,
while its launch may still be queued; the list is dropped after the step's final synchronize), NOT
inserted, and `(stage, b, topo)` is marked legacy: later calls capture it under the legacy key (or
run uncaptured under the memory reserve). A predicate drift costs performance and a warning, never
correctness.

### 2.6 Topology classes and the startup check

At load, derive per layer the topology-relevant facts over the UNION of every captured stage's
operands plus every `dlw` fact a host predicate reads inside a capture (predicates cross stages:
`kv_chain` branches on `attn_q_a.dtype`, forward_prefill.rs:5225, and pairs its quantize skip
with `q_chain`'s, 5010; today every such fact is an operand: 4995, 5010, 5026, 5225, 2824,
2855-2857, 2902): dtype, byte_len, the lengths of the scale / base / norm vectors, 16-B alignment.
Lane facts (alignment of the `bd` pointers) are not in the class: hipMalloc alignment plus 2.5's
guard. Layers with identical facts share a `topo` class id (expected: one class on V4.1); the
key carries the class id -- bounded, not a layer or address key -- so a mixed-dtype GGUF still
gets indirect graphs per class instead of falling back wholesale.

### 2.7 Baked process-static operands

`sd.*` and engine scratch stay direct (one per process, engine_worker.rs:1136). The first
capture records a fingerprint of the `sd` buffer addresses; `stage_cap_on` compares it BEFORE
`graphs.get` (or once at step entry), and a mismatch clears both the `stage_b` and the legacy
entries (legacy graphs bake `sd` too) after a device sync -- so no replay ever uses foreign
pointers. One host
thread submits to `ctx_dev` (documented invariant).

### 2.8 Device canary (tests and `V41_MS_CTX_CHECK=1`)

`ensure_ctx` stamps each entry with an increasing `seq`; every `_ind` kernel reads `seq` in its
prologue load sequence (the same path as its operands) and, when the log pointer is non-null (a
uniform branch COMPILED INTO the production twins, so the canary tests the production binary),
thread 0 of block 0 appends (seq, stage id) at an atomic cursor clamped at the log's end; the host
knows each graph's `_ind` launch count (recorded at capture, the vetted counter of 2.5) and checks
after the step that every indirect launch read the seq enqueued for its lane-layer. (A readback of the slot
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
- The taint path, exercised: `V41_MHC_FAST=0` sends `mhc_pre_attn` to the arm with the per-lane
  memcpy (~4963): expect the warning, legacy marking, a bit-exact result and a non-empty retire
  list.
- The fingerprint mismatch: a second `sd` in one process clears the cache before any replay.
- Rollout bar (section 4's A/B): `ms.step` p50 regression <= 1%, graphs ~64.

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

Review round 2 (reviewer: APPROVE WITH CHANGES):

1. Taint guard fails open: ACCEPTED -- whitelist of process-static ranges, vetted count at
   `launch_raw`, node count == vetted count (2.5).
2. Tainted executable dropped while queued: ACCEPTED -- per-step retire list (2.5).
3. Fingerprint only at capture: ACCEPTED -- checked before `graphs.get`, clears both caches (2.7).
4. `rope_dev` contents: ACCEPTED -- `device_args` outputs, n_rot asserted (2.1).
5. Shadow details: ACCEPTED (2.2).
6. Slot ids: ACCEPTED -- operand pointer = slot address, mask = dereference (2.3).
7. Topology over the union of stages: ACCEPTED (2.6).
8. Canary details: ACCEPTED (2.8).
9. Step 0 measurements / thresholds: ACCEPTED (2.0).
10. Gates for the new paths: ACCEPTED (3).
11. `attn_meta_fill` precedent: cited (2.2).

Review round 3 (reviewer: APPROVE):

1. Fallback rule for converted wrappers: ADOPTED -- 2.5 rules (a) / (b).
2. 1:1 launch-to-node mapping: ADOPTED -- Step 0 item 5 asserts it.
3. `matvec_bpack_ind` ignored `V41_GEMV_TB`: FIXED in code -- routes through rule (a).
4. Canary not yet in the twin: FIXED in code -- `ARENA_CTX_CANARY` in every twin (one extra
   `const ArenaCtx* canary` argument, null in production; the log pointer lives in the entry).
