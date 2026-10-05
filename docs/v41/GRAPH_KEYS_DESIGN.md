# Arena stage graphs keyed by (stage, rows) only

Status: DESIGN rev 4.1 (2026-10-05), rev 4 review round: APPROVE WITH CHANGES (applied, section 7). Rev 3.1 was APPROVED (3 rounds) and its Step 0 code
review-APPROVED (2 rounds); Step 0 runs 1-3 were NO-GO (run 3 narrowly: 1.03 / 1.08 / 1.23% of the
step at b = 1 / 4 / 8). Owner 2026-10-05: revise first (rev 4 = section 2.11), then merge the
production build (worktree-embed-phase, hub ce52ee3c).
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
Go / no-go (rev 3.4, after runs 1-2 and three code reviews): coherence clean, every twin
bit-exact, one node per launch, no drained queue (a drained run is re-run up to 3 times), no
BLOCKS, and the per-step budget at EACH operating point vs 1% of that point's `ms.step` p50 -- two
lanes (the binding case), b = 1 / 4 / 8 per lane at 60 / 121.6 / 196.8 ms (live; MULTI_LADDER_TWO;
its +9.4 ms/row extrapolation), each charging the twin deltas measured at its own b (rev 3.3 paired
b=8's delta with b=1's step):
writes (80 = 1 per lane-layer; presubmit is off in production, 160 printed as a what-if) x the
`ctx_store` GPU + host cost + graph launches (320 1-node + 320 multi-node = 8 stages x 80
lane-layers) x the K = 8 vs all-distinct relaunch delta of that size (GPU + host) + twin launches
per lane-layer (8 gemv-like at the q_b delta, 3 mhc, 7 small at the larger of the two) x 80. Each
timed run is preceded by a short spin covering the enqueue and real work (warm-up launches), so the
clocks are up when the events start.
Deltas are paired (21 alternating pairs); the median is charged when positive and a ~96.7%
upper-bound total uses the 16th of 21 paired differences. GO = medians and upper bounds fit;
MARGINAL = only the medians fit, or GPU + host does not fit while each alone does (the live A/B's
<= 1% `ms.step` bar decides); NO-GO otherwise. A 10% relaunch penalty alone would be ~4 ms = 7% of
the step, so SERIALIZES as a ratio bar was far too loose. Otherwise the design stops and is revised.

#### Step 0 run 1 (2026-10-04 22:07 UTC, 06bae25, hub down 57 s): NO-GO

| item | result |
|---|---|
| 1 write cost (per write, amortized) | ctx_store +2.67 us GPU / +1.22 us host; H2D +3.18 / +1.04; WriteValue32 +5.68 / +1.64. Live `ms.step` at 2 rows / 2 lanes ~60 ms: 80-170 writes = 0.5-1% -- passes, not by much |
| 2 BLOCKS | no (host time flat with the spin) |
| 2 SERIALIZES | 80 back-to-back launches of ONE 8-node exec: 71.7 / 74.1 us vs distinct 67.0 / 65.5 us (+7% / +13%) -- FAIL as measured; production's pattern (a stage graph recurs every ~8 launches) not measured -> run 2 measures round-robin K = 1, 2, 4, 8, 80 |
| 3 coherence | 10k rounds: 0 stale outputs, 0 wrong seq, 0 wrong resolved pointers |
| 4 gemv twin | b=1 -1.6%, b=4 +5.7%, b=8 +8.0% (FAIL), bit-exact |
| 4 mhc twin | +1.7 / +2.1 / +1.6% (b = 1 / 4 / 8; b=4 0.30 us vs a 0.30 us bar), bit-exact |
| 5 mechanism, nodes | A then B bit-exact; 4 launches -> 4 nodes |

Cause of item 4: the twins' body loads / stores compiled to FLAT (`flat_load`, waits merged into
`s_wait_loadcnt_dscnt`; gemv tB8_ind 28 flat_load vs 0, mhc 348 vs 2): a pointer read from memory
is generic. FIXED in `ARENA_CTX_DEREF` (2.3): the slot is read as an address_space(1) pointer, so
both incoming values of the resolved pointer are casts from global; offline the twins' bodies now
use exactly the direct kernels' global_load / global_store counts (tB8 27 / 8, mhc 344 / 48); only
the canary's cold path stays FLAT. Two attempts that did NOT work (kept here to save the next
person the time): a generic -> global -> generic cast round trip (folded away as a no-op pair) and
`__builtin_assume(!is_shared / !is_private)` (gone before the backend). Run 2 re-times the twins.

#### Step 0 run 2 (2026-10-04 22:25 UTC, 0122a52, hub down 57 s): NO-GO (budget 5.54%)

| item | result |
|---|---|
| 1 write cost | ctx_store +2.79 us GPU / +1.03 us host (median of 3) |
| 2 relaunch, K = 8 vs 80 (11 pairs) | 8-node GPU -0.72 us (4/11 > 0), host -0.04; 1-node GPU -0.37, host -0.04: no penalty at production's pattern. Even K = 1 (69.75 us) = K = 80 (69.47): run 1's SERIALIZES was drift between sequential runs. Per-launch times bimodal (~64 / ~74 us) -- the 50 ms idle spins let the dGPU clock down: run 3 spins only cover the enqueue |
| 3 coherence, 5 mechanism / nodes | clean |
| 4 gemv twin (FLAT fixed) | b=1 +0.15 us, b=4 +0.29 us (+0.9%), b=8 +4.05 us (+9.8%, 11/11) |
| 4 mhc twin | +0.14 / +0.50 / +0.43 us |

Cause of b=8: the canary record AFTER the body (code after the body forces the compiler to
reconverge the body's divergent tail: +24 s_nop, +8 s_wait_alu, extra moves in the body). Offline:
without canary code the twin's body matches the direct kernel's (tB8 1040 of 1081 instructions in
order; the rest is the dereference prologue); a record BEFORE the body is worse (1251). Design
change (2.8): production twins carry NO canary code; each twin has an `_canary` variant (same
dereference macro + the canary) that the wrappers pick when the canary is on (tests,
`V41_MS_CTX_CHECK=1`).

#### Step 0 run 3 (2026-10-04 22:48 UTC, 1d9af45, hub down 53 s): NO-GO, narrowly

| item | result |
|---|---|
| 1 write cost | ctx_store +2.90 us GPU / +0.83 us host: 80 writes = 298 us / step |
| 2 relaunch, K = 8 vs 80 (21 pairs, warm-up) | 1-node GPU +0.25 us (ub +0.91, 11/21 > 0), 8-node -0.05 (ub +0.12); host ~0. No penalty |
| 3 coherence, 5 mechanism / nodes | clean |
| 4 gemv twin (no canary code) | b=1 +0.18 us (+0.7%), b=4 +0.69 (+1.8%), b=8 +1.60 (+2.4%; run 2: +4.05) |
| 4 mhc twin | +0.08 / +0.47 / +0.51 us |
| budget, medians (upper bounds) | b=1 1.03% (1.47%) MARGINAL; b=4 1.08% (1.55%); b=8 1.23% (1.47%) -> NO-GO |

The residual twin cost is the indirection itself: one more dependent scalar load round per
workgroup (kernel argument -> slot -> body), exposed once per workgroup round, so it grows with the
gemv's workgroup rounds (b=8: ~4 workgroups resident per CU, 64 per CU). Of the b=8 total, 896 us is
the 7 small twins per lane-layer charged at the q_b gemv's delta (an assumption, not a
measurement); measured items alone: b=1 0.85%, b=4 0.77%, b=8 0.77%, plus the small twins.

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
pointer (2.8) come the same way, read in the same prologue load sequence. Direct kernels are untouched (prefill, single-token decode, uncaptured paths) -- INSTRUCTION-identical, checked offline
against the pre-twin build by disassembly: calling an inlined body from the direct kernel is NOT
enough (the kernel-argument noalias becomes scoped metadata and the schedule moves; Step 0's
`mhc_fast_batched`), so a kernel whose body is not already a device function keeps its original
signature and `#include`s its body file, which the twin's device function includes too
(`mhc_fast_body.inc`). Families:
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
and output_proj quantizes on `sd`, forward_prefill.rs:5013, 7576). Conversion happens at the
DISPATCHING entry (`matvec_batched`, not `matvec_bpack`), so `Dev` and `Ctx` always take the same
arm under every knob (`V41_GEMV_BPACK=0` sends both to `q8_0_gemv_batched_warp8`).
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

`ensure_ctx` stamps each entry with an increasing `seq`; every `_ind_canary` twin (the `_ind`
twin's dereference code plus the canary; picked by the wrappers when the canary is on --
production twins carry no canary code, Step 0 run 2: a record after the body reshaped the body,
gemv b=8 +9.8%; two conditions (review of 752a3c3): the canary mode is process-static
(read once, like the other startup knobs) or part of the graph key -- a captured graph bakes its
symbol, so a live toggle would replay production twins while the check expects records -- and the
bit-exact gate arms (3) run with the canary OFF, so the production symbols are what G5 exercises,
the canary being its own arm) reads `seq` and the log pointer with scalar loads right
after its operand dereferences, before any store (one `s_load_b128`, verified in the disassembly),
along with the XOR of its resolved operand pointers; after the body, thread 0 of block 0 appends
{seq << 16 | stage id, pointer XOR} at an atomic cursor clamped at the log's end (the XOR shows
what the launch computed on, not only that the slot was fresh); the host
knows each graph's `_ind` launch count (recorded at capture, the vetted counter of 2.5) and checks
after the step that every indirect launch read the seq enqueued for its lane-layer. (A readback of the slot
proves memory contents, not what the kernels read.)

### 2.9 Graph count, memory, cost

Indirect: 8 stages x distinct b (1..8; rows 9-16 per lane fall back to legacy) x topo classes (1) =
~64 graphs per process, whatever the stream mix (rev 4: 4 multi-node stages, ~32; the single-launch
stages run direct, 2.11). Cost per lane-layer: one `ctx_store` launch
(skipped when the entry is unchanged) plus one uniform load per operand per kernel; Step 0 gates
both. The eb84ebb reserve stays.

### 2.11 Revision 4: single-launch stages run direct; the context write rides along

Step 0 run 3 priced what is left after the FLAT and canary fixes: the indirection itself, one more
dependent scalar load round per workgroup (kernel argument -> slot -> real pointer), +0.1..1.6 us
per twin launch, plus the separate `ctx_store` dispatch at 2.90 us GPU + 0.83 us host per
lane-layer (298 us / step, the largest fixed item). Rev 4 removes the indirection where it buys
nothing and the separate write.

R1. Single-launch stages run DIRECT. On the production path (`V41_MS_MHC_SPLIT=0`, `V41_MHC_FAST`,
`V41_MHC_ARENA_FUSED`, `V41_MHC_FFN_LATE` on; b <= 8) four of the eight stage graphs of a
lane-layer hold exactly one kernel: `g.mhc_pre_attn` (`mhc_fast_batched`, forward_prefill.rs
4785), `g.mhc_pre_ffn` (`mhc_fast_batched`, collapse only, 7671), `g.router_matvec`
(`matvec_batched_router`, 7821) and `g.mhc_mix_ffn_late` (`mhc_fast_batched`, 8220). A one-node
graph saves no launch: in `stage_b` mode these four run uncaptured, with their direct kernels and
direct operands (no twin, no context read, no graph). A stage takes this path only when its body
is that single launch (the fast-path conditions it already tests); any other body (mHC split, a
non-fast mHC, b > 8) keeps today's legacy capture. Twin families drop `mhc_fast_batched` and the
router's f16 matvec; graphs per process drop to 4 stages (q_chain, kv_chain, output_proj, the shared
expert) x b x topo ~ 32. Verified per stage at every b <= 8 under the hub env (review): mhc_pre_attn
= `launch_fast` (`mhc_pre_scaled_for(b)` holds for b <= 8, counters = lane rows); mhc_pre_ffn = a
collapse-only `launch_fast` (`ffn_mix_late` on); router = one `matvec_batched_sym` h20 launch
(`router_wmma` off for b <= 64, z16 only above b = 48); mix_ffn_late = one `launch_fast`. A wrong
predicate here costs performance only (direct operands are always correct). `cap_ok` keeps its
meaning (`ffn_mix_late` reads it, 7670): R1 changes how a stage runs, not `cap_ok`. Running direct
puts the stage body's HOST code back on every lane-layer (a replay skips it): the env predicates it
reads per call -- `mhc_pre_scaled_for` (4794, 4843), `mhc_narrow_fallback_for` (7669),
`V41_ROUTER_WMMA` (7850) -- become LazyLock (process-static, as KNOB_AUDIT already lists them), and
Step 0b times the whole stage body against a graph replay.

R2. The context write rides along the first stage. In `stage_b` mode `mhc_pre_attn`'s direct launch
carries the lane-layer's entry BY VALUE (272 B; the launch's kernel arguments grow ~160 -> ~440 B,
well under 4 KB) and a twin of its kernel, `mhc_fast_batched_ctx`, runs ONE extra workgroup (grid
x + 1) that copies the entry into the slot and returns before the body; every other workgroup runs
the body. The entry and the slot pointer are the LAST kernel arguments (every original argument
keeps its kernarg offset, so the body's argument loads do not move), and the branch is
`if (blockIdx.x == carrier_x) { if (blockIdx.z == 0) copy; return; }` for every z, so the extra
workgroup never reaches the body's role dispatch or the counters (`n_part` comes from the
arguments, not `gridDim`). The kernel keeps the direct kernel's signature plus two arguments and
`#include`s `mhc_fast_body.inc` after the early-return branch, so its body is the direct body
(checked offline, 2.3's disassembly gate) and no store precedes the body's loads on any path. The
Sinkhorn tail's workgroup count (`n_part`) excludes the extra workgroup. `ensure_ctx` sets its
shadow to the carried entry, so the next stages see no change and enqueue nothing. Ordering: the
same stream, so q_chain's graph reads the slot after the carrier kernel completes. One write per
lane-layer holds in every arena driver (`_pipelined`, `_lanes`, `_ready_first`): each captured
stage of a lane-layer, the shared expert included (issued inside `pre_moe_chain` unless
`V41_PREFILL_PRESUBMIT`, forward_prefill.rs 8209-8212), runs inside ONE `pre_moe_chain` call; the
drivers interleave lanes only between the chain and the route / prep / launch phases, which read no
context (under presubmit the deferred shared expert, 9678, re-writes through `ensure_ctx`). Whenever the
carrier does not run (mHC split, the non-fast mHC path, a layer whose first captured stage is
another one), `ensure_ctx` falls back to the standalone `arena_ctx_store` (2.2) -- correctness never
depends on the carrier.

R3. Measure, do not assume (Step 0b, below): every twin at its call site's real shape, the grouped
`wo_a` twin and three small twins built for it, the carrier's cost, and a direct launch vs a 1-node
graph (R1's host and GPU cost).

R4. Fallback if b = 8 stays over after R1-R3 (design sketch only, not built unless needed): make the
slot address static so its loads issue WITH the kernel-argument loads instead of after them -- one
`__device__ ArenaCtx` per module that holds twins (the carrier writes each, ~6 x 272 B from one
workgroup) and compile-time slot indices per call site (template parameters), so each slot load is
PC-relative and independent of the kernel arguments. That removes the dependent round, the residual
cost of R1-R3. Prerequisite: the carrier lives in the mhc_fast module, so the other modules' slot
addresses come from `hipModuleGetGlobal` at load and reach the carrier as process-static arguments;
compile-time slot indices multiply symbols per call site (~14, acceptable).

Budget with R1-R2 on run 3's numbers, every remaining twin still charged at the q_b gemv's delta
(conservative): writes ~0 (the carrier adds one workgroup and 272 B of kernel arguments; host ~0.2
us), relaunch 320 multi-node launches at -0.05 us (ub 0.12), twins 80 x 14 (8 gemv-like + ~6
small): b = 1 0.36% (ub 0.45%), b = 4 0.64% (ub 0.91%), b = 8 0.91% (ub 1.03%). R3's real shapes
decide b = 8.

#### Step 0b (one hub-only window, tests/bench_graph_keys.rs extended)

1. Twins at the real shapes, b = 1 / 4 / 8, 21 paired runs each, bit-exact: the bpack gemv twin at
   q_a (1280 x 5120), q_b (32768 x 1280), the kv projection, wo_b (5120 x 8192) and the shared
   down (5120 x 2304) shapes; the grouped `wo_a` twin (8 x 1024 x 4096); the shared gate/up twin
   (`shared_gateup_swiglu_q8_tB{b}_r1`, 2 x 2304 x 5120: gemv-sized, charged as a gemv when not
   measured); `rms_quant_q8_1280_batched`, `rope_tail_batched_copy` and `kv_rms_rope_fp8` twins
   (built for this; the remaining small families are charged at the largest small-twin delta
   measured). Offline before the window (done for tB1..8 and mhc, 2026-10-05): every production
   twin has no FLAT op and its body matches the direct kernel's plus the dereference prologue
   (84-98% of instructions in order; the rest prologue and scheduling).
2. The carrier: `mhc_fast_batched_ctx` (entry by value, extra workgroup) vs the direct
   `mhc_fast_batched`, paired (GPU and host), bit-exact, and the slot holding the entry afterwards
   (a following `_ind` launch reads it: coherence with the canary variant).
3. R1: each single-launch stage's WHOLE body run direct (host code, cached predicates, wrapper,
   `events.stage`) vs a replay of its captured 1-node graph, paired, host and GPU per stage.
4. Sections 2 (relaunch, K = 8 vs 80, multi-node graphs only), 3 and 5 as in run 3.

Budget (2.0) recomputed: no standalone writes on the carrier path (the carrier's measured delta per
lane-layer instead); 320 graph launches (4 multi-node stages x 80) at the multi-node relaunch delta;
R1's direct-vs-graph delta x 320; per lane-layer the measured twin deltas of the q_chain, kv_chain,
output_proj and shared-expert kernels at their own shapes, unmeasured small families at the largest
measured small-twin delta. Same points (b = 1 / 4 / 8 at 60 / 121.6 / 196.8 ms), same verdicts.

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

- Offline, per build: every direct kernel instruction-identical to the pre-twin build
  (disassembly diff, trailing padding excluded); twins' VGPR / SGPR / spill / scratch.
- Per kernel (host GPU tests): every `_ind` twin vs its direct kernel bit-exact (extend
  q8_0_tb_bitexact, mhc_arena_bitexact, decode_fusion_bitexact) and timed.
- multistream_step, with `stage_b`: G5a-h; PLUS a `stage_b` vs `legacy` vs `V41_MS_GRAPHS=0` arm on
  the same schedule, logits and final state bit for bit (a consistent indirection bug passes the
  self-consistent G5 arms); counters asserting the mode was active (each (stage, b) captured once,
  replayed for >= 2 layers and >= 2 lanes; no silent startup disable); a 3-lane case; every b in
  1..8; `has_room()` forced false mid-run (direct fallback interleaved with indirect replays); a
  knob flip between steps; the canary (2.8) clean.
- Rev 4 counters (review of rev 4): per stage, direct / replayed / captured `stage_b` / captured
  legacy; writes, carrier vs standalone. At b <= 8 the `stage_b` arm asserts direct = 4 x
  lane-layers, `stage_b` captures <= 4 stages x distinct b, carrier writes = lane-layers,
  standalone writes = 0 (= lane-layers under presubmit). An arm with the carrier forced off (test
  hook) proves the standalone fallback bit-exact; the canary arm checks the carrier stamps `seq`
  exactly as `ensure_ctx` does.
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

Code review round 1, Step 0 code 4ba68da (reviewer: APPROVE WITH CHANGES):

1. `CanaryLog` header write length (blocker): FIXED (`slice_view_mut(0, 1)`).
2. Section 1 could overflow the AQL ring: FIXED -- N = 400 (1200 packets), a drained queue fails
   the item, max per-iteration host time logged.
3. `time_ab` host-bound: FIXED -- every round queued behind a 10 ms spin, drained rounds fail,
   median [min..max] reported.
4. Canary witness: FIXED -- uniform scalar loads after the dereferences (`s_load_b128` at +0x100,
   disassembly), pointer XOR logged, record written after the body (2.8).
5. Coherence on one cache line: FIXED -- slots 0 / 9 / 18 / 27.
6. Rescheduled direct `mhc_fast_batched`: FIXED -- `static` alone did not help; the direct kernel
   `#include`s `mhc_fast_body.inc` with its original signature: instruction-identical to the pre-twin
   build over all 3189 instructions (only trailing alignment padding differs); gemv tB1 / 4 / 8
   identical. GK_OLD_MHC dropped (2.3, 3).
7. Convert at the dispatching function: ADOPTED for the build (2.5).
8. ABI asserts: ADDED (HIP static_asserts, Rust const assert); `arena_ctx_store` scratch 0; block 32.
9. One-node graph in section 2: FIXED -- an 8-node graph.
10. `out_i` poisoned per b: FIXED.
11. Debug-assert no indirect operand of an absent mhc half: ADDED.

Code review round 2, 936363f (reviewer: APPROVE):

1. `ArenaCanaryLog::rec[2]` written past its bound: FIXED -- flexible array member `rec[]`.
2. A no-go exited 0: FIXED -- the bench's last line is `STEP0: GO` / `STEP0: NO-GO (items)`; the
   window script reports and logs it.

Review of a7dc914 (reviewer: APPROVE WITH CHANGES):

1. The ratio bars were too loose for the 1% budget: FIXED -- one per-step budget sum (2.0).
2. Run 2 could not resolve sub-1% deltas: FIXED -- K = 8 vs K = 80 and the twins as 11 alternating
   pairs with a sign test; section 1 median of 3.
3. AS1-typed slot load: sound (the slot only ever holds host-written device addresses).

Review of 0122a52 (reviewer: APPROVE WITH CHANGES) and run 2:

1. 160 writes: FIXED -- 80 (presubmit off in production), 160 printed as a what-if.
2. The ~7 small twins per lane-layer uncharged: FIXED -- charged at the larger of the gemv / mhc
   deltas.
3. Sign-test charging biased toward GO: FIXED -- the median is charged when positive; an
   upper-bound total (order statistic); GO / MARGINAL / NO-GO.
4. Relaunch by graph size: FIXED -- 320 launches at the 1-node delta, 320 at the 8-node one.
5. GPU-only / host-only totals: ADDED.
6. Run 2 (NO-GO 5.54%): the gemv b=8 twin's +4 us traced to the canary record after the body:
   production twins now carry no canary code (`_canary` variants, 2.8); spins shortened to the
   enqueue (the 50 ms idle spins made per-launch times bimodal); 21 pairs.

Review of 752a3c3 (reviewer: APPROVE WITH CHANGES):

1. The budget paired b=8's twin delta with b=1's 60 ms step (overstating run 2 about threefold):
   FIXED -- per operating point (b = 1 / 4 / 8 at 60 / 121.6 / 196.8 ms), every point must fit.
2. Canary off the production twins: ACCEPTED with two conditions for the build (2.8): canary mode
   process-static or in the key; gate arms run canary-off, the canary its own arm.
3. One OS stall would hard-fail the window: FIXED -- drained runs / pairs re-run up to 3 times.
4. Clock state: FIXED -- warm-up with real work before every timed run; per-arm SPREAD flag.

Design review round 4, rev 4 (reviewer: APPROVE WITH CHANGES):

1. Carrier entry first would shift the body's kernarg offsets: ADOPTED -- entry and slot pointer
   last; the carrier branch returns for every z (2.11 R2).
2. Step 0b item 3 measured the wrong host path: ADOPTED -- the env predicates the four stage bodies
   read per call become LazyLock; Step 0b times the whole stage body direct vs a graph replay.
3. Shared gate/up twin is gemv-sized: ADOPTED -- measured or charged as a gemv.
4. Gate counters for R1 / R2 and a carrier-off arm: ADOPTED (3).
5. R4 prerequisite: recorded.
6. All-B disassembly check (carried over): DONE offline for tB1..8 and mhc (2.11 Step 0b item 1).
