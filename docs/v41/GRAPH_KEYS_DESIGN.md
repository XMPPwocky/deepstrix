# Arena stage graphs keyed by (stage, rows) only

Status: DESIGN rev 1, 2026-10-04, for review. Branch `worktree-ms-dspark2` (on eb84ebb).

## 0. Problem

The arena decode step captures its per-layer dGPU stages as HIP graphs (f71a9d5: the 1-8-row
chain was host-launch-bound, ~40 kernels x 20-40 us per lane-layer; a graph replays each stage as
one launch). The cache key is `layer | b << 8 | (bd.residual address >> 8) << 24`
(forward_prefill.rs ~1921): one graph per (stage, LAYER, rows, LANE BUFFER ADDRESS), never freed.
Default knobs hold ~2240 graphs (8 stages x 40 layers x 7 (lane, rows) pairs), ~2560 with a lone
DSpark stream. Every new (rows, lane) shape costs ~320 more graph executables of device memory for
good. Two speculating streams brought new shapes on 2026-10-04 and took the dGPU to 99.5% VRAM
(the layer-major prefill window halved 8 times; a capture whose instantiate runs out of memory
fails its step). eb84ebb added a memory reserve below which new shapes run uncaptured -- a safety
net, not a fix: uncaptured stages are the host-launch-bound path the graphs exist to avoid.

Owner, 2026-10-04: "row count b? sure. keying off that is fine. keying off a buffer address seems
obviously bad, as does keying off layer." Goal: graphs keyed by **(stage, b)** -- one graph per
stage and row count, replayed for every layer and every lane -- with replay cost unchanged
(about one launch per stage) and the numerics bit-identical.

## 1. Facts (code map 2026-10-04, eb84ebb)

- Stages (12 names, 8 on the default path; mHC split is off): `g.mhc_pre_attn`, `g.q_chain`,
  `g.kv_chain`, `g.output_proj`, `g.mhc_pre_ffn`, `g.router_matvec`, `g.shared_expert`,
  `g.mhc_mix_ffn_late`; split-mode `g.mhc_mixes_attn` / `g.mhc_collapse_attn` / `g.mhc_mixes_ffn` /
  `g.mhc_collapse_ffn` (de.hc + compute). All captured stages of one lane-layer are enqueued inside
  ONE `pre_moe_chain` call (`chain!(i, l)`), contiguously on `de.compute` (except split mode).
- **Topology does not depend on the layer.** No host branch inside a capture reads the layer, the
  compress ratio, KV/index-source, CED, hash router (N_HASH_LAYERS = 0) or Engram. The only
  per-layer branch inputs are weight dtypes, and V4.1 requantises every dense projection to Q8_0 at
  load (hf_v41.rs:18). Grids depend on `b` and constants only.
- **What differs per layer:** the `dlw.*` weight pointers (each tensor its own allocation: no
  constant stride), and the rope scalars of three kernels (two classes: layers 0-1, 2-39).
- **What differs per lane:** fixed `bd.*` buffers (split, hc_pre_carry, after_attn_hc,
  ffn_input_norm, ffn_shared, pos_per_b) and `bd.residual`, which swaps every layer and is read
  only by the mHC attention stages. Shared `sd.*` buffers are one per process. No captured stage
  reads `RowTablesDev`.
- Launch plumbing: every launch goes through `launch_kernel!` -> `hipModuleLaunchKernel`; capture
  copies the argument values into the graph nodes. Kernels take raw typed pointers.
  `hipGraphExecKernelNodeSetParams` is bound and unused.

Default-path kernels with a per-layer or per-lane pointer or scalar (the ones this design touches):

| stage | kernel family | per-layer (L) / per-lane (P, R) inputs |
|---|---|---|
| mhc_pre_attn | `mhc_fast_batched` | L: hc_attn_fn, hc_attn_scale/base, attn_norm; P: split, hc_pre_carry; R: residual |
| q_chain | `q8_0_gemv_bpack_tB{b}` x2, `rms_quant_q8_1280_batched`, `rope_tail_batched_copy` | L: attn_q_a, q_a_norm, attn_q_b, rope scalars; P: pos_per_b |
| kv_chain | `q8_0_gemv_bpack_tB{b}`, `kv_rms_rope_fp8` | L: attn_kv, kv_a_norm, rope scalars; P: pos_per_b |
| output_proj | `rope_inv_quant_q8`, `q8_0_grouped_gemv_bpack[_tB{b}]`, `q8_0_gemv_bpack_tB{b}` | L: rope scalars, attn_output_a, attn_output_b; P: pos_per_b |
| mhc_pre_ffn | `mhc_fast_batched` | L: ffn_norm; P: after_attn_hc, ffn_input_norm, hc_pre_carry |
| router_matvec | `f16_matvec_batched_h20` | L: ffn_gate_inp; P: ffn_input_norm |
| shared_expert | `q8_0_quantize_f32_wave`, `shared_gateup_swiglu_q8_tB{b}_r1` (b 1-5) or gate/up gemv + swiglu (b 6-8), `q8_0_gemv_bpack_tB{b}` | L: gate, up, down; P: ffn_input_norm, ffn_shared |
| mhc_mix_ffn_late | `mhc_fast_batched` | L: hc_ffn_fn, scale/base; P: split, after_attn_hc, hc_pre_carry |

## 2. Design

### 2.1 The context entry

A flat device struct, one definition shared by Rust and HIP (`#[repr(C)]` / a header):

```c
struct ArenaCtx {
    const void* p[N_CTX_PTR];   // every L / P / R pointer a captured kernel reads, by field id
    float       f[N_CTX_F];     // the rope scalars of the layer (and any other per-layer scalar)
};
```

Field ids are an enum shared by both sides (e.g. `CTX_ATTN_Q_A`, `CTX_RESIDUAL`, `CTX_POS`,
`CTX_ROPE_THETA_SCALE`, ...): ~24 pointers + 6 floats, ~220 bytes.

ONE `ArenaCtx` slot lives in device memory at a fixed address (`ctx_dev`), allocated with the
engine. Before the first captured stage of each lane-layer -- at the top of `pre_moe_chain`, so
every driver gets it -- the host fills the entry for (lane, layer) with exactly the values the
uncaptured path would pass (that layer's `dlw.*` pointers and rope scalars, that lane's `bd.*`
pointers and its CURRENT `bd.residual`), into a pinned host ring slot owned by (lane, layer) for
this step, and enqueues ONE `hipMemcpyAsync` H2D of the entry into `ctx_dev` on `de.compute`.
Stream order makes every stage of that chain read that entry, and the next lane's (or layer's)
write land after them. The ring slot is not reused before the step's final sync.

### 2.2 Indirect kernel variants

Each kernel family in the table gets an `_ind` entry point: its body moves into a `__device__
__forceinline__` implementation taking plain pointers; the existing `__global__` keeps its
signature (prefill, single-token decode and every uncaptured path are untouched); the `_ind`
`__global__` takes `const ArenaCtx* ctx` plus, for each indirect operand, a field id (an int
baked into the graph, identical for every layer and lane), reads `ctx->p[field]` /
`ctx->f[field]` (thread-uniform loads, one cache line) and calls the same implementation. Shared
`sd.*` operands stay direct arguments. Same instructions on the same data: bit-identical.

Rust side: each launch wrapper involved (mhc_arena.rs, q8_0.rs, rms_norm.rs, rope.rs, f16.rs,
shared_fused, dispatch.rs forwarders) gains an indirect form selected by an `ArgSrc` the stage
passes down: `Direct` (today) or `Ctx(&ctx_dev)` with the field ids implied by the operand role.

### 2.3 The key

`stage_cap_on` keys indirect-mode graphs `(stage, b)` (no layer, no address). Replaying stage S at
rows b for layer l of lane i is correct because every layer- and lane-specific operand is read
from `ctx_dev` at run time, and `ctx_dev` holds (i, l)'s entry at that point of the stream.
Legacy-mode graphs keep today's key under distinct stage names, so both coexist.

### 2.4 When indirect mode applies

All of: `V41_MS_GRAPH_KEYS=stage_b` (new, live knob, default `stage_b`; `legacy` = today), the
arena layout (`cap_ok`), mHC split off (split stages run on `de.hc`; a single context slot is
ordered on `de.compute` only), and a STARTUP CHECK that the topology really is layer-uniform:
every layer's tensors read by captured kernels have the same dtype and shape (else indirect mode is
disabled with a warning). Any stage or variant without an `_ind` kernel (the unfused / split mHC
arms, the large-b WMMA arms) keeps legacy capture -- or runs uncaptured under the memory reserve.

### 2.5 Graph count and memory

Indirect mode: 8 stages x the distinct b values seen (1..8; up to 16 rows per lane allowed) =
~64-128 graphs for the process, whatever the stream mix. Legacy graphs exist only for configs that
fall back. The eb84ebb memory reserve stays as the safety net.

### 2.6 Cost

Per lane-layer: one H2D copy of ~220 bytes (one API call) added to the ~8 graph launches; per
kernel: a few uniform loads from one cache line. f71a9d5 measured 20-40 us per host launch on this
path, so the write costs up to ~80 x 30 us = ~2.4 ms per two-lane step (~2-3%) if an H2D call costs
what a launch does; measured by an A/B of the live knob (`legacy` vs `stage_b`) per turn. If it
shows, 2.7 (B) removes it.

### 2.7 Alternatives considered

- (A) Patch the per-layer arguments into one executable per (stage, b) with
  `hipGraphExecKernelNodeSetParams` before each replay: one host call per NODE per lane-layer
  (~40 per lane-layer), the launch overhead the graphs exist to remove. Rejected.
- (B) Zero extra host calls: upload a table `ctx[lane][layer]` once per step; each lane has a
  device counter, reset once per step, and the FIRST captured stage of every layer begins with an
  `advance` node (`counter[lane] += 1`) inside its graph; indirect kernels read
  `ctx[lane][counter[lane]]`. Costs: graphs keyed (stage, b, lane) (the counter address is per lane:
  bounded, 2-3 lanes), and a correctness hazard -- exactly one advance per lane-layer must precede
  every indirect kernel of the layer, whether the first stage replays, captures or runs uncaptured
  (memory reserve, `V41_MS_GRAPHS=0`), else a whole layer silently reads the wrong weights. Kept as
  the follow-up if 2.6's write is measurable.
- (C) A static per-step table plus a per-lane-layer 4-byte index write (`hipMemsetD32Async`): the
  same call count as the chosen design with a second indirection. No gain.
- (D) Weights at a constant per-layer stride (base + layer x stride): needs a weight re-layout and
  still a device-side layer index. Rejected.

## 3. Correctness

- Bit-exact gates: multistream_step G5a-h with `stage_b` (every arm), and a new arm comparing a
  `stage_b` run against a `legacy` run of the same schedule bit for bit (logits and final KV
  state).
- Context integrity: a debug check (`V41_MS_CTX_CHECK=1`) that reads the slot back after each write
  in tests and compares with the direct arguments; a host unit test that the field-id table covers
  every operand of every `_ind` kernel (Rust and HIP agree on ids and struct size).
- Ordering: all writes and indirect stages on `de.compute`; split mode excluded; the ready-first
  driver enqueues `chain!(i, l)` atomically, so no other lane's write can land inside a chain.

## 4. Out of scope

- The single-token decode path's `(name, layer)` graphs (forward_layer.rs): unchanged.
- Fusing stages into fewer graphs, device-side graph launch.
- The iGPU routed-MoE graphs (`igpu_graphs`).
