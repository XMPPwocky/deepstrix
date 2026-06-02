# ejpir/ds4-hip — how the Strix Halo fork hits ~200 tok/s prefill, and what we're missing

**Date:** 2026-06-02
**Source:** [antirez/ds4 issue #16](https://github.com/antirez/ds4/issues/16), fork
[`ejpir/ds4-hip` branch `rocm-upstream-shape-cyberneurova`](https://github.com/ejpir/ds4-hip/tree/rocm-upstream-shape-cyberneurova)
(commit `c5e3ae9` referenced in the thread). Clone analyzed at `/tmp/ds4-hip`.

---

## TL;DR

- The "200+ tok/s on just Strix Halo (no eGPU)" claim is **PREFILL**, measured at
  **B=2048-token chunks** with the `DS4_SERVER_FAST_FULL=1` preset on. It is **not**
  decode. Their **decode is only ~11–13 tok/s — our decode (16–26 tok/s) beats them.**
  Our "30 tok/s sustained decode is hardware-impossible" floor is untouched by any of this.
- The fork is a ChatGPT-authored hipify of upstream `ds4_cuda.cu` plus a `rocm/*.cuh`
  kernel set. The 200 number was initially a **buggy F16 state emitting Chinese**; after
  fixes it settled at ~197–207 and was reproduced by 2 third parties (ttimbrook,
  accemlcc once configured correctly). Real, but config-fragile. The headline numbers
  appear **only** with `FAST_FULL`, which flips on a stack of opt-in flags — bare kernel
  defaults are much slower (integer dp4a, no hipBLAS).
- **The structural gap in our code:** our prefill MoE (`iq2_xxs_pair_matvec_par.hip`,
  `_by_expert` / `_chunked`) reuses weights across batch members **via L2 cache only** —
  it **re-runs the IQ2 grid+sign unpack (VALU-heavy) for every member**. Since our own PMC
  says iq2 is **VALU-bound**, this is the wasted work, and it explains why B=64→512 only
  bought +9.4%. ejpir unpacks each weight octet **once and reuses it across 8 batch columns
  in registers** (`block8`).

---

## How they hit ~200 t/s prefill (the FAST_FULL recipe)

`DS4_SERVER_FAST_FULL=1` is the load-bearing flag. It expands to the env list below and
sets `perflevel=high`. Per-subsystem:

### 1. Big prefill batch — B=2048
`DS4_METAL_PREFILL_CHUNK=2048` (server uses 4096). 2048 is chosen because it's divisible by
both compression ratios (4 and 128), so each chunk leaves compressor state on clean row
boundaries. We use B_MAX=512. Prefill throughput scales with batch **only if the kernels
reuse weights across the batch dimension** — see the gap section.

### 2. Dense Q8_0 projections → FP16 dequant-once cache + hipBLAS/hipBLASLt
Files: `rocm/ds4_rocm_q8.cuh`, `ds4_rocm_matmul.cuh`, `ds4_rocm_hipblaslt.cuh`,
`ds4_rocm_runtime.cuh`.

- Q8_0 weights are dequantized to FP16 **once** and cached (`cuda_q8_f16_ptr`, keyed on
  weight byte-offset in an `unordered_map`; first touch `cudaMalloc`s + runs
  `dequant_q8_0_to_f16_kernel`, subsequent calls are free).
- The GEMM is `cublasGemmEx` (→ hipBLAS under HIP): **f16 in / f32 accumulate / f16 or f32
  out**, `CUBLAS_TF32_TENSOR_OP_MATH`. For the MoE dense/shared path they use explicit
  **hipBLASLt** with **cached plans** (keyed `(out,n_tok,in)`), `hipblasLtMatmulAlgoGetHeuristic`
  picking `heur[0]`, **zero workspace**.
- Eligible tensors: `attn_output_a/b`, `attn_q_b`, shared-expert `ffn_{gate,up,down}_shexp`,
  and exact shapes (4096,2048),(2048,4096),(4096,1024),(4096,512),(1024,32768).
- Plans are **warmed at the 2048-token shape** at load time so the first real prefill
  doesn't pay heuristic selection.
- Budget: `DS4_CUDA_Q8_F16_CACHE_MB`, `_RESERVE_MB` (512 MiB on ≥112 GiB cards); on OOM it
  frees the whole cache and silently reverts to Q8 kernels.

### 3. Q2_K routed experts → FP16 WMMA "hotlist" (rocWMMA)
Files: `rocm/ds4_rocm_moe.cuh`, `ds4_rocm_moe_launch.cuh`.

- **Important:** this is **dequant-to-f16 WMMA**, not integer WMMA. `MOE_WMMA_HOT=1`
  (FAST_FULL) dequants each *hot* expert's Q2_K weight to an FP16 device cache
  (`moe_q2K_dequant_expert_f16_kernel` → `g_moe_dense_hot_cache`), then runs
  `moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel` / `moe_down_q2K_hotlist_wmma_n2_kernel`
  using `rocwmma::fragment` 16×16×16 f16 MMA.
- "Hotlist" = only experts with token-count ≥ `MOE_WMMA_GATE_HOT` (default 8) go WMMA;
  the cold remainder runs the scalar path. `MOE_WMMA_MTILES=16` (FAST_FULL).
- Gate+up are **N-doubled** (each block computes two N-tiles for gate and up together) and
  **fused**; **SwiGLU and the router-weight multiply are folded into the epilogue**
  (writes only `mid`).
- **This directly contradicts our `iq2-bottleneck` "don't build iq2 WMMA" rule — but only
  the *integer* (WMMA-IU8) variant was killed. Dequant-to-f16 WMMA at large B is untested
  by us.**

### 4. Non-WMMA default: block8 dp4a
The bare default (no WMMA flag) is integer `__dp4a` on IQ2_XXS×Q8_K with a `block8` inner
loop: each IQ2 grid/sign octet is **unpacked once and dp4a'd against 8 batch columns**
(`dev_dot_iq2_xxs_q8_K_block8_deq_lut`), grid/sign LUT staged into LDS once per block.
This is the cheap version of the register-tiling idea we're missing.

### 5. Token grouping — counting sort by expert → padded tiles
`moe_count → prefix → scatter → build_expert_tiles` (block_m=8, plus a block_m=16 tiling for
down at B≥128). Same shape as our shipped `q2k-down by-expert`.

### 6. Fast attention prefill — FP32, NOT WMMA
File: `rocm/ds4_rocm_attention.cuh`, kernel
`attention_static_mixed_heads8_online_kernel`.

- Grid `(n_tokens, n_head/8)`, 256 threads = 8 warps, **warp w owns head group*8+w** — one
  block per query token, 8 heads/block.
- KV staged **4 rows at a time in LDS** (`__shared__ float4 kv_shared[4*128]`), **shared
  across all 8 heads in the block** → recovers the 1-KV-head MLA reuse. `float4` loads +
  `#pragma unroll 16` + warp-shuffle reductions.
- Two-pass online softmax (deliberately two-pass, not streaming — streaming crossed greedy
  near-ties on long prompts).
- **Hard 768-visible-key cap** (`if (n_score > 768) return;`): 128 sliding-window rows +
  bounded compressed rows. **This cap is what fixed their "throughput drop at 2048+"** — it
  keeps the per-token score buffer bounded regardless of chunk size.
- Auto-selected for any prefill chunk ≥128 tokens in non-quality mode (no env var needed).
- Fallbacks: hipBLAS `SgemmStridedBatched` materialized scores (quality mode / cap exceeded),
  or scalar reference.

### 7. qmix indexer fast path
The indexer score `Σ_h ReLU(q[t,h,·]·k[c,·])·weight[t,h]` is restructured: the per-head
weighting is pulled **out** of the compressed-row loop, collapsing each token to a single
`qmix[t,·]` (head_dim=128) once, then scoring `qmix·comp` as a plain inner-128 dot. Drops
the 64× head factor from the dominant `n_comp`-scaled term. Measured 190 ms → 3.4 ms on the
score stage. Gated `DS4_HIP_INDEXER_QMIX_FAST` (default on). (NOTE: the rocWMMA indexer
score kernels in the repo are `#if __CUDA_ARCH__>=700` / `nvcuda::wmma` — **compiled out on
gfx1151**; AMD uses scalar/qmix.)

### System-level
- Shared-expert dense GEMM overlapped on a **low-priority non-blocking HIP stream** with
  routed-expert dispatch (`ds4_gpu_shared_gate_up_swiglu_q8_0_async_tensor`).
- `perflevel=high` locks clocks.
- **Full COPY_MODEL into VRAM** — works because the Q2_K GGUF is ~81 GB and fits. **Does NOT
  apply to us**: our ~86 GB V4-Flash exceeds budget; mmap still required (see
  `strix-v4flash-memory-tightness`).
- Expert-prefetch-on-router-output is listed as a *future* idea, not implemented.

---

## The gap in our code

`crates/v4flash-kernels/kernels/iq2_xxs_pair_matvec_par.hip`, kernel
`iq2_xxs_pair_matvec_fused_swiglu_by_expert` (and `_chunked`):

```c
// weight row base computed ONCE per (expert, row_block)
const uint8_t* gate_row_base = gate_w_base + e*gate_bpe + row*n_blocks*BLOCK_IQ2_XXS_BYTES;
...
for (int m = 0; m < n_members; m++) {        // each token that picked this expert
    ... stage member's xq into LDS ...
    for (bl = block_lane; bl < n_blocks; bl += 16u)
        g_sum += dot_block_half_lds(w_g, y, ...);   // <-- re-unpacks IQ2 weight EVERY member
}
```

- The weight **bytes** are reused across members (L2-resident after the first member — the
  comment even says so).
- But `dot_block_half_lds` re-runs the **IQ2 grid lookup + sign expansion (VALU)** for every
  member. **L2 reuse does nothing for VALU.**
- Our PMC (`iq2-bottleneck`) says iq2 is **VALU-bound**, so the per-member re-dequant is the
  wasted work. This is the structural reason B=64→512 only bought +9.4%.

ejpir's `block8` unpacks each weight octet **once** and dp4a's it against **8 columns** in
registers — amortizing the VALU dequant 8×.

---

## Recommended priority order (roofline-framed)

| # | Change | Why it's the ceiling lever | Risk |
|---|--------|---------------------------|------|
| **1** | **Multi-column register tiling in the iq2/q2k MoE inner loop** — unpack each IQ2 grid/sign once, dp4a/FMA against N=8 batch columns held in registers (our `block8`) | Directly attacks the VALU-bound dequant our own PMC identified; ~N× less unpack VALU. Single highest-value change; oracle-checkable; prerequisite for raising B_MAX | Med — kernel rewrite |
| **2** | **Dequant-Q2_K/IQ2 → f16, then f16 WMMA for hot experts** at B≥512 | Untested by us — our "no iq2 WMMA" rule only killed *integer* WMMA. At large B the per-expert dequant amortizes; this is their actual 200 t/s path | Med-high — careful A/B; may or may not beat #1 |
| **3** | **Raise prefill B_MAX 512 → 1024/2048**, gated on a 768-visible-key attention cap | Bigger batch only pays after #1 makes weight-reuse real; the cap keeps attention bounded at large B | Low once #1 lands |
| **4** | **qmix-style indexer collapse** (weighting out of the comp-row loop) | 190→3.4 ms on their score stage; long-ctx prefill lever | Low-med |

Start with **#1** — grounded in our own VALU-bound finding, oracle-verifiable, and the
prerequisite that makes raising B_MAX worthwhile. Measure back-to-back vs the current
`_by_expert` kernel per our bench-A/B methodology.

---

## Reproduction notes (their numbers)

- Model: `cyberneurova/CyberNeurova-DeepSeek-V4-Flash-abliterated-GGUF` (Q2_K).
- Build needs the `rsqrtf` host-code fix (`rsqrtf` → `1.0f/sqrtf`) — 2 occurrences in
  `ds4_cuda.cu`.
- Invocation: `DS4_SERVER_FAST_FULL=1 scripts/start_ds4_cli_rocm_upstream.sh --tokens 8192`,
  ROCm 7.2.3 (7.2.1 reportedly didn't reproduce). Throughput is measured on 2048-token
  prefill intervals, not short prompts — small prompts don't saturate the batched kernels.
