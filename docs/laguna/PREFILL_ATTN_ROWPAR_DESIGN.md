# Prefill Global-Attention Kernel Redesign: `gqa_attn_prefill_flash_wmma_fa2_rowpar`

Read-only design study. Target: gfx1201 (RDNA4, 9070 XT), 64 CU, 64 KB LDS/CU,
wave32, 256 VGPR/lane, `v_wmma_f32_16x16x16_f16_w32_gfx12` (synchronous), 600 GB/s
DRAM, 64 MB Infinity Cache (MALL). Model: Laguna-S-2.1 **global full-attention
layers** (`il % 4 == 0`, 12 of 48): `n_head=48`, `n_kv_head=8`, `kv_group=6`,
`head_dim=128`, causal, f16 K/V.

This spec targets the same "escape the VGPR×LDS Pareto wall" goal that
`fa2_hg_packed` (`crates/v4flash-kernels/kernels/gqa_attention.hip:1527`) is stuck
against, by adopting the structural choices that let llama.cpp's
`fattn-mma-f16.cuh` reach 22–25 % of matmul-peak on this exact GPU. All llama.cpp
line/identifier references below are checkable in
`/home/claude-code/llama.cpp/ggml/src/ggml-cuda/`.

---

## 0. Why the current kernel is walled (recap, from source)

`fa2_hg_packed` packs all G=6 heads into one WG and keeps O in registers
(`O_reg[6]` = 48 VGPR, `gqa_attention.hip:1572`), but everything else round-trips
LDS every KV tile:

- **Q lives in LDS** `Qs[6*16*(128+2)]` = 24 KB (`:1563`) and is re-read into WMMA
  A-fragments each KV tile (`:1666`).
- **P (softmax probs) lives in LDS** `Ps[6*16*(32+2)]` = 6 KB (`:1566`), written by
  softmax (`:1726`) then re-read as the AV A-fragment **8× — once per d-tile wave**
  (`:1762`). This is the "P is re-read 8×" leak.
- **Scores `Sp` in LDS** = 12 KB f32 (`:1567`). Total LDS 60.5 KB → **1 block/CU**.
- The score matmul lays q-rows on the WMMA **M** axis and keys on **N**
  (`:1664` `row=lane&15`→q-row as C-m at `:1679`); the AV matmul then needs P as an
  A-fragment indexed `(m=q-row, k=key)`, but P was produced as a C-fragment indexed
  `(m=q-row, n=key)`. Feeding C→A requires a transpose, and because P is produced by
  the *score* waves (12 (head,coltile) tasks over 8 waves, `:1654`) but consumed by
  *every* d-tile wave (`:1746`), the transpose is **cross-wave** → forced through LDS.

Net: 253 VGPR **and** ~61 KB LDS simultaneously, ATT shows 82 % of stall on
`ds_load` for WMMA operands. Adding register caching is impossible (no VGPRs). This
is a tiling problem, not a tweak.

---

## 1. The unlock, stated precisely (from llama.cpp RDNA4 source)

llama.cpp's MMA FA kernel makes three moves that together dissolve the wall. Each is
verified in source:

**(A) Query columns stay on the WMMA N axis through BOTH matmuls; the transpose is
pushed onto V's LDS read, not P.**
- Score tile `T_C_KQ = tile<16,16,float>` is `[key(i) × qcol(j)]` — keys on **M**,
  query-columns on **N** (`fattn-mma-f16.cuh` RDNA4 `mma_tile_sizes`, the
  `#else` non-RDNA3 block: `T_A_KQ=tile<16,8,half2>`, `T_C_KQ=tile<16,16,float>`).
  **This is the opposite of our kernel**, which puts q-rows on M.
- AV is `mma(VKQ_C, V_A, P_B)` (`:986-988`) with V as the A operand and **P as the
  B operand**. P_B needs `[k=key × n=qcol]`; the score C is `[m=key × n=qcol]`. Since
  N=qcol is preserved and RDNA4 defines matrix-C as *the transpose of A&B*
  (`mma.cuh:213` comment; `get_i/get_j` for C and A/B are the identical formulas
  `get_i=threadIdx.x%16`, `get_j=ne*(threadIdx.x/16)+l`), the score accumulator's
  per-lane residency is *already* a valid P_B fragment. The unavoidable "other"
  transpose (V must present key on the contraction axis) is absorbed by
  `load_ldmatrix_trans` (`mma.cuh:895-913`) — a **structured strided LDS read of V**,
  which we were already doing for V anyway.

**(B) The P transition C→B is therefore a pure in-register `make_half2` pack — zero
cross-lane, zero LDS.** `mma.cuh:746-752` (`get_half2`, AMD non-RDNA3 path):
```
for (l0=0; l0<ne; l0+=2)  ret.x[l0/2] = make_half2(KQ_C.x[l0], KQ_C.x[l0+1]);
```
No `permlanex16`, no shuffle, no `ds_store/ds_load`. Contrast `get_transposed`, which
*is* `NO_DEVICE_CODE` on AMD (`mma.cuh:755-758`) — llama.cpp never needs a warp
transpose on RDNA4 precisely because of move (A). This is *cheaper* than the
`permlanex16` C→A sketch in the brief; that sketch is the fallback only if one insists
on P-as-A.

**(C) Q and O are register-resident for the whole KV loop; K and V share ONE LDS
buffer.** `Q_in_reg=true` for every DKQ=128 RDNA config
(`fattn-mma-f16.cuh:148-151`, last macro arg). After the one-time load, Q's LDS is
*reused* as the K/V tile buffer: `tile_K = Q_in_reg ? tile_Q : …` and
`tile_V = nstages>1 ? … : tile_K` (`:1177-1178`) — **K and V alias the same shared
allocation** on RDNA (nstages≤1), with a `__syncthreads` between K-consume and
V-load. O = `VKQ_C` is a register array (`:1183`), half2-accumulated via the
`DATA_LAYOUT_I_MAJOR_SCRAMBLED` C tile (`T_C_VKQ = tile<16,16,half2>`), which halves
O's register footprint vs our f32 `O_reg`.

**Why (A)+(B)+(C) frees the budget:** Sp (12 KB) and Ps (6 KB) vanish entirely
(scores + P are transient registers); Qs (24 KB) vanishes (Q in regs, its slot reused
for K/V). LDS drops from 60 KB to ~9–17 KB, and the 82 %-stall `ds_load` operand
traffic for P/Q disappears because those operands are already in VGPRs.

**RDNA4 runs single-buffered, synchronously.** `nstages = cp_async_available(cc) &&
ncols2>=2 ? target : 0` (`fattn-mma-f16.cuh:348-349`); `cp_async_available` is
NVIDIA-Ampere-only (`common.cuh:352-354`) → **nstages=0 on gfx1201**. The config
table's `nstages_target=1` is a Nvidia ceiling; RDNA uses plain `ggml_cuda_memcpy_1<16>`
LDS staging (`mma.cuh` RDNA4 `load_ldmatrix`, `:848-850`). No `cp.async`, no
`ldmatrix` PTX (that path is `TURING_MMA_AVAILABLE` only, `:788-793`).

---

## 2. Exact tiling for our shapes

> **SUPERSEDED BY §9.** The `ncols1=2 / nthreads=64 / Bc=32` config below was copied
> from llama.cpp's **small-batch** config line and is **wrong for prefill**: it carries
> **8× the K/V DRAM traffic** of `fa2_hg_packed` and would be ~3.5× *slower*, not
> faster. §9 re-derives the config on a DRAM-traffic × VGPR Pareto. The M/N/K mapping
> rationale in §2 is unchanged and still correct; take the *numbers* from §9.

The RDNA4 WMMA N tile is **always 16 wide** (`cols_per_warp = T_B_KQ::I = 16`,
`fattn-mma-f16.cuh:562`). Query "columns" are `(token, head)` pairs. We pack the whole
G=6 head-group of a query token contiguously on N so all 6 heads' P for that token live
in one warp (the whole precondition for move (A)/(B)).

**Chosen configuration (baseline, mirrors the RDNA `ncols=16` config line
`fattn-mma-f16.cuh:129`):**

| Parameter | Value | Rationale |
|---|---|---|
| `ncols2` (heads/tile-column-group) | **6** | full GQA group → one K/V load serves all 6 heads; all 6 heads' P in-warp |
| `ncols1` (query tokens/block) | **2** | 2×6 = 12 query-columns fit the 16-wide N tile |
| `ncols` (N used / N tile) | **12 / 16** | 4 dead columns (see §2.1 waste) |
| `Bc = nbatch_fa` (keys/KV tile) | **32** | RDNA `ncols=16` config value (`:129`); 2 WMMA M-subtiles of 16 |
| `nbatch_K2 / nbatch_V2` | **64 / 64** | = `head_dim/2`; whole head loaded per K/V batch (no D-splitting) |
| `nthreads` | **64** (2 warps, wave32) | RDNA `ncols=16` config (`:129`) |
| `np` (warps splitting KV rows) | **2** | `nwarps*cols_per_warp/ncols`; the 2 warps take keys 0–15 / 16–31 of each 32-key tile as independent stream-K accumulators |
| `occupancy` target | **2 blocks/CU** | config arg (`:129`); VGPR-bound, not LDS-bound |
| grid | `(n_kv_head=8, ceil(seq/ncols1), 1)` | one block = 1 KV head × 2 query tokens |
| block | `(64,1,1)` | |

**Head-group → WMMA M/N/K mapping:**
- **Score** `S = Kᵀ·Q`: M = keys (Bc=32 → 2×16 subtiles, split np=2 across warps),
  N = query-columns (12 of 16 = 2 tokens × 6 heads), K = head_dim (128 = 8 k-steps
  of 16). C = `[key × qcol]` f32, per-lane 8 floats.
- **AV** `O = V·P`: M = head_dim (128 → 8×16 d-subtiles), N = query-columns (same 12),
  K = keys (32 → 2 k-steps of 16). V is the A operand (transposed on LDS load), P the
  B operand (the score C, repacked). C = `VKQ_C` `[d × qcol]` half2, per-lane 8 half2
  × 4 d-fragments.

**Why these and not others:**
- N is fixed at 16 by the WMMA fragment; 6 heads means the only integer token count
  that fits is `ncols1 ∈ {1,2}`. `ncols1=2` doubles KV reuse per block vs 1 at no extra
  register cost (VKQ_C is per-N-tile, still one N-tile).
- `Bc=32` (not 64): keeps `KQ_C` at 1 fragment/lane and K/V LDS at ~9 KB so occupancy
  is set by VGPR (≈2 blocks/CU) rather than LDS. `Bc=64` is a §7 Stage-2 A/B.
- 2 warps + np=2 (stream-K over keys) avoids any per-tile cross-warp barrier: each
  warp is a self-contained flash accumulator over a strided key subset, combined once
  in the epilogue (§3 step 8). This is llama.cpp's np mechanism (`:564`, `:622`
  `i_KQ_00 += np*T_A_KQ::I`).

### 2.1 The G=6 tax (honest)
Power-of-2 GQA (llama.cpp's measured GQA-8) fills all 16 N-lanes; G=6 fills 12 →
**25 % of WMMA N-throughput is structurally wasted** on the 4 dead columns in both
matmuls. So our reachable ceiling is ~0.75 × llama.cpp's 22–25 % ≈ **17–19 % of
peak**, still ~1.7–1.9× the current 10 %. Padding the group to 8 (2 masked heads) does
not help — it wastes the same 2/8 = 25 %. Splitting the group across N-tiles to fill 16
would break the "whole group in-warp" precondition and is rejected.

---

## 3. Data flow per KV tile (step by step)

Persistent registers (whole KV loop): `Q_B[8]` (Q fragments), `VKQ_C[4]` (O, half2),
`KQ_max[cols]`, `KQ_rowsum[cols]` per lane. Loop bound `kb0 ∈ [kb0_start, kb0_stop)`
(§5). Each warp `w` (=`threadIdx.y`) with `np=2` owns keys `w*16 .. w*16+15` of the
32-key tile.

Prologue (once/block): load Q → `tile_Q` LDS → `Q_B` registers (scaled by
`scale`), then reuse `tile_Q`'s LDS as `tile_KV`. Zero `VKQ_C`, `KQ_max=-inf`,
`KQ_rowsum=0`.

Per KV tile `kb0`:
1. **K load** → `tile_KV` LDS: `ggml_cuda_memcpy_1<16>` vectorized loads, full head
   (`nbatch_K2=64` half2/key), zero-padded past `k_VKQ_sup`. Layout stride
   `nbatch_K2+4` (the "+4 half2" skew, our `HG_*STRIDE` analog for bank-conflict-free
   fragment reads). `__syncthreads` **[B1]**.
2. **Score WMMA**: for k-step `kk=0..7`: load `K_A` fragment from LDS
   (`load_ldmatrix`), `mma(KQ_C, K_A, Q_B[kk])`. Q_B already in registers → **no Q LDS
   read**. Accumulate into the single `KQ_C` fragment. (This warp does its 16 keys.)
3. **Causal/window mask + scale**: add per-(key,qcol) mask to `KQ_C.x[l]`; for pure
   causal we mask analytically from `key_abs > q_abs` (no mask tensor, cf. our
   `:1686`). Dead columns (12–15) are left as −inf so they never contribute.
4. **Online softmax, fully in-register/in-warp**: row-max over `KQ_C` lanes via
   `__shfl_xor` (offset 16 only — "2 threads per KQ column" on RDNA4,
   `fattn-mma-f16.cuh:815-818`); `KQ_max_new`; rescale `VKQ_C *= exp(KQ_max-KQ_max_new)`
   (`:888-907`); `KQ_C = exp(KQ_C - KQ_max_new)`; `KQ_rowsum` update (`:830-863`).
   **No `Sp`/`Ps` LDS, no barrier.**
5. **P transition C→B (THE unlock)**: `B[k] = get_half2(KQ_C[k])` (`:930-932`) =
   register `make_half2` pack. **Intra-wave, zero LDS** (proof: §4).
6. **V load** → `tile_KV` LDS (aliases K's buffer). `__syncthreads` **[B2]** (WAR:
   don't overwrite K until step 2 finished; and don't read V until loaded).
7. **AV WMMA**: for d-subtile `i=0..7`, for k-step (keys) `kk=0..1`: `A = V` via
   `load_ldmatrix_trans` from LDS, `mma(VKQ_C[i], V_A, B[kk])`. O accumulates in
   registers.
8. **Next-tile barrier** `__syncthreads` **[B3]** before `kb0+1` overwrites `tile_KV`.

Epilogue (once/block): combine the `np=2` warps' partial `(KQ_max, KQ_rowsum, VKQ_C)`
— write partials to the (now-free) `tile_KV` LDS, one `__syncthreads`, warp0 reads
warp1's, rescales by `exp(m1−m0)`, sums O and l. Normalize `O /= l`, write
`out[(token*n_head + head)*128 + d]` for the 12 real columns only.

**Barriers/tile = 2–3** (B1, B2, and B3 which can fuse with next B1) vs our current
**3–4** (K/V stage, after-score, after-softmax, after-AV). The after-score and
after-softmax barriers are gone because softmax + P are within-warp.

---

## 4. Proof the P transition is intra-wave (no LDS)

Precondition: for query token `t` and every head `g∈0..5`, the score result
`P[key, (t,g)]` must be produced and consumed within a single wave.

- The N axis holds `(t,g)` columns. One warp owns a full 16-wide N tile
  (`cols_per_warp=16`), so all 12 real `(t,g)` columns are that warp's private
  C-fragment lanes. **Never shared across warps.** (The np=2 split is over *keys* (M),
  not columns (N); each warp keeps its own full-N C-fragment.)
- The score C-fragment lane→index map on RDNA4: `get_i=threadIdx.x%16` (=key, M),
  `get_j=ne*(threadIdx.x/16)+l` (=qcol, N), `ne=8` (`mma.cuh:193-217`). The P_B
  operand (`tile<16,8,half2>`) has the **same** `get_i/get_j` formulas (RDNA4: "matrix
  C is the transposed matrix A&B", `mma.cuh:213`). Therefore element `l` of lane `L` of
  the score C is *already* the correct B-operand element `l` of lane `L` — the only
  work is packing two f32s into one half2:
  `B.x[l/2] = make_half2(KQ_C.x[l], KQ_C.x[l+1])` (`mma.cuh:746-752`). This touches only
  the lane's own registers. **No `ds_*`, no `__shfl`, no `permlanex16`.** QED.

If a future variant instead assigned P to the A operand (keys on N of the score, our
current orientation), the C→A step *would* need the `permlanex16` warp transpose from
the brief — but it stays intra-wave *only if the group is packed into N in the score's
producing wave*. The clean win is to flip the score orientation (keys→M, qcol→N) so P
is the B operand and no permute is needed at all. **This orientation flip is the single
most important code change vs `fa2_hg_packed`.**

---

## 5. Causal tile-skipping & SWA

**llama.cpp `mask_to_KV_max`** (`fattn-common.cuh:626-676`): a tiny pre-pass kernel,
one block per Q-tile, scans the mask tensor backward from the last KV tile and records
the highest KV tile that has any non-(-inf) entry into `KV_max[jt]`; the main kernel
then loops `kb0 ∈ [kb0_start, KV_max)` (`fattn-mma-f16.cuh:1139-1140`,
`:1272-1275`), skipping the fully-masked upper-triangular tiles.

**Our equivalent needs no mask tensor and no pre-pass** — we compute the bound
analytically, exactly as our existing kernels already do
(`gqa_attention.hip:1593-1598`):
```
max_q_abs   = q_offset + q_row_base + n_rows - 1
kb0_stop    = max_q_abs / Bc + 1            // causal upper bound
win_lo      = (swa && min_q_abs+1 > swa) ? min_q_abs+1-swa : 0
kb0_start   = win_lo / Bc                    // 0 for global layers
```
This is strictly better than the mask scan for our pure-causal global layers.

**SWA window=512 layers (36 of 48):** the same `kb0_start/kb0_stop` window math serves
them (set `swa_window=512`), and correctness is identical. **But the payoff is on the
global layers only** — SWA reads ≤512 keys = ≤16 KV tiles, so it is *not* the O(L²)
bottleneck, and the note in `gqa_attention.hip:1497` (SWA `kv_group` differs from 6)
means rowpar's G=6 packing does not even apply to them. **Recommendation: rowpar
handles global (`il%4==0`) only; leave SWA on `fa2`/`fa2_hg`.** Wiring: in
`laguna_het.rs:1513` (the `is_full && HG_PACKED` gate), add a `rowpar` branch ahead of
`hg_packed`, gated `LAGUNA_ATTN_ROWPAR`.

---

## 6. VGPR & LDS budget table (per block, wave32)

Config: 2 warps × 32 lanes, ncols1=2, ncols2=6, Bc=32, D=128, np=2.

### LDS (shared), per block
| Array | Elements | Bytes | Lives | Notes |
|---|---|---|---|---|
| `tile_Q` → reused as `tile_KV` | 32 keys × (64+4) half2 | **8 704** | LDS | K & V **alias** this (nstages=0); +4 half2 skew for conflict-free fragment loads |
| `tile_mask` | — | 0 | — | analytic causal, no mask tensor |
| epilogue np-combine scratch | 12 cols × (m,l,O-partials) | ~2 KB | LDS | reuses `tile_KV` after loop; not additive |
| **LDS total (peak)** | | **≈ 8.7 KB** | | **13.6 % of 64 KB** |
| *(K+V both resident, opt)* | 2 × 8 704 | **≈ 17.4 KB** | | Stage-1 A/B; still 27 % |

### VGPR (per lane, dwords)
| Register set | Size | Persistent? | VGPR/lane |
|---|---|---|---|
| `Q_B[8]` (Q in regs) | 8 frags × 4 half2 | yes | **32** |
| `VKQ_C[4]` (O, half2 scrambled) | 4 frags × 8 half2 | yes | **32** |
| `KQ_max`,`KQ_rowsum`,`KQ_max_scale` | ~3 × cols_per_thread(8) | yes | ~24 |
| `KQ_C[1]` (score C, f32) | 1 frag × 8 f32 | transient | 8 |
| `K_A`/`V_A`/`B` frags | ≤ 3 × 8 halves | transient | ~12 |
| addr/loop/misc | — | — | ~20–30 |
| **Estimated occupied** | | | **≈ 110–140** |

**Does it clear the wall?** Yes, decisively. Current: **253 VGPR AND ~61 KB LDS at
once** → 1 block/CU, LDS-`ds_load`-bound. Rowpar: **≈130 VGPR AND ≈9 KB LDS** —
neither resource is near saturation, and the P/Q/O operand traffic that caused the
82 % `ds_load` stall is now register-resident. Occupancy: at ~130 VGPR, RDNA4's 1536
VGPR/SIMD gives ~11 waves by VGPR and LDS gives ~7 blocks/CU — so occupancy is set by
the config's chosen `occupancy=2` blocks/CU (4 waves/CU), leaving huge headroom; the
kernel becomes WMMA-issue / DRAM-BW bound (the intended regime), not LDS-bound.

Even the pessimistic K+V-resident + Bc=64 Stage-2 variant (~200 VGPR, ~35 KB LDS at
1 block/CU) never hits *both* the 253/61 KB corner.

---

## 7. Honest risk list (NVIDIA-only deps → RDNA4 substitute)

| llama.cpp feature | NVIDIA-only? | RDNA4 substitute / consequence |
|---|---|---|
| `cp.async` double/triple-buffered K/V (`nstages≥2`) | **Yes** (`cp_async_available`=Ampere+, `common.cuh:352`) | RDNA runs `nstages=0`, synchronous `ggml_cuda_memcpy_1<16>` LDS staging. We keep our **register-prefetch** trick (`gqa_attention.hip:1605` `kbuf/vbuf`) to raise MLP instead — the one thing we already do better. Risk: **low**, this is already how our kernels hide DRAM latency. |
| `ldmatrix.sync` PTX fragment load | **Yes** (`mma.cuh:788-793`, Turing path) | RDNA "`load_ldmatrix`" is a plain vectorized LDS load with fragment-indexed addressing (`mma.cuh:848-850`); the `+4` half2 skew handles bank conflicts. Risk: **low**. |
| `ldmatrix.trans` for V | **Yes** | RDNA `load_ldmatrix_trans` does the transpose via strided scalar/half2 LDS reads (`mma.cuh:895-913`). Costs more LDS instructions than Nvidia; the V read is the residual LDS traffic. Risk: **medium** — V-load may become the new hot spot (cf. our `smwsum` V-staging DRAM-wait note). Mitigation: f16 V is already our cache format; measure V `ds_load` share after Stage 0. |
| 228 KB smem multi-stage pipeline | **Yes** | Irrelevant at nstages=0; our budget is 9–17 KB. **No degradation** — RDNA path never used it. |
| Register-count / occupancy heuristics tuned for 255-VGPR Nvidia | partial | HIP/LLVM VGPR allocation differs; the ~130 estimate must be confirmed with `--save-temps`/`llvm-mca` (NOT via GPU run here). Risk: **medium** — if the compiler spills `VKQ_C` or `Q_B` to scratch, the whole thesis collapses (same failure mode our `O_reg` avoids by static indexing). Mitigation: keep all fragment arrays statically indexed; verify 0 scratch in ISA. |
| half2 SCRAMBLED C accumulate for O (`T_C_VKQ` `DATA_LAYOUT_I_MAJOR_SCRAMBLED`) | RDNA4-native | This is an RDNA4 *feature* (fast C-transpose), `mma.cuh:83`. Using it halves O VGPRs but the scramble must be undone at write-out ("convert to float to unscramble"). Risk: **medium** — get the unscramble wrong and O is silently permuted. Mitigation: Stage 0 uses plain f32 `VKQ_C` (like our `O_reg`), adopt scrambled half2 only in Stage 2. |
| `ncols2` power-of-2 assumption in dispatch | Yes (`fattn.cu:92-108`) | We hard-wire `ncols2=6` in our own template; not bound by llama.cpp's compile-combinatorics limit. Cost = the 25 % N-waste (§2.1). Risk: **structural, quantified** — caps us at ~17–19 %. |

**Top-3 risks:** (1) compiler spills `Q_B`/`VKQ_C` to scratch → verify ISA has 0
scratch before trusting any perf; (2) V's `load_ldmatrix_trans` LDS traffic becomes the
new bottleneck (RDNA has no `ldmatrix.trans`); (3) the 25 % G=6 N-waste caps the
ceiling below llama.cpp's 22–25 %.

---

## 8. Staged implementation plan

Kill criterion throughout: if a stage cannot beat `fa2_hg_packed`'s measured
prefill tok/s by the stated margin, stop and keep `hg_packed` as default. All measured
with **`PIPELINE_LANES=2`** and rocprofv3 kernel-trace (per
`reference_rocprofv3_kernel_trace.md`), at 4K/32K/96K, parity-gated against ds4
(rel ≤ 1e-4, the `hg_packed` gate).

**Stage 0 — minimal correct (orientation flip + register P).**
Build: score with keys→M / qcol→N; ncols1=2, ncols2=6, Bc=32, f32 `VKQ_C` in regs,
K/V shared LDS, register P via `make_half2`, analytic causal, np=2 stream-K combine.
No prefetch, no scramble, no skew tuning.
Measure: (a) ISA shows **0 scratch**, VGPR ≤ ~160, LDS ≤ ~10 KB; (b) parity rel ≤ 1e-4
vs ds4 token; (c) ATT: `ds_load` operand stall drops from 82 % to <40 %; (d) per-kernel
tok/s.
Expect: **~15–18 % peak, ~1.6–1.8× current**. **Kill if <13 %** (i.e. not clearly
above `hg_packed`) or if it spills.

**Stage 1 — DRAM-BW hardening.**
Add: register K/V prefetch (reuse `kbuf/vbuf` shift-register from
`gqa_attention.hip:1605`); optionally K+V both LDS-resident (~17 KB) to drop the mid-tile
B2 barrier; try `ncols1` bigger only if VKQ_C stays 1 N-tile.
Measure: DRAM BW % of roofline (should approach the KV-read roofline at long ctx); ATT
V `ds_load` share.
Expect: **~18–20 %** at 32K/96K (where it's KV-BW bound). Kill Stage-1 additions
individually if any regresses (the "attn double-buffer trap",
`feedback_attn_double_buffer_trap.md`).

**Stage 2 — WMMA-issue tuning.**
Add: half2 SCRAMBLED `VKQ_C` (halve O VGPR → higher occupancy); Bc=64 A/B; LDS skew
sweep (`+2`/`+4`/`+8` half2); occupancy 2↔3 blocks/CU.
Measure: WMMA issue % (`SQ_BUSY`), waves/SIMD, final tok/s vs the G=6 ceiling.
Expect: **~19–22 %** (the G=6 ceiling). Kill any change that regresses e2e at
`PIPELINE_LANES=2`.

**Fallback:** at any stage, `LAGUNA_ATTN_ROWPAR` gates it OFF by default until it
beats `hg_packed` at all three depths; `hg_packed` remains the shipped default.

---

# 9. DRAM-traffic correction — config re-derivation (supersedes §2, §6, §8)

§2 was derived on a VGPR×LDS Pareto only. That is an incomplete objective: FA K/V
traffic scales as **1/Br**, and §2's `ncols1=2` is an 8× traffic regression vs
`fa2_hg_packed`'s `HG_BR=16`. This section fixes it.

## 9.1 The config I copied was llama.cpp's *small-batch* config

`ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1` (`fattn.cu:9-35`) selects `ncols` by
**query-token count** `Q->ne[1]`:
```
if (Q->ne[1] <= 16/ncols2)  -> ncols = 16      // fattn.cu:21-26
if (Q->ne[1] <= 32/ncols2)  -> ncols = 32      // fattn.cu:28-32
                            -> ncols = 64      // fattn.cu:34   <-- prefill lands here
```
Long-context prefill has `Q->ne[1]` in the thousands, so it **always falls through to
`ncols = 64`**. The DKQ=128 config actually used for prefill is therefore
`fattn-mma-f16.cuh:151`:
```
GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64,  128, 2,  64, 64, 64, 64, 1, true);
//                    DKQ  DV ncols  nthr occ  nbatch_fa ...
```
→ **nthreads=128 (4 warps), nbatch_fa (Bc) = 64, occupancy 2**, i.e. `ncols1 = 64/8 = 8`
query tokens for their GQA-8 case — *not* the `ncols=16 / nthreads=64 / Bc=32` line
(`:129`) that §2 copied. The 22–25 %-of-peak measurement is from the ncols=64 config.

## 9.2 K/V DRAM traffic arithmetic

Per layer, per KV head, a query tile of `Br` rows based at row `r0` reads causal keys
`0 .. r0+Br-1`. Summing over the `L/Br` tiles:

    keys_read(per KV head, per layer) = Σ_{t=0}^{L/Br-1} (t·Br + Br)
                                      = L²/(2·Br) + L/2

Bytes: one key row = K + V = 2 × 128 × 2 B = **512 B**. Times `n_kv_head = 8`.
Global layers = 12.

| | L = 32 768 | L = 65 536 |
|---|---|---|
| **Br = 16** (`fa2_hg_packed`, and rowpar §9.4) | | |
| keys/head/layer | 33 554 432 + 16 384 = **33 570 816** | 134 217 728 + 32 768 = **134 250 496** |
| bytes/head/layer | 17.19 GB | 68.74 GB |
| bytes/layer (×8 heads) | **137.5 GB** | **549.9 GB** |
| all 12 global layers | **1.65 TB** | **6.60 TB** |
| per token, per layer | 3.50 MB | 8.39 MB |
| **Br = 2** (§2 as written) | | |
| keys/head/layer | 268 435 456 + 16 384 = **268 451 840** | 1 073 741 824 + 32 768 |
| bytes/layer (×8 heads) | **1.100 TB** | **4.399 TB** |
| all 12 global layers | **13.2 TB** | **52.8 TB** |
| **ratio vs Br=16** | **8.0×** | **8.0×** |

## 9.3 Infinity Cache: measured reuse is ≈1×, not 16×

Two independent routes agree that the **raw** number above *is* the effective traffic.

*Route 1 — capacity.* At 64K, per-KV-head K+V = 65 536 × 512 B = **33.55 MB**; all 8
heads = **268.4 MB** vs a **64 MB** IC. Not even two KV heads' working sets fit. The
grid is `(n_kv_head, ceil(B/HG_BR))` (`gqa_attention.hip:1496`) with `blockIdx.x` the
**fast** dimension, so linear dispatch enumerates *all 8 KV heads* before advancing the
query tile — the ~128 resident blocks therefore stream **8 KV heads concurrently**,
not one. The lockstep-sharing argument only ever applies to the ~16 same-head blocks,
and they must stay phase-aligned across the whole sweep to realise it.

*Route 2 — cross-check against measurement.* 6.60 TB at 600 GB/s = **11.0 s** floor.
64K prefill at ~440 tok/s → ~149 s wall; global attention measured at 15–20 % of the
dGPU wall → ~22–30 s. That implies **≈264 GB/s ≈ 44 % of DRAM roofline** — squarely
inside the independently quoted 26–50 % band. If IC were delivering the 16× reuse the
lockstep argument predicts, effective traffic would be 0.41 TB → 0.7 s → <1 % of wall,
which contradicts the measured 15–20 %. **Empirical IC reuse on the K/V stream ≈ 1×.**

*Is a grid-order swap enough?* Already tried and already rejected. `grid_mode=1`
(`gqa_attention.hip:841-846`) is a KV-first remap built *explicitly* "so each KV head's
K/V stays 64 MB-MALL-resident across its redundant reads" — and it is A/B-gated **off**
by default (`LAGUNA_ATTN_KVFIRST=1`, `laguna_het.rs:1545`). The locality fix has been
attempted at the grid level and did not win. **Design on raw traffic; do not budget for
IC absorption.**

*Consequence.* Headroom to the DRAM roofline is only **600/264 ≈ 2.3×**. An 8× traffic
increase puts the kernel at ~3.5× *over* roofline: `ncols1=2` would be **~3.5× slower
than today**, and no amount of WMMA efficiency recovers it. §2's config is **dead**.

## 9.4 The structural fact that rescues the design: Br scales by WARPS, not registers

With `np = nwarps·cols_per_warp/ncols = 1`, each warp owns **its own** 16-wide N-tile
(`j0 = (threadIdx.y/np)*cols_per_warp`, `fattn-mma-f16.cuh:1246`) and holds only that
tile's `Q_B[8]` (32 VGPR, `:1182`) and `VKQ_C[4]` (32 VGPR, `:1187`). **All warps in
the block share the one K/V LDS tile.** So:

    Br = 2 × nwarps  (2 query tokens per N-tile at ncols2=6),   nthreads = 32 × nwarps
    VGPR/lane is CONSTANT in Br.   LDS is CONSTANT in Br.

This is precisely what `fa2_hg_packed` **cannot** do: its Q is LDS-resident
(`Qs[6·Br·130]`, `gqa_attention.hip:1563` — 24 KB at Br=16), so Br=32 would need 48 KB
of `Qs` alone and is structurally impossible. **Q-in-registers makes Br free.** Raising
Br is therefore not merely a way to *avoid* a traffic regression — it is a traffic
*lever* rowpar unlocks and `hg_packed` does not have.

Also note `np=1` removes the cross-warp stream-K epilogue combine described in §3
step 8 — no key-splitting, so that combine and its barrier disappear entirely.

## 9.5 The Pareto table

VGPR/lane (constant in Br): `Q_B` 32 + `VKQ_C` 32 (half2) or 64 (f32) + `KQ_C`
(Bc/16 frags × 8) + `B[]` (Bc/32 frags × 4) + `KQ_max/rowsum/scale` 3
(`cols_per_thread = 1` on AMD, `fattn-mma-f16.cuh:329-335`) + K_A/V_A ~8 + addr/loop ~20.
LDS = K/V aliased single tile, stride `nbatch_K2+4` half2: Bc·68·4 B.

| # | Br | nwarps / nthreads | Bc | VKQ_C fmt | VGPR/lane | LDS | blocks/CU | waves/SIMD | K/V traffic vs today | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 1 / 32 | 32 | half2 | ~119 | 8.5 KB | 7 (LDS) | 1.75 | **8.0×** | **DEAD** (§9.3) |
| 2 | 4 | 2 / 64 | 32 | half2 | ~119 | 8.5 KB | 7 | 3.5 | **4.0×** | dead |
| 3 | 8 | 4 / 128 | 64 | half2 | ~143 | 17.0 KB | 3 (LDS) | 6 | **2.0×** | dead (llama.cpp's own GQA-8 prefill cell — but their Br=8 comes from ncols2=8; ours would be Br=8 at ncols2=6) |
| 4 | 16 | 8 / 256 | 32 | half2 | ~119 | 8.5 KB | 6 (VGPR) | 12 | **1.0×** | viable — Stage 0 |
| 5 | **16** | **8 / 256** | **64** | **half2** | **~143** | **17.0 KB** | **3 (LDS)** | **6** | **1.0×** | **Stage 0 — BUILD THIS** |
| 6 | 16 | 8 / 256 | 64 | f32 | ~175 | 17.0 KB | 3 | 6 | 1.0× | fallback if scramble-unpack is buggy |
| 7 | **32** | **16 / 512** | **64** | **half2** | **~143** | **17.0 KB** | **2 (VGPR)** | **8** | **0.5×** | **Stage 1 — TARGET** |
| 8 | 64 | 32 / 1024 | 64 | half2 | ~143 | 17.0 KB | 1 | 8 | 0.25× | Stage 2 stretch; 1 block/CU + 32-warp barriers, high trap risk |

(blocks/CU = min(LDS limit, VGPR limit); VGPR limit = ⌊1536/VGPR⌋ waves/SIMD × 4 SIMD ÷ nwarps.)

**Recommended cell: #5 (Br=16, Bc=64, 256 threads, half2 O) for Stage 0; #7 (Br=32) as
the Stage-1 target.**

Why #5 first: it is **exactly traffic-neutral** vs `fa2_hg_packed` (same Br=16, same
grid, same causal skip), so Stage 0 measures the *compute* win in isolation with no
DRAM confound. It also reuses our existing `HG_BLOCK=256` block shape. Bc=64 matches
llama.cpp's actual prefill `nbatch_fa`.

Why #7 next: it halves K/V traffic — the lever `hg_packed` structurally cannot pull.

## 9.6 Revised expectations

At Br=16 the kernel today sits at ~44 % of DRAM roofline and ~10 % of matmul peak, so
**compute is the binding constraint and there is room for ≈2.3× before DRAM binds.**

- **Cell #5 (Br=16):** compute ~1.65–1.9× faster (§2.1 ceiling: 0.75 × llama.cpp's
  22–25 % ≈ 17–19 % of peak, from the 25 % N-lane waste at G=6). DRAM term unchanged,
  so the kernel lands at **~83 % of DRAM roofline** — the win is realised but it is the
  *last* compute win available at Br=16.
- **Cell #7 (Br=32):** same compute win **plus** DRAM demand halved to ~22 % roofline
  → the ~1.9 × compute gain is fully realised with headroom to spare, and further
  compute work stays worthwhile. Expected **~2.0–2.4×** on the global-attn kernel.

Caveat carried forward from §2.1: rowpar issues **1.33× more WMMA instructions** per
useful FLOP than `hg_packed` (12 of 16 N-columns used vs `hg_packed`'s fully-packed
16×16 tiles). The win is entirely in *issue rate* (killing the 82 % `ds_load` stall)
and must exceed 1.33× just to break even. This makes the Stage-0 kill threshold
sharper, not softer.

## 9.7 Revised kill criteria

- **Stage 0 (cell #5)** must show ≥ **1.35×** on the global-attn kernel (rocprofv3
  kernel-trace, `PIPELINE_LANES=2`) — below that the N-lane waste is eating the issue
  win and rowpar is not worth the rewrite. Also required: 0 scratch, VGPR ≤ ~160,
  ATT `ds_load` stall < 40 %.
- **Stage 1 (cell #7)** must show measured DRAM bytes ≈ 0.5× Stage 0. If not, the
  causal-skip or grid assumptions are wrong — stop and re-derive.
- If Stage 0 lands < 1.35×, **kill rowpar and keep `fa2_hg_packed`.**

## 9.8 Honest answer to "can rowpar beat `fa2_hg_packed` at long context?"

**Yes — but only at Br ≥ 16, and the §2 config as written would have lost badly
(~3.5×).** The coordinator's objection is correct and it invalidated the v1 config, not
the kernel concept. The concept survives because of §9.4: with Q in registers, Br scales
by adding warps at zero VGPR and zero LDS cost, so rowpar can match `hg_packed`'s Br=16
traffic *and then go past it* to Br=32 — which `hg_packed` cannot do at all, since its
LDS-resident Q would need 48 KB. The remaining honest risks are unchanged from §7, plus
the sharper 1.33×-WMMA-overhead break-even in §9.6.

---

## 10. OUTCOME: BUILT, MEASURED, **KILLED** (2026-07-26)

The kernel described above was fully implemented (Stage-0, cell #6: Br=16, Bc=64,
256 threads, f32 O accumulator), wired behind `LAGUNA_ATTN_ROWPAR=1`, validated, and
then **reverted**. It is ~1.75× SLOWER than `fa2_hg_packed`. Do not rebuild it without
reading this section.

### What passed
- **Gate 2 (scratch=0): PASS.** gfx1201 `-Rpass-analysis=kernel-resource-usage`:
  ScratchSize **0**, VGPR/SGPR spills **0**, VGPR **172** (spec predicted ~175 for the
  f32-O cell), LDS **17408 B** (spec said 17 KB). Occupancy 3 blocks/CU, LDS-limited.
  The thesis's #1 risk did not fire.
- **Gate 1 (correctness): PASS.** vs CPU reference at kv_group=6: max_abs 1.9e-4 /
  5.1e-5 / 2.7e-5 — same accuracy as `hg_packed`, far under TOL 2e-3. **The orientation
  flip and the register-P transition are mathematically correct.** §4's proof holds.
- **The unlock itself worked.** ATT: `s_wait_dscnt` went from dominant (the "82 % LDS
  operand" wall) to **~4 %**. The LDS-operand stall this design targeted was eliminated.

### What failed
**Gate 3 (≥1.35× vs `hg_packed`): FAIL, decisively.**

| ctx | hg_packed | rowpar | ratio |
|---|---:|---:|---|
| 4K  | 1933 µs  | 2464 µs  | 0.78× |
| 32K | 14955 µs | 26239 µs | 0.57× |
| 64K | 30041 µs | 53344 µs | 0.56× |

ATT after: `s_wait_loadcnt` **53 %**, stall/latency 78.9 %, 32.1M cycles (~2.9× the
baseline's 11.0M). **It removed an LDS stall and bought a worse DRAM stall.**

### Why — the three things this spec got wrong
1. **`hg_packed` is at 14.1 % of peak, NOT the ~10 % this document assumed.** With the
   G=6-adjusted ceiling at 17-19 % (§2.1), the maximum conceivable win from ANY
   rowpar-style rewrite was ~1.2-1.35× — **below the 1.33× G=6 WMMA-issue tax that §9
   itself identified.** The design was unwinnable the moment the baseline was
   remeasured. *Re-derive the baseline before authorising a rewrite premised on it.*
2. **The aliased single K/V LDS buffer structurally prevents DRAM-latency hiding.**
   Because V overwrites K, V's DRAM load cannot issue until all warps finish reading K
   (barrier B2), and the next tile's K cannot load until AV finishes → **two full
   unhidden DRAM-latency exposures per tile**. That is the 53 % `s_wait_loadcnt`.
   `hg_packed` hides DRAM precisely because it does NOT alias (separate ~8 KB Ks/Vs)
   and can afford register prefetch (smaller persistent register set).
3. **§7 risk-1's mitigation is self-contradictory at Bc=64.** It proposes keeping the
   register-prefetch trick to hide DRAM — but a Bc=64 K+V shift-register ring (PFD=2)
   is ~128 VGPR on top of the 172 already used → guaranteed spill. The one lever that
   would fix the exposed DRAM stall is unavailable in the recommended config.

### Standing conclusion
**Prefill global attention is CAPPED at `fa2_hg_packed`'s ~14.1 % of matmul-peak** for
parity-exact dense attention on this hardware. The register-P unlock is real and the
fragment-map analysis in §4 is sound and reusable — but it is **irrelevant to the
binding constraint**, because `hg_packed` was never purely LDS-walled in a way that
mattered; it already hides DRAM.

Unexplored rescue (NOT recommended): Bc=32 + register K/V prefetch, which would fit the
prefetch registers but double the barrier count and still carry the 1.33 % tax against a
14.1 %-of-peak baseline. Low expected value; see point 1.
