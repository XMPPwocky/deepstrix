# M51 — iGPU MoE FFN prefill optimization journal

Goal: cold-cache prefill 230 → 300+ tok/s at T=4K-32K (floor; real target = HW max,
realistic dp4a ceiling estimated ~445). Plan: `~/.claude/plans/hello-i-d-like-to-virtual-lark.md`.
Method: profile-first (ablations + ISA + PMC; ATT attempt bounded), then
S0 staged micro-fixes → S1 k-widened iq2 → S2 k-widened q2k. Every experiment
gets a journal entry here, committed BEFORE the run.

Baseline HEAD: `0959f65` (230 tok/s cold T=4K-32K per project_current_state).

---

## 2026-06-09 — Phase 0a: occupancy baseline (static)

`nix develop -c hipcc --offload-arch=gfx1151 -O3 -c -Rpass-analysis=kernel-resource-usage`:

| kernel | VGPR | SGPR | LDS B/WG | scratch | spills | waves/SIMD |
|---|---|---|---|---|---|---|
| iq2 `_chunked_staged` (prod) | 97 | 70 | 18944 | 0 | 0 | **12** |
| iq2 `_tile8_row32` (opt-in) | 192 | 56 | 3072 | 424 B/lane | **309 VGPR!** | 8 |
| q2k `_by_expert` (prod) | 70 | — | 0 | 0 | 0 | **16** (full) |

Findings:
- staged occupancy (12) is LDS-bound at 18.9 KiB/WG → S0b's −8.5 KiB LDS may
  raise occupancy directly.
- tile8 spills 309 VGPRs to scratch — a second reason (beyond the 4×-fewer-WGs
  grid) for its cold-cache regression.
- q2k by_expert is at FULL occupancy with zero LDS — its 2.6×-over-roofline is
  pure per-member re-unpack VALU overhead, not occupancy. Supports the S2 design.

Gate for all new variants: waves/SIMD ≥ 12 (iq2) / 16 (q2k), zero spills.

## 2026-06-09 — Phase 0c: ISA audit of staged member loop

Recipe (CCOB bundles need unbundling first):
```
nix develop -c hipcc -O3 --genco --offload-arch=gfx1151 -o /tmp/iq2_dev.hsaco crates/v4flash-kernels/kernels/iq2_xxs_pair_matvec_par.hip
nix develop -c clang-offload-bundler --unbundle --type=o --input=/tmp/iq2_dev.hsaco --output=/tmp/iq2_elf.hsaco --targets=hipv4-amdgcn-amd-amdhsa--gfx1151
nix develop -c llvm-objdump -d --mcpu=gfx1151 /tmp/iq2_elf.hsaco
```

Member loop is unrolled ×4. Per member iteration (the hot ~25 instructions):

| group | insts | notes |
|---|---|---|
| useful | 4× `v_dot4_i32_iu8` | the only roofline work |
| scale chain | 2× `v_mul_lo_u32` + 2× `v_cvt_f32_i32` + ~4× `v_mul/fmac_f32` | F3 — hoistable |
| **accumulator movrel** | **2× `v_movrels_b32` + 2× `v_movreld_b32` + `s_add m0`** | `lane_pacc_g/u[mi]` dynamic index → M0-indexed register moves. NOT in any source-level analysis. |
| LDS | `ds_load_2addr_b32` (q8) + `ds_load_b32` (yd) + 2× `s_waitcnt lgkmcnt` | misaligned q8 → 2×b32 (F2) |

≈16 VALU + 2 SALU issued per member vs 4 useful → static ~4×, recovered to the
measured 2.4× by partial VOPD (`v_dual_*` present but sparse).

**Key surprises vs plan:**
1. Weights v74-77 are ALREADY hoisted to registers across the member loop — the
   F1 LDS round-trip costs ~nothing in the dot phase. Its only real cost is the
   8.5 KiB LDS (occupancy: 12 waves/SIMD is LDS-bound). S0b still worth it for
   occupancy, not for issue count.
2. **NEW lever S0d**: specialize the member loop for full chunks
   (`n_members == 32`) with `#pragma unroll` → static accumulator indices →
   kills all movrel + M0 traffic. Dynamic-n fallback loop kept for tail chunks.
   At B=1024 most work items are full 32-member chunks.

Attribution settled statically: per-member overhead (movrel + scale chain +
LDS waits), NOT dequant. Subwarp/chunk-64 plan stays rejected; ablation-kernel
trio downgraded to optional (ISA already gives the answer). Proceeding to S0.

Post-S0 predicted member iteration: 1× `ds_load_b64` (aligned q8) + 1 yd load
+ 4 dot4 + 2 cvt + 2 mul + 2 fma ≈ 10 VALU + 2 LDS, no movrel → ~1.6× issue
reduction in the 82%-of-wall dot phase.

## 2026-06-09 — Baseline e2e bench at HEAD 0959f65 (PIPELINE_LANES=2, B=1024/iter)

| depth | tok/s (best iter) |
|---|---|
| 4096 | 229.7 |
| 8192 | 228.6 |
| 16384 | 223.4 |
| 32768 | 224.4 |

Log: /tmp/bench_baseline_0959f65.log (3 iters each, ±0.5%).

## 2026-06-09 — S0 staged_v2: NEGATIVE result, valuable attribution

Isolated A/B (bench_iq2_isolated, BENCH_B=1024 WI=256 CHUNK=32, interleaved,
min-of-20):

| variant | ms | VGPR | waves/SIMD | notes |
|---|---|---|---|---|
| staged (prod) | **78.0** | 97 | 12 | unrolled ×4 by compiler, lgkmcnt(1) pipelining |
| v2 full-unroll (S0d=1) | 115.2 | 157 | 9 | batched loads, no movrel — occupancy cliff |
| v2 no-unroll | 89.5 | ~95 | 12 | 2× lgkmcnt(0) stalls per member |
| v2 unroll-4 | 81.3 | 95 | **16** | best v2; still +4% vs staged |
| v2 unroll-8 | 97.8 | — | — | worse |

PMC (iters=3, totals over 4 dispatches; same SQ_WAVES 2.097e6):

| | staged | v2-u4 |
|---|---|---|
| SQ_INSTS_VALU | 2.688e10 | 2.595e10 (−3.5%) |
| SQ_BUSY_CYCLES | 1.828e10 | 1.886e10 (+3.2%) |
| SQ_INSTS_LDS | 3.17e9 | 3.83e9 (+21%! unexplained) |
| LDSBankConflict | ~0 | 0 |

**Conclusions:**
1. The dot loop is NOT VALU-issue-bound: −18% dot VALU + 16-vs-12 waves still
   ran +4% slower. It is **LDS round-trip latency bound** — each member's fma
   waits lgkmcnt on its q8/yd loads; hiding is already saturated at 12 waves.
   Full-unroll proved it: maximal load batching with 9 waves = disaster;
   latency hiding (occupancy + pipelining), not issue count, is what pays.
2. Scale-chain hoist, movrel elimination, aligned b64 q8, −5.4 KiB LDS: all
   real, all worth ~nothing in time. The staged kernel is at a sharp local
   optimum for its geometry.
3. ATT on gfx1151: SIGABRT in DispatchThreadTracer::resource_deinit, then
   timeout-hang with --att-consecutive-kernels. Confirmed unusable (matches
   [[rocprofv3-rdna4]] + user's warning). PMC singles DO work on gfx1151:
   use `nix-shell -p rocmPackages.rocprofiler-sdk` + rocmPackages.rocprof-trace-decoder
   for the (broken) ATT lib.
4. Chunk-size increase (user suggestion): the latency-bound story says bigger
   chunks don't reduce per-member LDS round-trips, and chunk=64 doubles
   accumulator VGPRs (→ the full-unroll occupancy cliff, measured −48%).
   Will revisit only if kwide's profile says otherwise.

**Status: S0 retired** (kept as IQ2_VARIANT=staged_v2 opt-in, IQ2_V2_UNROLL=4).

**Implication for S1 kwide:** its real win is not "45% fewer VALU" but
**4× fewer LDS round-trips per dot4** (1× b128 feeds 8 dot4s vs b64+b32 per 4).
Design adjustments from the S0 data:
- SB_GROUP=2 (lane owns 16 weights, half sub-block): keeps LDS at ~19.4 KiB →
  6 WGs/WGP = 12 waves/SIMD (SB_GROUP=4 would need 32 KiB → 8-10 waves, the
  proven-bad zone).
- s_q8 layout [member][sbh][16×uint4] (member-major interleave) — [sbh][member]
  layout makes half-waves collide on banks (computed, not measured).
- No s_partial: fold swiglu+output into the post-reduce lane-0 loop (−2 KiB).
- Member loop #pragma unroll 4 (empirically the pipelining sweet spot).

## 2026-06-09 — S1 kwide: +30% e2e. Floor target hit.

Kernel `iq2_xxs_pair_matvec_fused_swiglu_kwide` (IQ2_VARIANT=kwide): warp=row
unchanged; lanes split a PAIR of super-blocks — lane owns 16 weights (half
sub-block), one ds_load_b128 feeds 8 dot4s. 110 VGPR / 19840 B LDS / 12
waves/SIMD / no spills. Member-major s_q8 [mi][sbh][16×uint4] (the [sbh][mi]
layout would 2-way bank-conflict the half-waves). Swiglu fused into reduce
loop (no s_partial). Member loop #pragma unroll 4.

Isolated (B=1024, WI=256, chunk=32, min-of-20, interleaved):
  staged 78.3 ms → kwide **50.6 ms (−35.4%)**

Oracle (varying xq-d AND weight-d, full 32-member + partial 19-member chunks):
  kwide rel=1.1e-4, staged_v2 rel=7.9e-5 (tol 5e-3) ✓

PMC vs staged: VALU 2.69e10→2.11e10 (−22%), LDS insts 3.17e9→2.49e9 (−22%),
SQ_BUSY 1.83e10→1.18e10 (−35%). IPC 1.47→1.78. Now ~1.56× over dot4-only
roofline (was 2.4×).

**E2e cold A/B (PIPELINE_LANES=2, B=1024/iter, back-to-back):**

| depth | staged | kwide | Δ |
|---|---|---|---|
| 4096  | 230.3 | **299.9** | +30.2% |
| 8192  | 228.9 | **298.0** | +30.2% |
| 16384 | 223.9 | **292.4** | +30.6% |
| 32768 | 224.7 | **293.6** | +30.7% |

Remaining iq2 ideas (post-S2, diminishing): SBG=4 (32 KiB LDS → 8-10 waves,
risky), q8-direct-from-L2 (no staging/barriers), staging-loop tuning.

## 2026-06-09 — S2 q2k kwide: bit-exact, e2e +5% → 316 tok/s combined

`q2_k_matvec_par_by_expert_kwide` (Q2K_VARIANT=kwide): loops inverted —
weight quarter unpacked ONCE per (block, lane) with 4-bit group scale folded
at unpack (byte-safe: 3·15 < 256), f16 d/dmin converted once, per-lane member
accumulators pacc[32], one warp-reduce per member at the end. 92 VGPR / 0 LDS /
16 waves / no spills.

- Oracle vs by_expert (random sc/q2/d/dmin/bsums, full+partial chunks):
  **bit-exact** (max_abs_diff = 0.0) — folded integer scale is associative.
- Isolated (B=1024, chunk=32): 22.2-23.3 → **19.8 ms (−12%)**. Less than the
  VALU math promised → q2k is substantially cache-BW-bound on cross-WG q8
  re-reads (each row-WG re-pulls members' midq from L2/MALL). LDS staging
  can't fix cross-WG redundancy; geometry change (more rows/WG) is the tile8
  trap. Accepting ~19.8 ms as near-floor for this shape.
- E2e (IQ2=kwide fixed, q2k by_expert → kwide): 301.8 → **316.6 tok/s** @4K.

**Combined state: baseline 230 → 316.6 tok/s @4K (+37.6%); ≥309 at 16K/32K.**

Wall composition @4K (3234 ms): iq2 kwide ~2176 ms (67%), q2k kwide ~851 ms
(26%), other ~200 ms. iGPU still the pipeline wall. Next iq2 levers: member
unroll sweep, SBG=4 (more k per lane, 32 KiB LDS occupancy risk), q8-direct
from L2 (drop staging+barriers).

## 2026-06-09 — kwide tuning round: all knobs flat or negative; 50.5 ms stands

- Member-loop unroll {2,4,8}: 50.5/50.6/50.5 ms — no longer pipelining-limited.
- Full-chunk static unroll (movrel removal): 168 VGPR → 9 waves → 71.5 ms.
  With `__launch_bounds__(256,12)` cap: 64 VGPR + 272 B scratch SPILL → 90.6 ms.
  The ~10% movrel tax is structurally locked (dynamic member index needs
  movrel or VGPR explosion). Knob `IQ2_KW_FULLUNROLL` stays 0.
- q8-direct-from-L2: rejected on paper — LDS staging dedups 8× within the WG;
  direct reads would 8× the L2/MALL traffic (~68 GB/launch).
- SBG=4: needs 32+ KiB LDS → ≤8-10 waves (proven-bad zone); SBG=4+chunk16
  pencils to ~+3% net. Not pursued.

PMC says kwide is VALU-issue-bound at IPC 1.78 (issue 10,040/wave ≈ busy
5,635 cyc × dual-issue). Composition/wave ≈ 3,584 dot+tail, ~1k staging,
~1k movrel, ~400 dequant, rest addressing/loop. ~1.56× over dot4-only roofline.

## 2026-06-09 — GATES + PROMOTION

1. Oracles: iq2 kwide rel=1.1e-4, staged_v2 rel=7.9e-5, q2k kwide BIT-EXACT ✓
2. Cold e2e sweep (PIPELINE_LANES=2, B=1024/iter), kwide+kwide vs 0959f65
   baseline, back-to-back:

| depth | baseline | kwide | Δ |
|---|---|---|---|
| 4096  | 229.7 | **320.4** | +39.5% |
| 8192  | 228.6 | **319.5** | +39.8% |
| 16384 | 223.4 | **311.7** | +39.5% |
| 32768 | 224.4 | **312.5** | +39.3% |

3. Occupancy gate: iq2 kwide 110 VGPR/12 waves, q2k kwide 92 VGPR/16 waves,
   no spills ✓
4. Real-server gate: deepstrix-server with kwide variants, 3371-token cold
   prefill → **297 tok/s** in production logs (`tok_per_s="297.0"`);
   completions coherent AND content-accurate on both short and long prompts
   (correctly summarized the ejpir doc). The tile8 `content: null` scare
   reproduced once but was max_tokens exhaustion (160 < reasoning length),
   not a kernel bug — fine at max_tokens=1200. ✓
5. **Defaults flipped**: IQ2_VARIANT=kwide, Q2K_VARIANT=kwide
   (forward_prefill.rs). Verified default-config bench: 319.4 @4K, 311.3 @32K.

## Remaining headroom (for a future session)

Wall @4K ≈ 3200 ms: iq2 kwide ~2176 (68%), q2k ~851 (27%), other ~200.
- iq2 at perfect dot4 roofline → wall ~2560 → ~400 tok/s. The 1.56× residual
  is tail ops (cvt/mul/fma per member-pair) + staging + movrel + dequant; each
  individually 5-12%, all measured sticky.
- Past ~400 needs the ejpir FAST_FULL structural route (per-chunk f16 hot-
  expert cache + WMMA): eliminates per-member tail AND dequant from inner
  loop, pays 4× weight BW + cache build. Plausible but a multi-day rewrite.
- q2k is cache-BW-bound on cross-WG q8 re-reads; geometry changes are the
  tile8 trap. Near floor for this shape.
- dGPU becomes the wall only below ~1250 ms iGPU — far away.

## 2026-06-09 — Post-promotion: 64K/96K depth A/B (gate extension)

| depth | staged+by_expert | kwide+kwide | Δ |
|---|---|---|---|
| 65536 | 225.8 | **290.8** | +28.8% |
| 98304 | 223.4 | **261.1** | +16.9% |

No regression at long context (tile8 check extended to 96K). The shrinking
delta is the predicted pipeline crossover: staged is FLAT 64K→96K (iGPU so
slow it hides all depth-dependent dGPU work), kwide drops 291→261 — at ~96K
the dGPU attention/indexer side stops being fully hidden. **Long-context
prefill optimization now shifts to the dGPU attention side; the iGPU MoE
levers only pay below ~64K.**
