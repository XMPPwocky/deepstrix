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

## Next: S1 kwide implementation
