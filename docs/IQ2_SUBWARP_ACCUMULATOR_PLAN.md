# Path to 280–290 t/s prefill: sub-warp partitioned accumulator for iq2/q2k

**Date:** 2026-06-08
**HEAD when authored:** `3d9cdd0` (post ATTN_SCORES_STRIDE + B_MAX=1024, baseline 230 t/s cold-cache T=4K–32K)
**Target:** 270–290 t/s cold-cache, no model-file change, no persistent cache infrastructure
**Effort:** ~1–2 days kernel work + ~half day bench/tune

---

## 1. The constraint we're breaking

The staged kernel `iq2_xxs_pair_matvec_fused_swiglu_chunked_staged` amortizes
weight dequant cost across `CHUNK_SIZE` members per WG. At chunk=32, the
dequant work per output element is `D/32 + M` where `D` = per-(row,super-block)
dequant cost and `M` = per-(member,row,super-block) matmul cost.

From [project_iq2_roofline_2026-06](../../.claude/projects/-home-claude-code-deepstrix/memory/project_iq2_roofline_2026-06.md):
iq2 fused runs at **2.4× over compute roofline**. Solving `D/32 + M = 2.4M`
gives `D = 44.8M` — so the dequant cost is ~45× the per-element matmul work,
amortized 32 ways.

**Bigger chunk = better amortization.** At chunk=64: per-element work drops to
`D/64 + M = 0.7M + M = 1.7M`, a **32% iq2 cost reduction**. At chunk=128:
`D/128 + M = 1.35M`, a **46% reduction**.

The reason we currently can't go past chunk=32: the kernel holds an array of
`chunk` f32 partial sums *per lane* in VGPRs:

```c
float lane_pacc_g[IQ2_STAGED_MAX_CHUNK];   // 32 f32/lane at chunk=32
float lane_pacc_u[IQ2_STAGED_MAX_CHUNK];   // 32 f32/lane
```

At chunk=32, the kernel already uses 104 VGPR/wave (per `iq2-bottleneck` memory).
At chunk=64 the accumulator alone would need 128 VGPR/lane → crosses the
**128-VGPR occupancy cliff** (3 WGs/CU → 1 WG/CU). The exact regression that
killed M40-P8 per [feedback_fusion_wg_geometry](../../.claude/projects/-home-claude-code-deepstrix/memory/feedback_fusion_wg_geometry.md).

LDS would fit 64+ easily; it's per-lane register pressure that breaks first.

## 2. The fix: distribute the accumulator across the warp

Each warp owns one row of output. Within the warp, 32 lanes split the
`chunk_size` member accumulators between them:

```
lane L owns members where (member_idx % 32) == L
```

At chunk=64, each lane owns 2 members → **2 VGPRs/lane per matrix → 4 total**.
At chunk=128, 4 VGPRs/lane → 8 total. At chunk=256, 8 VGPRs/lane → 16 total.

Plenty of headroom either way.

The dp4a stays int8-on-int8 (no quantization format change). The accumulator
distribution mirrors what WMMA does automatically with its 16×16 C-fragment
layout — we just do it manually to keep the existing kernel structure and
avoid the f16 conversion overhead WMMA would require.

### Per-iteration cost change

**Current**: Per (super-block, member): each lane's dp4a accumulates to its own
`lane_pacc[member]` register. Warp-reduce happens ONCE per member at the very
end of the kernel.

**New**: Per (super-block, member): each lane's dp4a contributes to the same
member's accumulator. Cross-lane warp-reduce *every iteration* gathers all 32
lanes' contributions, then the **owner lane** adds the reduced value to its
register.

Extra cost: `n_super_blocks × chunk × 5 shfl` per WG = 16 × 64 × 5 = 5120
shuffles at chunk=64. ~1.5 μs per WG of shuffle overhead vs the dequant
amortization savings:

| chunk | dequant savings | shuffle overhead | net |
|-------|-----------------|------------------|-----|
| 64    | -32% (kernel)   | +0.5%            | **-31.5%** |
| 96    | -42%            | +0.7%            | **-41.3%** |
| 128   | -46%            | +1.0%            | **-45.0%** |

## 3. End-to-end math

Per `rocprofv3 --kernel-trace` at T=8K cold (HEAD 3d9cdd0):

| kernel | total ms | per (layer,chunk,lane) |
|---|---|---|
| iq2 staged | 27022 | 19.6 ms |
| q2k_by_expert | 8203 | 5.96 ms |
| iGPU MoE wall | ~35225 | 25.6 ms |

At chunk=64 with the new accumulator layout:
- iq2: 19.6 × 0.685 = **13.4 ms** (-6.2 ms)
- q2k (same trick, q2k roofline ~2.6× → D=51M, chunk=64 ratio 0.71): 5.96 × 0.71 = **4.23 ms** (-1.73 ms)
- Savings: **~7.9 ms per (layer,chunk,lane)**

Across 8K prefill: 8 ms × 43 layers × 16 chunks × 2 lanes = ~11 seconds saved
on 37s wall → **24-27s wall → 300-340 t/s**.

Conservative (warp-reduce cost 2× estimate, partial dequant savings, cold-cache
effects): **270–290 t/s**.

The 300 t/s line is plausibly in reach with chunk=96 if the kernel restructure
goes clean.

## 4. Why this beats WMMA for our case

| dimension | sub-warp partition + dp4a | WMMA f16 |
|---|---|---|
| FLOPS peak on gfx1151 | dp4a-IU8 peak | Same (per `iq2-compute-bound` memory) |
| Dequant amortization | ✓ same | ✓ same |
| Dequant cost per weight | ~5 ops (to int8) | ~7 ops (to f16, has scale conversion) |
| Activation handling | xq stays Q8_K | Need xq → f16 pre-pass kernel |
| Kernel restructure size | Modify existing staged | New kernel from scratch |
| Engineering risk | Low — incremental | Higher — WMMA fragment layout is finicky |
| Effort estimate | ~1–2 days | ~3–4 days |

WMMA only pulls ahead when paired with a persistent f16 weight cache (ejpir's
trick), which is a separate, larger project. For *per-chunk* fused dequant+matmul,
dp4a is cheaper to land and gets us the same amortization win.

## 5. Implementation phases

### Phase 1: subwarp variant in new kernel symbol — ~6 hours

Add `iq2_xxs_pair_matvec_fused_swiglu_subwarp` next to `_chunked_staged` in
`crates/v4flash-kernels/kernels/iq2_xxs_pair_matvec_par.hip`. Same signature so
it's drop-in replaceable. Hardcode `chunk_size=64` initially.

Per-WG accumulator state changes:
```c
// OLD: per-lane chunk-size array
float lane_pacc_g[64];   // 64 f32/lane → 64 VGPRs/lane → KILLS occupancy

// NEW: per-lane owns chunk/32 members
float lane_pacc_g[CHUNK / 32];   // 2 f32/lane at chunk=64
float lane_pacc_u[CHUNK / 32];
```

Inner loop change at the dot phase:
```c
for (unsigned int mi = 0; mi < n_members; mi++) {
    // dp4a contributions (each lane has different weights / activations)
    int32_t sumi_g = sudot4_par(w_g0, q8_0, 0);
    sumi_g = sudot4_par(w_g1, q8_1, sumi_g);

    // Warp-reduce sumi_g across all 32 lanes (5 shfl_xor steps)
    sumi_g = warp_sum_i32_par(sumi_g);

    // Only the owner lane accumulates
    const unsigned int owner = mi & 31u;
    if (lane == owner) {
        const unsigned int slot = mi >> 5;  // local accumulator index
        lane_pacc_g[slot] += (float)(sumi_g * ls_g_lane) * yd * gd;
    }
}
```

Final write-out: each lane writes its owned members. Lane L writes
`lane_pacc[mi >> 5]` for `mi` in `{L, L+32, L+64, ...} ∩ [0, n_members)`.

### Phase 2: oracle test — ~3 hours

`crates/v4flash-kernels/tests/iq2_subwarp_oracle.rs` — same shape as
`iq2_tile8_oracle.rs`. Validates `_subwarp` against `_chunked_staged` on
random inputs with **per-(token, super-block) varying d** (the test we
hardened after the tile8 d-cache bug). Expect bit-exact for integer order,
≤5e-3 relative diff for f32 reduction order differences.

### Phase 3: bench A/B + tune chunk_size — ~3 hours

```bash
for V in staged subwarp; do
  for T in 8192 32768; do
    PIPELINE_LANES=2 BENCH_T=$T BENCH_WARMUP=0 IQ2_VARIANT=$V \
      cargo test --release -p v4flash-kernels --test bench_prefill bench_prefill_chunked -- --ignored
  done
done
```

Sweep `CHUNK_SIZE` in `forward_prefill.rs` over {64, 96, 128}. Watch for:
- VGPR pressure crossing 128 cliff (likely at chunk=128 — check `--save-temps` GCN)
- Cold-cache regression at any depth (the gotcha that bit tile8)
- Compile-time success (HIP may refuse some shapes)

### Phase 4: same treatment for q2k — ~6 hours

`q2_k_matvec_par_by_expert` has the same structural pattern. Apply identical
sub-warp accumulator partition. New kernel symbol `_subwarp` flavor, swap in
behind `Q2K_VARIANT` env when verified.

### Phase 5: wire + ship — ~3 hours

- Add `IQ2_VARIANT=subwarp` dispatch path in `forward_prefill.rs`
- Update `project_current_state.md` with new baseline
- Default `subwarp` only after a full e2e text-quality run against the running
  server (the lesson from tile8)

**Total: ~1–2 focused days**.

## 6. Risks

| risk | mitigation | severity |
|------|-----------|----------|
| Per-iter warp-reduce overhead > dequant savings | Phase 3 chunk-size sweep; abort if no net win at 64 | Med |
| VGPR pressure higher than estimated → occupancy regression | Check `--mllvm -print-regalloc` output during compile; tune chunk down | Low |
| Cold-cache regression (tile8 pattern) | A/B at T=4K, 8K, 16K, 32K BEFORE shipping | High |
| q2k structure doesn't match — different kernel shape | Drop q2k port if it doesn't pencil; iq2 alone is ~75% of the win | Low |
| Production routing makes chunk=64 wasteful (most experts have <32 members) | Profile actual `group_count` distribution at B=1024; fall back to chunk=32 in dispatch for layers with no hot experts | Med |

## 7. What this does NOT do

- **Does not implement persistent dequant caching.** That's a separate path
  (per-prefill or per-session) and worth maybe +10-15% more if added on top.
- **Does not unlock 300+ t/s by itself.** The math says 270-290; the structural
  cache work is what gets us to 320+.
- **Does not change model file or quantization format.** Stays on
  IQ2_XXS + Q2_K + Q8_K activations.
- **Does not affect decode** — decode uses different iq2/q2k kernels and
  this work is prefill-only.

## 8. Files touched (estimate)

- `crates/v4flash-kernels/kernels/iq2_xxs_pair_matvec_par.hip` — new kernel ~250 lines
- `crates/v4flash-kernels/kernels/q2_k_matvec_par.hip` — new kernel ~150 lines (phase 4)
- `crates/v4flash-kernels/src/iq2_xxs.rs` — new launch wrapper ~50 lines
- `crates/v4flash-kernels/src/q2_k.rs` — new launch wrapper ~50 lines
- `crates/v4flash-kernels/src/het/forward_prefill.rs` — dispatch branch ~30 lines
- `crates/v4flash-kernels/tests/iq2_subwarp_oracle.rs` — new test ~180 lines
- `crates/v4flash-kernels/tests/q2k_subwarp_oracle.rs` — new test ~180 lines

## 9. Pre-work before starting

1. **Measure actual routing distribution at B=1024**: profile a real prompt
   through the running server, log per-layer `group_count[256]` to verify the
   Zipf assumption (top 30 experts get 70%+ of activations, average hot-expert
   member count >=64). Use the existing
   [project_m50_phase3_expert_stats](../../.claude/projects/-home-claude-code-deepstrix/memory/project_m50_phase3_expert_stats.md)
   instrumentation. If most experts have <32 members at B=1024, the chunk-size
   bump won't deliver — and we should reconsider.

2. **Profile current kernel's VGPR usage precisely**:
   ```bash
   nix develop --command hipcc --offload-arch=gfx1151 -O3 -Rpass-analysis=kernel-resource-usage \
     crates/v4flash-kernels/kernels/iq2_xxs_pair_matvec_par.hip 2>&1 | grep -A2 'staged'
   ```
   Confirms baseline VGPR count, sets margin for the new variant.

3. **Bench harness baseline run** at HEAD 3d9cdd0 across T={4K,8K,16K,32K,64K}
   so we have a clean BEFORE for the eventual A/B.

## 10. Related work

- [project_iq2_compute_bound](../../.claude/projects/-home-claude-code-deepstrix/memory/project_iq2_compute_bound.md) — full
  history of iq2 optimization attempts, including the rejected WMMA-IU8 path
  and the staged kernel that's our baseline
- [project_iq2_roofline_2026-06](../../.claude/projects/-home-claude-code-deepstrix/memory/project_iq2_roofline_2026-06.md) — roofline
  math showing iq2 is at 2.4× over compute peak (the headroom we're closing)
- [EJPIR_DS4HIP_PREFILL_ANALYSIS.md](EJPIR_DS4HIP_PREFILL_ANALYSIS.md) — the
  upstream fork's FAST_FULL recipe; theirs uses persistent f16 cache + WMMA,
  ours skips the cache to keep the change small
- [feedback_fusion_wg_geometry](../../.claude/projects/-home-claude-code-deepstrix/memory/feedback_fusion_wg_geometry.md) — the
  M40-P8 cautionary tale of pushing per-WG state past the occupancy cliff
- [project_iq2_tile8_2026-06](../../.claude/projects/-home-claude-code-deepstrix/memory/project_iq2_tile8_2026-06.md) — the
  ejpir tile8 port that warm-benched well and cold-regressed; reminder to
  validate on cold cache before promoting any new variant

## 11. Decision point

Ship the iq2 subwarp variant ungated as the new default if:
- Phase 3 shows positive cold-cache delta at T={4K, 8K, 16K, 32K, 64K}
- No depth regresses
- Oracle bit-matches to within 5e-3 relative

Otherwise: keep behind `IQ2_VARIANT=subwarp` opt-in alongside `staged` (current
default).
