# Laguna-S-2.1 — Decode Per-Kernel Scorecard (2026-07-26)

**HEAD** `ee557a0` (master).  Measurement, not a code change.
**Method**: `rocprofv3 --kernel-trace` (7.2.3) on `laguna_decode_bench`, ctx
4096 and 32768, greedy decode at jumped KV positions.  Fresh binary
`laguna_decode_bench-ed8b11944aa9f3fb`.  Parity gate passed (token 22718).
The prefill-WMMA rocprofv3 abort did **not** trigger — decode uses scalar
matvec + split-KV attention, and the 5-token parity prefill is tiny; the trace
completed cleanly (only benign timestamp-swap warnings).

**Config of the traced run**: bench loads **K=0 hot experts** (no
`LAGUNA_HOT_EXPERTS_DGPU`), so *all* routed MoE runs on the iGPU (the
`layer_moe` non-split path).  This is the pure-roofline picture.  The server
runs K=6–8 hot experts mirrored to the dGPU, which rebalances iGPU↔dGPU (see
levers).

**Wall (from the traced run, self-consistent denominator):**
- ctx 4096:  36.7 ms/token  (27.2 tok/s)
- ctx 32768: 40.1 ms/token  (25.0 tok/s)

(A separate `LAGUNA_HET_DIAG=1` run read 39.8 / 43.3 ms — diag adds explicit
host syncs; the rocprof-traced wall is the truthful one and is what the kernel
% are normalized against.)

## Normalization

29 forward passes were traced: 5 low-KV parity-prefill passes (used
`gqa_attn_single_query`) + 4 warmup + 20 timed decode passes.
Position-independent kernels ran on all 29 → per-token = total/29.  The
high-KV decode attention (`gqa_attn_decode_partial_hg` + `_combine`) ran only
on the 24 real decode passes → per-token = total/24.  Verified against
structure: partial_hg count 1152 = 48 layers × 24; router 1363 = 47 MoE
layers × 29; wide matvec 6612 = (4×48 + 36 SWA-gate) × 29.

Model: 48 layers (12 global `il%4==0`: n_head=48, full KV; 36 SWA: n_head=72,
512-key window), layer-0 dense, 256 experts / TOPK=10, FF_EXP=1024, HIDDEN=3072.

---

## The dGPU and iGPU run essentially SEQUENTIALLY in decode

Per-token GPU-busy: dGPU 15.5 ms + iGPU 16.1 ms = 31.6 ms; wall 36.7 ms.
The only overlap is the dGPU **shared expert** (q4/q6 `dense_gemv`, ~2.2 ms)
hiding behind the iGPU routed experts.  Everything else is serialized by the
per-layer dependency chain: dGPU attention front-half → push `fn_in` → iGPU
routed experts → push `ffn_out` → dGPU combine → next layer.  **So iGPU MoE is
EXPOSED (on the critical path), not hidden.**  This is the central correction
to the old "decode is dGPU-attention-bound" framing (that was V4-Flash, and
predates the attention speedup).

---

## Scorecard — ctx 4096 (36.7 ms/token)

| kernel | device | % wall | exp/hid | binding roofline | achieved | % of roofline | verdict |
|---|---|---:|---|---|---:|---:|---|
| f16_matvec_wide_vec (Q/K/V/O + SWA gate proj) | dGPU | **25.5%** | exposed | DRAM BW (600 GB/s) | 597 GB/s | **99%** | AT ROOFLINE / CAPPED |
| q4_k_pair_matvec_fused_swiglu (cold gate+up) | iGPU | **20.8%** | exposed | LPDDR5X BW (~230 GB/s) | 218 GB/s | **95%** (85% of 256) | AT ROOFLINE / CAPPED |
| q6_k_matvec_par_batched (cold down, Q6_K) | iGPU | **11.0%** | exposed | LPDDR5X BW (~230 GB/s) | 147 GB/s | **64%** | FIXABLE |
| q4_k_matvec_par_batched (cold down, Q4_K) | iGPU | **6.0%** | exposed | LPDDR5X BW (~230 GB/s) | 192 GB/s | **84%** | AT ROOFLINE |
| laguna_router_scores | iGPU | 3.5% | exposed | latency/occ | — | — | small |
| gqa_attn_decode_partial_hg | dGPU | 3.3% | exposed | latency-bound (KV=276 MB tiny) | 226 GB/s | 38% of 600 | CAPPED (latency, not BW) |
| q4_k_dense_gemv (shexp/dense0/logits) | dGPU | 3.2% | **mostly hidden** | — | — | — | hidden |
| q6_k_dense_gemv (shexp/logits) | dGPU | 2.9% | **mostly hidden** | — | — | — | hidden |
| rms_norm_weighted | dGPU | 2.0% | exposed | latency | — | — | small |
| gqa_attn_decode_combine | dGPU | 1.2% | exposed | latency | — | — | small |

## Scorecard — ctx 32768 (40.1 ms/token)

Only attention changes (global layers now stream 32768 keys); everything else
is position-independent and identical.

| kernel | device | % wall | exp/hid | binding roofline | achieved | % roofline | verdict |
|---|---|---:|---|---|---:|---:|---|
| f16_matvec_wide_vec | dGPU | 23.7% | exposed | DRAM BW 600 | 590 GB/s | 98% | AT ROOFLINE / CAPPED |
| q4_k_pair (cold gate+up) | iGPU | 19.1% | exposed | ~230 GB/s | 218 GB/s | 95% | AT ROOFLINE / CAPPED |
| gqa_attn_decode_partial_hg | dGPU | **10.4%** | exposed | DRAM BW 600 | 406 GB/s | **68%** (isolated ≈77%) | near roofline |
| q6_k_matvec_par_batched (cold down) | iGPU | 10.1% | exposed | ~230 GB/s | 147 GB/s | 64% | FIXABLE |
| q4_k_matvec_par_batched (cold down) | iGPU | 5.5% | exposed | ~230 GB/s | 192 GB/s | 84% | AT ROOFLINE |
| gqa_attn_decode_combine | dGPU | 1.6% | exposed | latency | — | — | small |

---

## Roofline arithmetic (shown, not asserted)

**f16_matvec_wide_vec** — B=1 f16 weight-streaming matvec, weight-BW-bound.
Reads wq,wk,wv,wo on all 48 layers + gate on 36 SWA layers, f16 (2 B/elem):
- global (12 lyr, n_head=48, n_embd_q=6144): 3072·(6144+1024+1024+6144)=44.19M elem
- SWA (36 lyr, n_head=72, n_embd_q=9216): 3072·(9216+1024+1024+9216)+72·3072=63.36M elem
- total = 2.80G elem × 2 B = **5.603 GB/token**
- 4K: 5.603 GB / 9.38 ms = **597 GB/s = 99.6% of 600 GB/s** → CAPPED
- 32K: /9.50 ms = 590 GB/s = 98%.  The single largest big call (wq SWA, 56.6 MB)
  runs in ~94 µs = 602 GB/s — literally at the DRAM ceiling.

**q4_k_pair (cold gate+up, Q4_K, 10 experts × 47 MoE layers)** —
2·(1024·3072/256)·144 B/expert · 10 · 47 = **1663 MB/token**.
1663 MB / 7.64 ms = **218 GB/s** = 95% of 230 (85% of the 256 GB/s nameplate).
This fused kernel *is* the practical iGPU BW ceiling. CAPPED.

**q6_k_down (Q6_K, 10 exp × 23 layers)** — (3072·1024/256)·210 B ·10·23 =
**594 MB/token**.  594 / 4.05 ms = **147 GB/s = 64% of 230** → FIXABLE
(same iGPU, q4_k_pair hits 218 GB/s; the `par_batched` split+reduce at B=1
leaves ~35% on the table).

**q4_k_down (Q4_K, 10 exp × 24 layers)** — 425 MB / 2.21 ms = 192 GB/s = 84%.
Near ceiling.

**gqa_attn_decode_partial_hg** — GQA-decode reads K+V once per KV head
(8·128·2·2 B/key), global layers full-KV + SWA 512:
- 4K: 277 MB / 1.23 ms = 226 GB/s = 38% of 600 → **latency-bound, not BW-bound**
  (KV too small to saturate; 3.3% of wall, don't touch).
- 32K: 1686 MB / 4.15 ms = 406 GB/s = 68% of 600 (isolated bench measured
  ~77%; e2e is diluted by the `_combine` reduction + dispatch).  Near roofline.

---

## Non-kernel residual

Wall 36.7 ms − (dGPU busy 15.5 + iGPU busy 16.1) = **5.1 ms (14%) minimum**
non-kernel time, assuming zero device overlap.  Because the dGPU shared expert
(~2.2 ms) *does* overlap the iGPU experts, the true critical-path GPU-busy is
lower and the real residual is **~15–20% (≈6–7 ms/token)**.  The peer copies
themselves are cheap (`__amd_rocclr_copyBuffer` ≈ 0.29 ms/tok each side), so the
residual is **dispatch gaps + cross-device event-wait serialization** across the
48-layer ping-pong (2 handoffs + 2 event waits per layer × 48).  This is a real,
optimizable chunk — not rounding error.

---

## What actually contradicts the old scorecard

- **Old: "decode attn 43.6% of per-token time." NOW: 3.3% @4K, 10.4% @32K.**
  The 1.65× on `gqa_attn_decode_partial_hg` *plus* the split-KV/HG structure
  demoted attention from the #1 cost to a minor one at 4K. Attention is no
  longer the decode bottleneck at any traced context.
- **Old: "f16_matvec projections 40%." NOW: 25.5% @4K — and it is at 99% of the
  DRAM roofline.** It is the largest single kernel, but there is **zero
  kernel-level headroom**; it is byte-bound. The old figure implied room to
  optimize the kernel; there is none.
- **Old framing "decode is dGPU-bound / iGPU idle."** With K=0 hot experts the
  iGPU routed MoE is ~41% of the wall and **exposed on the critical path**. The
  two devices run sequentially, not overlapped.

---

## Ranked levers (Amdahl: kernel share × plausible speedup)

1. **Quantize the f16 attention projection weights → Q8_0.** 25.5% of wall, at
   99% DRAM BW → the *only* way to cut it is fewer bytes. Q8 halves the 5.6 GB
   → ~9.4 ms would drop to ~4.9 ms. **Amdahl e2e ≈ +12% (27→30+ tok/s).**
   Biggest single opportunity, but it's a precision/format change (needs an
   accuracy gate), not a kernel tweak.
2. **Load hot experts on the dGPU (K=6–8, the existing het-split).** The two
   iGPU BW-capped expert kernels (q4_k_pair 20.8% + downs 16%) are at their
   *iGPU* roofline. Moving the K hottest experts to the dGPU relocates that
   traffic from 230 GB/s LPDDR5X to the 600 GB/s / mostly-idle-during-MoE dGPU.
   This is the highest-ROI *shippable* lever (no accuracy risk). Expected to
   convert a chunk of the ~13 ms exposed iGPU expert time into hidden/dGPU time.
   **Confirm on-hardware — this bench was K=0 and understates it.**
3. **Fix `q6_k_matvec_par_batched` down (64% → ~90% of iGPU BW).** 11% of wall,
   FIXABLE. The `par_batched` split+reduce wastes ~35% at B=1 vs the sibling
   q4_k_pair at 95%. Bring to q4-down parity → 4.05→~2.9 ms. **Amdahl ≈ +3% e2e.**
4. **Cut the 15–20% dispatch/handoff residual.** 48-layer cross-device
   ping-pong with per-layer event waits. Graph-capture the decode step or fuse
   handoffs → recover several ms. Architectural but large.
5. **Attention: STOP at 4K, marginal at 32K.** 3.3% @4K (latency-bound, tiny)
   — the 1.65× is banked, leave it. @32K it's 10.4% at ~68–77% roofline;
   pushing to 90% saves ~1 ms = ~2.5% e2e. Low priority.

**Bottom line:** decode is no longer attention-bound. It is split roughly
evenly between (a) dGPU f16 attention projections and (b) exposed iGPU routed
experts — **both already at ~95–99% of their respective bandwidth rooflines** —
plus a ~15–20% cross-device handoff residual. There is almost **no per-kernel
optimization left**: the real levers are *format* (Q8 attention weights),
*placement* (K hot experts to dGPU), and *scheduling* (kill the handoff
residual). The only genuinely under-roofline kernel worth a direct fix is
`q6_k_matvec_par_batched` (64%), worth ~3% e2e.
