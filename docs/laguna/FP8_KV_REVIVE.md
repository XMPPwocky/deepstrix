# Laguna-S-2.1 — FP8 (e4m3fn) KV-cache revive & re-measure (2026-07-27)

**Branch** `fp8-kv-revive` off `master @ 79c4f99`. Merged the shelved
`stash@{1}` ("fp8-kv-noship") forward over the `ee557a0` decode-attention
rewrite and re-measured decode / prefill / quality with back-to-back A/B on the
gfx1201 dGPU (16 GB) + gfx1151 iGPU. Model: `laguna-s-2.1-Q4_K_M.gguf`
(48 layers, 12 global `il%4==0` full-KV, 36 SWA 512-window; n_kv_head=8,
head_dim=128; 256 experts / TOPK=10).

Format is **e4m3fn** (matches the poolside HF card / vLLM recipe), **not** int8.
Per-`(token, kv_head)` symmetric scale = amax/448 (a f32 sidecar), K and V both
e4m3. Native gfx1201 `cvt_pk_*_fp8` convert; portable LUT fallback for gfx1151.

---

## 1. Conflicts resolved — and the `ee557a0` occupancy win SURVIVES

7 conflicts (2 in `gqa_attention.hip`, 5 in `gqa_attention.rs`), all from the
stash predating (a) the depth-2 half2 prefetch ring and (b) the `ee557a0`
register-O / f16-Qs decode rewrite.

- **`gqa_attention.hip:1369`** (prefill hg ring load): kept upstream's 2-D
  `kbuf[SLOT][_i]` ring, injected the `if constexpr (FP8)` per-element dequant.
- **`gqa_attention.hip:2237`** (decode kernel, on top of `ee557a0`): kept the
  full register-O / f16-Qs / half2 depth-2 ring; the stash's single-buffer fp8
  macro was discarded and fp8 re-injected into the ring's load point.
- **`gqa_attention.rs`**: upstream added `prefill_flash_wmma_fa2_hg_packed`
  at the same insertion point as the stash's three fp8 wrappers; kept BOTH.

**Occupancy win preserved — verified by compile (`-Rpass-analysis`):**

| decode kernel | VGPR | LDS B | occ | spill |
|---|---:|---:|---:|---|
| `gqa_attn_decode_partial_hg` (f16) | 166 | 21136 | 6 w/SIMD | 0 |
| `gqa_attn_decode_partial_hg_fp8`   | 166 | 21136 | 6 w/SIMD | 0 |

**FP8 is byte-identical in resources to f16.** This *contradicts the brief's
guess* that fp8 would shrink the LDS budget: the kernel dequants fp8 → the SAME
f16 `Ks`/`Vs` LDS staging and f16 register ring, so LDS/VGPR/occupancy are
unchanged from `ee557a0` (21136 B, down from the pre-commit 30352 B). FP8 is
**purely a DRAM-read-bytes lever layered on top of** `ee557a0`, not an LDS
change. The occupancy win did not regress.

### Bug found & fixed while merging
The stash's fp8 only ever covered what became the decode **narrow** + **serial**
load paths. The **default b128 wide-load path had NO fp8 branch** — it read 16 B
(8×f16) from a 1-B/elem buffer → all-NaN output + `hipErrorIllegalAddress` at
long ctx. Added the fp8 dequant to the wide path (single 8-B coalesced `uint2`
load + 8× `gqa_fp8_to_f32`). Isolated kernel A/B then passes with 0 NaNs.

---

## 2. KV bytes (arithmetic) — ~48% saved, confirmed by the OOM boundary

Global KV @192K (196640 tok): `12 layers × 196640 × 8 × 128 × 2 B × 2(K+V)` =
**9.665 GB** (f16). SWA layers (cap 512) add only ~0.075 GB.

FP8: K/V → 1 B/elem = **4.83 GB** + f32 scale sidecar
(`12 × 196640 × 8 × 4 B × 2` = 0.15 GB) = **~4.98 GB**.

| | f16 | fp8 | saved |
|---|---:|---:|---:|
| KV @192K (global+SWA) | ~9.74 GB | ~5.02 GB | **~4.72 GB (−48.5%)** |

**Empirically confirmed by the fit boundary** (16 GB dGPU, max_kv=192K):

| config | K=0 | K=4 | K=16 | K=24 |
|---|:--:|:--:|:--:|:--:|
| **f16** @192K | fits | **OOM** | OOM | OOM |
| **fp8** @192K | fits | fits | **fits** | **OOM** |

So at 192K f16 is stuck at **K=0**; fp8's ~4.7 GB frees exactly **K=16** hot
experts (≈4.6 GB) — not K=24. The brief's "buy back K=16 or more" holds at
*exactly* K=16, not more.

---

## 3. Decode tok/s — matched-K A/B (back-to-back, same session)

**Matched K=0 (both fit at all ctx):**

| ctx | f16 K=0 | fp8 K=0 | Δ |
|---|---:|---:|---:|
| 4096   | 31.18 | 31.10 | −0.3% (noise) |
| 32768  | 28.61 | 28.38 | −0.8% |
| 131072 | 23.27 | 21.56 | **−7.3%** |
| 196608 | 20.23 | 18.86 | **−6.8%** |

FP8 parity token at the 5-token prefill = **22718** (exact), both configs.

**The real end-to-end question at long ctx — fp8@best-fit-K vs f16@K=0:**

| ctx | f16 (max K=0) | fp8 K=16 | winner |
|---|---:|---:|---|
| 131072 | **23.27** | 22.05 | f16 |
| 196608 | **20.23** | 19.48 | f16 |

**fp8 loses even at its best-fitting K.** At long ctx decode is
attention-dominated, so the hot experts fp8 buys add only ~+3% (18.86→19.48 at
192K) while fp8's attention penalty costs ~−7%. The hot-expert lever cannot
repay the fp8 attention tax *at these depths* because MoE is a small slice of an
attention-bound token. This **refutes the brief's economic hypothesis for the
192K decode case.**

**The ONE genuine decode win is CAPABILITY, not speed:**

| ctx | f16 K=0 | fp8 K=0 |
|---|---|---|
| 262144 (256K) | **OOM (cannot run)** | **fits, 16.44 tok/s** |

f16 physically cannot serve 256K on 16 GB (KV ~12.9 GB + ~5.6 GB weights > 16);
fp8 halves KV and runs it. This is the stash's original "256K-enabler" framing
and it is real.

---

## 4. Isolated decode-attention kernel — halving bytes does NOT translate

`fp8_kv_ab` isolated A/B (random K/V, back-to-back f16/fp8, full-attn split):

| n_kv | f16 µs | fp8 µs | f16 GB/s | fp8 GB/s (of its halved bytes) |
|---|---:|---:|---:|---:|
| 32768  | 313.2 | 329.7 | 429 | 203 |
| 65536  | 547.5 | 632.8 | 490 | 212 |
| 100000 | 796.3 | 933.8 | 514 (86% of 600) | 219 |

**f16 attention is at ~82–86% of the 600 GB/s roofline; fp8 is 5–17% SLOWER and
gets worse with ctx.** fp8 reads *half* the bytes but sustains only ~205–219
GB/s of effective BW — ~2.7× off its own halved-byte roofline. The dequant +
per-key scale loads sit on the AV critical path, so the kernel becomes
latency/VALU-bound rather than DRAM-bound; the byte savings are stranded. This
is the exact "bytes ≠ time" trap the brief warned about, now confirmed for
**decode**, not just prefill. (A coalesced 8-B `uint2` load instead of 8×1-B
made no measurable difference — the compiler already coalesced; the pole is the
convert/scale path, not load width.)

---

## 5. Prefill wall @64K — regression CONFIRMED (partly a missing kernel)

`LAGUNA_PROF=1 LAGUNA_LONG_LEN=65536`, per-prefill wall:

| config | tok/s | vs fp8 |
|---|---:|---|
| f16 default (HGP-6 packed) | 444.8 | — |
| f16 HG-3 (matched geometry) | 377.0 | — |
| **fp8 (HG-3)** | **306.0** | — |

- fp8 vs **matched** f16-HG3: **−18.8%** (the intrinsic fp8 dequant cost).
- fp8 vs **default** f16-packed: **−31.2%** — but ~13 pts of that is that
  **fp8 has no `_hg_packed` variant** (it routes HG-3), so it forfeits the
  packed kernel's win. Both prefills produce the same argmax (→72).

Confirms the old "31–42% prefill regression"; the honest fp8-only cost is ~19%.
Prefill is compute/latency-bound (not KV-DRAM-bound), so halving KV bytes can't
help and the extra dequant just costs. **A `prefill_flash_wmma_fa2_hg_packed_fp8`
kernel would claw back ~13 pts** but never make fp8 prefill a win.

---

## 6. Quality — teacher-forced, measured properly

300-token teacher-forced A/B (f16's own greedy generation as the fixed context;
per-position argmax with NO greedy cascade). `LAGUNA_GEN_DUMP_ARGMAX`.

- **top-1 agreement: 263/300 = 87.7%** (12.3% divergence) — matches the old
  "~10% argmax" shelving reason.
- **mean top-5 overlap: 78.9%.**
- **Disagreements are mostly NOT near-ties:** median f16 top1–top2 logit gap at
  the 37 flips = **0.97** (p90 2.71, max 3.87); only 21.6% had gap < 0.5. Raw
  last-position logit-sum differs 18% (199927 vs 237296). So this is a **real
  distribution shift, not just near-tie flip noise** — contradicting that
  hypothesis for this data.
- **By position (entropy, not depth):** pos 0–99 = 70%, 100–199 = 93%,
  200–299 = 100%. Divergence concentrates on genuinely-uncertain content
  positions; it does NOT accumulate with KV depth (the repetitive tail agrees
  100%).

**Implementation looks correct** (per-`(token,head)` scale is *finer* than
per-tensor, srow indexing verified K-write ↔ decode-read consistent, K/V both
quantized post-RoPE). 12% is high for a "supported" config; unexplored levers if
pursued: **e5m2 for V** (heavier tails than K), finer per-block K scale, or
higher-precision K only. Note poolside's "supported fp8 KV" is a vLLM recipe on
their checkpoint; we post-hoc quantize an f16-trained Q4_K_M gguf, which can
legitimately diverge more.

---

## 7. Recommendation — DON'T SHIP as the long-ctx default; ship OPT-IN for >192K

**DON'T-SHIP as default** at 128K/192K:
- decode is *slower* even at fp8's best-fit K (192K: 19.48 vs 20.23; 128K:
  22.05 vs 23.27) — attention-dominance starves the hot-expert lever;
- prefill −19% (intrinsic) / −31% (as-wired);
- 12% top-1 quality divergence, mostly not near-ties.

**SHIP as an explicit OPT-IN for context > 192K** (the `LAGUNA_FP8_KV=1`
capability path): fp8 is the *only* way to fit 256K on the 16 GB dGPU (f16
OOMs), at 16.44 tok/s — "slower but exists" beats "cannot run." Gate it on ctx,
keep f16 default ≤192K.

### Contradictions raised vs the brief
1. fp8 does **not** reduce the decode LDS budget (dequants into f16 LDS);
   occupancy is unchanged, not improved.
2. The economics did **not** invert for 192K *decode*: hot experts fp8 enables
   (+3%) can't repay its attention tax (−7%) because decode is attention-bound
   at depth. The win is *capability (256K)*, not throughput at 192K.
3. Max K at 192K is **exactly 16** (K=24 OOMs), not "K=16 or more."
4. Quality divergence is **not** dominated by near-tie flips on this data.

### Follow-up levers (if the >192K path is pursued)
- `prefill_flash_wmma_fa2_hg_packed_fp8` (recover ~13 pts prefill).
- Broadcast the per-key scale across the 16 score lanes (kill redundant loads).
- e5m2-V / higher-precision-K quality ablation.

## Measurement notes
- All A/B back-to-back on the exclusively-owned dGPU (thermal-valid).
- Decode `laguna_decode_bench` iters=16 warmup=3; prefill `LAGUNA_PROF=1`.
- Kernel resources via `hipcc -Rpass-analysis=kernel-resource-usage` @gfx1201.
- Untouched (user WIP): `sampler.rs`, `softmax_sample.hip`, `sampler` test,
  `external/ds4`.
