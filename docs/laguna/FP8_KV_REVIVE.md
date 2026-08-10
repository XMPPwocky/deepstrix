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

---

# ADDENDUM 2026-07-28 — quality re-measured properly. Verdict unchanged, reasons corrected.

The quality numbers above were taken before the b128 load-width fix, and the
greedy first token (which used to flip off 22718) now holds. That made them
suspect, so they were re-measured on the fixed build. **They reproduce — the old
number was genuine quantization drift, not the bug.** But the old
*interpretation* was wrong in a way that matters.

## Method (better than the original)
Teacher-forced on **real natural-language text** (~900K tokens of public-domain
novels), not the pangram loop. Deep context built via the batched prefill path,
then a 384-512-position window measured with per-position full-logit dumps.
Two controls the original lacked:
- **f16-vs-f16 is bit-exact** (KL=0, 100% top-1) → the pipeline is deterministic,
  so every fp8 number is signal, not run-to-run noise.
- **Fake-quant mode** (`LAGUNA_FP8_FAKE`): quantize→dequantize into the *f16*
  cache, read by the *f16* kernels. Isolates pure numerics from the read kernel.

## Results by depth
| depth | top-1 | top-5 overlap | KL(f16‖fp8) | median gap@disagree | near-ties (<0.5) |
|---|---|---|---|---|---|
| shallow (~0-320) | 83.1% | 74.4% | 0.316 | 0.51 | 50% |
| 4K   | 62.5% | 68.2% | 0.491 | 0.50 | 50% |
| 32K  | 72.5% | 76.9% | 0.251 | 0.43 | 56% |
| 128K | 71.1% | **79.8%** | **0.192** | 0.27 | **68%** |

**fp8 error does NOT accumulate with depth — it shrinks.** KL falls and top-5
overlap rises from 4K to 128K (more keys attended → more averaging → less
error; corroborated by the isolated test, where attention-output error drops
0.0125 → 0.0010 as n_kv grows 34 → 4096). The 4K row is the worst point and is
content-confounded (different chapters at different depths; content cannot be
held fixed across depth with a linear corpus), which is why KL is the robust
signal rather than top-1.

**This corrects the original claim above.** It reported "disagreements are
mostly NOT near-ties, therefore real distribution shift" — but that was measured
in a shallow, low-entropy, in-distribution window. At the long context fp8
actually exists to serve, divergence is smaller (KL 0.19) and **68% of flips are
near-ties**.

## No read-kernel bug remains
fake-both@32K (72.5%, KL 0.245) is identical to real fp8@32K (72.5%, KL 0.251).
The two post-merge fixes (b128 NaN path, load width) fixed crashes/NaN at long
ctx and the first-token flip — not this drift.

## Attribution: no asymmetry to exploit
- **K vs V roughly symmetric.** 4K: K-only 58.0%, V-only 63.1%. 32K: K-only
  75.4%, V-only 74.4%. Neither dominates → no "quantize only the tolerant one" win.
- **Global vs SWA layers (4K):** global-only 56.5% / KL 0.53; SWA-only 60.9% /
  KL 0.47. Global layers hurt more per-layer, but the 36 SWA layers (which only
  ever attend 512 keys) contribute nearly as much in aggregate → this is broad
  per-layer KV sensitivity, NOT a deep-attention phenomenon.

## Every fix hypothesis tested and FAILED (4K, fake-both)
| variant | top-1 | KL |
|---|---|---|
| per-row (blk=128) e4m3 | 58.4% | 0.515 |
| per-32 e4m3 | 56.5% | 0.568 |
| per-16 e4m3 | 58.4% | 0.535 |
| per-128, V=e5m2 | 59.0% | 0.582 |
| per-16, V=e5m2 | 60.6% | 0.546 |
| per-16, K&V=e5m2 | 55.7% | 0.617 |

Finer scale granularity gives **no** improvement (slightly worse KL — fewer
elements per amax estimate). That **rules out the outlier/clipping hypothesis**:
the error is uniform across elements, the signature of correct-but-
resolution-bound quantization. e5m2-for-V doesn't help (extra range is not the
constraint; lower mantissa slightly hurts). No clipping is possible by
construction — the per-row scale is amax/448, so the max element maps to exactly
448. Write/read scale application verified consistent.

**There is no quality fix to ship.** The cause is the intrinsic ~3-bit mantissa
of 8-bit KV, amplified by 256-expert top-10 MoE routing sensitivity, on a
checkpoint that is ALREADY 4-bit-weight quantized and that we quantize post-hoc
— unlike poolside's natively-supported fp8 recipe on their higher-precision base.

## Recommendation (unchanged, better justified)
Keep fp8 KV as the **explicit opt-in >192K capability path**, not a general
long-context default.
- Below 192K, f16 fits AND is faster → no reason to pay any quality cost.
- At 256K, f16 physically OOMs, so "slightly lossy but runs at 16.44 tok/s"
  beats "cannot run". The drift there is the mild end of the range measured
  (KL ~0.19, 80% top-5 overlap, majority near-tie flips).

---

# ADDENDUM 2026-07-29 — quantized KV format swapped e4m3fn → INT8 symmetric (production)

The `LAGUNA_FP8_KV=1` path is now backed by **int8 symmetric**, not e4m3fn. The
env-var name is unchanged (it means "quantized KV"), and the wire layout is
**byte-identical**: 1 byte/elem + the same per-`(token, kv_head)` f32 scale.
Only the scale denominator and the encode/decode change:

- write (`laguna_quantize_fp8_kv`): scale = **amax/127**; `q = roundf(v/scale)`
  clamped to `[-127,127]`, stored two's-complement in the existing u8 buffer.
- read (`gqa_fp8_to_f32` / `gqa_fp8x2_to_f32`): `(float)(signed char)` widen ×
  the same row scale — replaces `v_cvt_pk_f32_fp8`. No arch intrinsic needed, so
  the e4m3 LUT fallback (128 f32 of `__constant__`) is deleted from
  `gqa_attention.hip`. `amax/127.5` was measured slightly worse; 127 is used.

**e4m3 is REPLACED, not added as a second option.** Two lossy KV formats where
one strictly dominates is pure maintenance cost. e4m3/e5m2 survive only in the
`LAGUNA_FP8_FAKE` diagnostic round-trip (fmt 0/1), alongside int8 (fmt 2/3).

## Why: int8 strictly dominates e4m3 at matched bytes

Fake-quant grid (teacher-forced, 320 positions, per-row blk=128, vs f16):

| depth | e4m3 KL | int8 KL | int8 better by |
|---|---:|---:|---:|
| 4K   | 0.5014 | 0.4388 | 12.5% |
| 32K  | 0.2897 | 0.2589 | 10.6% |
| 128K | 0.2485 | 0.2238 | 9.9% |

**Confirmed on the REAL production path** (128K, K=4, same f16 reference — this
is the decisive matched real-vs-real A/B, not real-vs-fake):

| config | top-1 | top-5 | KL |
|---|---:|---:|---:|
| real e4m3 | 68.4% | 73.9% | 0.2763 |
| **real int8** | **69.4%** | **75.3%** | **0.2421** |

int8 is **12.4% lower KL** with better top-1 AND top-5 — agreeing with the
fake-quant grid's ~10% prediction. Both real numbers sit ~8-11% above their
fake-quant counterparts (the quantized prefill routes HG-3 while f16 default
routes HGP-6 packed, so accumulation order differs), but the *ranking and margin
are preserved*, which is the point of the check.

Elementwise, the isolated decode A/B max_abs vs f16 improves 5-9×:
int8 0.0014/0.0004/0.0002/0.0004 vs e4m3 0.0125/0.0023/0.0010/0.0024.

Per-32 / per-16 blocking is slightly better on KL still (128K int8_16 = 0.1918)
but grows bytes/token and would put the K=16 hot-expert tier at risk — NOT used.

## Perf: no regression; a real win at shallow ctx

Isolated decode-attention kernel, one n_kv per process (cross-context thermal
accumulation otherwise penalises whichever format is timed second — that
artifact produced a spurious "-24.6%" in an early sweep):

| n_kv | f16 | int8 | |
|---|---:|---:|---|
| 32768  | 307.7 us | 278.6 us | **int8 1.10×** |
| 65536  | 540.7 us | 518.1 us | int8 1.04× |
| 100000 | 786.9 us | 783.8 us | 1.00× |
| 196608 | 1515.6 us | 1538.6 us | 0.99× (−1.5%) |

int8 is genuinely **faster than f16 below ~64K** and at parity by 100K. It does
NOT stay faster at depth: f16 climbs from 73% → 89% of the 600 GB/s roofline as
n_kv grows, while the quantized path plateaus at ~45% of its own halved-byte
roofline — the per-tile pipeline (LDS staging, barriers, score/AV) does not
scale with bytes and becomes the floor. −1.5% at 192K is inside noise and inside
the ±2% gate. e4m3 measured on the same harness was +0.1% @100K / −0.7% @192K,
i.e. int8 ≈ e4m3 on time while being materially better on quality.

## Resources unchanged — the LDS swizzle is format-invariant

| decode kernel | VGPR | LDS B | scratch | occ |
|---|---:|---:|---:|---:|
| `gqa_attn_decode_partial_hg` (f16) | 205 | 21136 | 0 | 6 w/SIMD |
| `gqa_attn_decode_partial_hg_fp8` (int8) | **203** | **21136** | **0** | **6 w/SIMD** |

The brief flagged a risk that the int8 ring's per-lane LDS stride might differ
from fp8's and invalidate `DEC_LDS_SWZ`'s conflict analysis. **It does not.** The
prefetch ring `dec_hw` is `_Float16 ext_vector(WELEM=16)` — it holds *dequantized
f16*, never the raw 1-byte codes. So the staging store is 16 f16 = 32 B/lane for
both formats, `WELEM`/`BLKPC`/`DEC_BLK` are untouched, and the 8-way→4-way XOR
swizzle argument carries over verbatim. int8 uses 2 *fewer* VGPRs (no LUT
address math), so the 709 B/WG headroom to the 3-WG/CU threshold is preserved.

## Capacity unchanged

Bytes/token (global layers) = `12×8×128×1×2` KV + `12×8×4×2` scale =
**25344 B**, exactly as before → 4.98 GB at 192K. **Verified by loading, not
arithmetic**: `LAGUNA_FP8_KV=1` + `laguna_hot_experts_k16.txt` at
`LAGUNA_BENCH_CTXS=196608` loads and decodes at **20.66 tok/s**. The K=16 tier
still fits.

Bonus: the 192K quantized run hit the greedy parity token **22718 exactly**
(`[OK parity/fp8]`) — e4m3 used to flip that near-tie. The f16 default path is
untouched and still asserts 22718.

## Does this change the DON'T-SHIP-as-default verdict? No.
int8 removes the *quality* objection's sharpest edge and removes the shallow-ctx
*speed* objection, but the ≤192K economics are unchanged: f16 fits and is at
parity-or-faster at the depths that matter, so there is still no reason to pay
any lossy-KV cost below 192K. int8 makes the >192K capability path (256K on a
16 GB dGPU, where f16 OOMs) meaningfully better than it was.

---

# ADDENDUM 2026-07-30 — the line is CLOSED: f16 decode attention is compute/DRAM BALANCED

Final experiment in the quantized-KV arc. Hypothesis: int8 sustains only ~45% of its
own halved-byte roofline at 192K while f16 sustains 89% of the full roofline, so
int8 must be bound by a byte-INVARIANT per-tile pipeline — because the LDS staging
holds DEQUANTIZED f16 (32 B/lane) for both formats, discarding int8's saving at the
LDS level. Fix: stage RAW int8 in LDS, dequantize at point-of-use, which halves LDS
per key and lets Bc go 32 -> 64 at constant occupancy, halving tile/barrier count.

**Built exactly as specified. REFUTED. Dropped.** (Artifact preserved on branch
`int8-lds`, commit 719c64e, dormant behind `DEC_I8_LDS`; NOT merged to master.)

## The lever moved nothing
int8 bandwidth stayed pinned at **42-44% of the halved-byte roofline in EVERY
config**:
| config | VGPR | LDS B | scratch | WG/CU | 192K us | vs f16 |
|---|---:|---:|---:|:--:|---:|---:|
| f16 (untouched) | 205 | 21136 | 0 | 3 | 1525 | — |
| int8-lds Bc=64 (PFD=1) | 165 | 21636 | 0 | 3 | 1566 | -2.7% |
| int8-lds Bc=32 (PFD=2) | 225 | 12036 | 0 | **5** | 1634 | -5.7% |
| int8-lds Bc=128 | — | 40836 | **1900** | 1 | dead | — |

Three independent attacks on "per-tile staging/barrier overhead" — raw-int8 LDS,
Bc=64 (tiles and barriers HALVED), and Bc=32 at 5 WG/CU (occupancy RAISED) — all
left the achieved bandwidth unmoved. The hypothesis was wrong.

## What actually binds (the decisive diagnostic)
Half-heads (`DEC_I8_HALFHEADS`, score+AV over 3 heads instead of 6) at 192K cut the
wall only **1634 -> 1326 us (-19%)** and left it at **52% of roofline** — still not
DRAM-bound. So the floor is **per-head score/AV COMPUTE** (~38%) plus a shared
per-tile pipeline — and that compute is exactly what f16 performs after its dequant.

## STANDING CONCLUSION — stop optimizing this axis
**f16 decode attention at depth is PERFECTLY BALANCED: compute ~= DRAM ~= 1525 us.**
Therefore **no KV quantization scheme can beat f16 at long context on this kernel.**
Halving bytes does not speed anything up; it exposes a compute wall that was already
there at the same height. Data placement — LDS format, tile size, occupancy — cannot
drop below it.

This retroactively explains the whole arc: e4m3 at parity, int8 at parity, MLP
inert, scale granularity inert. Every one of them rearranged the memory side of a
kernel whose memory side was already matched to its compute side.

The only remaining lever is cutting the score/AV FLOPs themselves. The shelved
`decode-attn-WMMA` stash already measured WMMA as BW-neutral on this path.

## Refuted-levers list for int8 decode attention (do not retry)
dequant/VALU cost · prefetch depth / MLP · load width (fixed, real) · scale
granularity · e5m2 for V · K/V format asymmetry · **LDS staging bytes** ·
**tile/barrier count (Bc)** · **occupancy (WG/CU)**.

## What quantized KV IS still for
Capacity, and only capacity: it halves KV (9.74 -> 5.02 GB at 192K), which buys the
K=16 hot-expert tier where f16 gets K=0, and it is the only way to fit 256K on 16 GB.
That remains the entire case for `LAGUNA_FP8_KV=1`, and it is unchanged.
