# M54 — decode roofline + gap attribution (2026-06-09)

## Measured (bench_decode_het_parallel, FAKE_POS, B=1)

| depth | ms/tok (p50-p70) | tok/s |
|---|---|---|
| 4096  | 39.6 | **25.3** |
| 98176 | 42.4 | **23.6** |

(96K bench note: FAKE_POS=98304 exactly trips an off-by-one in
indexer_topk allowed_bits sizing — 24577 comp rows need 769 words; use
98176. Worth a real fix.)

## Per-token byte traffic (from GGUF tensor sums; N_EXPERT_USED=6)

| component | GB/token | device |
|---|---|---|
| routed experts (6/256 of 77.19 GB) | 1.809 | iGPU |
| attention weights | 5.84 | dGPU |
| shared expert | 1.15 | dGPU |
| lm_head (~) | 0.56 | dGPU |
| other | 0.18 | dGPU |
| KV+indexer @4K / @96K | 0.03 / ~0.21 | dGPU |

## Roofline (BW-only; iGPU ~218 GB/s, dGPU 9070 XT ~576 GB/s achievable)

Strict serial: iGPU 8.3 ms + dGPU 13.6 ms ≈ 21.9 → 45.7 tok/s @4K.
Overlap-optimal (shared expert ∥ routed experts; the only legal intra-token
overlap at B=1 — layers are a strict attn→MoE→attn dependency chain):
dGPU-serial 11.6 + max(iGPU 8.3, shared 2.0) ≈ 19.9 ms →
**~50 tok/s @4K, ~48.5 @96K. Headroom ≈ 2.0× from measured.**

## Gap attribution (rocprofv3 kernel-trace @4K, per token; ~2650 dispatches!)

| bucket | ms | vs floor |
|---|---|---|
| iGPU MoE (iq2 5.98 + q2k 3.20 + quant 0.27) | 9.45 | floor 8.3 → **88% BW-eff, near-roofline** |
| dGPU weight matvecs (q8 gemv 9.16 + grouped 2.96 + f16 ~3.5) | ~15.6 | floor 13.6 → ~87% BW-eff |
| dGPU small-kernel zoo (rms 183×/tok!, rope 150×, quantize 259×, kv_append, copies, attn pieces, sinkhorn, hc, topk) | ~7.5 | ~0 BW — launch/latency-bound |
| host + sync + inter-device gaps (bench reports host 5.5 ms) | ~7-12 | 0 |

**Conclusions:**
1. The big kernels are NOT the problem (85-90% of BW). [[iq2-decode-profile]]'s
   "decode at its floor" is true per-kernel, false per-token.
2. The 2× headroom lives in: (a) ~2650 kernel dispatches/token — the small-
   kernel zoo + host/launch overhead (~13-19 ms combined); (b) shared-expert ∥
   routed-expert overlap quality.
3. Levers, in order: fuse/batch the per-layer small kernels (rms+rope+quantize
   chains), audit HIP-graph coverage (graphs exist but host is still 5.5 ms/tok),
   verify shared-expert overlap actually happens in HetParallel mode (pftrace),
   then micro-BW on the 87%-efficient matvecs (~+1.5 ms total available).
4. Hard ceiling without model/format changes: ~50 tok/s @4K. Past that needs
   MTP speculative decode (exists per [[mtp-validation]]) or expert-quant
   changes — out of scope.

## 2026-06-09 — M54 implementation round: honest results

**Pre-issue iGPU lane: NEGATIVE (kept off).** First version showed +27%
(25.5 → 32.5 tok/s) but the oracle caught a race: hipStreamWaitEvent
snapshots at CALL time, so waits enqueued before the token's records were
no-ops — the MoE ran on stale inputs. Fixed properly with
hipStreamWaitValue32/WriteValue32 (new v4flash-hip bindings; value waits
compare at EXECUTION time): correct version is 40.0 vs 39.3 ms — neutral to
slightly negative. **The MoE dependency wait IS the critical path; the
125 µs/layer "submission lag" was slack under the ~500 µs inter-layer
dependency gap.** Misread trace; good oracle. Value-signal infra retained
(engine.moe_signal + token_seq), DECODE_PREISSUE stays opt-in/off.

**Bench-vs-oracle note:** forward_prompt_batch_matches_sequential fails
IDENTICALLY at baseline 0959f65 (worktree-verified): batch_v2 1.7088e2,
first divergence (L1, b0). Pre-existing, now tracked separately. With the
value-signal fix, preissue=1 reproduces the exact baseline signature →
sequential path bit-restored.

## Remaining decode levers (sized, unstarted)

| lever | est. ms/tok | notes |
|---|---|---|
| kv_cache_append ring buffer | ~1.0 | steady-state slide = 127 rows × 254 barriers, single block, per layer; ring + head offset touches all raw_kv readers (attention raw section, compressor ingest, batched append) |
| rms_norm+quantize fusion (and rope_tail chains) | ~1-2 | 183 rms + 259 quantize launches/token; fuse pairs inside existing graphs |
| token-boundary device feedback | ~1.6 | vocab_matvec → sample → host → embed H2D round trip; embed-gather by device token id |
| shared-expert grouped_gemv at 60% BW | 0 e2e | hidden under routed MoE overlap |

Sum ≈ 3.5-4.5 ms → ~28.5-29 tok/s (+13%). Roofline remains ~50 (the rest
is the serial weight-BW chain itself).

## 2026-06-09 — Divergence investigation CLOSED: benign drift, miscalibrated oracle

Full 43-layer bisect of batch-vs-sequential: drift grows smoothly 7.9e-3 →
5.4e1 ABSOLUTE, but hc magnitudes grow 0.01 → ~2700 over the same span —
vector-scaled drift is a flat ~0.5-1.3% everywhere. The L33/L34 "explosions"
were magnitude growth, not error growth. At every layer: identical expert
selection, router weights matching to 3-4 decimals, kv diffs exactly one
fp8/f16 quantization step. Per-element relative ALSO misleads (cancellation:
hc elements sum O(1000) terms — a 0.6-valued element legitimately carries
O(1) noise from 0.5% drift on the terms).

Sources: the deliberately-shipped precision trade-offs (f16 WMMA scores,
fp8 KV, q8/q8k stages), each oracle-gated at the kernel level. The e2e test
used an ABSOLUTE 5e-2 tolerance from a pre-WMMA era; failure predates today
(worktree-verified at 0959f65).

Fix: oracles now compare max-diff / vector-scale (bound 5e-2, measured
3.5-4.3e-2 — intentionally thin, it's a drift BUDGET: new precision
trade-offs that add drift will trip it and force a conscious decision) and
the logits oracle additionally asserts argmax agreement (it matched:
token 260 both paths). **All 4 batch-vs-sequential oracles GREEN** — the
e2e gate that caught the pre-issue race is now usable for future work.

# M55 — decode round 2 (2026-06-09, late): measured lever results

Target ≥30 @4K; reached **26.9** (from 25.4). All oracles green throughout.

| change | predicted | MEASURED | status |
|---|---|---|---|
| monotonic KV append + raw_off (no 127-row slide) | −1.0 ms | **−2.1 ms** (39.3→37.2) | SHIPPED (default; also MTP-rollback prereq) |
| iq2 decode weight-staging (b128 via LDS) | −1.1 ms | **+3.3 ms REGRESSION** | opt-in only (DECODE_IQ2=wstage). Narrow u16 loads were already latency-hidden by the 1536-WG grid; staging serialized fetch→compute. ~187 GB/s is the pattern ceiling. |
| rms_w+quantize fusion (q_chain) | −0.3-0.4 | **−0.2 (≈noise)** | shipped (harmless); graph-internal nodes are already cheap |

96K decode: 23.6 → 24.9 tok/s.

**Honest correction to the M54 lever table**: the "small-kernel zoo ~7.5 ms"
estimate was trace-inflated. Kernels inside HIP graphs sequence cheaply;
only genuinely latency-SERIALIZED items pay big (the KV slide was one).
Remaining sized-by-measurement levers: token boundary (~1.6 ms, partly
host), q/kv-chain device-pos graph (~1.0), per-kernel grinding (~1-1.5
total at terrible ROI).

**Path to 30+ is MTP self-speculative decode** (probe + validation exist;
not wired): draft 1 token with the MTP head, verify with a 2-position
forward. Prerequisites now in place: monotonic KV makes raw rollback a
counter decrement. Design dodge for compressor-state rollback: SKIP
speculation at positions where accepting would cross a ratio-4/128
compressor boundary (~75% of positions still speculate). Expected
1 + 0.7·0.75 ≈ 1.5 tokens/step → **~38-40 tok/s** — the only mechanism
that clears 30 with headroom, and it stacks on any future kernel wins.
Verify-forward should reuse the q8 GEMM/BxN-MoE batched kernels (NOT the
full batch-prefill path — its fixed per-chunk overhead swamps B=2).

# M56 — heterogeneous MoE split (2026-06-10)

User challenged MTP (correctly: verify-2 nearly doubles iGPU expert reads
since adjacent tokens pick mostly-disjoint experts; IQ2 acceptance unknown).
Re-examination found the better lever: each device idles ~half of every
token — the serial chain is a device-ASSIGNMENT choice for weight-parallel
work.

**Per-token expert streaming BOTEC (user question): rejected.** PCIe ~25
GB/s → 283 µs to move one 7.08 MB expert vs 32 µs of iGPU read saved (9×
loss); selections for L+1 unknowable at L (no prefetch). Streaming pays only
for many-times-reused bytes = static placement. v2 extension: background
adaptive refresh (swap ~1 expert/few-hundred tokens over PCIe ≈ free).

**Static hot-expert placement (DGPU_HOT_EXPERTS=K + _FILE):** per-layer
top-K experts by measured decode selection frequency (DEEPSTRIX_EXPERT_STATS
probe, 128 tokens @4K, 70 cycled real residuals →
reference/decode_hot_experts.txt) packed dense into dGPU VRAM; hetsplit
kernel variants (shared remap[256], mode 0=iGPU misses / 1=dGPU hits); dGPU
computes its picks in its former MoE wait via a "dgpu_hot_moe" graph;
partials add at ffn_combine. Oracles: batch_v2 + last_only + pipelined all
PASS (within drift budget).

Measured (mean over 96+ tokens — per-token hit rate VARIES, see histogram):

| config | mean ms/tok | tok/s | note |
|---|---|---|---|
| off | 37.52 | 26.65 | |
| K=4 (1.2 GB) | 35.06 | 28.52 | **max K that coexists with 2-lane prefill scratch (server config)** |
| K=6+ | — | — | OOM next to prefill scratch |
| K=16 (4.9 GB) | 33.75 | 29.63 | decode-only process |
| K=24 (7.3 GB) | 33.58 | 29.78 | decode-only; diminishing |

Aggregate top-16 hit ~62%; per-token hit-rate HISTOGRAM (256 tokens, 5%
bins): bell around 55-60%, bulk 40-85%, outlier 15% — hence mean-not-median
reporting (time p99 only ~+4% over mean).

Caveat: calibration and bench share the input distribution; real-traffic
hit rates will sit lower until placement is recalibrated (or adaptive).

**Next lever to reach 30+ in SERVER config: dGPU VRAM audit** — 2-lane batch
prefill scratch currently leaves only ~1.5-2 GB free; trimming it (or
phase-alternating lane-B scratch with the hot set) raises K to 8-16 →
~29-30.5 server-side. MTP remains shelved per analysis above.

# M57 — dGPU VRAM audit (2026-06-10)

Tool: DEEPSTRIX_ALLOC_TRACE=1 (buffer.rs, ≥8MB allocs + #[track_caller]
source line). Findings (server config = 2 prefill lanes):

| allocation | size | verdict |
|---|---|---|
| token_embd dGPU copy | **1.06 GB** | **DEAD — removed** (server embeds host-side from mmap; zero kernel consumers; dtype still validated) |
| active_comp_kv | 537 MB/lane | real (CSA gathered top-512 × 512 f16 × B=1024); next structural trim target |
| attn_scores (f16 mode) | 268 MB/lane | real |
| q / q_normed / heads | 3×134 MB/lane | trimmable to f16 (−0.5 GB/2 lanes) if needed |
| indexer_scores | 101 MB/lane | grows with ctx (B × MAX_KEYS f32) |
| lm_head q8 | 563 MB | real (only head copy now) |

**Result: hot-expert K=8 fits the FULL server stack** (was K=4):
- worst-case oracle (3 scratch instances): K=8 PASS, K=12 OOM
- real server @ ctx 98304 + K=8: 15.62/17.1 GB used → **1.47 GB free**
- **128K ctx KV budget: +0.24 GB vs 96K (comp 705 MB + indexer 176 + raw 50
  + r128 21 ≈ 0.95 GB total) → fits with ~1.2 GB margin at K=8** ✓
- K=12 at 128K would be borderline; take K=8.

Decode (mean, 96 tokens): K=8 = **29.19 tok/s** (production-compatible).
Server e2e incl. sampling/overhead: ~27.2 tok/s on a real generation,
output coherent. Day-2 decode totals: 25.4 → 29.2 bench / 27 server.

Remaining to clear 30 bench-side: q→f16 trim → K=12 (+~0.3), token-boundary
(~1.6 ms), q/kv-chain device-pos graph (~1).
