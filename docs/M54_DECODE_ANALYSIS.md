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
