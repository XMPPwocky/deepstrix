# Design A — "move fewer bytes, launch fewer kernels" (independent architect, 2026-09-13)

**Anchor:** per-layer dGPU attention chain 0.50 ms ≈ 0.22 ms of weight bytes (136 MB @ ~620 GB/s) +
~0.28 ms in ~30 kernel boundaries ≈ **9 µs per boundary** (CP dispatch bubble + end-of-kernel L2
writeback + ramp/tail). HIP graphs were dead because they remove host submission cost, which was
already hidden; M59's kv_post 5→1 kernels (−0.55 ms) shows the boundary itself is the cost. Two
halves of equal size: ~0.23 ms/layer of bytes that cannot go away at zero loss, ~0.28 ms/layer of
boundaries that can.

## 1. Design

**A. Launches: 30/layer → 3/layer (persistent phases with device-side grid barriers), no host in
the per-layer loop.** Today's reuse-layer dGPU chain ≈ 32 launches (~50 on source layers).
Replace with two persistent dGPU kernels per layer + the existing WMMA attention launch:

| kernel | phases (grid barriers between) | note |
|---|---|---|
| **A: pre-attention** | (1) hc-mix dots + Σx² in one pass → (2) hc_pre collapse + Σx² for attn_norm → (3) normalize+quantize+wq_a, Σqr² via atomics → (4) wq_b rows ∥ wkv rows ∥ WG0 runs the 20-iter sinkhorn (mixes consumed only at hc_post) → (5) kv_post (1 WG) | 5 barriers, 1 launch |
| attention | score + smwsum (existing WMMA kernels; K-split) | 1–2 launches |
| **B: post-attention + FFN** | (6) rope⁻¹ + wo_a grouped → (7) wo_b with hc_post fused in the epilogue → (8) hc-mix-ffn dots+Σx² → (9) hc_pre + ffn_norm Σx² → (10) normalize+quantize + router gate matvec → (11) softplus/√/bias/top-6 (1 WG) **+ doorbell store** (activation 5 KB + 6 ids into the iGPU-polled buffer and the box-2 forwarder's buffer, `__threadfence_system`) → (12) shared + hot gate/up → (13) shared/hot down → (14) **spin on the partials doorbell**, combine, hc_post → residual_next | 8 barriers, 1 launch |

Barrier = agent-scope atomic + spin (`s_sleep`), all WGs co-resident via `hipLaunchCooperativeKernel`
(64 CUs × 4 WGs of 256). Expected 2–4 µs each vs ~9 µs per launch. Reduction-order drift only.

**iGPU side:** one persistent MoE kernel per token polling its local doorbell: gate/up → swiglu+q8 →
down (2 barriers, existing kwide-era bodies), writes 20 KB of partials into dGPU-visible memory +
doorbell. Cross-device sync: from the event/`hipMemcpyPeerAsync` path (~59 µs fixed, M54-B1) to two
posted PCIe writes over OCuLink (~5–10 µs). Each device polls only its own memory. Box 2: a host
thread on box 1 polls the same doorbell and forwards over USB4; the reply lands in the dGPU's
partials buffer — the persistent kernel does not care where a partial came from.

**Token boundary:** head + rms + hc_pre + sampler inside kernel B of layer 39; sampled id → host
doorbell; CPU forwarder copies the 10 KB F16 embedding row into the dGPU's layer-0 input buffer and
fires Engram prefetch for layers 1/14 (48 × 4 KiB random reads via io_uring, ~0.2–0.3 ms, under
layer 0's budget). No hipMemcpy/launch at the boundary.

**B. Bytes (zero-loss).**
1. **fp8-native attention projections** (wq_a/wq_b/wkv/wo_a/wo_b, shared expert, indexer wq_b) with
   the checkpoint's 32×32 e8m0 scales: 134 → 126.5 MB/layer (−6%) and deletes the measured
   Q8_0-from-fp8 argmax flip. At B=1 dequant-to-f16 + `v_dot2_f32_f16` matvec (exact); fp8 WMMA on
   gfx1201 pays at the B=3–4 DSpark verify width. 126.5 MB/layer is the floor (MALL 64 MB can't
   reuse across tokens).
2. Compressor/indexer weights f16 (measured exact except 0.2% subnormals); drop `attn.wkv` on reuse
   layers if M4 confirms unused (−2.6 MB × 36).
3. **Candidate-only indexer scoring on 24/28/32/36** (reference masks non-candidates to −∞ →
   identical): at 1M ctx 1.1 MB instead of 34 MB of keys per layer, top-512 over 16K not 500K;
   with a warm-started threshold select the long-context depth tax (single-WG bitonic sorts,
   60–75 µs each) goes away. Exact.
4. **Head:** Q8_0 of bf16 (700 MB, 1.1 ms); under DSpark read once per verify step (B=3 GEMV).
   Optional exact dGPU/iGPU split 496/204 MB with Gumbel-max merge: −0.25 ms/token.
5. KV bytes already ~0.2 MB/layer; nothing to take.

**C. Bytes (quantified-loss knobs, off by default, gated by §3b eval + argmax/top-5):**
- **|mid|-thresholded down projection** (drop w2 columns with |mid_j| < τ; w2 repacked K-major in
  row-slabs; per-column skipping at 50–70% kept saves 16–25% of expert bytes; exact zeros don't
  exist). CATS-class 0.3–2% task loss, unmeasured on V4.1.
- int6 block-32 attention projections (−18% attention bytes ≈ −1.5 ms/token). hc_fn bf16, head Q6_K (small).

## 2. Arithmetic (single stream, two boxes, RTT 100 µs)
Per layer A + attention + B: 137 MB @ 600 GB/s = 0.23 + 13 barriers × 3 µs 0.04 + 3 launches 0.03 +
tails 0.04 → **0.34 ms** (1.5× BW; today 0.55 = 2.4×); range 0.30–0.42. Expert phase: local 3 picks
0.264 + barriers/doorbells ≈ 0.28; remote 0.10 + 0.26 + 0.02 = 0.38; with a 4/2 placement max ≈ 0.35 → **0.36**.

| | per layer | ×40 | head | boundary + misses | ms/tok | tok/s |
|---|---|---|---|---|---|---|
| plan rev 4b | 0.55 + 0.38 | 37.2 | 1.4 | ~0 + 1.9 | 40.5 | 26 |
| this design | 0.34 + 0.36 | 28.0 | 1.1 | 0.05 + 1.9 | 31.1 | **32** |
| pessimistic | 0.42 + 0.38 | 32.0 | 1.1 | 0.1 + 1.9 | 35.1 | 28.5 |
| optimistic | 0.30 + 0.33 | 25.2 | 1.1 | 0.05 + 1.9 | 28.3 | 35 |

Every 1 µs of per-boundary cost across ~16 boundaries/layer = ±0.64 ms/token (±2%).
**With DSpark K=2 (2.31 tok/step):** B=3 through the same phases (chain ≈ 0.39); distinct experts
≈ 16/layer → 300 MB split 8/8 → local 0.72, remote 0.83 → phase ≈ 0.8; per layer 1.19 → 47.6 +
head@B=3 1.2 + drafter 1.6 + misses 3 = 53.4 ms / 2.31 → **~43 tok/s** (plan 37; range 38–48).
Under DSpark the expert phase is 2/3 of the layer → the |mid| knob is worth its eval (+3 tok/s).

## 3. Quality
Zero: all of A; B1 (restores reference parity); B2; B3 (identical by construction); B4 split
(Gumbel-max exact). ≈Zero: head Q8_0. Gated/unknown: |mid| skip, int6, hc_fn bf16, head Q6_K.

## 4. Cost and order
0. Exp-0 (2 days). 1. fp8 B≤4 matvec + grouped wo_a, oracle-gated (1 wk) — needed for quality
regardless. 2. Prologue/epilogue fusion in the existing structure → ~12 launches/layer (1–2 wk;
0.55 → ~0.45 alone). 3. Doorbells + persistent iGPU MoE kernel + CPU forwarder (1–2 wk; −40–50
µs/layer). 4. Persistent kernels A/B with grid barriers (3–4 wk; risks: co-residency vs prefill
lanes, matvec BW inside a max-VGPR kernel, `__threadfence_system` visibility over OCuLink on RDNA4).
5. Long-ctx exact items (1 wk). 6. Lossy knobs (1–2 wk each incl. eval).
Zero-loss core ≈ 7–10 weeks; items 1–3 alone ≈ 29 tok/s single stream.

## 5. Cheapest validating experiment
**Exp-0 (1–2 days):** synthetic persistent dGPU kernel running the chain's matvec skeleton on random
fp8 weights (wq_a 6.5 MB → barrier → wq_b 41.9 → barrier → attention stub → barrier → wo_a 33.5 →
barrier → wo_b 41.9 → barrier, ×40) vs the same five kernels launched per layer (graph-captured
and plain). Report ms/layer, µs/barrier, per-phase GB/s. **Kill:** persistent ≥ 0.75× of the
launched chain, or barrier > 6 µs, or matvec phases < 80% of standalone BW → item 4 dead, design
degrades to items 1–3 (+3–4 tok/s, not +6). Companion 1-day probe: dGPU→iGPU→dGPU doorbell round
trip through fine-grained host memory; > 30 µs kills item 3.

Skepticism: the 9 µs/boundary figure is inferred (0.50 − 0.22 over ~30 launches), not traced per
boundary; earlier notes put indexer launches at ~21 µs and M59 glue at ~3–6 µs, so the truth is a
spread and Exp-0 measures exactly that. The 214 GB/s expert term, the 1.9 ms miss placeholder and
the RTT are taken unchanged; the DSpark distinct-expert count (16) is from V4-Flash routing.
