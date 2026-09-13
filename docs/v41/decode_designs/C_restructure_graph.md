# Design C — "restructure around what V4.1 itself gives you" (independent architect, 2026-09-13)

Sources: ARCH_SPEC.md (all), PLAN.md §4–7b, ENGINE_PORT.md, model.py (Block, Attention/CSA2,
Indexer, Compressor, DSparkBlock/Attention, Transformer.forward/forward_spec), engram.py.

## 0. Placement verdict: nothing beats "attention on dGPU, experts EP by residency" for one stream
- **Encoder on box 1 / decoder on box 2:** box-2 iGPU attention chain = 168 MB ÷ 214 GB/s = 0.79 ms
  at bandwidth, ~1.0 launch-bound (vs dGPU 0.55). Box 2: 20×(1.0+0.53) = 30.6 ms; box 1:
  20×(0.55+0.53) = 21.6 + SSD misses (encoder experts 144 GB in ~85 GB → 59% resident) + head 1.4
  + 1 RTT → ≥ 59 ms → **~17 tok/s vs 26**. Saving 39 RTTs (3.9 ms) never pays for +9 ms iGPU
  attention and +10 ms of unsplit expert legs; dead at any RTT < ~700 µs. Same arithmetic kills any
  contiguous block on box 2.
- **EP only on the decoder half (20 RTTs):** forces encoder experts to ~59% residency → miss cost ≫
  2.4 ms saved. USB4 carries activations, never experts (18.8 MB = 12.5 ms > SSD 4.8).
- **Head-group tensor parallel over OCuLink** (wq_b/wo_a column-, wo_b row-parallel): dead by the
  measured 59 µs peer round trip vs ≤ 42 MB matvecs.
- **CED gives decode nothing:** decode runs all 40 layers per token by construction; bounded replay
  is a prefill approximation. The decoder's global store being one projection of H_20 only makes
  prefill store production cheap.
- **Async/stale remote experts:** not exact, routed experts ≈ half the FFN output — rejected.
- Already skipped by the model: 32/40 Reuse layers run no compressor/indexer; the 4 decoder
  Reindex layers (24/28/32/36) score only the 16,384-position candidate pool built at layer 20;
  window KV is per-layer and unavoidable.

So the graph stays: dGPU = 40 attention chains + router + shared + (new) DSpark drafter; box-1
iGPU / box-2 iGPU / dGPU-LRU / SSD-LRU = routed experts, three concurrent branches per layer.

## 1. Per-token graph changes

### A. mHC bridge split — exact (fp32 reassociation only)
§1.1: `H_{l+1} = ffn_post ⊗ f + ffn_comb·H'`, next attention input `attn_norm(Σ_j ffn_pre[j]·H_{l+1}[j])`,
and `ffn_pre/post/comb = hc_mixes(H')` are computed from the *post-attention* residual — **known
before the experts run**. So the post-expert critical path decomposes:
- **Precompute during the expert wait** (dGPU idle ~0.28 ms/layer): `C = ffn_comb·H'`, `r = Σ pre_j C_j`,
  `α = Σ pre_j post_j`, `v = hc_attn_fn·C_flat` [24], `Σ‖C_j‖²`; the 4-copy write of `H_{l+1}` is deferred.
- **On-path after f** (local + remote + shared sums): ONE fused kernel: `‖f‖²`, `f·C_j`, the 24-row
  projection of `post_j f`, rsqrt from the closed form `‖H_{l+1}‖² = ‖f‖²Σpost_j² + 2Σpost_j(f·C_j) + Σ‖C_j‖²`,
  sinkhorn, `z = αf + r`, `attn_norm(z)` → `wq_a`. Same trick at the attention→FFN boundary.
- Today's on-path chain = 3–5 launches at the ~17 µs floor. **Saving: 4–8 launches/layer ≈
  0.07–0.13 ms → 3–5 ms/token** (≈1 ms if `mhc_chain` already fuses each boundary). A dependency
  cut, not a byte cut.

### B. Store production at source layers cannot leave the decode path (checked)
`compress_lens = (pos+1)//ratio` includes the group the current token completes; deferring it
changes the attended set. Keep. Micro-item: at even positions the ratio-2 layers' `wkv/wgate` only
fill state → defer to the odd token (~0.1 ms, exact).

### C. DSpark — the drafter is strictly serial; its placement and internals are the item
The draft block is `[sampled token, noise×4]`; token 0 is the main model's sample at the last
accepted position — unknown until the head runs. So the drafter cannot overlap the verify pass; its
time D adds to every step. What *can* run during layer 39 + head: `main_x = main_norm(main_proj(cat(mean_copies h_37,38,39)))`
for all K+1 candidates (79 MB matvec) and DSpark's `kv_norm(wkv(main_x))` ring entries; commit only
the accepted prefix. Exact.
- **Placement:** iGPU-resident drafter ≈ **9–10 ms/step** (3×(0.3 + 1.3) + head 3.1 + 5 sequential
  markov reads). dGPU via het-split (DSpark non-routed 0.5 GB + markov 0.15 + ~3 GB LRU of DSpark
  experts; full 7.2 GB on box-1 iGPU) ≈ **3.5–5 ms**, hit-rate dependent; evicts the main model's
  dGPU hot experts (+~1 ms/token).
- **Free-approximation zone:** under speculative sampling (accept w.p. min(1, p_main/q_draft))
  drafter-side changes move only acceptance: markov bias over the drafter's top-256 instead of
  129280 rows; drafter experts IQ2/IQ3 (3.5–5.2 GB → all on the dGPU, D ≈ 2.7 ms). Verify exact.
- **Confidence-adaptive K:** the model ships a per-position confidence head; extend K ∈ {1..5}
  while marginal expected tokens / marginal ms ≥ current rate. Build after calibration is measured.
- **Engram with drafts:** hash is token-only, drafts known at step start → issue 48×(K+1) row reads
  + both `wkv` projections before layer 0 (lead ≥ 1.3 ms vs ~0.3–0.5 ms gather). n-gram→rows RAM cache
  (12 KB/position, exact).
- Correction to PLAN §7b.3: "verify batch routing known one step early" is wrong — routing at
  layer l needs the verify pass to reach layer l.

### D. Composition
Route prediction hides the RTT → expert phase → max(local, dGPU) ≈ 0.26 and A's bridge is a larger
share of what is left. Launch fusion: A's on-path kernel is one of theirs; the residual-side
precompute is what still moves into the idle window. DSpark verify batches move in lockstep so the
dGPU idle window persists — A fills a slice of it.

## 2. Arithmetic (RTT 100 µs, 32K)
- Baseline 38.6 ms → 25.9 tok/s. **+A: −3 ms (1–5) → 28.1 (26.6–29.8)**; honest single-stream
  **27–29 (+5–12%)**. Nothing on this axis touches the 22 ms attention chain or the 15 ms expert phase.
- DSpark K=2, a=0.75: −3 (A) +1.5 (D = 4.5) +1 (hot experts off dGPU) → 62.3 ms → **37.1 tok/s**;
  iGPU-placed drafter 33.8; quantized all-on-dGPU drafter 38.2 minus acceptance loss (likely a wash).
  Adaptive K: **+0–8%, unknown until measured. Summary: ~37 with DSpark, low 40s best case.**
- Prefill: A is noise at B=1024. DSpark ring seeding comes free from the decoder's 128-token
  replay. KV snapshots shrink to 4 global stores + 43 rings → prefix-cache save/restore ≈ free.

## 3. Quality
A exact (≤ 1e-6 rel); B exact; C: main distribution exact under speculative sampling; Engram
cache exact. Load-bearing elsewhere: bounded-replay prefill leaves approximate window rings at
layers 21–39 for the first 128 decode tokens — quantify with the oracle before trusting 1300 tok/s pp.

## 4. Cost and build order
1. **A** (2–3 days), oracle-gated at 2× floor; bank the number on V4-Flash first (same chain).
2. **DSpark verify** (~1 week): decode primitives (not the prefill path), speculative sampling,
   drafter on dGPU via het-split, candidate `main_x` precompute, Engram gather at step start.
3. Adaptive K (2 days), gated by calibration. 4. Engram row cache (1 day).

## 5. Cheapest experiment / kill result
The tok/s rests on DSpark acceptance ≥ 0.7 at draft position 1 with the native drafter, and
confidence↔acceptance correlation. No engine work needed: the CPU oracle streams one Block at a
time and model.py has `forward_spec`; run ~100 decode tokens of a real agentic transcript, log
per-position draft == main-greedy and the confidence head (~5 h CPU). **Kill:** position-1
acceptance < 0.55 → K=1 yield ≤ 1.55, DSpark ≤ +20%, ~31 not 37; confidence AUC < 0.6 → adaptive K
dead. For A: `bench_decode` on V4-Flash with the bridge kernels stubbed — if token time drops
< 1 ms, A is not worth its kernel. For the §0 negative: the all-iGPU decode attention stage time —
if < 0.6 ms/layer at V4.1 widths, box-2 contiguous placement becomes competitive at RTT ≥ 300 µs.
