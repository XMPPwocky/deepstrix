# Design B — "hide every latency: prediction, pipelining, every device busy" (independent architect, 2026-09-13)

**Verdict.** Routing prediction is dead on this hardware (measured on the trace); the RTT is
hideable; **SSD misses are the un-hidden latency and no scheduling design hides them**. The one big
overlap lever left is *intra-step*: run the speculative verify batch as a two-sub-batch **layer
wavefront** instead of one batched pass. Model: 33.5 → 47 tok/s at p=0.75 (no misses), and it
makes the design RTT-insensitive to ~300 µs. Scripts: `scripts/v41_sched/{trace_analysis,step_misses,sched_sim2}.py`
(routing predictability + LRU; miss Monte-Carlo; DAG/event list-scheduler over dGPU, 2 iGPUs, 2 NVMes, link).

## 1. Design
**A. Layer-wavefront verify.** Standard spec-decode verifies all K+1 positions in one pass with no
acceptance dependency inside the pass. Split them into two contiguous sub-batches A=[bonus,d1,d2],
B=[d3,d4,d5] and run B trailing A by exactly one layer. Legal: B depends on A only through the KV
that A's *attention* writes at the same layer; B never needs A's MoE output; A never needs B. Same
logits as batched verify. Per layer-slot: dGPU runs attn(B,l) while A's experts run
[iGPU1 ∥ link→iGPU2→link ∥ dGPU shared+hot]; then attn(A,l+1) while B's experts run. Each RTT sits
under the other sub-batch's attention; each iGPU always has a request queued; the link always has
one message in flight each way. Busy fractions (model): dGPU 0.91 / iGPU1 0.74 / iGPU2 0.85 /
link 0.89 (vs one token in flight: dGPU idle ~35%, iGPUs ~50%). Why not 3 stages: the dGPU chain
is launch-bound (0.55 ms per sub-batch regardless of b) → 2+2+2 is dGPU-bound at 92 ms/step; it
becomes right only if fusion gets the chain to ≤ 0.35 ms (then 2+2 → 55 tok/s, 2+2+2 → 48 @ RTT 300).
**B. Send-before-route.** Ship the post-attention normed activation the instant it exists; box 2
runs the replicated router (2–4 MB/layer); box 2 echoes its 6 ids, hub compares, re-requests on a
near-tie mismatch. ~30–50 µs/layer ≈ 5%. Last in build order.
**C. Misses are branches, but not hideable.** Issue at router time, 64-thread pread straight into
the owning box's device slot, all positions of a sub-batch-layer together; global residency
directory on the hub; **disjoint** LRUs; cold-tier ownership follows disk speed (box-2 plaintext
NVMe). A miss at layer l for position i stalls every position ≥ i at layers > l; misses at
different layers serialise; the wavefront covers only sub-batch-B misses (~65% of 4.8 ms) — A's are
exposed. Deferred correction is not exact (attn_norm, q_norm, hc_mixes are nonlinear).
**D. Adaptive K from DSpark's ConfidenceHead** — fewer wasted positions → fewer wasted expert bytes
and misses (at p=0.75, 45% of positions are dead weight; with misses, wave 2+2 beats 3+3).
**E. Cross-step bubble** (head 1.4 + accept + drafter ~5 ms ≈ 7–10%/step): early-exit — if A's head
already shows a rejection inside A, B is dead; draft immediately from A's main_x (44% of steps).
Engram rows for the drafts gathered during the drafter/head. True cross-step overlap needs a second
request (server).
**Rejected:** layer-split across boxes (residency collapse + iGPU attention ~2× slower); starting
layer l+1 on the local-only partial sum (nonlinear); prefetch-into-MALL from predicted ids.

## 2. Arithmetic (sim; attn 0.55+0.03(b−1)+0.03 router, pick 0.088 ms, shares 38/47/5, miss 4.82 ms
single-server per NVMe, head 1.4, drafter 5.2, tokens/step = (1−p^{K+1})/(1−p))
No misses, p=0.75:

| config | RTT 100 | RTT 200 | RTT 300 |
|---|---|---|---|
| single, no spec (plan 26) | 25.0 | – | 20.8 |
| batched K=2 (plan 37) | 33.5 | 31.7 | 30.0 |
| **wave 2+2** (K=3) | 42.8 | 42.8 | 41.5 |
| **wave 3+3** (K=5) | **47.2** | 44.6 | 42.3 |

Acceptance sensitivity, wave 3+3, RTT 100: p=0.60 → 34, 0.75 → 47, 0.85 → 60, 0.95 → 76.

With misses (RTT 100, p=0.75) — **the real regime**:

| residency → misses/position (trace) | batched K=2 | wave 2+2 | wave 3+3 |
|---|---|---|---|
| 75% → 1.5 (4.82 ms) | 25.9 | 31.3 | 32.0 |
| 66% → 3.5 (4.82 ms) | 21.5 | 26.5 | 25.0 |
| 75%, cold tier on box-2 plaintext (3.0 ms) | 29.3 | 36.4 | 37.5 |
| 66%, same | 24.1 | 30.0 | 29.1 |

Trace facts (`expert_trace_v4flash.bin`, warm LRU, global slots): misses/token at 50/60/66/75/85%
resident = 10.7 / 5.3 / 3.4–3.7 / 1.45–2.2 / 0.83–1.28. A 6-position step at 66% has ~21 distinct
misses touching ~15 of 40 layers — draft positions barely share cold experts: +60 ms on a 70 ms
step, serialised by the dependency chain. Busy fractions at 75% residency: 0.61 / 0.50 / 0.58 /
NVMe 0.24+0.23 — the idle is NVMe wait, visible as gaps on all four tracks at once.
Skepticism: trace is V4-Flash (256 experts, 43 layers); DSpark p unmeasured; drafter 5.2 ms a guess
(2→8 ms = 49→45 tok/s); attention at b=3 assumed +0.06 ms; the sim's dGPU is FIFO.

## 3. Quality — all exact
Wavefront = batched verify's math; rejection sampling distribution-preserving; send-before-route
guarded by id echo; adaptive K / prefetch change speed only.

## 4. Cost and build order
1. RTT ping-pong at real sizes (15 KB out / 60 KB back), MTU 1500 vs 65520, busy-poll — ½ day (sets
   the sub-batch count; 47 → 42 from 100 → 300 µs). 2. B=k verify on the *decode* graph — 2–3 weeks;
shared prerequisite for every DSpark design. 3. Two-sub-batch scheduler (two streams per device,
per-layer events, the single KV edge) — ~1 week; validate **single-box first** (iGPU1 + dGPU).
4. Expert RPC with 2 outstanding requests, residency directory, disjoint LRUs, misses at router time
to the faster NVMe — 1–2 weeks. 5. DSpark drafter + confidence-truncated K + early-exit — ~1 week.
6. Send-before-route + id echo — 2 days, +5%.

## 5. Experiments and kill results
- **Done: routing prediction from ids is dead.** Cross-layer table predictor, train EN → test ZH:
  recall of the 6 true picks within top-12 = 0.14–0.16 vs frequency baseline 0.09 (random 0.05),
  flat in k; catches 5% of LRU misses at 66% while issuing ~140 non-resident prefetches/token.
  Kill stands unless a hidden-state probe is ≥ 10× better and residency ≥ 75%. Cheap follow-up:
  apply gate_{l+k} to the copy-mean of residual_l from the t200 dump — half a day.
- **Biggest un-run assumption: DSpark acceptance p.** ~200 tokens of real traffic through the CPU
  oracle + drafter. p < 0.6 → wavefront ≈ 34 no-miss, tied with batched K=2 under misses.
- **LRU miss rate at V4.1's 384-expert routing** (trace hook once V4.1 runs): > 3.5/position at 66%
  → no schedule reaches 30 tok/s; residency (bytes) is the lever.
- Combined kill: p < 0.6 and misses > 3/position → wavefront ≈ batched ≈ 20 tok/s.
Composition: fusion shortens the launch-bound chain → more wavefront stages (0.35 ms → 3 stages,
55 tok/s at RTT 100); a smaller cold tier is the only thing that fixes the miss rows; a second
server request is the only cross-step filler.
