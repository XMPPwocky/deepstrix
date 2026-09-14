# Speculating against a degraded expert — and the better idea inside it
### user proposal + adversarial review, 2026-09-14

## The proposal

Keep all 15,360 experts resident at IQ2_XXS (~140 GB). Keep a separate MXFP4
cache paged from NVMe. On a cache miss, **do not stall** — compute that expert
with the tiny copy and let the token proceed while the accurate expert loads.
When it arrives, re-run the rest of the token with accurate weights and
rejection-sample the speculative token against it; on reject, roll back. In
effect, speculative decoding where the drafter is the same model with some
experts degraded, and the draft/verify boundary is set by **cache residency**
rather than a separate draft model.

## What is right about it

**It needs no prediction.** Routing prefetch died here on recall — 0.08-0.47 on
MISSES, because the router predicts well exactly the picks that are already
resident (`ROUTER_LOOKAHEAD_AND_PREFETCH.md`). This scheme acts on a miss that
has already happened, so its "recall" is 1.0 by construction. That is a real
structural advantage and it is why the idea deserved the arithmetic.

**The draft is only mildly degraded.** 139.9 GB of IQ2_XXS leaves 35 GB = 1,862
MXFP4 experts = 12.1% residency, and an LRU at 12.1% still hits **0.769** (warm,
measured on the routing trace). So ~55 of ~240 picks/token run tiny — a hybrid
model, not the IQ2 model. Acceptance would be high.

**Rejection cascade is not the problem.** A 6.6 ms verify against a >=33 ms token
leaves the engine under one token ahead; at ~0.9 acceptance that is ~10% redo.

## Why it still loses: it converts latency into bandwidth

Verification needs the accurate weights of exactly the experts that were
degraded. So the scheme does not remove the read — it defers it, and by dropping
MXFP4 residency from 61% to 12% it **triples** it:

    config                            MB/token off disk   at 4.47 GB/s   ceiling
    today, 61% resident, ~18 miss       338              75.7 ms       13.2 tok/s
    proposal, 12% resident, ~55 deg     940             210.3 ms        4.8 tok/s
    mixed precision, all resident         0                  --        no disk

**Hard bound ~4.8 tok/s, below today's 4.1-4.6 and 6x short of the goal.** The
drive is flat at 4.47 GB/s across queue depth 1-16 (measured), so asynchrony buys
nothing against a saturated link, and the 23x batching win is a COMPUTE win —
batching 5 tokens dedups distinct experts only 3.2x and does not raise disk
bandwidth at all.

## The better variant, which the proposal contains

Drop the speculation; keep the two precisions. Allocate the whole budget:

    maximise x s.t. 18.80x + 9.11(15360 - x) <= 175,000 MB
    => x = 3,619 experts (23.6%) at MXFP4, 11,741 at IQ2_XXS, 175.0 GB exactly

  * **Zero disk in the decode path.** Every expert is resident in one precision
    or the other. No misses, no stalls, no prefetch, no drafter, no rollback.
  * **2.58 bits/weight average** against uniform IQ2_S's 2.50 — the same
    footprint, but the bits are allocated by pick frequency instead of
    uniformly.
  * At 23.6% residency an LRU hits **0.887 warm**, so **~89% of picks run at full
    MXFP4** and only the cold ~11% are degraded. Uniform IQ2_S degrades *every*
    pick including the hottest.

Decode then falls to the ~88 ms zero-miss floor (11.4 tok/s), and conventional
DSpark at E=4.13 on top projects to 38-42 tok/s.

## The one thing that must be adaptive

The MXFP4 tier must be chosen by an **LRU/adaptive rule, not a static frequency
file**. Static placement retained only ~30% of its in-sample coverage held-out in
this system (`STATIC_PLACEMENT_DOES_NOT_GENERALIZE.md`); the 0.887 above is an
LRU number and a static tier would deliver far less. This also means the tier
needs both precisions materialised on disk and a cheap swap, since "which expert
is hot" moves.

## Verdict

Do not build the speculative version — it is dominated on its own binding
constraint. Build **mixed precision by residency**. It keeps the no-prediction
property that made the idea attractive, removes disk from decode entirely, and
needs no acceptance sampling, no rollback and no drafter.

Keep the speculative form in reserve for exactly one contingency: if IQ2_XXS
quality on the cold 11% proves unacceptable, speculative upgrade is the way to
recover it, and it is still the only miss-hiding mechanism here that requires no
prediction.
