# Static expert placement does not generalize; use a victim cache
### held-out evaluation, 2026-09-14

Two levers died here and a better one replaced them. The cause in both cases was
the same methodological error: **a placement fit and scored on the same trace is
an oracle bound, not a deployable policy.**

## The measurement

Fit frequency ranks on tokens 0..1728 of the routing trace, score on 1728..3456.

    k/layer  VRAM GB  IN-SAMPLE  HELD-OUT  oracle(test)  retained
          4      3.0      15.2%      5.0%         16.4%     30.5%
          6      4.5      19.9%      6.6%         22.0%     30.0%
          8      6.0      23.8%      8.1%         26.6%     30.6%
         16     12.0      35.1%     13.4%         39.9%     33.7%
         22     16.5      41.2%     16.9%         46.5%     36.4%
         32     24.1      49.3%     22.5%         54.8%     41.0%

**A static placement retains only ~30% of its in-sample coverage.** Routing here
touches a mean of 352 distinct experts per layer out of 384 and drifts between
text segments, so "hottest in the past" is a weak predictor of "hottest next".

## What this kills

**1. "Refit `hot_experts.txt`" — VOID.** `project_v41_dgpu_hot_tier_policy` measured
the shipped file at 6.9% of picks against a 29.9% oracle and concluded the policy
captured "only 23% of achievable" and should be refit before doing the plumbing.
A freshly refit placement scores **6.6% held-out** at the same k=6. The shipped
file was not badly chosen. The 23% "gap" WAS the in-sample/held-out gap. There is
no refit to do, because there is no policy defect — the approach is the defect.

**2. The static exclusive band — overstated.** `DECODE_MISSES_ARE_GEOMETRY.md`
priced box 1 pinning the next k experts per decoder layer at 51 ms (k=48, 960
slots) from ranks fit and scored on the same half. Held-out it is **41 ms**. Real,
and ~80% of the claim, but the claim was not measured the way it was stated.

## What replaces them: a victim cache

Box 1 catches what box 2 **evicts**. Exclusive by construction — box 1 can only
ever hold what box 2 does not — and adaptive, so it tracks drift instead of
assuming stationarity. No placement file, no refit, no trace needed.

    box1 slots    GB | STATIC band  ms | VICTIM cache  ms | b1 serves/tok
             0   0.0 |        20.4 134 |         20.4 134 |          0.0
           320   6.0 |        18.1 119 |         16.9 112 |          8.2
           640  12.0 |        16.1 106 |         14.2  94 |         17.6
           960  18.0 |        14.2  94 |         12.0  80 |         27.4
          1280  24.1 |        12.4  82 |         10.4  68 |         33.6
          1920  36.1 |         9.3  62 |          7.9  52 |         42.4

All held-out. The victim cache wins at every size — **54 ms at 960 slots against
the static band's 41 ms** — while needing strictly less machinery.

### Why it is affordable

A box-1 fill costs one read from box 1's own NVMe: **4.10 ms**, measured, on the
drive that is otherwise completely idle during decode. Decode's structure is
`max(box1 iGPU MoE, box2)` with box 2 the long pole (35 ms vs 18.2), so box 1 has
tens of ms of exposed wait per token in which to fill, at zero wall-clock cost —
and every expert it catches removes a request from the long pole.

### Implementation sketch

Box 2 already knows which picks missed; the response needs a miss bitmap (or an
eviction list) so the hub can fill. Box 1's decode LRU then pages those from its
own disk in the shadow of the next layer's remote wait. The catch-all decision in
`forward_layer.rs` consults box 1's LRU first, which it already does under mode 1
— but a victim cache is history-dependent in the same way mode 1 is, so it needs
the mode-2 determinism analysis re-run before shipping, or an explicit accept of
history-dependent partitioning.

**Blocking constraint, unchanged:** box 1's pool is 2,766 slots of which 2,688 are
prefill's pinned windows (`forward_layer.rs:2320`: "Box 1's decode LRU is only ~25
slots"). 52 GB is the measured ceiling, 76 GB OOMs. So the sizes above are not
free — 960 slots costs 7 of 21 windows. **This lever is worth building AFTER the
hub swap**, when box 1 becomes a pure expert executor with no dense weights, no KV
and no prefill windows to defend.

## Method note

Third time this project has been bitten by evaluating a policy on its own fitting
data (see the "tautological oracle" in `INDEXER_PORT_PLAN.md`'s review). The rule
that would have caught all three: **a placement, ranking or threshold fit from a
trace must be scored on a DIFFERENT trace, and the in-sample number must never be
quoted as the expected result.**

Reproduce: `scratchpad/holdout.py`, `scratchpad/adaptive.py`.
