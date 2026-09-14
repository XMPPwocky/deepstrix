# The router one layer early: accurate, and useless for prefetch
### `route_probe.py` on a 1,006-token dump, 2026-09-14

**Question (user):** how accurate is layer L's router if run one layer "too
early" — on the residual entering L-1 rather than L? It is a residual stream, so
the input should barely change.

**Answer: essentially lossless, and it does not help.** Both halves matter.

## Accuracy — the intuition is right

Predictor = layer (l+k)'s OWN gate applied to the residual after layer l
(mean over hc copies, that layer's `ffn_norm`, gate + sqrtsoftplus + bias).
Recall of the true top-6 within the predicted top-6 / top-12:

    k=0 (same layer, via the residual shortcut)   0.709 / 0.862
    k=1 (ONE LAYER EARLY)                         0.703 / 0.846
    k=2                                           0.617 / 0.761
    k=3                                           0.555 / 0.694
    k=5                                           0.465 / 0.592
    k=8                                           0.382 / 0.498
    random                                        0.016 / 0.031

**k=0 -> k=1 costs 0.006.** One layer of lookahead is free; the ~30% that is
missing at k=0 is the mean-over-copies shortcut in the predictor, not the
lookahead. Decay only becomes real past k=2, and even k=8 is 24x random.

## Economics — why it still loses

`prefetch_economics()`, 66% residency, warm half, 3.58 misses/token:

    k  rule   recall-ON-MISSES   prefetch reads/token   MB/token
    2   -1         0.082                 0.6              12
    2   -2         0.160                 1.9              36
    2   -3         0.246                 3.9              73
    2    0         0.467                14.8             278
    2  +.05        0.557                29.5             556
    3   -3         0.238                 4.8              90
    5   -3         0.216                 7.2             135

At k=2 the best-recall rule catches 46.7% of misses for **278 MB/token** = 62 ms
at the measured 4.47 GB/s, against the 23.6 ms those 3.58 misses cost. Every
cheaper rule is also net-negative: rule -3 spends 16 ms of reads to save 5.8 ms.

**The reason is structural.** Overall recall is 0.703 but recall ON MISSES is
0.08-0.47. A miss is by definition an expert the cache chose not to keep — the
cold, unpredictable tail. The router predicts the hot picks well, and those are
already resident. **The prediction is accurate exactly where it is useless.**

This is the same shape as the inclusive-cache failure measured today
(`WHY_THE_BIG_POOL_REGRESSED.md`): anything that targets "what is likely to be
picked" ends up targeting what is already there. Only structures keyed on what
the other tier LACKS — the victim cache — are exclusive by construction.

## What would change the verdict

The economics are precision-bound, not recall-bound: each speculative read is a
full 18.8 MB expert. Two things move it:

  * **A smaller expert.** At ~2.5 bits/weight (11.06 MB) the same rules cost 41%
    less — still negative at these ratios, but it compounds with residency.
  * **A cheaper miss.** These numbers price a read at box 2's 4.47 GB/s. Box 1's
    idle drive does 4.93 GB/s and both could run at once (~8.4 GB/s aggregate),
    which roughly halves the prefetch cost. Rule -3 at k=2 would then spend
    ~8 ms to save 5.8 — close, still short.

Neither makes prefetch a lever on its own. Recording it so the "predict routing
and prefetch" idea, which has come up repeatedly, is closed with numbers.

Reproduce: `nix-shell -p python3Packages.{torch,numpy,safetensors,pillow,transformers}
--run 'python3 scripts/v41_oracle/route_probe.py ~/.cache/deepstrix/v41/agentic/main'`
