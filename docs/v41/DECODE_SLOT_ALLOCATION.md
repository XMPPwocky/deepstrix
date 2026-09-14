# Box 2's decode misses are an ALLOCATION artifact (2026-09-14)

Found by an architecture review challenging the assumption, then verified against
box 2's own load log and re-simulated. **This invalidates every residency
simulation in `DECODE_CAPACITY_WALL.md`**, which modelled 154 slots/layer uniform.

## What actually runs

    expert shard: L 0 268 experts ... slots    0..268
    ...
    expert shard: L19 268 experts ... slots 5092..5360
    expert shard: L20  40 experts ... slots 5360..5400
    ...
    expert shard: L39  40 experts ... slots 6120..6160

**L0-19 get 268 slots each; L20-39 get 40.** Total 6160, which the daemon reports
as "154 slots/layer (avg)" — and that average is what got modelled as uniform. The
word "(avg)" was in the log line all along.

The source is `~/box2_placement.txt`: lines 1-20 carry 268 comma-separated ids,
lines 21-40 carry 40. `--experts-k 268` takes at most 268 per line, so the decoder
lines' 40 is what binds. The `EXPERTS` default says the same thing:
`L0-L19:116-383` is 268 experts, `L20-L39:344-383` is 40.

That shape is right for CED PREFILL (encoder layers 0-19 run over all tokens; only
a 128-token replay goes through 20-39). It is badly wrong for DECODE, which
traverses all 40 layers every token, so half of them thrash a 40-slot cache over
384 experts.

## What it costs

Converged 1,152-token trace, warm quarter, 9.0 ms/miss:

    allocation          miss/tok   ms/tok |  L0-19   L20-39
    real (268/40)          33.3     299.3 |    2.7   30.5  (92% of misses)
    uniform 154            17.6     158.1 |   10.0    7.5
    optimum ~180/128       17.5     157.2 |    7.3   10.1

**92% of decode misses come from the 20 layers holding 10% residency.** The real
config also calibrates far better against production (33.3 modelled vs ~26
measured) than the uniform 154 that was assumed (17.6) — the model was being
"validated" against the wrong geometry.

## Options, at box 2's 128 GB

    config            slots     GB | miss/tok  ms/tok | production saving -> tok/s
    today   268/40     6160  115.8 |    33.3   299.3  |        --            3.24
    spare   268/69     6740  126.7 |    22.2   199.7  |      ~78 ms          4.3
    spare   268/98     7320  137.6 |    16.5   148.1  |   DOES NOT FIT (128 GB)
    rebal   180/128    6160  115.8 |    17.5   157.2  |     ~111 ms          5.1
    rebal   154/154    6160  115.8 |    17.6   158.1  |     ~110 ms          5.0

(Production scale factor 0.78 = 26 measured / 33.3 modelled on today's config.)

**The spare-RAM-only path is capped.** 268/98 needs 137.6 GB against 128 GB, and
even 268/69 at 126.7 GB + ~0.4 GB overhead leaves under 1 GB of headroom on a box
that has OOM'd before.

So the effective change is a **rebalance at constant memory**, and it is a genuine
TRADE, not a free win: encoder residency falls 268 -> 180 (70% -> 47% of each
layer's experts), and prefill is exactly what encoder residency serves. Both decode
and prefill are goal clauses, so this must be measured on BOTH before adoption.

## Next

1. Regenerate `box2_placement.txt` with 180 ids on lines 1-20 and 128 on lines
   21-40, frequency-ranked per layer (the decode trace gives per-layer frequencies
   for all 40 layers).
2. Restart expertd, 1,200-token decode -> `page stats` miss rate and ms/token.
3. Re-run prefill at 32K and 100K on the SAME daemon and compare against
   356/395 tok/s. If prefill regresses more than decode gains in goal terms, try
   220/108 or make the split phase-aware (a code change).
