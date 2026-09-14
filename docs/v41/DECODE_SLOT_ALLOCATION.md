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

## MEASURED (2026-09-14): a hard prefill constraint caps the fix

Ran the rebalance for real. The optimum the simulation picked **breaks prefill**:

    180/128 -> all four 32K prefill passes return HTTP 500:
      "expert pager: L0 union needs > 128 slots (window 0, stride 128).
       Raise V41_PAGER_STRIDE or give box 2 more of this layer."

Box 1's prefill window is `stride` = 128 wide, so box 1 can page at most 128
experts per encoder layer and **box 2 must own >= 384 - 128 = 256 of every encoder
layer**. Today's 268 clears it; 180 does not. That constraint was not in any model.

Re-solving inside it, with box 2's measured 12 GB of spare RAM:

    config      slots     GB | miss/tok | decode      prefill@32K
    268/40       6160  115.8 |    33.3  | 3.24 tok/s  466 tok/s   (baseline)
    260/68       6560  123.3 |    22.6  | 3.65 tok/s  489 tok/s   <- ADOPTED
    180/128      6160  115.8 |    17.5  | 4.16 tok/s  HTTP 500    <- illegal

**Adopted 260/68**: decode **3.24 -> 3.65 tok/s (+13%)**, prefill unchanged.
Prefill medians were [66.1, 66.0, 59.0] before and [62.9, 59.1, 62.9] after — both
carry a 59 s pass, so read that as NEUTRAL, not a gain.

Two honest caveats on the decode number:

* The model predicted ~5.1 tok/s for the constant-memory rebalance and the illegal
  180/128 actually delivered 4.16 — the simulation over-predicts by ~35%.
* Per-miss cost ROSE as the miss count fell: 9.00 -> 11.16 ms (read 7.86 -> 10.04).
  Fewer, more isolated misses appear to lose read pipelining, which eats part of
  the gain. Any further residency work should expect this.

**What now caps decoder residency is box 1's `V41_PAGER_STRIDE`, not box 2's RAM.**
Widening box 1's prefill window would let box 2 own less encoder and more decoder;
that spends box 1 memory and is the next thing to price.

## Box 2's expert compute is NOT a kernel problem

Per layer request box 2 streams 6 x 18.80 MB = 112.8 MB and takes 651 us p50:

    112.8 MB / 651 us = 173 GB/s

against ~200 GB/s achievable on Strix Halo LPDDR5X — **~87% of achievable**. There
is no kernel win there; at B=1 decode the expert FFN is pure weight streaming.

The consequence is worth stating, because it cuts against the "wall" framing: a
token needs 40 x 112.8 MB = **4.5 GB of expert weights**, and two boxes at 173 GB/s
aggregate to ~346 GB/s = **13 ms/token if everything is resident**. Bandwidth is
NOT what blocks 30 tok/s. Misses are.
