# Is expert prefetch plausible? — trace study, 2026-09-18

Source: `~/logs/picks-20260918-1501.trace` (`V41_PICK_TRACE`), 2.74 M rows =
**19,139 decode tokens** x 40 layers x 6 picks, plus 101 prefill chunks.
Scripts: `scripts/prefetch_study.py`, `scripts/prefetch_control.py`
(per-token grouping, which `scripts/analyze_picks.py` explicitly does not do).

Caveat: captured 15:01, i.e. BEFORE the ctx-truncation fix (b154602). The fix
changes what the model attends to and therefore its routing, so treat the
*absolute* rates as a snapshot; the *shape* results below are what matter and
they reproduce the live miss rate to within 20%.

## Verdict

**Speculative prefetch for decode is dead, for a structural reason.** Not "the
correlation is weak" — the correlation is real. The cache is already large
enough to absorb every short-range reuse, so what is left over is, by
construction, the part no recency signal can see.

**Two things that ARE alive:** `V41_B1_PREFETCH=1` (built, default off, never
A/B'd since the partition landed) and layer-ahead paging in PREFILL, which
needs no prediction at all.

## 1. Histogram — skewed, but with no small hot set

Decode, over all 15,360 (layer, expert) pairs:

    pairs touched at all        14,387 / 15,360   = 93.7%
    top  1% of pairs (143)      18.6% of picks
    top 10% of pairs (1438)     57.3%
    top 50% of pairs (7193)     94.7%

Zipfian enough that caching works, flat enough that *pinning* does not: you
need half the address space to cover 95% of demand. The live miss histogram
agrees from the other side — `top16_pct=0.6`, `by_layer` flat from 723 to
1518 across all 40 layers. There is no hot set among the missers to pin.

## 2. Miss curve — we sit on the steep part

LRU miss rate from the exact stack-distance distribution:

    C =  4,454 slots   12.54%
    C =  6,000          7.16%
    C = 10,614          1.15%
    C = 15,360          0.31%

Production is box 1 4,070 + box 2 6,160 = **10,230 slots**. Simulating box 1's
partition (39.7% of ids by hash) against its own 4,070 slots gives **1.43% miss
= 1.35 misses/token**, against a live heartbeat p50 of **1.67** — the model
reproduces production.

Reading the curve where we actually sit: **+50% slots is -3.7x misses.** That
is a capacity lever, and it is much stronger than anything prefetch offers.

### What it costs today

178 live decode heartbeats:

    tok/s        p10  7.80   p50 10.30   p90 12.60
    miss/tok     p10  0.70   p50  1.67   p90  3.38
    ms/miss      p10  8.39   p50  8.85   p90  9.29
    => blocking box-1 disk    p50 14.7 ms/token   p90 28.7 ms/token
       as a share of a token  p50 14.9%           p90 23.1%

`ms/miss` is remarkably tight (8.4-9.3) — that is the dm-crypt pread, not
queueing. The variance is all in *how many*.

## 3. Correlations — real, and measured against the right null

**Same layer, token to token.** 1.70 of 6 picks persist into the next token.
Against a null that draws independently from each layer's OWN empirical
marginal (0.67), that is **2.6x** — genuine temporal structure, not just skew.

**Across layers, within one token.** Against a uniform null this looks
spectacular (15-27x). Against the correct null — a predictor that ignores the
source layer and just names the target layer's 6 most frequent experts — it
collapses:

    src -> dst    conditional   marginal-only   gain
      0 ->  1        0.347         0.151        2.30x
     19 -> 20        0.420         0.302        1.39x
      0 -> 20        0.368         0.302        1.22x
     20 -> 21        0.296         0.271        1.09x
     30 -> 39        0.236         0.148        1.59x

Most of the apparent cross-layer signal was Zipfian marginals. Real residual
correlation is 1.1-2.3x, strongest at layer 0->1 and at the CED seam 19->20.

## 4. Why prefetch cannot pay — the decisive measurement

For every real box-1 capacity miss, how long since that expert was last picked?

    p1  :   536 tokens        within last  1 token  : 0 misses
    p25 :   936               within last  8 tokens : 0
    p50 : 1,265               within last 32 tokens : 0
    p90 : 3,222               within last 64 tokens : 0
    p99 : 8,761

**Zero misses anywhere inside 64 tokens.** Not "few" — zero, out of 25,885.

This is structural. A 4,070-slot LRU fed by ~7.6 new distinct keys per token
retains roughly 500 tokens of history, so *anything recent is already
resident*. A recency prefetcher's entire target set is already in the cache;
it would issue reads for nothing. (22% of misses are first-touch cold, which
no predictor of any kind can see.)

Scoring the cross-layer predictor on the only population that matters — the
accesses that actually MISS:

    L19 -> L20   recall@32  0.167   (marginal-only 0.023)
    L0  -> L20   recall@32  0.076
    L0  -> L39   recall@32  0.035

7x over marginal, so the correlation survives on the miss subset — but
recall@32 means reading **32 x 18.8 MB = 600 MB speculatively** to catch ~17%
of one layer's misses, i.e. to save ~3 MB of critical-path read. **~200x
underwater.** No tuning rescues that.

## 5. What to do instead

### a. `V41_B1_PREFETCH=1` — built, off, and now clearly the right shape

Not speculative prefetch: **miss-triggered async fill**. A box-1-share pick
that is not resident goes to box 2 for THIS token (~14 ms RTT) while box 1's
disk read runs on a background thread and is admitted at a token boundary.

Today, with it off, that pick is computed locally and `ensure` does the pread
**synchronously on the layer's critical path** — the 14.7 ms/token (p50) above.

It was landed default-off (c346a09) because the first A/B showed zero miss
reduction: box 1's smaller LRU was a strict subset of box 2's. **That reason no
longer applies** — `V41_T2_PARTITION=1` (fedbea2, later) gives the two boxes
disjoint id spaces, which is a stronger version of the exclusivity the
residency hints were added to buy. It has not been A/B'd since.

Expected: removes blocking disk from decode; costs box-2 link traffic and
~1.3 ms x 8 admissions per token of H2D. Needs a back-to-back A/B, one binary.

### b. Prefill: layer-ahead paging needs no prediction

Distinct experts touched by ONE 512-row chunk at ONE layer:

    L0  p50 318 / 384  (83%)      L20 p50 107  (28%)
    L10 p50 259        (67%)      L30 p50 129  (34%)
    L19 p50 322        (84%)      L39 p50 154  (40%)

An encoder-layer chunk touches ~83% of the layer. So "page all of layer L+1
while layer L computes" is **100% recall at ~83% precision with zero
prediction** — the union is nearly the whole layer anyway. Decoder layers
20-39 are narrow (28-40%) and would not pay.

Prefill hit is 0.88-0.90, the worst of the two phases, so this is where the
remaining paging headroom actually is.

### c. Capacity still beats both

+50% slots = -3.7x misses (section 2). Nothing above comes close.
