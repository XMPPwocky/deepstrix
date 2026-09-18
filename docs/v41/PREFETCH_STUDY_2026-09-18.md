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
it would issue reads for nothing.

**CORRECTION (same day).** The "22% of misses are first-touch cold" figure above
is a cold-START artifact, not a property of the workload. Measured per 2,000-token
window with the cache never reset, cold share runs 83.4% / 15.3% / 7.9% / 2.9% /
1.5% / 0.5% — it decays to a **settled 1.6%**, and 22.2% is just the first window
dominating a whole-trace average taken from an empty cache. Production's pool
persists across requests, so it sits in the settled regime. So ~98% of steady-state
misses are CAPACITY misses, not compulsory ones. That makes the irreducible floor
far smaller than stated and slightly *helps* the case for prediction; it does not
change the conclusion, which is that recall on the cold tail is the wall.
Cold misses are also irreducible only for a DEMAND-FILLED cache — preloading
eliminates them, and box 1's pool has a load step.

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

### b. Prefill layer-ahead — BUILT, and MEASURED INERT ON BOX 1

Implemented behind `V41_PREFILL_READAHEAD=1` (default off) as a page-cache
hint, not a device-side prefetch: a background thread fadvises `WILLNEED` on
the byte ranges layer L+d will miss on, while layer L's MoE runs. It touches no
pool slot, no remap and no LRU, so it cannot race the queued MoE kernel of the
layer it runs ahead of, and cannot evict a resident expert.

**It cannot pay on box 1 as configured, and the reason is one number:**

    total 93 GB   used 92 GB   buff/cache 1.5 GB   available 1.1 GB

A 78 GB pager pool on a 93 GB box leaves ~1.5 GB of page cache against 289 GB
of experts (0.5%). There is nowhere to read ahead *into* — a hinted page is
reclaimed long before the layer that wants it arrives. The code is committed
because it is correct, safe and default-off, and becomes live on any box with
real page cache; **do not enable it here expecting a win.**

The premise that motivated it still holds and is worth recording: one 512-row
chunk touches ~83% of an ENCODER layer's experts, so a layer-ahead fill needs
no prediction at all. It is the *destination* that is missing, not the signal.

### b2. O_DIRECT on box 1 — TESTED, NOT A WIN (1.01x)

The same 1.5 GB page cache suggests box 1 should take box 2's `O_DIRECT` path
(box 2 measured 4.70 ms vs 12.55 ms buffered for one expert, 2.7x). It does
not transplant. Alternating arms read-by-read with a FRESH random offset every
read, so no read ever warms another, 18.8 MB each, box under live load:

    buffered   min 10.21  p50 15.89  p90 56.17  max 102.06 ms   (1.19 GB/s)
    O_DIRECT   min 10.02  p50 15.78  p90 38.57  max  57.00 ms   (1.20 GB/s)

**1.01x at p50**; only the tail improves (p90 1.46x, max 1.79x). A first pass
that reused offsets across arms showed "buffered 40.97 -> 5.08 ms" and was
pure self-warming — the arms shared page cache. Worth knowing the tail is
tighter, not worth a default change on this evidence.

Note production's live `ms_per_miss` is 8.85 ms, FASTER than either arm here:
it splits one expert across `V41_EXPERT_PREAD_THREADS=8`. The read path is
already close to what this drive gives under load, which is the argument for
moving the read OFF the critical path rather than trying to make it quicker.

### b3. Prefill: the signal, for whoever has the RAM



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

### d. HOW MUCH could prediction buy? The economics, from first principles

Scripts: `scripts/belady_bound.py`, `scripts/prefetch_economics.py`,
`scripts/ced_predictor.py`.

**Prefetch cannot reduce the fetch COUNT.** At a fixed capacity C the minimum
number of disk fetches over a trace is exactly Belady's OPT; a prefetched
expert occupies a slot, and OPT already assumes optimal slot use. So prefetch
moves a fetch EARLIER, never removes it. The objective is therefore exposed
latency, not miss rate.

**Bandwidth is not the constraint — this is the surprise.** We issue 1.17
demand misses per token against a drive that serves ~13 expert-reads in a 97 ms
token: `rho = 0.09`. Modelling a demand miss as `c_s/(1-rho)` with
`c_s = 8.85*(1-0.15) = 7.50 ms`, net gain over recall h and precision p:

                p:    1%     2%     5%    10%    25%    50%   100%
        h= 25%      SAT    SAT   -42%    +3%   +19%   +23%   +25%
        h= 75%      SAT    SAT    SAT   +24%   +68%   +73%   +75%
        h=100%      SAT    SAT    SAT  +100%  +100%  +100%  +100%

**Break-even precision is ~10%.** A predictor may be wrong nine times out of
ten and still pay. Below ~5% the prefetch stream saturates the drive and the
1/(1-rho) term punishes every miss it failed to hide. Precision was never the
blocker; RECALL on the cold tail is.

**The ceiling is +10% tok/s.** A perfect oracle takes exposed latency from
9.66 to 0.02 ms/token, i.e. 10.3 -> 11.4 tok/s. That bounds every decode
prefetch scheme, including all of the above.

**Every id-based predictor in this trace is dead**, measured end to end with
pollution charged (prefetched experts take slots and evict):

    BASELINE                    demand/tok 1.17                    9.66 ms
    ORACLE (p=1)                demand/tok 0.00  prefetch 1.50     0.02 ms  +99.8%
    recency (last token)        demand/tok 1.17  prefetch 0.00     9.66 ms    0.0%
    frequency top-32/layer      demand/tok 1.17  prefetch 0.01     9.65 ms   +0.1%

Recency and frequency issue ~NOTHING: every expert they name is already
resident. That is section 4's dead zone restated as economics.

### e. The CED-seam predictor (L19 -> L20..39) — closest miss

Layer 20 is the encoder/decoder seam and 19->20 was the strongest cross-layer
pair (7x over marginal on the miss subset), so predicting the WHOLE decoder
from the seam gives half the network as lead time: 17 of 20 decoder layers
clear the 8.85 ms read.

     n_pred  demand/tok  prefetch/tok  precision    rho   exposed   vs base
          0        1.17          0.00          -  0.091    9.66ms   baseline
          8        1.17          0.21       5.1%  0.107    9.86ms     -2.1%
        128        1.17          0.23       4.8%  0.109    9.89ms     -2.3%

The first predictor that FIRES at all (0.23 non-resident candidates/token vs
0.00 for recency), but 4.8% precision is just under break-even, so it lands at
-2%. Demand misses do not move: its hits are cancelled by what its admissions
evict. It needs ~25x the recall and ~2x the precision.

**Caveat, and the one live variant.** This bounds ID-BASED prediction only —
the trace carries 6 discrete ids per layer, which is all the predictor above
gets. The router works from the 5120-dim hidden state. Untested and not
refutable from a trace: at layer 19, speculatively run layers 20-39's GATE
matrices on layer 19's hidden state (20 x 5120 x 384 x 2 = 78 MFLOP, nothing on
a GPU) and prefetch their top-k. At a 10% break-even precision, gate_L(h19)
does not have to rank much like gate_L(h_L) to pay. Needs a model run.

### c. Capacity still beats both

+50% slots = -3.7x misses (section 2). Nothing above comes close.
