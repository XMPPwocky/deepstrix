# Decode is capacity-bound, not policy-bound (measured 2026-09-14)

Decode sits at **271 ms/token (3.7 tok/s)**, of which `remote_rtt` is 230 ms (85%).
This prices every remaining cache-policy lever and finds that none of them reach
30 tok/s, because the binding constraint is RAM, not policy.

## Where box 2's 230 ms goes

`--log-every 2000` on the expert daemon, during a 200-token generation:

    expertd B=1  n=2000 | us p50/p90/p99: read 1/2/9  queue 3/28/122
                          h2d 51/106/186  gpu 651/744/950  d2h 17/22/52
    expertd: page stats requests=48431 misses=5991 hit=0.8763
             ms_per_miss=9.46 (read 8.30 h2d 1.16)

Per request box 2 spends only ~0.65 ms computing. The cost is the **miss path**:
12.4% of picks miss, each costs **9.46 ms** (8.30 ms reading 19.25 MB off its own
NVMe at ~2.3 GB/s, plus 1.16 ms H2D). At ~160 picks/token that is ~20 misses
= ~188 ms/token, which is the observed `remote_rtt`.

The submit-mask fix made this WORSE on purpose: box 2's hit rate fell 0.9735 ->
0.8763 because it now actually computes every pick it is handed instead of
silently dropping 77% of them. The old hit rate was measuring a box that wasn't
doing the work.

## Pricing the global pool

`remote_experts.rs` calls per-layer LRU regions a limitation and a global pool
"strictly better". Replaying the Sep-13 256-token decode trace through both
policies at equal total capacity, scoring only the **last quarter** (the trace is
still discovering 9.3 new (layer,expert) pairs/token at its end, so whole-trace
numbers are dominated by cold start — an earlier pass of this analysis reported
20 ms/token from exactly that contamination):

    slots  /lyr | per-layer hit  miss/tok  ms/tok | global hit  miss/tok  ms/tok |  saved
     5000   125 |        0.8942      25.4   240.2 |     0.9035      23.2   219.2 |  21.0ms
     6160   154 |        0.9280      17.3   163.5 |     0.9415      14.0   132.9 |  30.6ms   <- box 2 today
     8000   200 |        0.9541      11.0   104.2 |     0.9594       9.7    92.1 |  12.1ms
    12000   300 |        0.9594       9.7    92.1 |     0.9594       9.7    92.1 |   0.0ms

The model predicts 0.9280 at box 2's actual 154 slots/layer against **0.8763
measured**, so it is a fair if slightly optimistic instrument.

**The global pool is worth ~30 ms/token**: 271 -> 241 ms, 3.7 -> 4.1 tok/s. Real,
cheap in principle, and nowhere near the goal. It is also not free to build: the
wire format hands the executor a contiguous `[base_slot, base_slot+n)` range per
layer, so a global pool changes the executor contract, not just the policy.

## The wall

    per-expert         19.25 MB   (box 2 log: 6160 experts = 115.81 GB)
    full expert set    288.8 GB   (40 layers x 384)
    RAM, both boxes    224.0 GB   (96 + 128)

**The two boxes together cannot hold the expert set at Q8_K.** That is the whole
story. Every caching policy is rearranging 224 GB of shelf space around a 289 GB
problem, so some fraction of picks must always come off disk at ~9.5 ms.

If every expert *were* resident the miss path disappears entirely, and decode is
just compute: box 2 measured 0.65 ms/layer x 40 = 26 ms/token with box 1's leg
concurrent, i.e. **~30-40 ms/token = 25-33 tok/s** — the goal, without DSpark.

Fitting the set needs 12.6-13.3 MB/expert (5.6-5.9 bits/weight) depending on how
much of the 224 GB the non-expert weights, KV and scratch take:

    minus 25 GB non-expert -> 199 GB for experts = 13.3 MB/expert = 5.9 bits/wt
    minus 35 GB non-expert -> 189 GB for experts = 12.6 MB/expert = 5.6 bits/wt

Q8_K is ~8.5 bits/weight. **Q5_K (~5.5) fits; Q6_K (~6.6) does not.** So the path
to 30 tok/s decode is a 5-bit expert format, not a better LRU.

## Consequences

1. **Requantising experts to ~Q5_K is the decode goal's critical path.** It is the
   only change measured here that reaches 30 tok/s. Quality impact is unmeasured
   and is the obvious risk — experts are the bulk of the model.
2. The global pool is a genuine ~11% decode win and is worth doing on its own
   merits, but should not be mistaken for progress toward 30 tok/s.
3. Do not tune the LRU further at Q8_K. The plateau above is flat from 300
   slots/layer; the policy is not what is costing the time.
4. `ms_per_miss` 9.46 is ~2.3 GB/s for a 19.25 MB read, roughly single-stream
   NVMe. A second drive or striped read would cut the miss cost but not the miss
   COUNT; at 14-17 misses/token even a free read leaves the ~26 ms compute floor
   plus queueing. Worth far less than making the misses not happen.

## Method note

The first version of the pool comparison scored the whole 256-token trace and
reported "global saves 20 ms". That number was cold-start contamination: with
10000 slots every miss in the trace was a first-ever touch, which should have been
the tell that the working set had not converged. Always check whether a residency
trace has reached steady state before quoting a hit rate from it.
