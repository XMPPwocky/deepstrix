# The 260/68 -> 164/164 re-split: +22% decode, -37% prefill
### measured 2026-09-14. The cheap falsifier for the global-pool thesis.

## Result

Identical total capacity (6,560 slots, 123.33 GB) on box 2, reallocated from
260 encoder / 68 decoder to a uniform 164/164 built by frequency rank from the
routing trace. Box 1's `V41_PAGER_STRIDE` raised 128 -> 220, since box 1 now owns
384-164 = 220 per layer and the stride must cover it.

    config      decode tok/s             box2 miss/token   prefill cold / warm
    260/68      4.94 (4.40/5.44/4.99)      15.8 - 17.3        345 / 496 tok/s
    164/164     6.04 (5.57/6.50)            9.65              210 / 313 tok/s
                     +22%                    -40%                  -37%

Prefill measured identically in both arms: a 7,208-token prompt with
`max_tokens=1`, two runs (cold then warm), same server build, box 2 restarted
between arms.

## What it proves

**The global-pool mechanism is real and worth ~+22% decode.** The simulation said
per-layer 260/68 gives 20.4 misses/token and a uniform/global allocation at the
same capacity gives 10.4-11.5. Measured: **15.8-17.3 -> 9.65**. Same direction,
same magnitude, third independent validation of that simulator.

Crucially this needed **no code at all** — a placement file and one env var. The
executor-contract rewrite described in `GLOBAL_POOL_FEASIBILITY.md` is NOT
required to capture the win, only to capture it *without the prefill cost*.

## What it costs, and why

**Prefill -37%.** Box 1's pinned prefill windows went from 21 (covering encoder
layers 0..20) to **12** (layers 0..11), because the pool is a fixed 2,969 slots
and raising the stride 128 -> 220 buys fewer windows. Eight encoder layers lose
their pin and must re-page every request.

This is the `encoder >= 256` floor, and it is now measured rather than inferred:
earlier, 180/128 hard-failed prefill with HTTP 500; 164/164 does not fail but
costs 37%.

## The conclusion

The static re-split is a **Pareto trade, not a win** — it buys decode with
prefill. But it establishes both halves of the case for the real fix:

  * the decode win is worth +22% and is driven purely by slot allocation;
  * the prefill cost is caused by a *static* partition that must serve both
    phases at once.

**A phase-aware pool gets the +22% without the -37%**: per-layer regions sized
for prefill's union during prefill, a balanced or global allocation during
decode. The phases never overlap within a request, the slot CONTENTS survive the
switch, and only the bookkeeping changes. That is the version worth building, and
this experiment is its justification.

## Caveats

  * Decode n=2 for the 164/164 arm against n=3 for the baseline, and e2e tok/s
    here has a ~+/-8% noise floor. The +22% clears it and the miss counter
    corroborates independently, but a third run would be cheap insurance.
  * Prefill was measured at 7,208 tokens inside CTX=8192, not at the 100K the
    goal names. The RATIO should hold since the mechanism is window pinning, but
    the absolute numbers are not comparable to the 606 tok/s @100K figure.
  * The 164/164 placement is frequency-ranked from a 3,456-token trace. Static
    placement retains only ~30% of in-sample coverage held out
    (`STATIC_PLACEMENT_DOES_NOT_GENERALIZE.md`), so a *dynamic* balanced
    allocation should do at least as well and probably better.
