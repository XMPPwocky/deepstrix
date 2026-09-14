# Which kernel regressed under the 81.6 GB pool
### `DEEPSTRIX_TOKEN_PROFILE=1` stage diff, 2026-09-14

The 81.6 GB pool is a reproduced loss. Rather than keep guessing at a cause
(I floated page-cache starvation and withdrew it), here is the stage table.

Both arms: same prompt, same warm-up request, 256 tokens, T=0, profiler on.
Steady-state half only.

    pool 52          256 tok / 62.6 s = 4.09 tok/s
    pool 76 + cap3   256 tok / 75.0 s = 3.41 tok/s

    device stage                              A(52) ms  B(76) ms    delta
     igpu  igpu.routed_moe                      15.64     29.61   +13.97
     igpu  igpu.peer_push_ffn_moe.wait           6.56     20.15   +13.59
     dgpu  dgpu.ffn_combine.wait                 6.31     19.60   +13.29
     igpu  igpu.moe.wait   (waiting for box 2)  21.01     20.22    -0.79
     dgpu  dgpu.peer_push_ffn_input_norm.wait   19.40     18.94    -0.46
     ... every other stage moves < 0.1 ms

**`igpu.routed_moe` — box 1's own iGPU MoE kernel — +13.97 ms/token, +89%.**
The next two rows are the SAME event observed from the devices that block on it
(`feedback_exposed_wait_is_not_work`), not independent regressions. Nothing else
moved.

## The cap worked; that is the problem

`V41_LOCAL_CLAIM_MAX=3` did what it says. At pool 52 box 1 held ~25 experts and
computed ~0.5 picks/layer; at pool 76 it holds 1,396 and now computes up to 3.
The extra ~2.5 picks/layer x 40 layers is the +13.97 ms. So this is not a bug in
the cap — it is the cap being *reachable* for the first time.

## What it prices

    box 1 iGPU   140 us per expert per layer   (13.97 ms / 100 extra picks)
    box 2         87 us per expert             (measured, 216 GB/s, ~94% of achievable)

**Box 1's iGPU is ~1.6x SLOWER per expert than box 2's**, because it shares the
device with the entire dense chain — attention, norms, head — while box 2's iGPU
does MoE and nothing else. Moving MoE work to box 1 is a loss on compute alone.

And it bought almost nothing on the other side: box 2's leg fell **0.79 ms for
100 picks moved = 8 us/pick**, against 140 us/pick paid. A 17x bad trade.

## Why box 2 barely benefited — and what the fix has to be

Box 1 can only serve a pick it HOLDS. Its LRU fills with what it sees, which is
the same hot set box 2's LRU holds, so the picks box 1 took were overwhelmingly
ones box 2 would have **hit** (87 us) rather than **missed** (6.6 ms). Paying
140 us to save 87 us is the 17x trade above.

This is the inclusive-duplication failure predicted by the held-out simulation
(`STATIC_PLACEMENT_DOES_NOT_GENERALIZE.md`: 1024 inclusive slots absorb 96.8
hits/token and remove 1.6 misses), now confirmed at kernel granularity.

**The break-even is unambiguous.** Box 1 should serve a pick only when box 2
would MISS it:

    serve a pick box 2 would hit    140 us spent,  87 us saved   ->  -53 us
    serve a pick box 2 would miss   140 us spent, 6687 us saved  -> +6547 us

So residency must be EXCLUSIVE — box 1 holding what box 2 evicted, i.e. the
victim cache — and the claim cap must stay, because even correct picks cost
140 us each against box 1's own leg. Residency, exclusivity and claim budget are
three separate decisions and all three have to be right; getting two of them
right (big residency + a claim cap, as here) still loses.

## Caveat

The stage table accounts for ~14 ms of a ~49 ms/token wall-clock regression
(244 -> 293 ms). Stage rollups are known not to sum to the token here — an
earlier measurement put ~31 ms/token outside every instrumented phase. So
`igpu.routed_moe` is identified as THE regressed kernel, but it is not the whole
of the wall-clock delta, and the remainder is uninstrumented.
