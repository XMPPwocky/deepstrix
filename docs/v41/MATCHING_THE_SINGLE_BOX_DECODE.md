# Matching the single-box decode number (2026-09-15)

A public report of DeepSeek V4.1-Flash on ONE 128 GB Framework Desktop (Strix
Halo iGPU, no dGPU) quotes 395 tok/s prefill and **16 tok/s decode**, with peak
GTT 113 GB — and, two days earlier, **6.3 tok/s decode warm / 4.8 cold, "still
the main gap"**. That earlier figure is exactly where this engine sat, on
strictly more hardware. Two things closed it.

## 1. The measurement was wrong: the working set is PER PROMPT

`DEEPSTRIX_EXPERT_TRACE` over 384 decode tokens (40 layers, top-6), counting
DISTINCT experts actually touched:

    first  32 tokens   66.4 /layer    48.8 GB
    first  64 tokens   96.0 /layer    70.5 GB
    first 128 tokens  132.0 /layer    96.9 GB
    first 256 tokens  165.2 /layer   121.3 GB
    first 384 tokens  189.7 /layer   139.3 GB     (full set: 15,360 = 282 GB)

It GROWS and does not saturate. So "warm" is not a property of the server, it is
a property of the PROMPT: every earlier measurement here warmed on a 300-token
essay and then timed a different prompt, leaving the cache warm for the wrong
working set. Warming on the prompt actually being measured:

    run 1  178.75 ms/tok   2349 box-2 misses
    run 2  120.27          1106
    run 3  115.97          1086
    run 4  116.28          1072

35% of the apparent gap was methodology, not the engine.

## 2. The per-layer pool partition wastes the capacity

Box 2 owns 260 slots on encoder layers and **68 on decoder layers**. The trace
says decode actually needs **202/layer encoder and 177/layer decoder**. So the
decoder half runs a 68-slot cache against a 177-expert working set while the
encoder half is over-provisioned — and capacity cannot migrate, because
`V41_B2_POOL_FLOOR` (default 0.90) guarantees each layer 90% of its own region.

Dropping the floor lets the pool serve the working set globally. Clean arms,
each a fresh box-2 + box-1, six same-prompt runs, last runs reported:

    floor 0.90 (default)   115-141 ms/tok  (median ~122) = 8.2 tok/s   hit 0.9433
    floor 0.00 (global)     66-70  ms/tok                = 14.6 tok/s  hit 0.9787

**66.2 ms/tok = 15.1 tok/s at best, against the post's 16.** The mechanism is
visible in the hit rate: 5.7% miss -> 2.1%. A ~80-100 GB working set FITS in box
2's 123 GB pool globally, and does not fit once carved into 260/68 per layer.

The gain is workload-dependent, as it must be: on a second prompt with a larger
working set the same config gave 98.6-125.5 ms/tok (10.1 tok/s). Their 16 tok/s
is a warm, favourable-workload number; so is this.

## The trade

`b2_pool_floor`'s own frontier (measured by an earlier session, decode 512 tok):

    floor      decode   miss/tok   prefill warm
    0.90        4.92      14.15       526        <- chosen as "free"
    none        5.57      10.44       393        (-25% prefill)

Today's decode delta is far larger than that table's +13%, because the table used
a 512-token generation (working set ~140 GB, fits under NO configuration) while a
100-token generation's working set fits globally but not partitioned. **The
floor's cost is prefill; its benefit scales with how well the working set fits.**

Default left at 0.90 deliberately — the earlier session picked it as the point
that costs prefill nothing, and lowering it is a real prefill trade that belongs
to the operator. For decode-focused work: `V41_B2_POOL_FLOOR=0`.
