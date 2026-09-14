# `V41_LOCAL_CLAIM_MAX=0` is a no-op at the shipped pool — a retracted result
### 2026-09-14

I measured `V41_LOCAL_CLAIM_MAX=0` at +7.8%, confirmed it at +10.7% over two runs
per arm, and was about to make it the default. **It is a no-op.** Recording the
retraction and, more usefully, the measurement that settles it.

## What looked like a win

Paired runs, same prompt, 256 tokens, `DEEPSTRIX_TOKEN_PROFILE=1` on both arms:

    pool 52, uncapped   4.09, 4.07 tok/s
    pool 52, CLAIM_MAX=0 4.41, 4.62 tok/s        -> "+10.7%"

with a stage diff that looked mechanistic:

    igpu.routed_moe              15.64 -> 12.88   -2.75
    igpu.peer_push_ffn_moe.wait   6.56 ->  3.97   -2.59
    dgpu.ffn_combine.wait         6.31 ->  3.74   -2.57
    igpu.moe.wait (for box 2)    21.01 -> 21.57   +0.56

## What settles it

Same prompt, 512 tokens, no profiler, box 2's own counters bracketed around the
request:

                     tok/s   box2 picks/token   box2 misses/token
    DEFAULT           4.06        237.1              17.27
    CLAIM_MAX=0       4.00        237.1              17.32

**`picks/token` is identical to four significant figures.** The cap flips local
picks to remote, so if it were doing anything box 2's pick count would RISE. It
does not move. Box 1 was claiming nothing to begin with: at pool 52 its decode
LRU is 25 slots (`forward_layer.rs:2320`) and essentially none of what it holds
gets picked.

A counter that must move if the knob works, and does not move, beats a tok/s
delta. The `routed_moe` swing was history variance — which of box 1's 25 slots
happen to be picked differs run to run.

## The variance floor this exposes

    pool 52 default      n=3   4.06-4.09 tok/s   spread  1%
    pool 52 CLAIM_MAX=0  n=3   4.00-4.62 tok/s   spread 16%

Two arms that are provably the same configuration differ by up to 16%.
**E2E decode tok/s here has roughly +/-8% run-to-run precision, so a single-run
delta under ~15% is not evidence.** Everything quoted from e2e tok/s needs either
n>=3 or a counter that corroborates the mechanism.

What survives that bar from today, because it rests on tight counters rather than
wall-clock: box 2's miss cost 10.57 -> 6.60 ms (expertd bench, p50 over 2,400
iterations), the batched-submit fix at 11-13% (bench p50 over 200 iterations per
B), and the 81.6 GB pool regression (5 runs, 4.06-4.09 vs 3.26-3.51 — separation
well beyond the noise band, and corroborated by `igpu.routed_moe` +89%).

## What is still true

The break-even from the pool-regression stage diff is unaffected: box 1 costs
140 us/expert against box 2's 87 us, so serving a pick box 2 would HIT is -53 us
and serving one it would MISS is +6547 us. And the point that box 1 claiming a
pick denies box 2's LRU a touch of it — so box 2 may evict it and pay a miss
later — is sound reasoning. **Neither is measurable at the shipped config,
because box 1 claims nothing there.** Both become live only once box 1 has real
exclusive residency, which is the victim cache.
