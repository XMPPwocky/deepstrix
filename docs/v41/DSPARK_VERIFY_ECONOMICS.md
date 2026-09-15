# DSpark verify: the fixed cost is the blocker, not the expert traffic
### measured 2026-09-15, box 1 + box 2, production config (V41_CED=1)

The drafter works (see `DSPARK_DESIGN.md`; acceptance validated against the CPU
oracle). This is about whether a speculative step can PAY.

## The arithmetic

A speculative step with K drafts verifies a batch of `B = K + 1` tokens and
advances the sequence by `1 + n` positions, where `n` is the accepted prefix.
So the step is a win only when

    cost(verify at B)  <  (1 + E[n]) x cost(one decode token)

`1 + E[n]` is exactly the oracle's `expected_tokens`, and it SATURATES with K
(1.843 / 2.573 / 3.247 / 3.843 / 4.382 at K=1..5, agentic text, seeded). So the
verify cost must be strongly sublinear in B, or speculation cannot win.

## Measured cost curve

`V41_VERIFY_PROBE=1,2,3,4,6,8 V41_VERIFY_BATCHED=1` — one batched forward over B
tokens appended to the live KV, then `rollback_kv`. Median over ~14 samples per
B, steady-state decode:

| B | verify (ms) | ms per token in the batch |
|---|---|---|
| 1 | 588 | 588 |
| 2 | 937 | 468 |
| 3 | 974 | 325 |
| 4 | 958 | 239 |
| 6 | 1065 | 178 |
| 8 | 999 | 125 |

**The cleanest read of the fixed cost is the B=1 row, not the fit.** A decode
token through `forward_token_paged` costs **192 ms**; the SAME single token
through `forward_prefill_pipelined` costs **588 ms**. Identical work, identical
batch size, no batching confound — the ~**396 ms** difference is pure per-call
entry fee. (A least-squares fit over the whole curve gives
`cost(B) ~ 740 + 45 B ms`, but its intercept is inflated by noise at large B;
trust the 396 ms.)

Two things follow, and they point in opposite directions:

**The good news: batching amortises very well at the margin.** An extra token
inside the batch costs ~45-56 ms against **192 ms** for a standalone decode
token (`het.token.summary total_us` in steady state). That is ~4x, and it
refutes the worry that a sparse MoE cannot amortise because each token routes to
different experts — it amortises fine.

**The bad news: the fixed cost is ~400 ms, i.e. 2 decode tokens.** At
B=6 the batch costs ~1065 ms and advances 4.382 positions = 243 ms/token
against a 192 ms baseline — a LOSS. Even at the oracle's best acceptance the
step only breaks even, and with the no-seed acceptance we actually implement
(E[tok] 3.281 agentic / 2.58 measured on code prose) it is well underwater.

## So the lever is the fixed cost, not acceptance

Raising acceptance cannot rescue this: `expected_tokens` is bounded by `B` and
saturates well below it, so no achievable E beats a 740 ms floor. Removing the
floor does: at `cost = 45 B`, B=6 costs 270 ms for 4.382 tokens = **62 ms/token,
a 3.1x speedup**. Every ms of fixed cost removed is worth more than any further
drafter work.

The floor is `forward_prefill_pipelined`'s per-call setup, not anything
fundamental — this path exists to move 512-row chunks, and a 6-row verify pays
its whole entry fee. That is what "the verify path must be a batched DECODE
path, not `forward_prefill`" means concretely. Suspects, in the order worth
measuring: CED replay (`V41_CED=1`; a probe run with it off could not be
completed — the batched path then fails `Engram rows not staged`, a harness
limitation), union expert paging, per-call lane/scratch setup, and the Engram
row gather.

## A second constraint the probe found

`rollback_kv` legitimately REFUSES when the speculative ingest wraps the KV
window: `layer 0 wrapped since the mark (raw_off 0 < marked 1); the
eviction-down copy moved the window`. An accept/reject loop has to either avoid
speculating across that boundary or fall back to a plain decode step there. It
fails loudly rather than silently serving wrong KV, which is the behaviour we
want, but it is a real case to handle.


## Follow-up 2026-09-15: CED ruled out; it is per-layer host scheduling

The first curve was measured with the probe passing `last_only=true`, which is
NOT the shape of a real verify — a verify needs per-token logits to compare
against the drafts, i.e. `last_only=false`, and that also turns CED off
(`ced = ced_enabled() && last_only`). Under CED the stack really is traversed
twice, which the per-stage profile shows plainly: every dGPU stage at 80 calls
for 40 layers while the iGPU MoE runs 40, and `lo=0 hi=21 KvSourceOnly` followed
by `lo=20 hi=40 Replay` in the lane debug. So the first measurement priced a
pass a real verify does not do.

Re-measured with `last_only=false` (one `lo=0 hi=40 Exact` pass, request
completes, 80 tokens):

| B | verify (ms) | ms per token in the batch |
|---|---|---|
| 2 | 934 | 467 |
| 4 | 1056 | 264 |
| 6 | 1097 | 183 |
| 8 | 1165 | 146 |

**cost(B) ~ 857 + 38.5 B ms**, against a 177.5 ms decode token in the same run.
So removing CED did NOT remove the fixed cost — it is ~860 ms either way, about
**4.8 decode tokens**. CED is not the answer; neither is expert traffic (the
marginal batched token is 38.5 ms, **4.6x cheaper** than a standalone decode
token).

**What it actually is: per-layer host scheduling.** With CED off no single stage
dominates — the largest is `dgpu.mhc_pre_attn` at 53 ms, where under CED
`dgpu.ffn_combine` alone was 149 ms. The cost is spread across ~40 layers x
several host-side RAII scopes, each with its own launches and waits, with almost
no GPU work to hide them at B<=8. This is precisely the tax the DECODE path
already pays down with captured graphs — `forward_layer.rs` fuses layer N's
`ffn_combine` with layer N+1's `mhc_pre_attn` into one graph specifically to
close a "~115 us/layer host-scheduling gap". `forward_prefill_pipelined` has no
such capture: it is built to move 512-row chunks, where 860 ms of scheduling
disappears against the GPU work.

**So "the verify must be a batched DECODE path" is now measured, not asserted,
and the mechanism is named.** At `cost = 38.5 B` a B=6 verify is 231 ms for
4.382 tokens = **53 ms/token, a 3.4x speedup** over the 177.5 ms baseline. That
is the prize, and it is entirely in scheduling, not in arithmetic.


## SHIPPED: single lane below B=8 (-10 to -16% on the verify)

The two-lane pipeline exists to overlap one lane's GPU work with the other's
host scheduling. At verify batch sizes there is no GPU work to hide behind — the
cost IS the host scopes — so splitting just runs every per-layer scope twice.
Below `V41_PREFILL_SINGLE_LANE_MAX` (default 8, `0` restores the old path) the
whole chunk goes in lane A and lane B is skipped. Guarded on `spans.is_empty()`,
since image spans are the reason `lane_split` has to cut carefully.

A/B back-to-back in ONE binary, verify probe:

| B | two-lane | single-lane | delta |
|---|---|---|---|
| 2 | 817 ms | 684 ms | -16.3% |
| 4 | 947 ms | 828 ms | -12.6% |
| 6 | 1055 ms | 949 ms | -10.1% |
| 8 | 1081 ms | 1022 ms | -5.4% |

Fixed cost **729 -> 571 ms**. The gain shrinks as B grows and there is finally
work to overlap, which is why the default stops at the edge of the measured
range.

That still leaves ~571 ms, i.e. **~14 ms per layer** at B=6 against the decode
path's ~4.4 ms per layer (177 ms / 40) while doing 6x the tokens. The rest is
the captured-graph gap, and closing it is the remaining 3.4x.

## RETRACTION 2026-09-15: the "small-B offload" was skipped work

I reported a 5.3x verify speedup (950 -> 180 ms) from having tiny batches page
no experts on box 1 and handing box 2 the whole layer. **The speedup was real
and the arithmetic was wrong.** Retracted.

It was caught by the accept loop, because DSpark's acceptance rate IS a
correctness check on the verify: the drafter predicts the MODEL's next token, so
if the verify's logits are right, acceptance should match what shadow mode
measured independently.

| verify expert split | E[tokens/step] |
|---|---|
| unmodified (correct) | **2.12** |
| box 2 takes every pick, masked submit | 1.10 |
| same + `V41_T2_CATCHALL=2` | 1.23 |
| same + `submit_unmasked` | 1.01 (and 1194 ms/tok) |
| box 1 keeps residents, box 2 takes only MISSES | 1.11 |

Every variant that moves expert work off box 1 halves acceptance. All of them
returned `HTTP 200` with no error, and `verify_routing_exactly_once` passed
throughout — it validates the hub's own `owns_eff` vector, not what box 2
actually computed, so an under-computed expert is silent.

`V41_SMALL_B_OFFLOAD_MAX` stays at its default of 0. What survives is the
diagnosis: **94% of a B=6 verify is box 1's expert paging** (`prefill_requests`
~316, `prefill_misses` ~250, `prefill_read_ms` ~700 of ~950 ms), because a
verify's DRAFT tokens route to experts the decode LRU has never touched. Any fix
has to move that work without changing what gets computed — and must be scored
with acceptance, not wall time.
