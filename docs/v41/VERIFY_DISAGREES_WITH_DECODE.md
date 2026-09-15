# The verify path does not reproduce the decode path, and it gets worse with B
### measured 2026-09-15, direct argmax cross-check

## The test

`V41_DSPARK_XCHECK=1` with the verify probe. Row 0 of the probe's batch sits at
`pos` with `next` as its input — exactly the same work the decode forward on the
next line does — so `argmax(probe_logits[0])` MUST equal the token decode then
samples. Any disagreement is the verify computing something different from what
decode would.

This is a DIRECT correctness test. DSpark acceptance is only a proxy, and a bad
one: changing the box1/box2 split changes f32 summation order, which changes the
generated TEXT, and acceptance is strongly content-dependent (the oracle
measures 2.2x between agentic and prose). E moving does not prove corruption.

## The result

| verify batch | argmax agreement | mean cos(verify row 0, decode logits) |
|---|---|---|
| B=2 | 0.784 / 0.922 / 0.829 / 0.927 | **0.913 / 0.872** |
| B=6 | 0.490 / 0.588 / 0.610 / 0.537 | **0.790 / 0.746** |
| B=6, decoder-only catch-all | 0.157 / 0.235 | — |

**The cosines settle what the argmax rate could not.** Near-tie flips would show
cos ~0.9999 with an occasional different top-1. At **0.75-0.91** the two paths
are producing genuinely different vectors, and the gap widens with B.

**Agreement degrades with batch size**, and at the B=6 a 5-draft DSpark step
needs, the verify disagrees with decode about 45% of the time.

**It is not rollback contamination.** The probe's whole purpose is that the
generation must come out unchanged, and it does: same prompt, `V41_VERIFY_PROBE`
on vs off, output sha `e4a5a27da0eedbf2` both ways. So the state is restored
correctly and the divergence is in the verify's own arithmetic.

**Row 0 should not depend on the batch at all.** Causal attention means row 0
attends only to history and itself; rows 1..B-1 are strictly later positions. A
batch-size-dependent result for row 0 therefore points at a reduction whose
ORDER depends on batch composition — the MoE partial sums across the batch are
the obvious candidate (the group builder batches by expert, not by row).

## Why this matters more than the paging

It explains the thing that had no explanation: DSpark acceptance caps around
E=2.878 while the CPU oracle measures 4.38 on comparable content. If the verify
disagrees with decode ~45% of the time, then even a PERFECT drafter — one that
predicts exactly what decode would emit — gets its drafts rejected at roughly
that rate. The drafter was never the limit.

It also means **byte-identical validation is impossible with this verify**, no
matter how the expert split is arranged: the accepted tokens are not the tokens
non-speculative decode would have produced.

And it reframes "the verify must be a batched DECODE path" from a performance
argument into a CORRECTNESS one. Making the verify cheaper is pointless while it
is answering a different question than decode does.

## Ruled out

* **The f16 WMMA MoE path.** The prefill MoE runs "f16 activations end to end,
  no Q8_K quantize" (`forward_prefill.rs:4364`) where decode quantizes to Q8_K —
  a plausible culprit, but `IGPU_MOE_WMMA=0` does not fix it (0.415 / 0.049).
* **Per-row attention masking.** `n_raw_after[i] = n_raw_before + i + 1` and the
  kernels take `n_raw_per` per row, so row 0 attends to its own prefix only.
* **Rollback contamination** and **the two-box split** — below.

## It is NOT the two-box split

MEASURED: `V41_REMOTE_SPLIT=0 V41_REMOTE_SPLIT_DECODE=0` — box 1 does ALL the
MoE, box 2 is uninvolved — gives **0.516**, statistically the same as the 0.49 /
0.59 with the split on. So the divergence is entirely WITHIN box 1, between the
batched-prefill implementation and the B=1 decode implementation.

That retroactively explains why none of the four expert-split changes could
raise acceptance: the ceiling was never about the split.

## ROOT CAUSE: the two paths run different MoE implementations

The verify inherits the PREFILL MoE; decode uses its own. They are different
kernel families, not two settings of one:

| | decode (`forward_layer.rs`) | verify/prefill (`forward_prefill.rs`) |
|---|---|---|
| gate/up | `moe_gate_up_batch_hetsplit` | `moe_gate_up_chunked` (+ by-expert kwide) |
| down | `moe_down_batched_hetsplit` | `q2k_down` by-expert |
| activations | quantized to Q8_K | f16 end to end on the WMMA path (`forward_prefill.rs:4364`) |
| iteration | per token, over its own picks | by EXPERT, accumulating across the batch |

Both are deliberate: the by-expert prefill chain is the 2026-09-08 WMMA program
(448 -> 727 tok/s) and `q2k_down`'s (B, expert) inversion (+14%). Neither was
ever required to agree with decode, because **prefill only consumes the LAST
row's logits** — a speculative verify is the first consumer of all of them.

Over 40 layers those differences compound into the measured cos 0.75-0.91, and
the batch dependence follows from `moe_gate_up_chunked` chunking over B.

## TESTED: swapping in decode's MoE is necessary but NOT sufficient

`V41_VERIFY_DECODE_MOE=1` (default off) recomputes the batched path's local MoE
with decode's kernels — Q8_K activations, `moe_gate_up_batch_hetsplit` +
`moe_down_batched_hetsplit` per token — overwriting what the by-expert chain
produced. One self-contained block, so the diagnosis could be tested before
paying for the surgery to skip the wasted chain.

| | argmax agreement | mean cos |
|---|---|---|
| off | 0.561 / 0.585 | 0.666 / 0.614 |
| on | 0.561 / 0.829 | **0.730 / 0.734** |

**cos 0.64 -> 0.73.** Real, and in the right direction — but nowhere near the
~0.9999 two identical computations would give. **The MoE is roughly a third of
the gap, not the gap.**

So the divergence is not one component: the batched forward differs from the
decode forward throughout — attention (batched `*_htiled_wmma` variants vs the
B=1 pair), the q/kv chain, the output projection, mHC. Each was tuned for its
own phase and none was ever required to agree with the other.

**That settles the design question.** A verify cannot be obtained by patching
the prefill path component by component; it needs to BE the decode path, batched.
"The verify must be a batched decode path" is now a measured conclusion with a
falsified alternative behind it, not a preference.

## What to do next, in order

1. **Give the verify decode's MoE.** In `forward_layer_pre_moe_v2`, behind a
   verify-only flag, replace the chunked/by-expert chain with the hetsplit
   per-token pair looped over the B rows — the exact call the decode path makes
   (and the one `het/mtp.rs` already makes per row). Numerically identical to
   decode by construction. It costs B x decode's MoE, so it is a CORRECTNESS
   fix first; the speed work only becomes meaningful afterwards.
2. **Then re-measure acceptance.** If the diagnosis is right, E should move from
   2.878 toward the oracle's 4.38, because the drafter is already validated at
   d1 0.744 vs 0.764 — it has been predicting decode while being scored against
   a different function.
3. **Check the reduction order** if it does not. `moe_gate_up_batch_hetsplit` /
   `moe_down_batched_hetsplit` accumulate per-expert across rows; compare a
   B=1-per-row loop against the batched call on identical inputs, the way
   `mtp_batched_kernels_b5.rs` does for the other batched kernels. That test
   already exists as a template and found a real kernel bug once.
2. Only then revisit expert placement or paging for the verify.

## CORRECTION: the verify divergence is NOT what caps acceptance

I claimed the verify's disagreement with decode capped DSpark's acceptance at
2.878 against the oracle's 4.38, and that "the drafter was never the limit".
**That was wrong**, and building the faithful verify is what disproved it.

`V41_VERIFY_DECODE_PATH=1` runs the verify through decode's own per-layer
function, layer-major over the B rows — numerically what decode computes, by
construction. If the divergence were the cap, acceptance should have risen
toward 4.38. It did not:

| verify | E[tokens/step] |
|---|---|
| batched prefill driver (diverges from decode) | **2.878** |
| decode's own per-layer function, layer-major | 2.282 / 1.913 / 2.068 |
| same, under `V41_T2_CATCHALL=2` | 1.294 / 1.692 / 1.692 |

The faithful verify is not better. And SHADOW mode already had the answer:
scoring the drafter against decode's ACTUAL output on this content gives
per-depth `[0.744, 0.492, 0.360, 0.256, 0.152]`, mean accepted 1.258,
**E ~ 2.26**. Both verifies bracket that. The drafter's own accuracy — no
prefill window seeding, on this content — explains E ~ 2.3 directly, with no
appeal to verify fidelity.

**So the divergence (cos 0.75-0.79, argmax agreement 0.49-0.61) is real but
costs throughput almost nothing.** It costs OUTPUT FIDELITY: the tokens a
speculative run emits are not the tokens non-speculative decode would emit, so
byte-identical validation is still impossible. That is a correctness problem,
not the performance problem, and I conflated the two.

What actually caps DSpark here, in order:
1. **Acceptance ~2.3-2.9**, set by the drafter without seeding on non-agentic
   content. The oracle's 4.382 is agentic text WITH window seeding; its
   no-seed number is 3.281 and its prose number is 1.99.
2. **Verify cost**: ~950 ms batched (94% box-1 expert misses), or B x decode
   for the faithful path.

Both must improve together. Neither is the "one blocker" I described.
