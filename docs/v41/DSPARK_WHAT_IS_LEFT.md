# DSpark: what is built, what blocks 20 tok/s, and what to do next
### state as of 2026-09-15

## Built and validated

* **Drafter** (`het/mtp.rs`, `het/weights.rs`): 3-layer stack at B=MTP_BLOCK=5,
  entry projection parity `cos 0.999958` against the CPU oracle, exit with the
  tied head + rank-256 markov head. Acceptance **d1 0.744** against the oracle's
  no-seed **0.764** on agentic/code content. Tests: `mtp_entry_parity`,
  `mtp_forward_smoke`, `mtp_exit_smoke`, `mtp_load`, `mtp_batched_kernels_b5`.
* **Accept loop** (`V41_DSPARK=accept`): batched verify over `[head, d0..d4]`,
  longest agreeing prefix kept, partial rollback via a `KvMark` advanced by rows
  kept, re-draft from the verify's own per-row captured residual. Correct and
  stable: **E = 2.878** warm, identical across consecutive runs.
* **Instruments**: `V41_DSPARK_XCHECK=1` (verify-vs-decode argmax + logit cosine,
  the tool that found the blocker), `V41_LAYER_MISS_HIST=1` (per-layer expert
  misses, split at `CED_DECODER_START`), `V41_LAYER_HOST_TIMING=1` (per-layer
  host phases), `V41_DSPARK=1` shadow mode (acceptance without acting).

## The numbers

| | value |
|---|---|
| baseline decode, warm | **6.4 tok/s** (144-169 ms/token) |
| DSpark accept, warm | **2.4 tok/s** (413-417 ms/token) |
| E[tokens per verify step] | 2.878 |
| verify(B=6) | ~950 ms, of which ~700 ms is box-1 expert misses |
| verify misses | ~250/step, **93.7% in the decoder half** |
| verify-vs-decode agreement | 0.49-0.61 at B=6, cos **0.75-0.79** |

## The blocker

**The verify does not compute the same function as decode.** Not a tuning
issue — the two are different kernel families throughout, each tuned for its own
phase, and neither was ever required to agree because prefill only consumes the
LAST row's logits. A speculative verify is the first consumer of all of them.

Measured, with alternatives falsified: not rollback contamination (same output
sha probe on/off), not the two-box split (`V41_REMOTE_SPLIT=0` still 0.516), not
the f16 WMMA MoE path alone (`IGPU_MOE_WMMA=0` does not fix it), not per-row
attention masking (verified correct). Swapping in decode's MoE
(`V41_VERIFY_DECODE_MOE=1`) moves cos 0.64 -> 0.73 — about a third of the gap —
so attention, the q/kv chain, output projection and mHC differ too.

**This caps acceptance.** A verify that disagrees with decode ~45% of the time
rejects a perfect drafter at that rate, which is why E sits at 2.878 against the
oracle's 4.38. The drafter was never the limit, and neither was expert paging.

## What 20 tok/s requires

Step = drafter (~35 ms) + verify. At E=4.38 a step must fit **~219 ms**.

1. **A batched decode path.** Not a patched prefill path — component-by-component
   patching was tried and falls short. Take `forward_layer.rs`'s per-layer
   schedule (captured graphs, pre-submit reorder, hetsplit MoE, B=1 attention
   pair) and give it a batch dimension, keeping the REMOTE submit batched (box
   2's cost is `105 + 20B + 87D`, so batching wins there while box 1's local MoE
   loops per row). This fixes correctness AND cost together: it removes the
   ~700 ms of box-1 misses by construction, since decode's residency is what the
   pager is tuned for.
2. **Prefill window seeding** for the drafter: the oracle measures no-seed 3.281
   vs seeded 4.382 E[tok]. Worth ~33% once the step is small. The per-row capture
   (`bd.mtp_src`) already exists; seeding needs the last 128 prompt positions.
3. Only then revisit expert placement, pool size, or lane structure.

## Two rules this session paid for

* **Wall time cannot validate a change to this path.** The cheapest way to go
  faster is to compute less. Two "wins" (small-B expert offload 5.3x, single-lane
  -16%) were both retracted after scoring them on acceptance. Score with
  `V41_DSPARK_XCHECK` or acceptance, never the clock alone.
* **Discard the first request after a restart.** It is cold
  (`miss_per_tok=1.25` vs `0.00`) and up to 2.5x slow, which silently invalidated
  every env-gated A/B until it was found.

## Pricing the batched decode path from measured components

Decode-only trace + `het.token.summary`, warm, no probe (139 ms/token):

    total_us=138760  remote_rtt_us=92372 (67%)  sel_sync_us=28746 (21%)
    engram_us=7938 (6%)  pager_ensure_us=2648
    remote.expert track: busy 79.1 ms/tok (33.8%), gap 154.8 ms/tok

Per-layer, from the trace's own labels (`wait L_n rtt=.. link=.. remote=..`):

| component | per token | per layer | scales with B? |
|---|---|---|---|
| link (USB4 wire time) | ~24 ms | 0.6 ms | **NO** — fixed per layer |
| sel_sync (host pick readback) | 29 ms | 0.72 ms | **NO** — one sync per layer |
| box 2 compute | ~9 ms | 0.2 ms | yes (distinct experts) |
| box 2 misses | ~10 ms | 7.7 ms x 1.25/tok | yes |
| engram | 8 ms | — | yes |

**Roughly half a decode token is per-layer FIXED cost** — wire latency and the
pick-readback sync — which a batched verify pays ONCE for B tokens. That is the
whole economic case for speculation on this deployment, and it is why the answer
is a batched DECODE path rather than anything done to the prefill path.

Rough price of a B=6 verify on that schedule: ~24 (link) + ~29 (sync) +
~48 (box 2, 6x) + ~48 (engram, 6x) + box 1's local share = **~200-250 ms**.
Step = verify + ~35 ms drafter, at E=4.38:

    250 + 35 = 285 ms / 4.38 = 65 ms/token = 15.4 tok/s
    200 + 35 = 235 ms / 4.38 = 54 ms/token = 18.6 tok/s

So the batched decode path plausibly lands **15-19 tok/s** — a 2.4-3x gain over
the 6.4 tok/s baseline, at or just under the 20 target rather than past it.
Getting clear of 20 would additionally need the per-layer sync removed (the
picks are read back to host purely to build the remote submit) or the drafter's
~35 ms trimmed. Neither is free, and neither is worth pricing until the verify
agrees with decode.

## The batched decode path is cheaper to build from the PREFILL driver

`forward_prefill_pipelined` is ALREADY layer-major and batched — that is exactly
the schedule a batched decode path needs, and it is what keeps the link round
trip and the pick-readback sync per-LAYER rather than per-token. What is wrong is
only which KERNELS it calls inside each layer. So the work is not "add a batch
dimension to `forward_layer.rs`"; it is "swap the prefill driver's per-layer
kernels for decode's, one component at a time, scoring each with
`V41_DSPARK_XCHECK`".

Decode's kernels are per-token, which is fine: looping them B times keeps the
per-layer link and sync shared, and that is where the amortisation lives.

**Proven on the MoE** (`V41_VERIFY_DECODE_MOE=1`): one self-contained block,
cos 0.64 -> 0.73, about a third of the gap.

**Remaining components, and what each costs to swap:**

| component | prefill uses | decode uses | what the swap needs |
|---|---|---|---|
| MoE | `moe_gate_up_chunked` + by-expert kwide/`q2k_down` | `moe_*_hetsplit` per token | DONE, flag-gated |
| attention score | `launch_score_batched_htiled_wmma*` (stride `ATTN_SCORES_STRIDE` 3072) | `launch_score_b1_htiled_wmma` (stride **`ATTN_MIXED_MAX_KEYS` 82176**) | a separate `[N_HEAD, 82176]` = **21 MB** scores buffer; `sd.attn_scores` cannot be sliced |
| attention wsum | `..._batched_htiled_wmma_ldsv*` | `wsum_b1_htiled_ksplit_ldsv` + `reduce_partials_apply_inv` | decode's K-split partials buffers |
| q/kv chain, output_proj, mHC | batched variants | B=1 variants | per-component scratch |

So each swap replicates a piece of `DgpuScratch` into `BatchDgpuScratch`. That is
the real cost of the project, and it is why it is multi-session: not algorithmic
difficulty, but scratch plumbing, with a correctness gate after each step.

**Sequence:** swap one component, run `V41_DSPARK_XCHECK=1` (one ~4-minute
server run), keep it only if cos moves toward 1.0. When cos reaches ~0.999,
acceptance should jump from 2.878 toward the oracle's 4.38 on its own, because
the drafter is already validated — and only then is the verify's ~700 ms of
box-1 expert misses worth attacking, since decode's residency is what the pager
is already tuned for.
