# The verify step, measured — and why the number does not yet test the projection
### 2026-09-14

## What was built

`V41_VERIFY_PROBE=K[,K...]` with `V41_VERIFY_BATCHED=1` makes the decode loop, on
every steady-state token, run a real batched forward over K tokens appended to the
live KV, then roll it back. It sweeps a comma list per token, so one server load
measures every verify width. Sits before the embed — the only point where
clobbering `residual` and the device scratch is harmless.

## What it measured

CED off, catch-all mode 2, 150-200 token generations, p50 over ~20 samples per B:

    B   step p50 ms   ms/token    E      tok/s
    2       621.1       310.6   1.93      3.1
    4       721.8       180.5   3.57      4.9
    5       775.5       155.1   4.13      5.3
    6       860.2       143.4   4.94      5.7
    8       867.6       108.5   5.60      6.5

    reference: B=1 sequential decode = 246 ms/token = 4.1 tok/s
    with CED on: B=5 was 791.4 ms, so the CED replay is NOT the cost (-2%)

## Why this does NOT refute the 38-42 tok/s projection

**The batched path routes experts through the wrong tier.** `forward_prefill` pages
a per-layer union out of box 1's `ExpertPager`, i.e. box 1's own dm-crypt disk.
Decode routes experts to box 2 under T2 catch-all, where a resident expert costs
87 us. So this measurement prices a verify step in which box 1 pages ~19 distinct
experts per layer locally — the exact configuration the two-box split exists to
avoid, and which took decode from 7.0 to 14.3 tok/s when it was removed.

That is also why B=2 at 621 ms is WORSE than two sequential decodes (492 ms): the
batched path's expert sourcing is strictly worse per token than the decode path's,
and only the ~10 ms/token of dense compute is amortised.

So the honest status is: **the projection is untested, not disproven.** What this
run establishes is a requirement, not a ceiling.

## The requirement it establishes

**DSpark's verify step needs a batched DECODE path — batched attention and head
over B tokens, with expert routing through T2 catch-all — not `forward_prefill`.**
That path does not exist. Reusing prefill for verify was the cheap hope and it is
measurably wrong.

## Second requirement, found by the rollback guard

`HetModelState::rollback_kv` refused after the prefill path ran: *"layer 0 wrapped
since the mark (raw_off 0 < marked 1)"*. Prefill's post-chunk eviction physically
relocates the live KV window and resets `raw_off`, so mark/restore of
`(n_raw, raw_off)` cannot undo it. The guard caught a real incompatibility rather
than a hypothetical one.

**So a prefill-based verify is unrollbackable as well as slow.** A batched decode
path that appends monotonically — the scheme `state.rs:279` describes — is
rollback-compatible by construction.

## Method note

I first reported "B=5 verify = 286.7 ms -> 14.4 tok/s" from this probe. That was
wrong: the request had 500'd with "Engram rows not staged", and the nine
`total_us=` samples I parsed were `het.token.summary` lines, not probe output.
**Check the request succeeded before parsing any metric out of its log**, and
grep for the probe's own message rather than a field name that other events share.
