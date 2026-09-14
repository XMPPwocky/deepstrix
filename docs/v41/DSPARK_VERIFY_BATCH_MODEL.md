# DSpark verify: how box 2 scales with batch size (MEASURED 2026-09-14)

Bottom line: **DSpark at B=5 projects 39-43 tok/s and clears 30 even at E=3.0** —
*provided decode first reaches its zero-miss floor*. The batch multiplier is
favourable, not hostile, and the reason is that the dominant per-request costs do
not scale with B.

## The measurement

`deepstrix-expert-bench` against the LIVE box-2 daemon (260/68 placement, 6560
resident experts, 123.3 GB). No box-1 weight load. Four arms, back-to-back, same
binary, `--iters 200`, `--picks 3`, `--depth 1`.

`--pool 3` pins every token to the SAME 3 experts, so distinct-experts D is constant
in B and the per-token term is isolated from weight bandwidth. `--pool all` lets
D grow ~3B, exposing the per-expert term.

    arm  path      D          B=1   B=2   B=4   B=5   B=6   B=8      (srv p50, us/layer)
    C    batched   3 const    386   396   444   465   487   527
    B    decode    3 const    388   738  1387    --    --    --      (decode_max_b=4)
    D    batched   ~3B        383   689  1258  1539  1790  2349

### Fit

    srv_us(B, D) = 105 + 20*B + 87*D          per layer, batched path

Residuals under 3% across all 12 batched points (e.g. B=8,D=24: model 2353 vs
measured 2349; B=4,D=12: 1229 vs 1258).

* **87 us per distinct expert** = 18.8 MB / 87 us = **216 GB/s**, ~94% of achievable
  on this part. Box 2's marginal cost is pure weight bandwidth at near-peak. There is
  **no kernel headroom on box 2** — this supersedes the earlier, conservative 173 GB/s
  reading. Do not spend time optimising box 2's expert FFN.
* **20 us per token** on the batched path — nearly free.
* **105 us fixed** per request.

## Finding 1: box 2's DECODE path re-reads weights per token

Arm C vs arm B is the same work — identical expert set, identical picks — and differs
only in which path serves it:

    per-extra-token cost, expert set held CONSTANT:
        batched path    +20 us      <- groups tokens by expert, one weight read
        decode path    +333 us      <- ~3 experts x 87 us: re-reads per token

**16x.** The decode path does not group by expert, so batching buys it nothing. The
batched path is already better at B=2 and 2.7x better at B=8.

`decode_max_b = 4` currently forces B>=5 onto the batched path, which is accidentally
the right thing. **DSpark verify must use the batched path at every B**, including
B<=4 (at B=4 the batched path is still 1.24x cheaper end to end). Either raise the cap
and route verify to the batched kernel, or teach the decode path to group by expert.

## Finding 2: real routing dedups weakly

From the 3,456-token routing trace (`expert_trace_routing_ds.bin`, 4 domains),
distinct experts per layer over B CONSECUTIVE tokens — which is exactly a verify batch:

    B        1     2     3     4     5     6     8    12    16
    distinct 6.00  9.97 13.36 16.38 19.19 21.73 26.52 34.73 41.82
    vs B=1   1.00  1.66  2.23  2.73  3.20  3.62  4.42  5.79  6.97
    saving      -   17%   26%   32%   36%   40%   45%   52%   56%

Routing is flat (top-15 = 28% of picks), so dedup saves only 36% at B=5. The
per-expert term therefore grows 3.2x at B=5. This is the pessimistic half of the story
and it is still not enough to hurt, because of what follows.

## Projection

Composition of the MEASURED 71 ms zero-miss floor:

    step = box1_serial + max(box1_igpu_moe, box2_leg)
         = 35.9        + max(18.2,          35.0)      = 70.9 ms   (B=1)

Box 2's leg per layer = overhead + 20*B + 87*D(B). Calibrating the overhead
(link + wake-up + the hub's inter-layer gap) so leg(1) = 35 ms gives 594 us/layer, and
it is **B-independent**: there are 40 request rounds per step regardless of B.

    B   D/layer  box2 leg  box1 moe  step ms  @E=4.13  @E=3.0  serial+25%
    2      4.98      42.7      30.2     78.6     52.6    38.2        47.2
    4      8.19      55.5      49.7     91.4     45.2    32.8        41.2
    5      9.60      61.2      58.2     97.1     42.5    30.9        38.9
    6     10.86      66.4      65.9    102.3     40.4    29.3        37.1
    8     13.26      76.3      80.4    116.3     35.5    25.8        33.0

At B=5 the two legs balance to within 5% (58.2 vs 61.2) — the `max()` is tight, which
is the most efficient point available. **B=5 is the recommended verify width.**

The result is robust: it survives E dropping from 4.13 to 3.0, and survives a 25%
heavier serial chain. It does not survive B=8 at low E, because box 1's iGPU MoE
overtakes box 2 there (80.4 vs 76.3) — and box 1's MoE kernels run at 54%/35% of
achievable bandwidth, so *that* is where kernel headroom exists if B>5 is ever wanted.

## The binding precondition

Every number above is at the **zero-miss floor**. Today decode is 274 ms/token:

    274 ms  today (3.65 tok/s)
     71 ms  measured zero-miss floor (14.1 tok/s)
    ------
    203 ms  pure expert-miss overhead = 74% of decode

Misses do **not** amortise under speculation — a B=5 verify touches 3.2x the distinct
experts, so it takes ~3.2x the misses for 4.13x the tokens. Closing the miss gap is
therefore required for DSpark to pay, and it is the entire pre-DSpark program. No
other lever comes close: everything else on the board lives inside the 71 ms.

## Method note

An earlier framing here asked for a single "batch multiplier m" and concluded m>=2
would make 30 tok/s unreachable. That framing was wrong in shape: m applies to the
*step*, which decomposes into a B-invariant serial chain, a B-invariant per-layer
overhead, and only one term that actually scales. m at B=5 is 97.1/70.9 = 1.37, and
the 4.13x token yield dominates it. Decompose before multiplying.

Reproduce: `scratchpad/dedup.py` (trace -> dedup table), `scratchpad/proj.py` (model
-> projection), and the four bench arms above.
