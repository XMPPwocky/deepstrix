# Half the decode misses are allocation geometry, not capacity
### simulated against the real routing trace, validated against the live measurement, 2026-09-14

`DECODE_MISS_PATH_AND_CAPACITY.md` concluded the decode wall was capacity and the
only exit was a ~2.5-bit expert format. **That conclusion was drawn from one
number — 20.2 misses/token — without asking why it was 20.2.** It should have
been ~5. Most of the gap is where the slots are, and two of the four devices in
this cluster are doing nothing during decode.

## The contradiction

`lru_curve.py` replays the 3,456-token routing trace through a GLOBAL LRU:

    slots     GB  %resident | miss/tok (warm)
     9326  175.3      60.7% |      4.9      <- today's total capacity
    12000  225.6      78.1% |      2.1
    15360  288.8     100.0% |      0.7

At today's capacity a global pool misses **4.9/token**. We measure **20.2**. A 4x
gap that capacity cannot explain.

## Where it goes: per-layer regions

Box 2 runs an INDEPENDENT LRU per layer, sized by what it owns on that layer —
260 slots on encoder layers 0-19, **68 on decoder layers 20-39** (17.7% of 384).
Modelling exactly that geometry reproduces the measurement:

    allocation                       slots  enc/dec  miss/tok  ms/tok @6.6
    TODAY 260/68, per-layer LRU       6560  260/68      20.4       134   <- measured 20.2
    uniform 164/layer                 6560  164/164     11.5        76
    encoder-min 256 / decoder 328     6560  256/72      19.4       128
    TODAY's slots, ONE GLOBAL pool    6560  260/68      10.4        69

**20.4 simulated vs 20.2 measured** — the model is sound. And the same 6,560
slots, pooled globally instead of partitioned per layer, halve the misses. That
is ~65 ms/token available with no extra RAM and no format change.

`uniform 164` is NOT reachable: prefill's window stride is 128, so box 2 must own
>= 384-128 = 256 per encoder layer or prefill dies (the 180/128 attempt returned
HTTP 500). `encoder-min 256` shows why that constraint hurts — holding encoders
at 256 leaves decoders 72 and buys nothing.

## The bigger one: box 1 is idle during decode

Box 1 has a 52 GB expert pool (~2,688 slots at stride 128, `V41_PAGER_WINDOWS=21`)
**pinned to prefill's CED encoder windows**. During decode those hold the wrong
experts: box 1 serves a measured **0.5 of ~6.5 picks/layer** and forwards 260.5
picks/token to box 2. Its `decode_hit` is 1.0000 with `miss_per_tok` 0.00 — it
never even reads. So during decode:

  * box 1's 52 GB of resident experts are ~unused,
  * box 1's iGPU does almost no MoE work,
  * box 1's NVMe is **completely idle**,
  * and box 2's single drive carries 100% of the I/O.

Box 1's drive is not the slow one. At the miss's exact shape (3 threads x 6.3 MB):

              buffered   O_DIRECT
    box 1       4.10 ms    4.86 ms     4.93 GB/s   (YMTC PC411, on dm-crypt)
    box 2       5.89 ms    4.84 ms     3.43 GB/s   (Crucial T500)

**Box 1's idle drive is 44% faster than box 2's at this access pattern.**

Pointing box 1's pool at the decoder layers box 2 starves:

    config                                          miss/tok  ms/tok
    today                                              20.4     134
    box 2 one global pool                              10.4      69
    + box 1's 2688 slots over DECODER layers            9.4      62   <- best
    + box 1's 2688 slots over ALL 40 layers             41.3     272  <- thrashes
    + box 1 on decoders AND box 2 uniform 164          13.7      90

Targeting matters: spreading box 1 over all 40 layers makes it a too-small
first-level cache that evicts faster than it serves, and is 2x WORSE than doing
nothing. Over decoder layers only it is the single best configuration found.

## What this is worth

20.2 -> 9.4 misses/token = **~71 ms off a 246 ms token -> ~175 ms = 5.7 tok/s**,
from 4.06. And the picks box 1 serves are computed on box 1's idle iGPU
*concurrently* with box 2, so some of box 2's compute leg leaves the critical
path as well. No format change, no new hardware, no prefill regression — box 1's
pool is repurposed only in the decode phase, which is exactly the "phase-aware
split" `DECODE_M8_PLAN.md` lists as step 0 and which was never built.

## Correction

The capacity framing was not wrong about the arithmetic (380 MB/token at 4.47
GB/s really is 85 ms). It was wrong about two premises:

1. **20.2 misses/token is not a property of the model.** It is a property of a
   per-layer partition that gives decoder layers 17.7% residency. The same RAM,
   pooled, gives 10.4; adding box 1's idle pool gives 9.4.
2. **4.47 GB/s is not the cluster's read bandwidth.** It is one drive's. The
   other, faster drive is idle. Aggregate is ~8.4 GB/s.

Re-pricing the wall with both corrected: 9.4 misses x 18.8 MB = 177 MB/token
across ~8.4 GB/s = **21 ms/token**, against a 33 ms budget for 30 tok/s. The
I/O is no longer over budget at all. The expert format goes from "the only
exit" to "one lever among several" — still the biggest single one, but no
longer load-bearing for reaching the goal.

Reproduce: `scratchpad/geom.py` (per-layer vs global vs two-tier),
`~/.cache/deepstrix/v41/lru_curve.py` (global curve), `scratchpad/buf_vs_direct.py`
(both drives).
