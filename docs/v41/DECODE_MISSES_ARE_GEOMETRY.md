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

### CORRECTED — box 1's slots are not free, and duplication does not help

The first pass of this section modelled "box 1's 2688 slots over decoder layers
-> 9.4 miss/tok" and called it free. **Two errors, both now fixed.**

**Error 1: those slots belong to prefill.** `forward_layer.rs:2320` says it
outright — *"Box 1's decode LRU is only ~25 slots (pool minus the packed prefill
windows)"*. The 52 GB pool is 2,766 slots, of which 2,688 are 21 packed windows
at stride 128 pinning all 20 CED encoder layers, worth prefill 159 -> 255 tok/s.
Decode gets the ~25-78 remainder. That also explains why
[[project_v41_pool_split_not_a_decode_lever]] measured resizing the decode LRU
inert: 3.5x of 25 slots is 87 slots, 2 per layer.

**Error 2: a front cache duplicates what box 2 already holds.** Modelled properly
— box 1 checked first, both caches filling independently — an INCLUSIVE box 1
cache is nearly worthless:

    box1 slots  arrangement          miss/tok  b1 hits/tok  saved
          25    today                   20.4         0.0      0ms   <- validates
        1024    inclusive front cache   18.8        96.8     11ms
        2688    inclusive front cache    8.6       113.3     78ms

At 1024 slots it absorbs **96.8 hits/token and removes 1.6 misses**. Almost every
hit was one box 2 would have served anyway. Only at 2,688 — box 1's entire
prefill window set — does it pay, and that trade is prefill 255 -> ~159 tok/s.

### What actually works: an EXCLUSIVE, frequency-ranked band

Box 2 keeps the top-68 per decoder layer. Box 1 **statically pins the NEXT k by
frequency** — the same freq-ranked placement already shipped for encoder layers
([[project_v41_freq_ranked_encoder_placement_2026-09-14]]). A pick in box 1's band
never reaches box 2, and box 1 never pages, so neither box touches disk for it:

    k/decoder  box1 slots    GB  prefill win  miss/tok  saved
            0           0   0.0    21 of 21      20.4     0ms
           16         320   6.0    19 of 21      17.4    19ms
           32         640  12.0    16 of 21      14.8    36ms
           48         960  18.0    14 of 21      12.7    51ms
           64        1280  24.1    11 of 21      10.8    63ms
           96        1920  36.1     6 of 21       7.8    83ms

**960 exclusive slots buy 51 ms; 1024 inclusive slots buy 11 ms.** Same RAM, ~5x
the return. The lever is DISJOINTNESS, not capacity — which is also why every
previous attempt to size box 1's decode cache measured flat.

The prefill cost is real and unpaid-for above: k=48 leaves 14 of 21 windows, so 6
encoder layers lose their pin. Box 1's pool cannot simply grow to cover both —
52 GB is the measured ceiling and 76 GB OOMs. A full phase swap (release windows
at decode start, reclaim at prefill) costs 18 GB of re-paging at 4.93 GB/s = 3.7 s
per switch, too slow per request. **So this needs a prefill A/B before shipping**,
at k=16 and k=32 where only 2-5 windows are given up.

## What this is worth

At k=32 (640 slots, 5 windows given up): 20.4 -> 14.8 misses/token = **~36 ms off
a 246 ms token -> 210 ms = 4.8 tok/s**, from 4.06 (+18%). At k=48, 51 ms -> 5.1
tok/s (+26%). The picks box 1 serves are also computed on its idle iGPU
*concurrently* with box 2, so some of box 2's compute leg leaves the critical path
too — not modelled above, so these are lower bounds on wall-clock.

**The hub swap changes this arithmetic favourably.** With box 2 (128 GB) as hub
holding dense weights and KV, box 1 becomes a PURE expert executor: no dense
weights, no KV, so most of its 96 GB and all 17.1 GB of idle dGPU VRAM become
expert slots — roughly 4,000 + 900 against today's 2,766, and with no prefill
windows to defend, all of it available for an exclusive decode band. Re-run this
simulation against that split before committing to a k.

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
