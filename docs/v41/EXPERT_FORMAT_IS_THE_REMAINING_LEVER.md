# The expert format is the remaining lever (2026-09-15)

Scoped from the session's measurements. This is the only identified path to
20 tok/s; everything else on the decode and verify paths has been measured and
refuted. It is a PROJECT, not a change — written down so it can be started
cleanly rather than half-built.

## The measured case

Decode is 180-230 ms/token warm. Where it goes, from the token summary and box
2's own counters:

    remote_srv   121-213 ms   box 2's service, which is ~all page misses
    sel_sync      21-27 ms    box 1 syncing to read router picks, per layer
    remote_link    3-4 ms     the LINK IS NOT THE BOTTLENECK (3%)
    box-1 misses   0          decode's catch-all keeps box 1 off its own disk

Box 2 misses ~20 experts/token at 7.1-8.0 ms each. That is a CAPACITY fact, not
a policy one:

    expert set   15,360 experts (384 x 40)
    MXFP4 today  19.25 MB each  ->  289 GB
    RAM          box2 ~124 GB + box1 pool ~52 GB  =  ~176 GB
    residency    ~43%  ->  ~20 misses/token

Refuted as levers, each by measurement, this session: box-2 slot rebalance
(164/164 is 48% slower than 260/68), box-1 pool size (three times, and bigger is
slower — cost is max(box1, box2)), leg balance (`V41_LOCAL_PICKS=5` is 3x worse
because box 1 pages what it does not hold), the link, and the adaptive victim
cache (untested: its admission path fired 7 times in a run).

## The arithmetic that makes format the lever

    format     bpw    per expert    full set    fits 176 GB?
    MXFP4     4.25     19.25 MB      289 GB     no  (43% resident today)
    IQ3_XXS   3.06     13.86 MB      208 GB     no
    IQ2_S     2.50     11.32 MB      170 GB     YES, ~6 GB margin
    IQ2_XXS   2.06      9.33 MB      140 GB     yes, comfortably

At 100% residency box 2's service collapses to its compute term — 40 layers x
~0.6 ms = ~24 ms/token — and the token becomes roughly

    24 (box2 compute) + 22 (sel_sync) + 4 (link) + box-1 dense chain

i.e. **50-90 ms/token = 11-20 tok/s non-speculative**, versus 5.2 today. That is
the goal's range without speculation at all, and it is also the precondition that
makes speculation worth revisiting: the verify's dominant term is the same box-2
expert service.

## What makes it a project, not a change

1. **The runtime already exists.** `iq2_s_pair_matvec.hip` is in tree and
   `het/dispatch.rs` dispatches per-tensor `GgufType`, already mixing formats
   within a layer. This is the cheap part.
2. **The weights do not.** V4.1 loads HF safetensors whose experts are natively
   MXFP4. Requantising means dequant -> imatrix -> IQ2_S for 15,360 experts, and
   2-bit experts on a 552B model carry real quality risk that must be measured
   against the CPU oracle, not assumed.
3. **It forces an architectural trade.** 170 GB does not fit on box 2 alone
   (124 GB); it needs ~46 GB of box 1's pool. Box 1's pool is currently 21 pinned
   prefill windows worth prefill 159 -> 255 tok/s. So full expert residency is
   bought partly with prefill residency, and that trade has to be priced, not
   waved through.

## Order of work

1. Price IQ2_XXS too (140 GB fits box2+box1 with room, and may fit a smaller
   box-1 donation). Cheaper on RAM, worse on quality — measure both.
2. Requantise ONE layer's experts, check output against the CPU oracle
   (`project_v41_oracle`), and measure box-2 hit rate on that layer alone.
3. Only then do the full set, and re-price the prefill-window trade.

## Measurement discipline (learned the hard way today)

Box 2's LRU converges over hundreds of thousands of requests. Two IDENTICAL
baselines bracketing one arm differed by 10.4% today. Warm every arm to
convergence or interleave arms in one process; corroborate any delta with box
2's own `page stats` miss delta, never tok/s alone.
