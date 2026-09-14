# Brief: get V4.1-Flash decode to 30 tok/s WITHOUT speculative decoding

Everything below is MEASURED on this hardware 2026-09-14 unless marked. Current
decode is **3.24 tok/s (308.6 ms/token)**, steady state, warm pool. Target 30 tok/s
= **33 ms/token**. That is a 9.3x gap, so incremental tuning is not the answer and
the frame itself should be attacked.

## Hardware

* box 1 "lumi-brain" — Strix Halo, **96 GB** unified. iGPU gfx1151 + RX 9070 XT
  (gfx1201, 16 GB) over OCuLink. Runs the hub/engine. NVMe is **dm-crypt** (~6.9 ms
  to read one expert).
* box 2 "lumi-brain2" — Strix Halo, **128 GB** unified, iGPU only, at 10.99.0.2.
  Runs `deepstrix-expertd`. NVMe is **plaintext** (~7.9 ms/expert = ~2.38 GB/s).
* link: USB4, measured **~724 MB/s**. Shipping an 18.8 MB expert over it costs
  ~26 ms — worse than either box reading its own disk. Weights never cross it.

## Model

DeepSeek-V4.1-Flash. 40 layers, **384 routed experts/layer, top-6**, 1 shared
expert, hidden 5120, moe_intermediate 2304, 64 KV heads. Vision tower attached.

* per-expert = 3 x 5120 x 2304 = 35.4 M weights. Stored ggml **MXFP4: 17 B per 32
  weights = 4.25 bits/weight = 18.80 MB**. Source `expert_dtype='fp4'` with one
  ue8m0 scale per 32x32 block (4.008 b/w), so tighter packing buys only 5.7%.
* **full expert set = 15,360 experts = 288.8 GB.** Total RAM across both boxes is
  224 GB, of which maybe ~180 GB is usable for experts. Holding all of them would
  need **2.65 bits/weight** — a different model, not a requant.
* Engram table: 98 GB on disk, gathered per token.
* CED prefill: encoder layers 0-19 over all tokens, then a 128-token replay through
  decoder layers 20-39.

## Where the 308.6 ms goes (per token, `het.token.summary`)

    total_us      308,626      remote_rtt_us  ~230,000-300,000   (85%)
    sel_sync_us    20,431      engram_us        7,451
    gap_us          7,689      pager_ensure_us  3,688
    sync_us         1,558      sample/embed/stream  ~250

Box 2's daemon, per B=1 request (40 requests/token, one per layer):

    queue 3 us | h2d 51 us | gpu 651 us | d2h 17 us     <- compute is NOT the problem
    hit 0.876 steady | ms_per_miss 9.00 (read 7.86 + h2d 1.12)

Box 2 currently receives **all 6 picks of every layer** (page stats increment 6
expert requests per network request) because box 1's decode LRU is 25 slots — the
packed prefill windows own 2,944 of its 2,969.

So: ~240 picks/token, ~12% miss, ~26 misses/token x 9.0 ms ~= **230 ms/token of
blocking disk read**. That is the whole problem.

## What has already been measured DEAD (do not re-propose)

1. **Cache policy.** Belady OPT at box 2's 6160 slots: per-layer LRU 18.6
   misses/tok, global pool LRU 15.3, **OPT 15.0**. A global pool is within 2% of
   optimal; total policy headroom over today is 32 ms/token = 3.24 -> 3.6 tok/s.
   Nothing cleverer than LRU is worth building.
2. **More capacity, by policy.** >=63% of real misses are capacity misses, proved by
   counting: 41,603 misses in a 1,200-token run against only 40x384 = 15,360
   (layer,expert) pairs that exist, so at most 15,360 can be first touches.
3. **Rebalancing the legs.** Box 2's miss COUNT is invariant (~17/token) whether it
   handles 240 picks or 48: moving picks to box 1 moves the HOT ones that were
   hitting anyway and leaves box 2 the cold tail that misses. Worth 4.2 -> 4.4 tok/s.
4. **Requantising experts.** They are already fp4/MXFP4 at 4.25 b/w. Q5_K is BIGGER.
5. **Streaming weights over USB4.** 26 ms/expert, worse than local disk.
6. **Engram.** Row cache took it 7.9 ms -> 23 us on repetitive text but it is ~7.5 ms
   on novel n-grams; it is 2.4% of the token either way. Not the lever.

## The question

**Find a path to 33 ms/token that does not use speculative decoding / DSpark / MTP.**

Attack the frame, not the constants. Some directions that have NOT been ruled out,
offered only to show the kind of move that counts — find better ones:

* **I/O path, not I/O volume.** 7.86 ms for 18.80 MB is **2.38 GB/s**, which is
  slow for Gen4 NVMe (5-7 GB/s). The read is a synchronous `pread` of a whole
  expert. What does io_uring, O_DIRECT, larger queue depth, or striping across a
  second drive actually buy? If reads hit 7 GB/s the miss cost drops 9.0 -> ~3.8 ms.
  Is 26 misses x 3.8 = 99 ms/token, and what else then binds?
* **Overlap instead of block.** Misses are synchronous inside the layer request.
  The router's picks for layer L are known only after layer L-1, but the two boxes
  are pipelined and there are 40 layers — is there any lookahead that is NOT
  speculation (e.g. the shared expert, the attention chain, Engram) to hide reads
  behind?
* **Reduce picks, not bytes.** 6 of 384 per layer. Is the 6th pick worth its
  latency? What does routing weight mass look like across the top-6 — is there a
  measured quality/latency curve for top-4 or top-5?
* **Change what a "miss" costs.** Must a miss fetch the whole 18.8 MB expert, or
  can the FFN be computed from a partial/streamed read, or a lower-precision
  resident copy with the fp4 copy as a refinement?
* **Change residency granularity.** Experts are cached whole. Routing is Zipfian
  within a layer. Is a sub-expert (row-block) cache viable?
* **Use the dGPU.** 16 GB of gfx1201 VRAM is currently not a decode expert tier.

## Rules

* Do NOT edit the NixOS flake at ~/lumi-brain/lumi-brain for box-2 changes.
* Rust dependencies need sign-off — stop at Cargo.toml.
* One weight load per test run; the box has OOM'd, run ONE heavy job at a time.
* Quote prefill/decode at the context the goal names, and check a measurement's
  regime before transferring it (this session lost hours to a decode profile quoted
  as prefill, and to a hit rate read off a pool that had never filled).

Deliverable: a ranked set of concrete, measurable proposals with predicted
ms/token, each with the cheapest experiment that would falsify it. Say plainly if
you conclude 30 tok/s is not reachable pre-DSpark on this hardware, and give the
number that proves it.
