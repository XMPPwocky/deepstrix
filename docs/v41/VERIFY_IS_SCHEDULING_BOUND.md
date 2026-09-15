# The verify is 80% idle: a costed path to 20 tok/s DSpark
### perfetto trace + analyzer, 2026-09-15, B=6 verify probe

## Every track is idle

`~/scripts/analyze_pftrace_gaps.py` over a trace containing 6 verify probes at
B=6, 14 decode tokens and one prefill (span 10.2 s):

| track | busy | % of span |
|---|---|---|
| remote.expert (box 2) | 1694 ms | 16.7% |
| dgpu.compute | 1085 ms | 10.6% |
| igpu.compute | 608 ms | 6.0% |
| expert pager | 148 ms | 2.1% |
| dgpu.xfer | 38 ms | 0.4% |
| igpu.xfer | 16 ms | 0.2% |

Nothing is busy. Total busy across all tracks is 3.6 s inside a 10.2 s span, and
the largest single track is 1.7 s. Corroborated off-trace (perfetto gaps have
been an artifact here before, see [[project-decode-at-floor]]): the decode
`het.token.summary` independently reports `dgpu_busy_us=0 igpu_busy_us=0` with
`host_us` ~= `total_us`.

## The critical path is one gap, and it is the same gap on both devices

dGPU, top gap by total time:

    k.shared_expert.down_matvec -> k.ffn_combine.vec_add
      4,980,390 us total | 239x | avg 20,838 us

239 = 6 verifies x 40 layers. **Each verify layer stalls 20.8 ms** with the dGPU
waiting for the MoE partial before it can combine. 40 x 20.8 = 833 ms, which is
the whole 949 ms verify.

iGPU, top gap:

    igpu.q2k_down -> igpu.q8k_quantize_pre_iq2
      5,495,030 us total | 234x | avg 23,483 us

Same 234 ~= 6 x 40. The iGPU finishes its MoE and then idles 23.5 ms waiting for
the next layer's input. **The two devices are waiting on each other**, and the
real work inside that window is ~5.6 ms/layer (iGPU 1.6 + box 2 ~4.0).

## It is not the things it looks like

* **Not expert I/O.** `pager_misses=0`, `pager_read_us=0`, `pager_h2d_us=0`; the
  pager track is 97.9% idle.
* **Not the link.** The trace labels the remote round trips directly:
  `rtt=861us link=612us remote=216us` at b=8. Under 1 ms per layer.
* **Not box 2's throughput.** It is 83.3% idle.
* **Not CED**, and **not the two-lane split** (both measured separately — see
  `DSPARK_VERIFY_ECONOMICS.md`).

It is host-side serialization inside the prefill driver's per-layer loop, in
code that no `events.stage(..)` scope covers — which is why it shows up as a gap
BETWEEN slices rather than as a stage.

## What this makes the goal

**Decode already solves this same ping-pong at 2.58 ms/layer** (its version of
the dGPU gap, 560x = 14 tokens x 40 layers). The prefill/verify path does it at
20.8 ms/layer — 8x worse for 6x the tokens. Decode gets there with captured HIP
graphs (`dgpu_graphs.run(..)`, which exists specifically to close a
"~115 us/layer host-scheduling gap") and the pre-submit reorder; the prefill
driver has neither, because it was built to move 512-row chunks where this
disappears against real GPU work.

Budget at 20 tok/s = 50 ms/token. With the oracle's E=4.38 at B=6 a step must
fit in ~219 ms:

| verify per-layer latency | verify(B=6) | step (+35 ms drafter) | tok/s |
|---|---|---|---|
| 20.8 ms (today) | 833 ms | 868 ms | 5.0 |
| 5.6 ms (its own work) | 224 ms | 259 ms | 16.9 |
| 2.58 ms (what decode achieves) | 103 ms | 138 ms | **31.7** |

So **20 tok/s does not need a faster machine or a better drafter — it needs the
verify's per-layer latency to land between decode's 2.58 ms and its own 5.6 ms
of work.** Closing that gap is the entire project, and it is scheduling, not
arithmetic.

Order of attack, by measured size:
1. Port the decode path's captured-graph scheduling to the verify's per-layer
   loop. This is the 20.8 -> ~3 ms lever.
2. Pipeline the remote submit across layers (submit L+1's picks as soon as its
   router is done, instead of after L's wait). The trace shows box 2 starved at
   16.7%; this is the "race to tell box 2" idea, and it helps DECODE too.
3. Only then revisit acceptance (prefill window seeding, E 3.28 -> 4.38), which
   is worth ~33% once the step cost is actually small.
