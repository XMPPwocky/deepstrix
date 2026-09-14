# Prefill @100K, measured stage profile (2026-09-14)

First run of `DEEPSTRIX_PREFILL_PROFILE=1` at the context the goal names. Indexer
ON (`V41_INDEX_K=1`), box 2 on the adopted 260/68 placement.

    96,935 prompt tokens, ced_replay total_s = 160.0  ->  606 tok/s

(The 395 tok/s figure carried earlier predates today's changes and is not a
controlled comparison; 606 is the current number. Part of the difference is
probably the 260/68 placement: the CED replay runs 128 tokens through decoder
layers 20-39, which previously had 40 slots each and now have 68 — replay
elapsed_s is 12.4 of the 160.)

## Where it goes

    igpu  igpu.pair_kwide                51,394 ms
    igpu  igpu.q2k_down                  42,642 ms
    dgpu  dgpu.ffn_combine               39,621 ms
    dgpu  dgpu.attn_compute              13,605 ms
    dgpu  dgpu.mhc_pre_attn               6,936 ms
    igpu  igpu.peer_push_ffn_moe          6,027 ms
    dgpu  dgpu.peer_push_ffn_input_norm   5,920 ms
    dgpu  dgpu.prefill_indexer_reuse      4,050 ms   <- S2
    dgpu  dgpu.prefill_indexer            3,164 ms   <- S1
    dgpu  k.attn.smwsum                   3,470 ms
    dgpu  k.attn.score                    2,803 ms

    device totals:  dGPU 93,634 ms   iGPU 102,067 ms

## PREFILL IS iGPU-MoE-BOUND, NOT ATTENTION-BOUND

The two MoE kernels are **94.0 s of the iGPU's 102.1 s (92%)**, and the iGPU is
the LONGER pole. Attention (`attn_compute` + `mhc_pre_attn` + `smwsum` + `score`
= 26.8 s) is ~29% of the dGPU — the SHORTER pole.

**Therefore S3 is not worth building for prefill throughput.** Removing attention
entirely cannot move a wall set by the other device. The sparse indexer is still
required for long-context VALIDITY and for the VRAM its scratch flip reclaims, but
it should no longer be sold as a prefill-throughput lever. This also corrects the
"attention is ~59% of prefill at 100K" figure quoted earlier today; the honest
number is ~29% of the dGPU and ~13% of the combined stage time.

## The prefill lever is the iGPU MoE kernel bandwidth

`project_v41_prefill_moe_roofline_2026-09-14` recorded these two kernels running at
**124 GB/s and 80 GB/s against ~230 GB/s achievable** on this iGPU. Applying that:

    pair_kwide  51.4 s at 124 GB/s -> 27.7 s at 230
    q2k_down    42.6 s at  80 GB/s -> 14.8 s at 230
    iGPU 102.1 s -> 50.6 s

    optimistic (wall = max of device totals):  93.6 s -> 1,035 tok/s
    conservative (saving applied to the 160 s measured wall): 108.5 s -> 893 tok/s

**Use 893.** The measured wall (160.0 s) exceeds max(dGPU, iGPU) = 102.1 s, so
~58 s is host time, gaps and pipelining slack that a kernel fix would not touch.

So closing the iGPU MoE roofline gap is worth roughly **606 -> ~890 tok/s** and is
the single biggest prefill lever on the board — the only measured path that gets
within reach of the 1000 tok/s clause.

Caveats: the 124/80/230 GB/s figures are from an earlier roofline pass and were
NOT re-measured today; re-verify before committing effort. And the unexplained
58 s between the stage totals and the wall is itself worth a look — it is 36% of
prefill and nothing currently accounts for it.

## RESOLVED AGAINST THIS DOCUMENT — the ~890 tok/s projection is WITHDRAWN

The 2.1x disagreement below is settled, and the earlier analysis was right.
`reference_rocprofv3_kernel_trace` states the rule plainly: **stage timings
include intra-stage stream-sync idle and must NOT be used for compute cost** —
with a recorded case where a stage read 185 ms against ~40 ms of actual kernel
time, the remaining 123 ms being device idle inside the stage span.

`igpu.pair_kwide` and `igpu.q2k_down` are stage timings. I used them for exactly
the question the rule forbids.

Against TRUE per-kernel time (`bench_v41_kernel_roofline`, 39.244 + 31.778 ms per
call at B=1024, all 384 experts), box 1's share at 124/384 ownership:

    (39.244 + 31.778) ms x 0.323 x 20 layers x 95 chunks = 43 s

    stage figure quoted above       94 s   -> inflated 2.2x
    true share of the 160 s wall    27%    (this document claimed 59%)
    closing the 2.1x roofline gap   saves ~23 s -> 706 tok/s, NOT 893

**So the MoE kernel program is not the prefill lever, and the earlier analysis
that said so stands.** Everything below is retained as the record of how the
error was made; the numbers in it are stage-scoped and overstate GPU cost by
~2.2x. The remaining candidate lever is the encoder SPLIT (box 2 carries 260/384
encoder experts on an identical 256 GB/s iGPU while box 1 also does attention,
projections and the head), which is a config change rather than a kernel program
— and the 58 s gap between wall and max(device totals) is consistent with that:
it is idle, not unaccounted compute.

## (superseded) this contradicts the earlier roofline analysis by 2.1x

`project_v41_prefill_moe_roofline_2026-09-14` reached the OPPOSITE conclusion —
"the MoE kernel program is NOT the prefill lever at 100K" — by scaling its
synthetic bench by production ownership:

    box 1 MoE = 72.24 ms x 116/384 x 20 layers x 101.4 chunks ~= 44 s   (~15% of wall)

Today's direct measurement says box 1's two MoE kernels are **94.0 s**, i.e. 2.1x
that estimate and 59% of the 160 s wall. Both cannot be right, and the ~890 tok/s
projection above depends entirely on which is.

Candidate explanations, none verified:

1. **Ownership drift.** The estimate assumed box 1 owns 116/384 encoder experts.
   Under the T2 catch-all box 1 may be computing more than it nominally owns.
2. **Stage timings include waiting.** `igpu.pair_kwide` is a GPU-event-scoped
   stage; if it brackets stream idle it overstates kernel cost. The same
   suspicion applies to the 58 s that separates the 160 s wall from
   max(dGPU 93.6, iGPU 102.1) = 102.1 s.
3. **Chunk count.** 96,935 / 1024 = 95 chunks assumed; the CED encoder pass may
   issue more.

If (2) holds, the MoE kernels are cheaper than they look and closing the roofline
gap buys much less than 890 tok/s — and the earlier analysis was right that the
lever is the SPLIT (box 2 owning 260/384 encoder experts on an identical iGPU),
not the kernels.

**Do not start a kernel program on the strength of this document.** The cheap
discriminator is `rocprofv3 --kernel-trace` over one 100K prefill (recipe in
`reference_rocprofv3_kernel_trace`), which gives true per-kernel GPU wall time and
settles both the 2.1x gap and the unexplained 58 s at once. That is the honest
next step, and it is one run.

Recorded because the earlier analysis carried its own warning that this exact class
of inference — scaling a synthetic bench by ratios — had already produced three
wrong conclusions in one day. Today's number is a real aggregate rather than an
extrapolation, which is why it is quoted first, but a measured disagreement of 2.1x
is not resolved by preferring the newer measurement.

## CORRECTION 2: the kernel program IS the lever — on BOX 2, which emits no stages

The withdrawal above is half right and over-corrects. Box 1's stage timings were
indeed inflated 2.2x, but "the MoE kernel program is not the prefill lever" does
not follow, because **box 2 runs the same two kernels on twice the experts and
does not appear in box 1's stage log at all.**

Box 2's daemon reports its OWN per-request GPU time (not a stage timing). For the
same 100K prefill:

    expertd B=512  n=3760  us p50: read 2453  h2d 116  gpu 22056  d2h 501  write 3499

    n = 3760 ~= 20 encoder layers x 189 chunks (96,935/512 = 189)   <- accounts exactly
    box 2 MoE GPU      82.9 s
    box 2 non-GPU o/h  24.7 s   (read 9 s, write 13 s)
    box 2 TOTAL       107.6 s   of the 160 s wall  <- the prefill critical path
    box 1 MoE (true)   43.0 s

**Cross-check:** 82.9 / 43.0 = 1.93 against an ownership ratio of 260/124 = 2.10.
Two independently measured quantities — box 2's daemon timings and box 1's
bench-derived kernel time — agree to 8%. That is what makes this trustworthy where
the stage figure was not.

Applying the recorded roofline gap (124 and 80 GB/s vs ~230 achievable = 2.20x on
the chain):

    box 2 MoE 82.9 s -> 37.6 s, saving 45.3 s  ->  up to ~845 tok/s

**Read that as an upper bound, not a projection.** Box 1 and box 2 run
CONCURRENTLY, so the wall is bounded by max(box 1 path, box 2 path); once box 2's
MoE drops to 37.6 s its total falls to ~62 s and box 1 becomes binding. Box 1's
true path is unknown — its dGPU stage total (93.6 s) is inflated by the same 2.2x
effect and has not been re-measured with `rocprofv3 --kernel-trace`. So the gain
is real and large but its ceiling is not established. Do NOT sum the two boxes'
savings: that would give 1,063 tok/s and is wrong, because they overlap.

**Net: the MoE kernel roofline gap is the prefill lever after all, and it is worth
up to ~45 s of a 160 s wall.** The earlier analysis was right that box 1's share is
too small to matter (43 s) and wrong to stop there — it priced box 1's ownership
and never priced box 2's, which is 68% of the encoder.

Secondary, and cheap: box 2's per-request `write` is 3.5 ms x 3760 = **13 s** and
`read` 2.45 ms x 3760 = **9 s**, together 22 s of the wall in pure serialisation
and I/O around the kernels. That is worth more than closing the whole attention
path and needs no kernel work.
