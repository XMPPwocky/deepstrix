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
