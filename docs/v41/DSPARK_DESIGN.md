# DSpark: the drafter is three transformer layers, and every kernel exists
### checkpoint inventory + build plan, 2026-09-15

## What is actually in the checkpoint

`mtp.0`, `mtp.1`, `mtp.2` — **2.65 / 2.57 / 2.71 GB, 7.93 GB total**, so the whole
drafter is RESIDENT with no paging. Each stage is structurally ONE transformer
layer, and every tensor maps onto a kernel the engine already runs:

    main_proj.weight      F8_E4M3  (5120, 15360)   <- 3 x 5120 residuals -> 5120
    main_norm.weight      BF16     (5120,)
    attn_norm.weight      BF16     (5120,)
    attn.wq_a/.wq_b       F8_E4M3  (1280,5120)/(32768,1280)
    attn.wkv              F8_E4M3  (512, 5120)
    attn.wo_a/.wo_b       F8_E4M3  (8192,4096)/(5120,8192)
    attn.q_norm/.kv_norm  BF16     (1280,)/(512,)
    attn.attn_sink        F32      (64,)
    ffn_norm.weight       BF16     (5120,)
    ffn.gate.weight       BF16     (128, 5120)     <- 128 experts, not 384
    ffn.gate.bias/.bias_vl F32     (128,)
    ffn.experts.0..127    I8+F8_E8M0               <- MXFP4, same layout as main
    ffn.shared_experts    F8_E4M3  w1/w2/w3
    hc_attn_base/fn/scale F32      (24,)/(24,20480)/(3,)   <- mHC
    hc_ffn_base/fn/scale  F32      same
    confidence_head.proj  BF16     (1, 5376)

There are NO shared interface tensors outside the stages: each stage carries its
own `main_proj` and `main_norm`.

## The draft step

    x = concat( mean-over-hc-copies of the residual ENTERING layers 37, 38, 39 )   [15360]
    h = main_norm( main_proj(x) )                                                  [5120]
    h = layer_forward(h)      // attn_norm -> MLA -> mHC -> ffn_norm -> MoE -> mHC
    logits = tied_head(h)

Three stages give **three draft tokens**, so the verify batch is **B=4**. That is
exactly the width the checkpoint is built for, and `dspark_accept.py` measures
**E = 3.57 accepted tokens per verify at K=3** (1.93 / 2.77 / 3.57 / 4.94 at
K = 1/2/3/5).

## Build order

  * **L. Loader** — `mtp.{0,1,2}.*` into resident device buffers. Mirrors the
    per-layer loader; the expert tensors are the same MXFP4 layout the shard
    already reads, just 128 of them. Testable in isolation by shape assertion
    against `config` constants.
  * **F. Forward** — `forward_mtp_stage(stage, x15360, pos) -> logits`, composed
    entirely of existing kernels. Testable against `scripts/v41_oracle`'s
    reference for a fixed input.
  * **V. Verify** — batched DECODE forward over B=4 with experts through T2
    catch-all. **NOT `forward_prefill`**: measured 775 ms at B=5 because it pages
    a per-layer union off box 1's own disk, and it is unrollbackable (its
    post-chunk eviction relocates the KV window, which `rollback_kv` refuses).
  * **A. Accept/reject** — compare draft tokens against the verify logits,
    accept the longest matching prefix, `rollback_kv` the rest. Rollback is DONE
    and verified byte-identical (`c0e3c8e`) and REQUIRES `V41_T2_CATCHALL=2`
    (mode 1's residency-based partition is history-dependent and diverges).

## Correctness gates, in order

  1. Loader: every `mtp.S.*` tensor consumed, no unconsumed tensors, shapes match.
     The 0.44-acceptance bug was exactly an unloaded-tensor bug (`ffn_norm` left
     at 1.0, a 4-6x input scale error) that produced *plausible* tokens — so a
     sanity check on output quality is NOT sufficient here.
  2. Forward: stage-0 logits match the Python oracle for a fixed residual.
  3. Verify: a B=4 batched decode forward must equal 4 sequential `forward_token`
     calls on the same tokens (this is the continuation-parity property; the
     existing `forward_prompt_batch_matches_sequential` test is V4-Flash GGUF and
     tests batch-from-scratch, not continuation).
  4. End to end: greedy decode with DSpark must be byte-identical to greedy
     decode without it. Speculative decoding is EXACT — if the output differs,
     the accept/reject is wrong.
  5. Only then measure: acceptance rate against the oracle's 3.57, and tok/s.

Gate 4 is the one that matters and the one this project is equipped for: the
`V41_VERIFY_PROBE` harness already exercises speculative ingest + rollback and
was validated byte-identical under mode 2.

## Why this is smaller than it looked

No new kernels. The drafter is 7.9 GB resident, so no pager, no remote shard, no
link traffic — it runs entirely on box 1 alongside the dense chain. The expensive
unknown is V (the batched decode path), which is also the one piece that pays off
independently of DSpark.
