# DSpark: the drafter is three transformer layers, and every kernel exists
### checkpoint inventory + build plan, 2026-09-15

## What is actually in the checkpoint

`mtp.0`, `mtp.1`, `mtp.2` — **2.65 / 2.57 / 2.71 GB, 7.93 GB total**, so the whole
drafter is RESIDENT with no paging.

**They are a 3-LAYER draft MODEL, not three independent drafters** — corrected
after the loader test caught it. Only the entry layer has `main_proj`/`main_norm`
and only the exit layer has the final norm and heads:

    mtp.0   main_proj + main_norm -> layer                    (entry)
    mtp.1   layer                                             (middle)
    mtp.2   layer -> norm -> confidence_head + markov_head     (exit)

Run autoregressively to emit K draft tokens. Each layer is otherwise a standard
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
    confidence_head.proj  BF16     (1, 5376)      <- exit layer only
    markov_head.embed/.head                        <- exit layer only, 256-dim hidden
    norm.weight           BF16     (5120,)        <- exit layer only

VERIFIED by `crates/v4flash-core/tests/mtp_weights_present.rs`: all three layers
present with the right geometry, the router 128-wide, and `ffn_norm` means
**0.1571 / 0.2005 / 0.2405** — the trained gains, not RMSNorm's init of 1.0.

## Geometry, from the checkpoint's own config (not inferred)

    dspark_block_size          5        draft tokens per speculation step
    sliding_window           128        the drafter's KV ring depth
    dspark_n_routed_experts  128        vs the main model's 384
    dspark_num_experts_per_tok 3        vs the main model's 6
    dspark_target_layer_ids  [37,38,39] residuals the drafter eats
    num_nextn_predict_layers   3        drafter layers
    dspark_markov_rank       256        auxiliary n-gram head width

## The draft step

    x = concat( mean-over-hc-copies of the residual ENTERING layers 37, 38, 39 )   [15360]
    h = main_norm( main_proj(x) )                                                  [5120]
    h = layer_forward(h)      // attn_norm -> MLA -> mHC -> ffn_norm -> MoE -> mHC
    logits = tied_head(h)

`dspark_block_size = 5`, so a speculation step drafts **5 tokens** and the verify
batch is 6. `dspark_accept.py` measures **E = 1.93 / 2.77 / 3.57 / 4.94 at
K = 1/2/3/5**, so K=5 is both the configured and the best-measured width.

### The attention, which is NOT what it looks like

`DSparkAttention.forward(x, start_pos, main_x)` — corrected after reading the
reference rather than inferring from the tensor names:

  * `main_x` is the **KV source for EVERY layer**, not just the first. `h` (the
    query stream) comes from `forward_embed` of the input token. I had this
    backwards.
  * The ring holds `kv_norm(wkv(main_x))`, roped at the MAIN position, written at
    `start_pos % window_size`.
  * The block's own KV comes from `x`, roped at the block position, and is
    CONCATENATED after the ring.
  * `sparse_attn` with `get_dspark_topk_idxs` is **not sparse**: the indices are
    `[0 .. min(win, start_pos+1)) ++ win+[0 .. block_size)`, identical for every
    query. It is dense attention over "valid ring prefix ++ current block", which
    is `attn_swa`'s exact shape at block_size 1.
  * An **inverse rope is applied to the attention OUTPUT** before `wo_a`/`wo_b`.
    `RopeTail::launch_inverse_pdev` already exists for the main model's
    equivalent.

### The drafter runs at B=5, and there is no causal mask inside the block

Two more things that the `h = layer_forward(h)` sketch above hides, both
confirmed against `forward_spec` / `forward_embed` / `get_dspark_topk_idxs`:

  * The drafter is **not autoregressive**. `forward_embed` builds
    `draft_input_ids = [real_token, noise, noise, noise, noise]` and embeds all
    `block_size = 5` of them, so ONE pass over the 3 layers emits all 5 drafts.
    Every buffer in `MtpState` is therefore `[5, ...]`, not `[1, ...]`. Running
    it as 5 sequential B=1 steps would be a different (and much slower) model.
  * `get_dspark_topk_idxs` `.expand(bsz, block_size, -1)`s ONE index vector over
    every query, so all 5 positions attend the identical set — **the block is
    bidirectional, not causal**. `n_kv` is the same for all 5 queries and no
    mask kernel is needed. (The noise placeholders carry no information, so
    there is nothing to leak.)

Two consequences for the ring layout. The valid ring entries are the prefix
`[0, min(win, start_pos+1))` — while filling, `start_pos % win == start_pos`;
once full, all `win` are valid — so `n_valid(pos)` is right either way. And
because RoPE is baked into each key at cache time, attention is
**permutation-invariant over the KV set**: the block's transient KV can be
compacted to `[n_valid, n_valid + 5)` instead of the reference's fixed
`[win, win + 5)`, which keeps the keys contiguous for `attn_mixed` and costs
nothing. Anything at or past `n_valid` is scratch, so the next step's `main_kv`
write at `pos % win` legitimately overwrites it.

### Landmine: the batched attention kernels are dGPU-only

The drafter runs on the **iGPU (gfx1151)**, and every
`attention_mixed_*_batched_htiled_wmma*` kernel guards its weighted-sum phase
behind `#if defined(__gfx1200__) || defined(__gfx1201__)`. On the iGPU the
softmax phase still runs and writes `scores`, and the weighted sum is compiled
out, so `out` is never written — attention returns **exactly zero** with no
error. The drafter then looks healthy from the outside: `h` is finite, it varies
across the block, and only the MoE is actually contributing.

Two things that caught it, both cheap and worth keeping on any new stage:

  * attention output of *exactly* zero is diagnostic on its own — MLA shares K
    and V, so a zero output means the value path never ran (or the ring is
    empty), never "the weights came out small";
  * **assert the output depends on `start_pos`**. Different positions rope at
    different angles and see a different `n_valid`, so identical output at
    pos 37 and pos 900 is proof that attention is not contributing.

Use `attention_mixed_score` / `attention_mixed_softmax_wsum` (the B=1 pair, no
arch guard) in a loop over the five queries. Note their scores stride is
`ATTN_MIXED_MAX_KEYS` (82176), **not** `n_kv`: one shared `[N_HEAD, 82176]`
buffer, not a per-query slice sized to the key count, which writes ~600x past
its end.

## Build order

  * **L. Loader — DONE.** `mtp.{0,1,2}.*` presented through `V41HfWeights` under
    `mtp.{s}.*` names. The expert `Kind` was generalised from a layer index to a
    name prefix + count, so the drafter's 128 experts reuse the same MXFP4 read
    path as the main model's 384. Gate 1 passes.
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
