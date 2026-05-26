# M50 — Full Batched Prefill: Progress

## What this is

V4-Flash decode is single-token at 27-28 tok/s (35.6 ms floor). For prefill (turning N prompt tokens into N forward-passes worth of KV cache + comp_kv state), naive sequential decode means N × 36 ms. A 200-token prompt = 7.2 seconds wall before generation starts.

M50 implements **layer-major batched prefill** — process all B tokens through layer 0, then all B through layer 1, etc. Stateless kernels run B-wide in one batched launch; stateful per-position kernels (KV append, compressor) run serially in an inner loop. Target: 150-250 tok/s prefill at B=64.

Plan: `~/.claude/plans/read-design-doc-agile-hollerith.md`.

## Status (May 2026)

### ✅ Phase 1: Scaffolding + correctness oracle
- `src/het/batch_scratch.rs` — `BatchScratch` (shared scratch + per-token residual buffers)
- `src/het/forward_prefill.rs` — `forward_prompt_batch` driver, loops `forward_layer_pair_mode` per batch element
- `tests/forward_prompt_batch_matches_sequential.rs` — passes bit-identical (max diff 0.0) at B=1, 4, 7
- `tests/bench_prefill.rs` — measures 77 ms/tok at B=7 (slower than 36 ms/tok single because pair_mode disables M30 combined graphs)

### ✅ Phase 2: dGPU batched kernels + v2 integration — PASSING ORACLE

`forward_prompt_batch_v2` + `forward_layer_batch_v2` in `src/het/forward_prefill.rs` — **bit-identical to sequential reference at B=4 and B=7** (max diff 0.0). Three bugs fixed during the integration:

1. **Stream race**: Stage 11's peer-pushes on `de.xfer` raced with Stage 9's `router_topk` writes on `de.compute` because there was no sync. Added `de.compute.synchronize()` before the per-batch peer-push loop.
2. **`hc_weighted_sum_batched` weights stride**: kernel assumed `weights[B, n_hc]` packed (stride 4) but callers pass `split[B, HC_MIX_DIM]` (stride 24). Added `w_stride` kernel parameter.
3. **Causal attention violation**: Stage 5 used the final post-loop `ls.n_raw` for every batch element's attention, letting token i attend to future tokens. Captured per-batch `n_raw_after[i]` / `n_comp_after[i]` snapshots during Stage 4 and use them in Stage 5.

Also: the wide single-token `f16.matvec` is used per-batch for the router (Stage 9) instead of the batched narrow kernel, to keep router-logit float-reduction order matching the single-token path. Saves us from float-drift causing topk to pick different experts. A future batched WIDE f16 matvec would replace this loop.

Bisect harness lives in `tests/forward_prompt_batch_matches_sequential.rs::forward_prompt_batch_v2_bisect_layer` (run-by-run B-by-B per-layer comparison; supports per-stage debug compare via `V2_DBG_LAYER=N`).

**Building blocks** (all committed):

| kernel | hip file | rust wrapper | validated by |
|---|---|---|---|
| `q8_0_gemv_batched_warp8` | `kernels/q8_0_matvec.hip` | `Q8_0Matvec::matvec_batched` | unit test + v2 oracle |
| `q8_0_grouped_gemv_batched` | `kernels/q8_0_grouped_matvec.hip` | `Q8_0GroupedMatvec::matvec_grouped_batched` | v2 oracle |
| `f16_matvec_narrow_batched` | `kernels/f16_matvec_narrow.hip` | `F16Matvec::matvec_narrow_batched` | v2 oracle (HC stages) |
| `q8_k_quantize` (trivial) | existing `kernels/q8_k_quantize.hip` | use `launch(n_blocks = B × per_token)` | v2 oracle |
| `rms_norm_weighted_batched` | `kernels/rms_norm.hip` | `RmsNorm::launch_weighted_batched` | v2 oracle |
| `rms_norm_no_weight_batched` | `kernels/rms_norm_no_weight.hip` | `RmsNormNoWeight::launch_batched` | v2 oracle |
| `hc_weighted_sum_batched` | `kernels/hc_weighted_sum.hip` | `HcWeightedSum::launch_batched(.., w_stride, batch)` | v2 oracle (after stride fix) |
| `hc_sinkhorn_par_batched` | `kernels/hc_sinkhorn_par.hip` | `HcSinkhorn::launch_batched` | v2 oracle |
| `hc_post_batched` | `kernels/hc_post.hip` | `HcPost::launch_batched` | not used by v2 yet |
| `hc_post_from_split_batched` | `kernels/hc_post.hip` | `HcPost::launch_from_split_batched` | v2 oracle |
| `hc_sigmoid_bias_batched` | `kernels/hc_sigmoid_bias.hip` | `HcSigmoidBias::launch_batched` | not used by v2 |

All v0 designs: `grid.z = B` parallel WGs, no W-amortization across batch (each WG re-reads weights independently). A v1 pass per kernel can pack multiple batch elements per WG to amortize W reads — that's the next-level perf optimization, but v0 already enables dGPU-side WG concurrency which should give 1.5-3× wall improvement.

**Remaining for Phase 2:**
1. **Unit tests** for the un-tested kernels (mirror the q8_0 test pattern).
2. **`split_unpack_batched` helper kernel** to gather `post[B, n_hc]` and `comb[B, n_hc, n_hc]` out of `split[B, 2*n_hc + n_hc²]`. Lets `HcPost::launch_batched` consume sinkhorn output.
3. **`BatchDgpuScratch`** struct in `src/het/batch_scratch.rs` — replace the current `shared_dgpu + per_token_residual` hack with a single struct of B-extended contiguous buffers (~26 MB at B=64).
4. **`forward_layer_batch_v2`** in `src/het/forward_prefill.rs` — ~500 LoC stage-by-stage rewrite, using batched kernels for stateless work, serial inner loop for stateful (kv_append, compressor). Stage-by-stage cheat sheet in `memory/project_m50_prefill_state.md`.
5. **Validate** via `forward_prompt_batch_matches_sequential` oracle at B=4, 8 (tolerance ~1e-3 — float reduction order will differ from single-token).
6. **Bench**: expect 70-100 tok/s at B=64 with dGPU batched + iGPU still serial.

### ✅ Phase 3: iGPU MoE batching

Added `iq2_xxs_pair_matvec_fused_swiglu_batch_BxN` and `q2_k_matvec_par_batched_BxN` (grid.z = B). Added `BatchIgpuScratch` with B-extended buffers. Stage 11 in `forward_layer_batch_v2` now does a single batched peer-push + 4 batched iGPU kernel launches + single batched peer-push back — replaces the per-batch serial loop. Bit-identical oracle at B=7. Batching is **by token** (each WG handles one (row_block, slot, token) triple) — no SGLang-style expert-major grouping, because at B=64 / 256 experts / top-6 the expected unique-experts-touched is ~78% (avg reuse ≈ 1.9×), so per-expert amortization is small.

### ✅ Phase 4: per-token causal attention

`attention_swa_batched` + `attention_mixed_batched` (grid `(n_head, B, 1)`) consume per-token `n_raw_per[B]` / `n_comp_per[B]` device buffers. Stage 5 in v2 now uploads those snapshots via `copy_from_host_async` (FIFO with the subsequent attention launch — sync `copy_from_host` would fence the device and erase the win). Bit-identical oracle at B=7. Bench: B=64 = 67.7 tok/s (+9% over Phase 3).

NB: gain is modest because the per-WG work is uneven (token b=0 attends to 1 KV row, token b=B-1 to B). Wave-quantization tail latency means ~B× WGs don't fully parallelize. A future variant could decompose the kv-axis sum across multiple WGs and reduce; deferred until profiler tells us it's worth it.

### ✅ Phase 5: HcSinkhorn batched kernel
Done as part of Phase 2's kernel batch.

### 🔲 Phase 6: chunked prefill driver
`forward_prefill(prompt, pos0, last_only)` that chunks at CHUNK_SIZE=64. State carries across chunks. `last_only=true` skips per-token head except for the very last batch element.

## File layout

```
crates/v4flash-kernels/
  kernels/                          # HIP source
    q8_0_matvec.hip                 # + q8_0_gemv_batched_warp8
    q8_0_grouped_matvec.hip         # + q8_0_grouped_gemv_batched
    f16_matvec_narrow.hip           # + f16_matvec_narrow_batched
    rms_norm.hip                    # + rms_norm_weighted_batched
    rms_norm_no_weight.hip          # + rms_norm_no_weight_batched
    hc_weighted_sum.hip             # + hc_weighted_sum_batched
    hc_sinkhorn_par.hip             # + hc_sinkhorn_par_batched
    hc_post.hip                     # + hc_post_batched
    hc_sigmoid_bias.hip             # + hc_sigmoid_bias_batched
  src/
    q8_0.rs                         # + matvec_batched / matvec_grouped_batched
    f16.rs                          # + matvec_narrow_batched
    rms_norm.rs                     # + launch_weighted_batched / launch_batched
    head.rs                         # + launch_batched on all 4 hc structs
    het/
      batch_scratch.rs              # NEW (Phase 1 — to be replaced by Phase 2 v2)
      forward_prefill.rs            # NEW (Phase 1 driver — to add v2 path)
  tests/
    forward_prompt_batch_matches_sequential.rs  # Phase 1 oracle (passing)
    bench_prefill.rs                # Phase 1 bench (passing)
    q8_0_matvec_batched.rs          # Phase 2 unit test (passing)
```

## Gotchas

- **HIP graph captures bake in scratch pointers.** Phase 1 hit this — allocating two separate scratches makes graph replays read freed memory of the first. Workaround: use ONE shared scratch for runs that share captures. Phase 2's `forward_layer_batch_v2` should NOT use captures (direct launches only) since the batched scratch pointers differ from the single-token captures.
- **`set_current_cached` stale** if you call `Device::set_current` directly elsewhere. Workaround: `engine.current_device.store(-1, …)` at start of `forward_prompt_batch`. See forward_prefill.rs.
- **`forward_layer_pair_mode`** is the entry point for batched (disables M30 combined cross-layer graph).
- **Q8_0 weight layout** is M18-repacked: per row = `[blocks×2 scales][blocks×32 quants]` flat, NOT interleaved.

## Bench reference

- Master single-token decode (M31): **27.95 tok/s** sustained (35.78 ms/tok p50)
- Phase 1 prefill at B=7: 12.94 tok/s (77 ms/tok — pair_mode overhead, no batching)
- **Phase 2 v2 measured** (`bench_prefill_v2`) — pre-Phase-3:

  | B  | best tok/s | median tok/s | ms/tok (best) |
  |----|-----------:|-------------:|--------------:|
  |  4 |      37.48 |        36.86 |          26.7 |
  |  7 |      41.22 |        37.25 |          24.3 |
  | 16 |      44.09 |        43.06 |          22.7 |
  | 32 |      44.97 |        43.06 |          22.2 |
  | 64 |      42.97 |        41.69 |          23.3 |

  Plateaued at ~43-45 tok/s (~1.5× single-token). Bottleneck: iGPU MoE
  + attention still in serial per-batch loops.

- **Phase 3 v2 measured** (`bench_prefill_v2`) — iGPU MoE batched, attention still serial:

  | B  | best tok/s | median tok/s | ms/tok (best) | vs Phase 2 |
  |----|-----------:|-------------:|--------------:|-----------:|
  |  7 |      52.41 |        51.90 |          19.1 |       1.27× |
  | 16 |      61.30 |        60.84 |          16.3 |       1.39× |
  | 32 |      63.93 |        62.90 |          15.6 |       1.42× |
  | 64 |      62.15 |        61.55 |          16.1 |       1.44× |

  ~62-64 tok/s = **2.3× single-token**. Plateau from attention still per-batch
  + per-batch wide router matvec.

- **Phase 4 v2 measured** — all stages batched (attention via grid `(n_head, B, 1)`,
  per-token causal `n_raw_per[B]` device buffer):

  | B  | best tok/s | median tok/s | ms/tok (best) | vs Phase 3 |
  |----|-----------:|-------------:|--------------:|-----------:|
  |  7 |      55.77 |        55.75 |          17.9 |       1.06× |
  | 16 |      62.94 |        62.93 |          15.9 |       1.03× |
  | 32 |      66.96 |        66.84 |          14.9 |       1.05× |
  | 64 |      67.68 |        67.65 |          14.8 |       1.09× |

  ~68 tok/s = **2.4× single-token**. Modest gain because per-WG attention
  work is uneven (b=0: 1 KV row; b=63: 64 rows) → wave-quantization tail
  latency dampens the launch-overhead savings.

- Phase 6 target on 200-token prompt: ≥ 150 tok/s with `last_only=true`
