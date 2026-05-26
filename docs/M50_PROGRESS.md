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

### ⏳ Phase 2: dGPU batched kernels DONE, full v2 integration FAILING ORACLE

`forward_prompt_batch_v2` + `forward_layer_batch_v2` in `src/het/forward_prefill.rs` — committed but currently fails its oracle by ~8.9e3 (outputs ~10× smaller than expected). See `memory/project_m50_v2_failing_oracle.md` for bisect strategy + suspect list. 8 of 10 batched kernels are unverified; bug is in at least one of them or in v2's orchestration.

**Building blocks** (all committed):

**All batched kernels written and building cleanly:**

| kernel | hip file | rust wrapper | tested? |
|---|---|---|---|
| `q8_0_gemv_batched_warp8` | `kernels/q8_0_matvec.hip` | `Q8_0Matvec::matvec_batched` | ✓ B=4 bit-identical |
| `q8_0_grouped_gemv_batched` | `kernels/q8_0_grouped_matvec.hip` | `Q8_0GroupedMatvec::matvec_grouped_batched` | no |
| `f16_matvec_narrow_batched` | `kernels/f16_matvec_narrow.hip` | `F16Matvec::matvec_narrow_batched` | no |
| `q8_k_quantize` (trivial) | existing `kernels/q8_k_quantize.hip` | use `launch(n_blocks = B × per_token)` | trivially works |
| `rms_norm_weighted_batched` | `kernels/rms_norm.hip` | `RmsNorm::launch_weighted_batched` | no |
| `rms_norm_no_weight_batched` | `kernels/rms_norm_no_weight.hip` | `RmsNormNoWeight::launch_batched` | no |
| `hc_weighted_sum_batched` | `kernels/hc_weighted_sum.hip` | `HcWeightedSum::launch_batched` | no |
| `hc_sinkhorn_par_batched` | `kernels/hc_sinkhorn_par.hip` | `HcSinkhorn::launch_batched` | no |
| `hc_post_batched` | `kernels/hc_post.hip` | `HcPost::launch_batched` | no |
| `hc_sigmoid_bias_batched` | `kernels/hc_sigmoid_bias.hip` | `HcSigmoidBias::launch_batched` | no |

All v0 designs: `grid.z = B` parallel WGs, no W-amortization across batch (each WG re-reads weights independently). A v1 pass per kernel can pack multiple batch elements per WG to amortize W reads — that's the next-level perf optimization, but v0 already enables dGPU-side WG concurrency which should give 1.5-3× wall improvement.

**Remaining for Phase 2:**
1. **Unit tests** for the un-tested kernels (mirror the q8_0 test pattern).
2. **`split_unpack_batched` helper kernel** to gather `post[B, n_hc]` and `comb[B, n_hc, n_hc]` out of `split[B, 2*n_hc + n_hc²]`. Lets `HcPost::launch_batched` consume sinkhorn output.
3. **`BatchDgpuScratch`** struct in `src/het/batch_scratch.rs` — replace the current `shared_dgpu + per_token_residual` hack with a single struct of B-extended contiguous buffers (~26 MB at B=64).
4. **`forward_layer_batch_v2`** in `src/het/forward_prefill.rs` — ~500 LoC stage-by-stage rewrite, using batched kernels for stateless work, serial inner loop for stateful (kv_append, compressor). Stage-by-stage cheat sheet in `memory/project_m50_prefill_state.md`.
5. **Validate** via `forward_prompt_batch_matches_sequential` oracle at B=4, 8 (tolerance ~1e-3 — float reduction order will differ from single-token).
6. **Bench**: expect 70-100 tok/s at B=64 with dGPU batched + iGPU still serial.

### 🔲 Phase 3: iGPU MoE batching
Extend `iq2_xxs_pair_matvec_fused_swiglu_batch` and `q2_k_matvec_par_batched` to grid.z = B, with per-batch routing buffers. Same v0 pattern as Phase 2. Expect: routed MoE wall scales 4-8× at B=64.

### 🔲 Phase 4: per-token causal attention
`attn_swa_batched` and `attn_mixed_batched` with per-batch `n_kv[b]` array. Each token attends to a different KV prefix (causal). KV cache is shared.

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
- Phase 2 target at B=64: 70-100 tok/s (dGPU batched, iGPU serial)
- Phase 3 target at B=64: 150-250 tok/s (full batched)
- Phase 6 target on 200-token prompt: ≥ 150 tok/s with `last_only=true`
