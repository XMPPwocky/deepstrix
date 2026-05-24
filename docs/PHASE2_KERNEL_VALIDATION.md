# Phase 2 — Kernel validation framework

Per-kernel ports validate against the **M2 activation dump** (the canonical ds4-CPU forward trajectory). This doc is the template every future kernel milestone (M3+) follows.

## Where the oracle lives

`reference/v4flash-cpu-activations/` (persistent btrfs; gitignored). Produced by `external/ds4-dump/ds4-dump-activations` against the M1 prompt with the 0002 ds4 patch active. SHAs pinned in `docs/PHASE1_REFERENCE.md`.

- 14,792 tensors, 489 MB on disk
- 51 token positions (7 prompt + 44 generated, before EOS) × 43 layers
- 6 activation tags per (layer, token) plus 2 weight tags per layer
- Bit-deterministic across reruns

## Manifest schema

`manifest.json` at the dump root:

```json
{
  "meta": {
    "model_path": "...",
    "prompt": "DeepSeek-V4 Flash is",
    "n_tokens_arg": 50,
    "backend": "cpu"
  },
  "tensors": [
    {
      "tag": "attn_input_norm",
      "layer": 17,
      "token": 5,        // -1 for weight tensors
      "dtype": "f32",    // f32 | f16 | fp8 | i32
      "shape": [4096],
      "bytes": 16384,
      "path": "L17/T0005/attn_input_norm.bin",
      "is_weight": false
    },
    ...
  ],
  "n_tensors": 14792,
  "n_logit_rows": 51,
  "vocab_size": 129280,
  "prompt_len": 7
}
```

- Binary files are raw native-endian bytes (little-endian on x86-64). No header.
- Weight tensors live under `L<LL>/weight/<tag>.bin` and are emitted once per layer (deduped — the ds4 patch fires the `weight:` hook only at `pos == 0`).
- Per-token activations live under `L<LL>/T<TTTT>/<tag>.bin`.

## Tag catalog

Per layer L=0..42 (regular layers, in `L<LL>/T<TTTT>/`):

| Tag                       | Shape                              | Dtype | When           |
|---------------------------|------------------------------------|-------|----------------|
| `layer_input_residual`    | `[n_hc, n_embd]`                   | f32   | per (L, T)     |
| `attn_cur`                | `[n_embd]`                         | f32   | per (L, T)     |
| `attn_input_norm`         | `[n_embd]`                         | f32   | per (L, T)     |
| `q_a_out`                 | `[n_lora_q=1024]`                  | f32   | per (L, T)     |
| `q_a_normed`              | `[n_lora_q=1024]`                  | f32   | per (L, T)     |
| `q_b_out`                 | `[n_head*n_head_dim=32768]`        | f32   | per (L, T)     |
| `q_head_normed`           | `[32768]`                          | f32   | per (L, T)     |
| `q_post_rope`             | `[32768]`                          | f32   | per (L, T)     |
| `kv_raw_out`              | `[n_head_dim=512]`                 | f32   | per (L, T)     |
| `kv_normed`               | `[512]`                            | f32   | per (L, T)     |
| `kv_post_rope`            | `[512]`                            | f32   | per (L, T)     |
| `kv_cached_row`           | `[512]`                            | f32   | per (L, T)     |
| `attn_heads`              | `[n_head*n_head_dim=32768]`        | f32   | per (L, T)     |
| `attn_heads_inv_rope`     | `[32768]`                          | f32   | per (L, T)     |
| `attn_out_low`            | `[n_groups*rank=8192]`             | f32   | per (L, T)     |
| `attn_out`                | `[n_embd=4096]`                    | f32   | per (L, T)     |
| `comp_kv_row`             | `[n_head_dim=512]`                 | f32   | sparse: ratio==4 every 4 tokens from T=3 |
| `comp_allowed_mask`       | `[n_comp]`                         | i32   | ratio==4, per (L, T) from T=3      |
| `comp_state_kv_row`       | `[comp_width]`                     | f32   | per (L, T) for ratio>0 layers      |
| `comp_state_score_row`    | `[comp_width]`                     | f32   | per (L, T) for ratio>0 layers      |
| `comp_pool_out`           | `[head_dim=512]`                   | f32   | sparse on boundary (ratio>0)       |
| `comp_pre_fp8`            | `[head_dim=512]`                   | f32   | sparse on boundary (ratio>0)       |
| `comp_post_fp8`           | `[head_dim=512]`                   | f32   | sparse on boundary (ratio>0)       |
| `index_comp_*` (×5)       | (indexer dims)                     | f32   | parallel set, ratio==4 only        |
| `indexer_{q,head_weights,scores}` (×4) | (indexer dims)         | f32   | per (L, T) on ratio==4 layers; only when n_comp > top_k (never on 57-tok prompt) |
| `ffn_cur`                 | `[n_embd]`                         | f32   | per (L, T)     |
| `ffn_input_norm`          | `[n_embd]`                         | f32   | per (L, T)     |
| `layer_output_residual`   | `[n_hc, n_embd]`                   | f32   | per (L, T)     |
| `weight:attn_norm`        | `[n_embd]`                         | f32   | per L (pos=0)  |
| `weight:ffn_norm`         | `[n_embd]`                         | f32   | per L (pos=0)  |
| `weight:q_a_norm`         | `[1024]`                           | f32   | per L (pos=0)  |
| `weight:kv_a_norm`        | `[512]`                            | f32   | per L (pos=0)  |
| `weight:rope_params`      | `[6]`                              | f32   | per L (pos=0)  |
| `weight:attn_sinks`       | `[n_head=64]`                      | f32   | per L (pos=0)  |

`weight:rope_params` packs `[freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow]` — `beta_fast=32`, `beta_slow=1` are hardcoded DeepSeek-V4 defaults in the 0004 patch (ds4 does not parameterise these). `n_ctx_orig` is `DS4_ROPE_ORIG_CTX=65536` when `ext_factor != 0`, 0 otherwise; the dump does not carry it (it's layer-invariant and only consulted in the compressed-layer branch).

Synthetic L=43 (head bucket — added by 0003 patch, in `L43/T<TTTT>/` and `L43/weight/`):

| Tag                       | Shape                | Dtype | When           |
|---------------------------|----------------------|-------|----------------|
| `output_flat`             | `[n_hc, n_embd]`     | f32   | per T          |
| `output_pre`              | `[n_hc]`             | f32   | per T          |
| `output_hc_weights`       | `[n_hc]`             | f32   | per T          |
| `output_embd`             | `[n_embd]`           | f32   | per T          |
| `output_norm`             | `[n_embd]`           | f32   | per T          |
| `weight:output_norm`      | `[n_embd]`           | f32   | pos=0          |

**Logit-row ↔ output_norm-token mapping** (for Q8_0 oracle):
- `logits.f32` row k (0..50) was produced from `L43/T<6+k>/output_norm.bin`.
- Prefill consumes positions T0..T6 (no logits written); decode positions T7..T56 each produce one logit row matching the eval after that position. Row 0 = "predict the next token after seeing the prompt" = output of the eval at T6.

Future milestones add hook sites for Q/KV/RoPE (inside `layer_q_projection_with_lora_*`), attention (inside `layer_attention_*`), MoE (inside `layer_routed_moe_*` and `layer_shared_ffn_*`), mHC (`hc_*` helpers). See `docs/DESIGN.md` and `ds4.c:7438` for the canonical operations.

## Per-kernel test pattern

Each kernel ported lives at `crates/v4flash-kernels/{src,kernels,tests}/<name>.{rs,hip,rs}`.

The standard test shape:

1. `ActivationDump::open(reference/v4flash-cpu-activations)`
2. Pick a HIP device (prefer `gfx1151` iGPU for V4 Flash — UMA is the production target)
3. Load the kernel via `<KernelName>::for_arch(gcn_arch_name)`
4. For each `(layer, token)` position in the dump:
   - Load input tensors via `dump.tensor(input_tag, L, T)` and `dump.read_f32(...)`
   - Load weight tensor via `dump.weight(weight_tag, L)` (deduped lookup, layer-only)
   - Load expected output via `dump.tensor(output_tag, L, T)`
   - Upload to `DeviceBuffer<f32>`, launch kernel, sync, download
   - Compare to expected, track max_abs_diff
5. Assert `overall.max_abs_diff < THRESHOLD`

See `crates/v4flash-kernels/tests/rms_norm.rs` for the template.

## Thresholds

`max_abs_diff` between our HIP kernel output and the ds4-CPU reference. Float32 arithmetic, so a few ULPs (~1e-6) are expected; the budget per kernel depends on accumulation depth and order sensitivity.

| Kernel                        | Threshold | Measured (gfx1151) | Extra gate            | Notes                                                            |
|-------------------------------|----------:|-------------------:|-----------------------|------------------------------------------------------------------|
| `rms_norm_weighted`           |    1.0e-4 |             3.8e-6 | —                     | n=4096; double partials, f32 final scale                         |
| `q8_0_matvec` (output head)   |    5.0e-3 |             3.8e-5 | argmax 51/51          | k=4096, n_rows=129280; ds4-faithful Q8_0                         |
| `rms_norm_no_weight`          |    1.0e-4 |             3.8e-6 | —                     | n_rows=64, n=512; reused by M10 for output_flat (n_rows=1, n=16384) |
| `rope_tail` (KV stripe)       |    5.0e-5 |             6.9e-6 | —                     | n_head=1, head_dim=512, n_rot=64; YaRN ramp on 41/43 layers      |
| `rope_tail` (Q stripe)        |    5.0e-5 |             2.4e-5 | —                     | n_head=64, head_dim=512, n_rot=64; same kernel, larger fan-out   |
| `q_lora_chain` (M4 chain)     |    5.0e-2 |             1.7e-2 | mean<1e-4             | matvec→rms→matvec→rms_nw→rope; rms amplifies spiky matvec noise  |
| `kv_chain` (M4 chain)         |    5.0e-3 |             1.7e-3 | mean<1e-5             | matvec→rms→rope; same amplification mechanism on spiky kv_raw    |
| `attention_swa`               |    1.0e-3 |             2.9e-6 | —                     | sink-aware softmax + axpy; L=0,1 only (pure SWA); f32-ULP floor  |
| `q8_0_grouped_matvec`         |    2.0e-2 |             1.2e-2 | mean<1e-4             | attn_output_a; n_groups=8 × group_dim=4096 → 8 × rank=1024       |
| `swa_attention_chain` (M5)    |    1.0e-1 |             2.2e-2 | mean<1e-3             | attention_swa → inv_rope → grouped_q8_0 → q8_0; L=0,1 only       |
| `attention_mixed` (L=0,1)     |    1.0e-5 |             2.9e-6 | —                     | regression vs attention_swa; n_comp=0, mask=None                 |
| `attention_mixed` (ratio==128)|    1.0e-5 |             6.2e-6 | —                     | 20 layers, n_comp=0 in 57-tok prompt (compressor never fires)    |
| `attention_mixed` (ratio==4)  |    5.0e-4 |             7.9e-6 | —                     | 21 layers, n_comp grows to 14, dumped i32 mask consumed          |
| `mixed_attention_chain` (M6)  |    1.0e-1 |             3.4e-2 | mean<1e-3             | attention_mixed → inv_rope → grouped_q8_0 → q8_0; L=2,4,…,42     |
| `f16_matvec` (M7)             |    1.0e-3 |             1.2e-5 | mean<1e-5             | F16 weight × F32 input → F32, warp-per-row                       |
| `compressor_pool` (M7)        |    1.0e-4 |             9.5e-7 | mean<1e-6             | per-output-dim softmax-weighted avg; ratio==4 split-layout         |
| `fp8_e4m3fn_quantize` (M7)    |    1.0e-5 |               0    | bit-exact             | E4M3FN lookup-based encode/decode; matches CPU bit-for-bit       |
| `compressor_end_to_end` (M7)  |    2.0e-1 |             1.0e-6 | mean<1e-5             | matvec×2→APE→pool→rms→rope→fp8 vs comp_post_fp8; single FP8 bucket flip dominates max |
| `indexer_compressor` (M7)     |    1.0e-3 |             1.3e-5 | mean<1e-5             | same as compressor but head_dim=128, no FP8 step                 |
| `indexer_score` (M7)          |    1.0e-3 |             4.2e-7 | bit-exact (CPU x-val) | synthetic-input self-consistency; ds4 dump path unreachable in M1 prompt |
| `indexer_pipeline` (M7)       |     0     |                  0 | exact bool match      | early-permit branch (n_comp ≤ top_k) returns all-1s              |
| `hca_chain` (M7 close-loop)   |    1.0e-1 |             3.4e-2 | mean<1e-3             | our compressor+indexer+M6 mixed_attn+M5 outproj; attn_out vs ds4 |
| `router_hash_gate` (M8)       |    5.0e-5 |             5.8e-7 | exact selection       | L=0,1,2 hash-gate; tid2eid lookup + softplus-sqrt + normalise   |
| `router_learned` (M8)         |    5.0e-5 |             1.1e-5 | exact selection       | L=3..42 learned router; matvec + softplus + sqrt + topk + normalise |
| (future) `iq2_xxs_matmul`     |     TBD   |                TBD | —                     |                                                                  |

For Q8_0 specifically, the *argmax-match* check on every logit row is the production correctness gate — FP threshold backs it up but the discrete check is what guarantees "we pick the same greedy token as ds4 in every position."

For the M4 chain oracles, the **mean_abs** is the regression signal, not max_abs. The chain composes 3 (KV) or 5 (Q) kernels; max_abs is dominated by single elements at "spiky" (L, T) positions where a 15σ outlier in the matvec output gets amplified ~10–17x by the subsequent rms_norm (which divides by RMS — small RMS = large 1/RMS = large scale on any noise). The mean_abs stays at f32-ULP (~1e-5 / ~1e-6) across all 1.1–71.9 million element comparisons, confirming the bulk of every output matches ds4 bit-for-bit. The per-stage diagnostics in `tests/attention_setup_chain.rs` log exactly where the amplification kicks in; downstream attention's softmax tolerates ~1% noise on Q/KV values that are O(1)–O(10) in magnitude.

`rms_norm_no_weight`'s standalone test exercises the per-head case (n_rows=64, n=512). The same kernel handles `output_flat` in M10 (n_rows=1, n=16384) — the per-row strided loop has no n cap beyond what fits in 256-thread shared-memory reductions.

`rope_tail`'s inverse path (one extra kernel arg flipping `sin_sign`) is now exercised by M5's attention chain (inverse RoPE before the grouped output projection).

For M5: `attention_swa` is the validated SWA-only softmax — it sits at f32-ULP because the value-side axpy order matches CPU (parallelised across `head_dim`, not across rows), and the dot-product reduction's tree-vs-sequential order difference is microscopic post-softmax. Coverage is L=0, L=1 only (ratio==0 layers in V4 Flash); M6 extends the same softmax body to L=2..42 by adding compressed-KV rows in `layer_attention_mixed_one`. `kv_cached_row` represents the f16-roundtripped post-FP8 KV that the cache stores — the Rust test accumulates rows across tokens to reconstruct the cache without needing an FP8 implementation.

For M7: the CSA producer loop is closed. Our compressor (composing F16 matvec×2 + APE add + state machine + pool + RMS norm + RoPE + FP8 quantize + F16 roundtrip) reproduces ds4's `comp_kv_row` and `index_comp_kv` end-to-end. The indexer pipeline (matvec + RoPE + projection + scoring + top-K) produces `comp_allowed_mask` matching ds4 exactly for the early-permit branch (n_comp ≤ top_k); the scoring kernel is cross-validated against a CPU reimplementation on synthetic inputs since our 57-token M1 prompt never reaches the n_comp > 512 regime where ds4 emits `indexer_scores`. The full HCA chain feeds our compressor + indexer outputs into M6's mixed attention + M5's output projection and produces `attn_out` matching ds4 at the same 3.4e-2 max-abs as M6's chain (which used dumped comp_kv_row directly) — meaning the compressor's FP8 noise doesn't compound through the attention chain. With M7 done, every operation in V4 Flash's attention path (M4 setup + M5 SWA + M6 mixed softmax + M7 CSA producers) is our own kernel, validated.

For M6: `attention_mixed` is the generalised softmax that subsumes the SWA case. When `n_comp == 0` and `mask == None`, it reduces bit-for-bit to `attention_swa` (verified in block 1 of the oracle: max_abs=2.86e-6 matches M5). The masked-comp path uses `expf(-INFINITY - max_score) = 0` to drop masked rows from both denom and axpy — no explicit branching in phase 3/4. M6's scoping is **consumer-first**: `comp_kv_row` and `comp_allowed_mask` are dumped from ds4 and consumed directly by the kernel test, so the mixed_one softmax is validated standalone before the *producers* (compressor + indexer) land in M7. After M7, the same chain test runs end-to-end with our own kernel-produced comp_kv + comp_allowed, closing the loop.

Rationale per threshold should land in each kernel's test docstring.

## When a threshold fails

Don't relax the threshold unless you understand *why* it's being exceeded. Default investigation flow:

1. Is the kernel using a different *algorithm* than ds4's CPU reference, or just a different *FP reduction order*? Reading ds4.c is the answer (the function definitions for `rms_norm_weight`, `matvec_*`, etc. are short).
2. Is our kernel using f32 where ds4 uses double for the accumulation? (RMSNorm: ds4 accumulates `ss` in double — our kernel matches with double per-thread partials.)
3. Is the reduction order wildly different (e.g. tree vs sequential)? Mostly fine for sums; for softmax/log it matters more.
4. Is the bug in our port's launch config? (e.g. wrong block size relative to reduction layout.)

If a real algorithmic divergence shows up, fix the kernel — don't raise the threshold.

## Re-baselining

If `external/ds4` is updated (e.g. antirez merges fixes) such that CPU outputs change:

1. `git submodule update --init`
2. `external/apply-patches.sh` (reapplies 0001 + 0002)
3. Rerun `ds4-dump-activations` against the canonical prompt
4. Verify SHAs differ deliberately; update `docs/PHASE1_REFERENCE.md` with new SHAs and a one-line note on what changed upstream
5. Reruns of every kernel test should still pass within tolerance — if any *now* fail with a wider divergence, that's the moment to investigate.

Don't silently re-baseline. Every M2-dump SHA change is a deliberate event.

## Notes for future-us

- The ds4 patch fires hooks only on the **single-token CPU forward path** (`forward_token_raw_swa_cpu_decode_scratch`). To stay on this path the dumper bypasses `ds4_session_sync` and uses `ds4_session_eval` for every prompt token too — this is the source of the per-token-vs-batched-prefill trajectory divergence noted in `PHASE1_REFERENCE.md`.
- Per-arch hsaco blobs (`gfx1201`, `gfx1151`, `gfx1100`) all compile via `crates/v4flash-kernels/build.rs`. Only `gfx1201` (dGPU) and `gfx1151` (iGPU) are wired into the Rust dispatch in `rms_norm.rs`; gfx1100 builds for future portability but isn't selected at runtime on this hardware.
- ds4 itself rebuilds whenever its sources change; the dump_activations binary depends on patched `ds4.o` so a re-patched ds4 means re-running the dump capture for full consistency.
