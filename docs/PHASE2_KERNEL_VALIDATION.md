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

| Tag                       | Shape                | Dtype | When           |
|---------------------------|----------------------|-------|----------------|
| `layer_input_residual`    | `[n_hc, n_embd]`     | f32   | per (L, T)     |
| `attn_cur`                | `[n_embd]`           | f32   | per (L, T)     |
| `attn_input_norm`         | `[n_embd]`           | f32   | per (L, T)     |
| `ffn_cur`                 | `[n_embd]`           | f32   | per (L, T)     |
| `ffn_input_norm`          | `[n_embd]`           | f32   | per (L, T)     |
| `layer_output_residual`   | `[n_hc, n_embd]`     | f32   | per (L, T)     |
| `weight:attn_norm`        | `[n_embd]`           | f32   | per L (pos=0)  |
| `weight:ffn_norm`         | `[n_embd]`           | f32   | per L (pos=0)  |

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

| Kernel                        | Threshold | Measured (gfx1151) | Extra gate            | Notes                                       |
|-------------------------------|----------:|-------------------:|-----------------------|---------------------------------------------|
| `rms_norm_weighted`           |    1.0e-4 |             3.8e-6 | —                     | n=4096; double partials, f32 final scale    |
| `q8_0_matvec` (output head)   |    5.0e-3 |             3.8e-5 | argmax 51/51          | k=4096, n_rows=129280; ds4-faithful Q8_0    |
| (future) `rope_tail`          |     TBD   |                TBD | —                     |                                             |
| (future) `iq2_xxs_matmul`     |     TBD   |                TBD | —                     |                                             |
| (future) `attention_*`        |     TBD   |                TBD | softmax-stable        | Softmax tightens the budget vs raw matmul   |

For Q8_0 specifically, the *argmax-match* check on every logit row is the production correctness gate — FP threshold backs it up but the discrete check is what guarantees "we pick the same greedy token as ds4 in every position."

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
