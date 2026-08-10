# unsloth DeepSeek-V4-Flash-0731 UD-IQ2_XXS support

Branch `unsloth-ud-iq2xxs`, 2026-08-09/10. Second quant mix of the same 0731
checkpoint, runnable via `--gguf` (NO default switch — explicit decision).
Plan: `~/.claude/plans/make-a-plan-remember-piped-kite.md`.

## Files

- Model: `/persist/lumi/models/ds4f-unsloth/UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-0000{1,2,3}-of-00003.gguf`
  (84.6 GiB, 3-shard; pass ANY shard path — the loader derives siblings).
  sha256 verified vs HF LFS oids.
- Reference dump: `reference/v4flash-cpu-activations-unsloth/` (77880 tensors),
  produced by the patched ds4 dumper (patches 0010/0011) from the dumper
  variant GGUF `/home/claude-code/models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-dumper-variant.gguf`
  (dense roles pre-dequanted F32/F16; pending `sudo mv` into /persist/lumi/models/ds4f-unsloth/).

## Quant-mix diff vs antirez (identical 1328 tensors; `gguf-diff` reproduces)

down_exps Q2_K→IQ3_XXS×41+MXFP4×2 (blk.26,42); gate/up +IQ2_S (blk.26);
shexp Q8→Q5_K/Q6_K; q_a Q8→Q5_K; head+embd→Q4_K; compressor/indexer
F16→Q8_0 (converted back to F16 at load); router BF16→F16 at load.
blk.26 is the sensitive layer (all exceptions). MXFP4 is the official
checkpoint's native expert format — those two layers are plausibly lossless.

## Validation (all green, 2026-08-10)

- Weight contract: both files validate clean; wrong mixes fail with a full
  enumerated list (`weight_contract_models`).
- Kernel oracles vs scalar CPU references: IQ3_XXS/MXFP4/IQ2_S/Q5_K
  (gemv+GEMM) all ≤ f32-roundoff.
- ds4 dumper regression: patched build reproduces the antirez dump
  BIT-FOR-BIT; unsloth dump generated from the variant GGUF.
- Engine vs unsloth dump: forward_l0_t0_stages, forward_per_layer_vs_ds4
  (43 layers), head_to_logits (argmax 51/51), routed_moe, shared_expert,
  q_lora, mhc/head chains, compressors, indexer, routers — green with
  ZERO tolerance changes. Antirez direction re-verified.
- `deepstrix-vector-test` vs official API vectors: antirez 16/17 (94.1%),
  unsloth 16/17 (94.1%) — parity.

Env overrides for the suite: `DEEPSTRIX_GGUF=<shard1>` +
`DEEPSTRIX_DUMP_DIR=$PWD/reference/v4flash-cpu-activations-unsloth`.

## Perf A/B (back-to-back, K=6 het-split, PIPELINE_LANES=2)

| metric | antirez | unsloth | Δ |
|---|---|---|---|
| decode @4K | 28.85 | 28.54 tok/s | −1.1% |
| decode @64K | 27.88 | 27.55 tok/s | −1.2% |
| prefill @4K | 504.6 | 457.2 tok/s | −9.4% |
| prefill @32K | 472.1 | 431.9 tok/s | −8.5% |
| dGPU alloc | 8791 | 7549 MB | **−1.24 GB** |
| iGPU alloc | 77913 | 83869 MB | +5.96 GB (fits) |

Analysis: decode parity — the predicted −535 MB/token BW win is absorbed by
the correctness-first scalar-f32 K-quant dense gemv kernels (q5/q6/q4
dense); a dp4a Q5_K×Q8 formulation is the standing lever. Prefill −9%:
IQ3_XXS down reads 1.167× bytes (kernel itself is BW-bound, ratio 1.088 in
isolation) + K-quant dp4a dense GEMM vs Q8 WMMA; per-kernel rocprofv3
attribution still pending. dGPU freed 1.24 GB → K=8@192K plausibly fits
with unsloth even before fp8 KV.

## Running it

```
GGUF=/persist/lumi/models/ds4f-unsloth/UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf \
  ~/run_deepstrix.sh --bg
```
Per-model snapshot dirs mean first launch starts a cold KV-snapshot cache.
Hot-expert placement self-regenerates from the stats sidecar (fingerprint
differs → fresh stats).

## Follow-ups

- dp4a K-quant dense gemv (decode); prefill per-kernel trace + tuning.
- Phase 4: fp8 e4m3 main KV (vLLM `fp8_ds_mla` semantics: 448B NoPE e4m3 +
  64×f16 RoPE + inline block-64 scales, stride padded to 592/608) + fp8/fp4
  indexer K cache. Capacity win guaranteed; latency win is hypothesis
  (FP8_KV_REVIVE prior).
- MTP acceptance re-measure (optional); franken-mix exploration.
