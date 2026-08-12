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

## Perf-lever pass (2026-08-10, post-A/B)

- Q4_K head + Q6_K shexp-down decode switched to dp4a GEMM@B=1 (Q8_K
  activations): decode @4K 28.54 → 28.69 tok/s (−0.7% vs antirez =
  parity). Q5_K stays on the f32 gemv (measured parity).
- Prefill gap attributed: ~2% = K-quant dp4a GEMM vs Q8 WMMA
  (0.70 vs 0.37 ms @B=512 per shexp matmul); the rest is IQ3_XXS's
  1.167× down-projection bytes — the quality trade itself.
- **Server @ K=8, 192K ctx works with unsloth** (antirez caps at K=6):
  the freed 1.24 GB dGPU converts to +2 hot experts/layer.
- fp8 main-KV DEFERRED by explicit decision: the plan's 4.5 GB saving
  estimate was wrong (KV is compressed; real total @192K ≈ 1.4 GB, fp8
  saves ~0.6–1.2 GB) and the latency prior is against it
  (FP8_KV_REVIVE). Revisit as capacity play only.

## Follow-ups

- WMMA-ize the K-quant dense GEMM (recover ~2% prefill); iq2s/iq3
  prefill variant tuning.
- fp8/fp4 indexer K cache (small, ~0.4 GB incl. gather scratch). If fp8
  main KV is ever revisited, the pinned layout is vLLM `fp8_ds_mla` V4
  semantics: 448 B NoPE e4m3 + 64×f16 RoPE + 7×ue8m0 per-64 scales
  (amax/448, exp2-ceil), row stride padded to 592/608.
- MTP acceptance re-measure (optional); franken-mix exploration.

## M63 iGPU hot-expert de-dup (2026-08-12)

The iGPU held all 256 experts/layer while M56 also mirrored the K hottest
onto the dGPU, and the iGPU MoE kernels *skip* exactly those slots. The
duplicate bytes were live in one case only: >`DGPU_HOT_CAP` (4) of a
token's 6 picks resident, whose overflow fell back to the iGPU at the raw
expert id. Pinning the cap at `N_EXPERT_USED` removes that branch, so the
copies can go. `IGPU_DEDUP_HOT=1` (default in `run_deepstrix.sh`).

Encoding: remap's miss branch carries the iGPU slot as `-(slot+1)`;
kernels decode `-remap[e]-1`. Without de-dup `slot == id`, so the default
path is unchanged. Residency predicate (`remap[e] >= 0`) is untouched.

Measured (unsloth UD-IQ2_XXS, K=8, server placement):

| | off | on |
|---|---|---|
| iGPU GTT | 78.27 | **75.91 GiB (−2.36)** |
| dGPU VRAM | unchanged | unchanged |
| decode @4K | 27.36 | 27.34 tok/s |
| prefill @4K | 433.0 | 432.7 tok/s |

`forward_per_layer_vs_ds4` (43 layers), `head_to_logits` and
`forward_prompt_batch_v2` are **bit-identical** to a cap-6 non-de-dup
baseline — the device partition is the same, only the weight index moved.

## Corrections to the claims above (2026-08-12)

- **"full oracle suite green" was overstated.** With `DEEPSTRIX_GGUF` +
  `DEEPSTRIX_DUMP_DIR` and no placement file (which is what `cargo test`
  gives you — `hot_expert_file_path()` is CWD-relative and does not
  resolve from the crate root), `forward_prompt_batch_v2_matches_sequential`
  FAILS on unsloth at 7.8475e-2 vs a 5e-2 bound. Verified **pre-existing**:
  the pre-M63 commit produces the identical value. antirez passes at
  3.72e-2; unsloth with the server's placement passes at 2.18e-2.
- **`reference/decode_hot_experts.txt` is stale** (Jun 10, pre-0731, and
  derived from antirez's quant). It is also the CWD-relative default, so
  anything without an explicit placement silently gets a mismatched one.
  At K=8 it makes cap=4 fail that oracle at 1.79e-1 while cap=6 passes at
  4.71e-2; the server's own derived placement is insensitive to the cap.
- **The 16/17 vector-test figure is not a quality metric.** Its
  `short_code_completion` case measures formatting style: step 0 (a ```
  fence) misses on every quant we have, and step 3 penalises answering the
  question as asked. With that case disabled, UD-IQ2_XXS and UD-Q2_K_XL
  both score **13/13**. Do not use this harness to choose between quants.

## UD-Q2_K_XL (default since 2026-08-12)

Same 0731 checkpoint. Differs from UD-IQ2_XXS in exactly two roles:
gate/up experts IQ2_XXS→IQ2_XS ×42 (IQ2_S→IQ3_XXS on blk.26), and
token_embd Q4_K→Q5_K. `ffn_down_exps` is byte-identical. Contains no Q2_K
tensor — the name is an unsloth size tier, not a format.

New kernels: `iq2_xs_pair_matvec` (IQ2_XS = XXS's 7-bit ksigns codebook +
IQ2_S's nibble scales over a 512-entry grid) and `iq3_xxs_pair_matvec`
(IQ3_XXS at gate/up needs the fused-SwiGLU pair form; the in-tree IQ3_XXS
family is down-projection only). CPU reference pinned against llama.cpp's
own `ggml_vec_dot_iq2_xs_q8_K_generic` via `tests/ref/iq2_xs_gen.c`.

| | UD-IQ2_XXS | UD-Q2_K_XL |
|---|---|---|
| size | 84.6 | 90.2 GiB |
| iGPU GTT (K=8, de-dup) | 78.66 | **81.64 GiB** |
| dGPU VRAM @192K | ~14.97 | 15.08 / 15.92 GiB |
| decode @4K | 27.34 | 27.01 tok/s |
| vector-test | 13/13 | 13/13 |

**The quality case is UNMEASURED, not demonstrated.** Every instrument we
have scores the two files identically. The theory (gate/up is 44.6 of
84.6 GiB and goes 2.06→2.31 bpw) is untested; settling it needs
teacher-forced top-1/top-5/KL against a reference dump, which needs the
ds4 dumper taught IQ2_XS. Costs are certain: +3.0 GiB iGPU net of de-dup,
−1.2% decode.
