# Phase 1 reference output

This is the immutable baseline. Every Phase 1 milestone (M2+) validates ported kernels against the captured logits here. Re-baselining means we acknowledge a numerical drift in the upstream; **do not silently regenerate this file**.

## Status

- [x] M1.1 ds4 rocm submodule + nix build wired up
- [x] M1.2 ds4_cli smoke test
- [x] M1.3 phase1 crate skeleton
- [x] M1.4 first model load (CPU; GPU declared out of scope, see PHASE1_DS4_ISSUES.md)
- [x] M1.5 ds4-dump-logits wrapper built
- [x] M1.6 capture reference logits (CPU bit-identical x2)
- [x] M1.7 this doc
- [x] M1.8 PHASE1_DS4_ISSUES.md populated

## Hardware + toolchain

- Machine: NixOS + ROCm 7.2.3 (via nix flake, see `flake.nix`)
- Reference rig:
  - iGPU: AMD Radeon 8060S Graphics (Strix Halo, gfx1151) — HIP device 1
  - dGPU: AMD Radeon RX 9070 XT (gfx1201) — HIP device 0 (NOT used for inference; 16 GiB VRAM is too small for 86 GiB model)
- ds4 submodule: `external/ds4` @ branch `rocm` (initial: 7a751eb)
- ds4 patches applied: `external/ds4-patches/{0001-expose-logits-buffer, 0002-activation-dump-callback, 0003-output-head-hooks, 0004-q-kv-rope-hooks, 0005-attention-hooks, 0006-comp-mask-hooks, 0007-compressor-indexer-hooks, 0008-moe-router-hooks}.patch`

## Model

- Source: `antirez/deepseek-v4-gguf` (HuggingFace)
- Filesystem path: `/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- Model SHA256: `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`
- File size: 86,720,111,488 bytes (80.76 GiB on-disk; doc-equivalent "86 GiB" in decimal-GB)
- Architecture: `deepseek4` (43 blocks, 256 experts, top-6 used, vocab 129280)

## Reference prompt

- Prompt text: `DeepSeek-V4 Flash is`
- N generated tokens: 50
- Sampling: greedy (argmax via `ds4_session_argmax`)
- System prompt: NONE (`ds4-dump-logits` uses `ds4_tokenize_text` which is plain BPE, no chat-template wrapping)

## Reference outputs

### CPU backend (authoritative correctness baseline)

- Backend: `DS4_BACKEND_CPU`
- prompt_tokens (7): `[53091, 4374, 1465, 13582, 22, 32958, 344]`
- generated_tokens (50): `[260, 1017, 9353, 294, 8281, 9924, 5363, 14, 778, 477, 6558, 304, 3052, 13058, 305, 850, 8281, 34788, 362, 270, 22651, 4374, 1465, 13582, 22, 4923, 16, 983, 344, 260, 11367, 3051, 2645, 396, 14449, 22805, 14, 4105, 14134, 14, 305, 5665, 107718, 22213, 16, 983, 344, 554, 260, 103345]`
- vocab_size: 129280
- n_logit_rows: 51 (prefill + 50 generated)
- logits.f32 bytes: 26,365,440 (51 × 129280 × 4)
- logits.f32 SHA256 (both runs identical): `d294856e3a732d15ba5585d1a52320820428ff846a88848c7888173c416fcd2a`
- Wall time per 50-token run: ~155 s
- Storage:
  - `reference/v4flash-cpu-run1/{tokens.json,logits.f32}`
  - `reference/v4flash-cpu-run2/{tokens.json,logits.f32}` (determinism witness; bit-identical)

### GPU backend (iGPU, gfx1151)

- **Out of scope for M1.** ds4's GPU path is structurally incompatible with V4 Flash on this hardware (see `docs/PHASE1_DS4_ISSUES.md`). Phase 2+ replaces ds4's GPU path with our own kernels and allocator anyway, so a working ds4 GPU baseline was never load-bearing. CPU reference above is authoritative.

### Activation dump reference (M2 oracle for kernel ports)

Captured by `external/ds4-dump/ds4-dump-activations` using the 0002 patch (`ds4_set_activation_dump` callback). This is the oracle every Phase 2 kernel port validates against.

- Backend: `DS4_BACKEND_CPU` — same as the M1 reference above
- **But a different evaluation path**: this dumper feeds prompt tokens one at a time via `ds4_session_eval`, not via `ds4_session_sync`'s batched prefill. Both paths are valid ds4 outputs and both are deterministic, but they accumulate floating-point reductions in slightly different orders, which is enough to flip the greedy argmax at one low-confidence transition (position 5 of the generated sequence). Downstream tokens then diverge.
  - M1 reference (batched prefill): `[..., 8281, 9924, 5363, 14, 778, ...]`
  - M2 activation dump (per-token):  `[..., 8281, 10192, 39, 940, 15890, ...]`
  - Both are bit-deterministic across reruns of their respective paths.
- The M2 dump becomes the canonical reference for kernel ports because every intermediate tensor is captured along *one consistent* trajectory.
- Storage: `reference/v4flash-cpu-activations/` (gitignored, persistent btrfs)
  - **M7 (current) dump:**
    - `manifest.json` SHA256: `255b917f93ffbbe7bef2940566b1207f6521ea30fdce7076e97c0a1589175bc8`
    - logits.f32 SHA256: `bd3176ea0644067caf7a455db332874e62c23838963f561cac2d2b4b59a2ea0a` (unchanged from M2-M6)
    - aggregate SHA: `e675e77326c3efd230a83003af72972e1aed61ea7c0e6db4691350eba20dcddb`
    - 57,373 tensors (M6 + 5 compressor intermediates × 41 layers + 5 indexer-compressor intermediates × 21 layers + 4 indexer-scoring tags — last group never fires on this prompt since n_comp ≤ 14 < top_k=512)
  - Previous (M6) dump: manifest `0b10242d…`, aggregate `fb740e8f…`
  - Previous SHAs preserved for archaeology:
    - M3: manifest `a2e8b85b...`, aggregate `d033b20460bc...`
    - M4: manifest `df0e4f36...`, aggregate `95cd30f2...`
    - M5: manifest `4a825d66...`, aggregate `5fb7896c...`
    - All bit-identical for prior tag subsets (spot-checks confirm).
- Tag set per layer L=0..42 (regular layers), per token position:
  - `layer_input_residual` (n_hc × n_embd = 4 × 4096 f32)
  - `attn_cur` (n_embd f32)
  - `attn_input_norm` (n_embd f32) — output of layer→attn RMSNorm
  - **M4 attention setup (added by 0004 patch):**
    - `q_a_out` (n_lora_q = 1024 f32) — post `matvec_q8_0(q_a)`
    - `q_a_normed` (n_lora_q = 1024 f32) — post `rms_norm_weight` with `q_a_norm`; the Q LoRA reduction
    - `q_b_out` (n_head × n_head_dim = 64 × 512 = 32768 f32) — post `matvec_q8_0(q_b)`
    - `q_head_normed` (32768 f32) — post `head_rms_norm_inplace` (no-weight RMSNorm per head); RoPE input
    - `kv_raw_out` (n_head_dim = 512 f32) — post `matvec_q8_0(kv)`
    - `kv_normed` (n_head_dim = 512 f32) — post `rms_norm_weight` with `kv_a_norm`; RoPE input
    - `q_post_rope` (32768 f32) — post `rope_tail_layer_inplace`; attention kernel input
    - `kv_post_rope` (n_head_dim = 512 f32) — post `rope_tail_layer_inplace`; pre fp8 KV-cache quant
  - **M5 attention compute + output projection (added by 0005 patch):**
    - `kv_cached_row` (n_head_dim = 512 f32) — last row of `cache->raw_kv` after `kv_cache_push_raw` (f16-roundtripped post-FP8 value that attention actually consumes)
    - `attn_heads` (n_head × n_head_dim = 32768 f32) — per-head attention output post softmax; fires for both `rows_one` and `mixed_one` branches
    - `attn_heads_inv_rope` (32768 f32) — post inverse `rope_tail_layer_inplace`
    - `attn_out_low` (n_groups × n_lora_o = 8 × 1024 = 8192 f32) — post grouped Q8_0 matvec (`attn_output_a`), pre `attn_output_b` back-projection
    - `attn_out` (n_embd = 4096 f32) — post full grouped output projection (= input to HC-post composition, M10)
  - **M6 mixed-attention side channels (added by 0006 patch):**
    - `comp_kv_row` (n_head_dim = 512 f32) — sparse per-token: fires only when the compressor pushes a row (ratio==4 layers: every 4 tokens from T=3; ratio==128: never in this 57-token prompt). f16-roundtripped post-FP8 value that mixed_one consumes.
    - `comp_allowed_mask` (n_comp i32) — only at ratio==4 layers (L=2,4,…,42), per token from T=3 onwards (when n_comp ≥ 1). i32 0/1 re-encoding of the indexer's bool mask. **Only dump tag using `DS4_DUMP_DTYPE_I32`.**
  - `ffn_cur` (n_embd f32)
  - `ffn_input_norm` (n_embd f32) — output of layer→ffn RMSNorm
  - `layer_output_residual` (n_hc × n_embd f32)
- Tag set at synthetic L=43 (head bucket — 0003 patch), per token position:
  - `output_flat` (n_hc × n_embd f32) — post `rms_norm_no_weight` of the HC residual
  - `output_pre` (n_hc f32) — post `matvec_f16` against `output_hc_fn`
  - `output_hc_weights` (n_hc f32) — post sigmoid; the 4 HC mixing weights
  - `output_embd` (n_embd f32) — post `hc_weighted_sum_one`
  - `output_norm` (n_embd f32) — post `rms_norm_weight`; **M3 Q8_0 matvec oracle input**
- Per-layer weights (in `L<LL>/weight/`, deduped):
  - `attn_norm.bin`, `ffn_norm.bin` — per-layer RMSNorm scales (L=0..42)
  - **M4 (added by 0004 patch):**
    - `q_a_norm.bin` (1024 f32) — Q LoRA factor norm
    - `kv_a_norm.bin` (n_head_dim = 512 f32) — KV factor norm
    - `rope_params.bin` (6 f32) — `[freq_base, freq_scale, ext_factor, attn_factor, beta_fast=32, beta_slow=1]`; YaRN parameters per layer, computed via `ds4_layer_compress_ratio(il)` etc. inside the patch so the Rust RoPE port is independent of ds4 compile-time constants
  - **M5 (added by 0005 patch):**
    - `attn_sinks.bin` (n_head = 64 f32) — per-head sink scores; contribute to the softmax denominator (no value vector)
  - `output_norm.bin` — head RMSNorm scale (L=43 only)

**Logit-row ↔ output_norm token mapping** (used by M3 Q8_0 oracle):
- `logits.f32` row k (k=0..50, 51 rows total) ↔ `L43/T<6+k>/output_norm.bin` — the `output_norm` at position `6+k` is the input that ds4's Q8_0 matvec consumed to produce logit row k.
- Determinism: rerunning produces bit-identical manifest + every `.bin` file (verified by spot-check SHA on `L17/T0005/attn_input_norm.bin`).
- Reproduction:
  ```bash
  nix develop -c ./external/ds4-dump/ds4-dump-activations \
      /persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
      "DeepSeek-V4 Flash is" \
      reference/v4flash-cpu-activations \
      50
  ```
  Wall time: ~26 s with warm page cache, ~3 min cold.

## Tokenizer cross-check

`external/ds4-dump/ds4-dump-tokenize` (vocab-only) on `"DeepSeek-V4 Flash is"` returns the identical 7 token IDs as `ds4_tokenize_text` in the engine path:

```
[53091, 4374, 1465, 13582, 22, 32958, 344]
```

This confirms `ds4_dump_text_tokenization`'s `tokenize_rendered_chat_vocab` path degenerates to plain BPE for marker-free input. Rust-side `BpeVocab::encode()` comparison is still TODO (needs a small CLI binary in `v4flash-core`).

## Reproduction command

```bash
nix develop -c bash -c '
  ./external/apply-patches.sh
  make -C external/ds4 rocm ROCM_ARCH=gfx1151
  make -C external/ds4-dump
  mkdir -p /home/claude-code/deepstrix/reference/v4flash-cpu
  DS4_DUMP_BACKEND=cpu ./external/ds4-dump/ds4-dump-logits \
      /persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
      "DeepSeek-V4 Flash is" \
      /home/claude-code/deepstrix/reference/v4flash-cpu \
      50
'
```

## Determinism

CPU runs #1 and #2 produced **bit-identical** logits (verified via SHA256). Greedy decode on CPU is fully deterministic at the byte level on this hardware.

## Notes for future milestones

Thresholds from `docs/DESIGN.md` §9:

- **Top-1 (greedy) match ≥ 90% on first 50 tokens** is the floor
- **Top-5 set match must be 100%** — divergence in the top 5 indicates a real bug
- **KL divergence < 0.01 nats/token** averaged — small FP-accumulation differences are OK; larger drift is a regression

Validation harness: `phase1 compare <ref-dir> <test-dir>` — reads `tokens.json` + `logits.f32` from each, computes the three metrics, exits non-zero on threshold fail.

Roll-back protocol: if a port diverges materially and we can't fix it in the same session, revert and open an issue rather than re-baselining.
