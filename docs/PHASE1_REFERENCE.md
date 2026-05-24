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
- ds4 patches applied: `external/ds4-patches/0001-expose-logits-buffer.patch`

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
