# Phase 1 reference output

This is the immutable baseline. Every Phase 1 milestone (M2+) validates ported kernels against the captured logits here. Re-baselining means we acknowledge a numerical drift in the upstream; **do not silently regenerate this file**.

## Status

- [x] M1.1 ds4 rocm submodule + nix build wired up
- [x] M1.2 ds4_cli smoke test  
- [x] M1.3 phase1 crate skeleton
- [ ] **M1.4 first model load** (gated on weights download finishing)
- [x] M1.5 ds4-dump-logits wrapper built
- [ ] **M1.6 capture reference logits** (gated on M1.4)
- [x] M1.7 this doc — scaffolded, populated as M1.4 + M1.6 finish

## Hardware + toolchain

- Machine: NixOS 26.05 + ROCm 7.2.3 (via nix flake, see `flake.nix`)
- iGPU: AMD Radeon 8060S Graphics (Strix Halo, gfx1151)
- Reference run is on **iGPU only** (--cuda backend, ROCm via ds4_rocm.h shim)
- ds4 submodule: `external/ds4` @ branch `rocm` (initial: 7a751eb)
- ds4 patches applied: see `external/ds4-patches/` (currently 0001-expose-logits-buffer)

## Model

- Source: `antirez/deepseek-v4-gguf` (HuggingFace)
- HF cache path: `/persist/hf_cache/models--antirez--deepseek-v4-gguf/`
- Model SHA256: _TBD — fill in M1.4_
- Approx size: ~86 GiB (IQ2_XXS routed experts + Q2_K W_down + Q8_0 attention/shared)

## Reference prompt

(To be set in M1.4 — keep short to bound runtime since CPU/iGPU on V4 Flash is not fast)

- Prompt text: `"DeepSeek-V4 Flash"` (placeholder — finalize at M1.4)
- N generated tokens: 50
- Sampling: greedy (argmax)

## Reference outputs

(All TBD until M1.6 lands.)

- `reference/v4flash-rocm.tokens.json`
  - prompt_tokens: _TBD_
  - generated_tokens: _TBD (50)_
  - vocab_size: _TBD_
  - SHA256: _TBD_
- `reference/v4flash-rocm.logits.f32`
  - Shape: (n_logit_rows, vocab_size) row-major float32
  - Total bytes: _TBD_
  - SHA256: _TBD_

## Reproduction command

```bash
nix develop -c bash -c '
  ./external/apply-patches.sh
  make -C external/ds4 rocm ROCM_ARCH=gfx1151
  make -C external/ds4-dump
  mkdir -p reference
  ./external/ds4-dump/ds4-dump-logits \
      <model-gguf-path> \
      "<prompt>" \
      reference/v4flash-rocm \
      50
'
```

## Determinism check

Greedy decode + identical inputs should produce bit-identical logits. M1.6 runs the dump twice and asserts SHA256 equality. If they differ, that's a finding to flag (e.g. nondeterministic kernel, racey init).

## Notes for future milestones

When any later milestone's ported kernel produces logits that differ from this reference:

- **Top-1 (greedy) match ≥ 90% on first 50 tokens** is the floor per DESIGN.md §9.
- **Top-5 set match must be 100%** — divergence in the top 5 indicates a real bug.
- **KL divergence < 0.01 nats/token** averaged over the 50 tokens — small FP-accumulation differences are OK; larger drift is a regression.

Roll-back protocol: if a port diverges materially and we can't fix it in the same session, revert and open an issue rather than re-baselining.
