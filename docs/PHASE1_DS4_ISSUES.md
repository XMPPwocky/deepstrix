# ds4 rocm — known issues and local patches

deepstrix uses antirez's ds4 (rocm branch) as the Phase 1 correctness baseline. The branch is community-maintained — antirez doesn't have AMD hardware — so we expect to find and work around issues. This file tracks them.

## Local patches

Patches live in `external/ds4-patches/` and are applied by `external/apply-patches.sh` (idempotent — safe to re-run; checks `git apply --reverse --check` first). Each `.patch` is a unified diff of one logical change.

Re-apply after any `git submodule update --init`:
```bash
external/apply-patches.sh
```

### 0001-expose-logits-buffer.patch

**What**: Adds `ds4_session_logits_buffer(s, &ptr, &n)` to ds4's public API. Returns a const pointer to `s->logits` and the vocab size.

**Why**: ds4's public surface only exposes `top_logprobs(k)` (O(k²) sort) and `token_logprob(i)` (single token). Neither is workable for dumping full-vocab logits across N tokens for KL/top-5 numerical validation. Internal `s->logits` is already populated; we just expose it.

**Risk**: Patch is additive — adds one function, no modifications to existing logic. Safe to rebase on top of upstream.

**Status**: applied (M1.5)

## Build issues encountered

### Issue 1: hipcc doesn't search HIP_PATH/include for downstream libs

**Symptom**: `fatal error: 'hipblas/hipblas.h' file not found` during ds4_cuda.cu compile.

**Cause**: ds4's Makefile passes no `-I` flags. hipcc looks at its own install for HIP runtime headers but not the merged tree.

**Fix**: Set `CPATH=${rocmJoin}/include` in the nix dev shell so the underlying clang finds them. Also set `LIBRARY_PATH=${rocmJoin}/lib` for the linker.

**Status**: fixed in flake.nix at M1.1 commit.

### Issue 2: ROCm dependencies span multiple nixpkgs derivations

**Symptom**: ds4's Makefile assumes `$ROCM_PATH/bin/hipcc` and `$ROCM_PATH/lib/lib*.so` all live in one tree. Nix splits clr, hipcc, hipblas, hipblas-common, rocblas, rocsolver, rocwmma, etc. across separate derivations.

**Fix**: `pkgs.symlinkJoin` (`rocmJoin` in flake.nix) merges them all into one tree. Export `ROCM_PATH` and `HIP_PATH` pointing at the merged tree.

**Status**: fixed in flake.nix at M1.1 commit.

## Runtime issues (pending M1.4)

This section will be populated when the V4 Flash model arrives and we actually run inference. Likely-failure-modes list:

- **FP8 KV cache path**: ds4 uses FP8 KV; RDNA 3.5 (gfx1151) lacks FP8 WMMA. ds4 likely emulates in FP16, but worth verifying via runtime output that no kernel falls into a #ifdef branch we don't support.
- **Memory pressure**: ~86 GiB weights vs ~80 GiB GPU budget — see `memory/project-strix-v4flash-memory-tightness`. Must use mmap-based load (default), not `DS4_CUDA_COPY_MODEL=1`. If even mmap fails, fall back to `--cpu`.
- **rocwmma vs Tensor Core gap**: ds4_cuda.cu defines/imports rocwmma but doesn't appear to use it; CUDA WMMA intrinsics are shimmed. Expert-tile kernels may be slower than CUDA equivalents — acceptable for Phase 1 (correctness, not perf).
- **Determinism**: greedy + same prompt should be bit-identical. M1.6 verifies by running twice.

## Upstream-tracking strategy

If antirez ships a meaningful update on the rocm branch:

1. `cd external/ds4 && git pull origin rocm`
2. `cd ../.. && external/apply-patches.sh` — reapply our patches; if conflicts surface, regenerate patches from the new base
3. Re-baseline `docs/PHASE1_REFERENCE.md` explicitly (changed upstream = changed numerics)

Don't `git submodule update` blindly without rerunning the apply step.
