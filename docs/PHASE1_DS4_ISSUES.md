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

## Runtime issues (M1.4 / M1.6)

### CPU backend — works, bit-deterministic

V4 Flash IQ2XXS loads and decodes on the CPU backend (`DS4_BACKEND_CPU`) with no issues. Two independent 50-token runs with greedy decoding produced bit-identical logit dumps (SHA256 `d294856e3a732d15ba5585d1a52320820428ff846a88848c7888173c416fcd2a`). Wall time ~155 s per 50 tokens. This is the authoritative M1 reference; see `PHASE1_REFERENCE.md`.

### GPU backend — does not work on this hardware against this model

ds4's GPU path is unusable for V4 Flash IQ2XXS on Strix Halo + our setup. **Out of scope for M1; CPU reference is sufficient.** Symptoms recorded here for future-us if we ever revisit.

#### HIP device enumeration is counter-intuitive

`rocm-smi` and the dev-shell preamble print `gfx1151` first then `gfx1201`, but **HIP enumerates the opposite**:

- HIP device **0**: gfx1201 (9070 XT dGPU, 16 GiB) ← what ds4 grabs by default; can't fit the 86 GiB model
- HIP device **1**: gfx1151 (Strix iGPU, 137 GiB reported)

Confirmed via `phase0 toolchain` (`results/toolchain.json`). Set `HIP_VISIBLE_DEVICES=1` explicitly to target the iGPU.

#### Copy-model path (default; allocates model into device-owned arena)

```
DS4_BACKEND_CUDA, HIP_VISIBLE_DEVICES=1, no DS4_CUDA_DIRECT_MODEL
```

Reproducibly OOMs at the same spot, even with a freshly-reset GPU and no competing processes:

```
ds4: CUDA loading model tensors into device cache
ds4: CUDA model arena alloc failed for tensor-span:12 (1792.00 MiB chunk): out of memory
ds4: accelerator failed to cache model tensor span 12 at offset 7262399296
```

i.e. ~7.26 GiB of weights cached, then can't allocate the next ~1.75 GiB contiguous chunk. Suggests an allocator cap (BIOS-carved iGPU VRAM partition? GTT vs VRAM split? `amdgpu.vramlimit`-style?) well below the 137 GiB the device reports. Not investigated — `/sys/class/drm/card*/device/mem_info_{vram,gtt}_total` would be the first thing to check.

#### Direct-mmap path (`DS4_CUDA_DIRECT_MODEL=1`)

Skips the device-cache copy; exposes host mmap to the GPU via SVM. Fails differently:

```
ds4: CUDA host registration skipped: invalid argument
ds4: cuda backend initialized for graph diagnostics
ds4-dump-logits: prompt tokenized to 7 tokens
[silence; 100% CPU spin; no log output]
```

dmesg shows the cause:

```
amdgpu: SVM mapping failed, exceeds resident system memory limit
amdgpu 0000:c8:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:1024)
   in page starting at address 0x00007ffe... from client 10 (TCP)
```

amdgpu caps per-process SVM resident size below the 86 GiB ds4 wants. GPU shaders then fault forever trying to dereference unmapped host pointers; ds4 spins on CPU presumably in a kernel-launch retry loop. Faulting addresses are stack pointers, suggesting metadata structs in kernel launch args are what the GPU dereferences first.

Recovery requires `amdgpu` GPU reset (auto-eventual or via `rocm-smi`); subsequent GPU work may be degraded until reset.

#### Knobs reviewed but not deeply explored

- `DS4_CUDA_WEIGHT_PRELOAD_SPAN_MB=N` (clamped [64, 4096], default 1024): smaller chunks would just postpone the copy-model OOM, not fix it.
- `DS4_CUDA_Q8_F16_PRELOAD`, `DS4_CUDA_Q8_F32_PRELOAD`: opt-in dequant caches; would make memory pressure worse.
- `--warm-weights` (not exposed by `ds4-dump-logits`): would change timing, not the allocator ceiling.

### Why this doesn't block us

ds4's GPU path uses all-or-nothing whole-model placement. The design doc (§5/§6) plans per-tensor placement spanning iGPU UMA + dGPU VRAM + spill-to-mmap, picked by tensor hotness — that strategy is *unavailable* to ds4 by construction. Phase 2+ replaces ds4's GPU path entirely with our own kernels and our own allocator. The CPU reference is bit-deterministic and is the contract for every later milestone.

### If we ever revisit ds4 GPU

In rough order of likelihood-to-help:

1. Check `/sys/class/drm/card*/device/mem_info_{vram,gtt}_total` for the actual iGPU memory partition.
2. BIOS option for UMA frame buffer size, if exposed.
3. Read amdgpu kernel source for the "resident system memory limit" string to find the cap and any sysctl/kernel-param to lift it.
4. Patch ds4 to use chunked `hipMemcpyAsync` from mmap — but this is exactly what we'll build in Phase 2 anyway.

## Upstream-tracking strategy

If antirez ships a meaningful update on the rocm branch:

1. `cd external/ds4 && git pull origin rocm`
2. `cd ../.. && external/apply-patches.sh` — reapply our patches; if conflicts surface, regenerate patches from the new base
3. Re-baseline `docs/PHASE1_REFERENCE.md` explicitly (changed upstream = changed numerics)

Don't `git submodule update` blindly without rerunning the apply step.
