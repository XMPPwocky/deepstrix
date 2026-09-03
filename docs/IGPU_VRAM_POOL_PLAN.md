# iGPU VRAM carve-out pool — findings and plan (2026-09-02)

Goal: move ~1.8 GiB of iGPU-resident data out of GTT (= host RAM) into the
iGPU's 2 GiB UMA carve-out so a bigger expert quant fits. Everything below was
measured on this box (NixOS, kernel 7.1.5, ROCm 7.2.3, Strix Halo gfx1151 =
card2 / renderD129 / HIP dev 1, 9070 XT gfx1201 = card1 / renderD128 / HIP dev 0)
with the server down. Test programs: `~/.claude/jobs/dfe8905e/tmp/vrampool/`
(`probe.hip`, `gem_probe.hip`, `gem_va_probe.hip`, `vram_pressure.cpp`,
`build.sh`, `dev.sh` — build/run via `./dev.sh './build.sh && ./probe'`).

## TL;DR

1. **No HIP/HSA allocator can put memory in the carve-out on this kernel.** Not
   a ROCr choice: kernel 7.1.5 sets `adev->apu_prefer_gtt` (amdgpu_ttm.c,
   `real_vram_size < gtt_size`) and then KFD forces `domain = GTT` for every
   `ALLOC_MEMORY_OF_GPU` *and* for every dma-buf import
   (`amdgpu_amdkfd_gpuvm.c`: `alloc_memory_of_gpu`, `import_obj_create`).
   KFD topology exposes one 134 GiB "VRAM" bank (= `ttm.pages_limit`), rocminfo
   shows no 2 GiB pool. Verified: `hipMalloc`, `hipExtMallocWithFlags`
   (Default/Finegrained/Uncached/Contiguous; SignalMemory = invalid argument),
   `hipMallocManaged`, `hipMemCreate` (VMM), `hsa_amd_memory_pool_allocate`
   from both iGPU pools — all land in GTT (`mem_info_vram_used` delta 0).
   `hipImportExternalMemory` of a VRAM GEM BO **migrates the BO to GTT**.
2. **What does work (route R2): a DRM GEM BO mapped straight into the compute VM.**
   `DRM_IOCTL_AMDGPU_GEM_CREATE(domain=VRAM)` on the iGPU render node lands in
   the carve-out; mapping it with `DRM_IOCTL_AMDGPU_GEM_VA` on the render fd
   that **libhsakmt itself opened** (KFD acquired that fd's VM via
   `AMDKFD_IOC_ACQUIRE_VM`, so it is the VM the HSA queues run in) at a VA
   reserved with `hipMemAddressReserve` makes it addressable by iGPU kernels.
   Stays in VRAM (tested 256 / 512 / 1600 / 1664 / **1720 MiB**), streaming-read
   **240 GB/s = GTT (241 GB/s)**, CPU upload via `GEM_MMAP` (WC) 10–12 GB/s
   single-thread, dGPU can map the same BO via dma-buf + `GEM_VA` and
   read/write it correctly over PCIe. HIP's copy/memset APIs do **not** work
   on such pointers (kernels only).
3. **R2 has one serious, measured failure mode:** if any other DRM client
   requests VRAM the kernel evicts the BO to GTT and the compute-VM PTEs go
   stale — **silent 100 % data corruption, never self-heals** (1600 MiB BO,
   second client asked for 1024 MiB → `verify mismatches = 419430400`). It is
   safe only on a box where nothing else touches iGPU VRAM, with headroom and
   a fail-fast canary.
4. **Recommended first move (route R1, no engine change): shrink the carve-out.**
   `/sys/class/drm/card2/device/uma/carveout` (root-writable) selects the
   UMA size for the *next boot* via ACPI (`amdgpu_device.c` `carveout_store` →
   `amdgpu_acpi_set_uma_allocation_size`). Options on this board:
   `0: Minimum (512 MB)`, 1 GB, 2 GB (current), 4/8/16/32 GB. Writing `0` +
   reboot returns **1.5 GiB to Linux** (MemTotal), zero risk to the engine
   (driver uses 162 MiB of VRAM; GTT path unaffected; `apu_prefer_gtt` stays
   true). That is 0.3 GiB short of the 1.8 GiB target but free. Do R1; do R2
   only if the firmware rejects R1 (`EINVAL` "lack of Custom/Auto flag") or the
   remaining 0.3 GiB really matters — and note R1 and R2 are mutually exclusive
   (after R1 the pool is ~350 MiB).

## 1. Where allocations land (step 1)

`probe` results, 256 MiB each, iGPU, deltas of `/sys/class/drm/card2/device/mem_info_{vram,gtt}_used`:

| allocator | vram Δ | gtt Δ | kernel touch | read BW |
|---|---|---|---|---|
| `hipMalloc` | +0 | +256 MiB | ok | 240 GB/s |
| `hipExtMallocWithFlags(Default=0)` | +0 | +256 | ok | 241 |
| `hipExtMallocWithFlags(Finegrained=1)` | +0 | +256 | ok | 186 (fine-grain penalty) |
| `hipExtMallocWithFlags(SignalMemory=2)` | — | — | `invalid argument` (256 MiB and 64 MiB) | |
| `hipExtMallocWithFlags(Uncached=3)` | +0 | +256 | ok | 241 |
| `hipExtMallocWithFlags(Contiguous=4)` | +0 | +256 | ok | 241 |
| `hipMallocManaged` (+SetCoarseGrain) | +0 | +0 (HMM pages, still host RAM) | ok | 241 |
| `hipMemCreate` VMM (location=device 1) | +0 | +256 | ok | 232 |
| `hsa_amd_memory_pool_allocate`, iGPU pool 1 (COARSE, 134 GiB, location=GPU) | +0 | +256 | ok | 238 |
| `hsa_amd_memory_pool_allocate`, iGPU pool 2 (EXT FINE, 134 GiB) | +0 | +256 | ok | 242 |
| GEM `DRM_IOCTL_AMDGPU_GEM_CREATE` domain=VRAM, flags=0 (render node) | **+256** | +0 | (see §2) | |
| … then `hipImportExternalMemory(OpaqueFd)` | **−256** | **+256** | ok, but now GTT | 242 |

Why: KFD topology node 2 (iGPU) has one `mem_banks/0` with `heap_type 1`
(FB_PUBLIC) and `size_in_bytes 144039215104` = `ttm.pages_limit × 4096`; the
kernel reports system memory *as* the iGPU's frame buffer. Kernel code
(v7.1, `amdgpu_amdkfd_gpuvm.c`):

```
amdgpu_amdkfd_gpuvm_alloc_memory_of_gpu():
    if (adev->apu_prefer_gtt) { domain = AMDGPU_GEM_DOMAIN_GTT; alloc_domain = AMDGPU_GEM_DOMAIN_GTT; ... }
import_obj_create():
    (*mem)->domain = (bo->preferred_domains & AMDGPU_GEM_DOMAIN_VRAM) && !adev->apu_prefer_gtt
                     ? AMDGPU_GEM_DOMAIN_VRAM : AMDGPU_GEM_DOMAIN_GTT;
amdgpu_ttm.c amdgpu_ttm_init():
    if (adev->flags & AMD_IS_APU) if (adev->gmc.real_vram_size < gtt_size) adev->apu_prefer_gtt = true;
```

There is no module parameter for `apu_prefer_gtt` (`/sys/module/amdgpu/parameters` has none).
The only way to flip it is `gttsize < 2 GiB`, which would destroy the 83 GiB
GTT the engine lives on. Env knobs in libhsa-runtime64 / libamdhip64
(`HSA_*`, `ROC_*`, `GPU_*`, `HIP_*`, incl. `HSA_LOCAL_MEMORY_ENABLE`,
`HSA_ZFB`, `HSA_ALLOCATE_QUEUE_DEV_MEM`) contain nothing that selects a pool
that does not exist. Dead end confirmed at every layer.

## 2. Route R2 measurements (steps 2–3)

`gem_va_probe <MiB> [dgpu]`:

| item | result |
|---|---|
| `GEM_CREATE(VRAM)` 256 / 512 / 1600 / 1664 / 1720 MiB | all +N MiB `vram_used`, 0 GTT; 1720 MiB → `vram_used` 1881.9 / 2048 MiB (`AMDGPU_INFO_MEMORY` says usable 1878 MiB, driver baseline 162 MiB, so ~1.7 GiB is the ceiling) |
| VA reservation | `hsa_amd_vmem_address_reserve` and `hipMemAddressReserve` (no HSA link) both work; VA is in ROCr's SVM range (`0x7ffd…`), PROT_NONE on the CPU |
| `GEM_VA MAP` on libhsakmt's iGPU fd (`/proc/self/fd` → `/dev/dri/renderD129`, fd 8 in the probe) | OK; iGPU fill+verify 0 mismatches; BO still in VRAM after touch |
| streaming read, back-to-back | VRAM BO 240.1 / 239.9 GB/s vs `hipMalloc` GTT 242.0 GB/s (one later run 220 vs 147 — clock noise; always A/B back-to-back) |
| upload | `GEM_MMAP` + CPU `memcpy` (WC): 10.6–12.6 GB/s single thread (32 GB/s for warm 256 MiB); GPU verify 0 mismatches. 1.5 GiB ≈ 0.15 s of the ~100 s load |
| `hipMemcpy` H2D onto the VA | `invalid argument` |
| `hipMemcpyAsync` D2D / `hipMemsetAsync` onto the VA | **SIGSEGV** (CLR treats an unregistered pointer as host memory) |
| `hipPointerGetAttributes` | type=0 (unregistered) |
| dGPU: dma-buf export → `PRIME_FD_TO_HANDLE` on libhsakmt's dGPU fd → `GEM_VA` at the same VA | OK, **no migration** (vram/gtt Δ 0); dGPU kernel writes 64 MiB, iGPU verifies 0 mismatches and vice versa |
| `hsa_amd_memory_async_copy(dGPU buf → VA)` | works (0 mismatches) — the reserved VA is known to ROCr |
| `hipMemcpyPeerAsync` onto the VA | not usable (same unregistered-pointer path); peer traffic would need the dGPU-side `GEM_VA` mapping + a kernel or the HSA copy |
| cleanup | `GEM_VA UNMAP` + `GEM_CLOSE` → `vram_used` back to baseline every run; a SIGSEGV'd process also released its BO (kernel frees on fd close) |

**Eviction experiment** (`HOLD_SECS=14 ./gem_va_probe 1600` + `./vram_pressure 1024 5` from a second process):
t=1..4 s verify OK, `vram_used` 1762 MiB; the moment the second client's
1024 MiB `GEM_CREATE(VRAM)` succeeded, ours was evicted (`vram_used` → 1186,
`gtt_used` +1600) and **every word read through the compute-VM mapping was
wrong (419 430 400 / 419 430 400)** — the VM still pointed at the old VRAM
pages, now owned by someone else. After the pressure client released, the BO
stayed in GTT and the mapping stayed stale. DRM only refreshes such mappings
at the next command submission of the *DRM* client, which a compute process
never issues. No GPU fault, no log line: silent.

Side findings (control buffers, plain `hipMalloc` GTT):
`hipMemcpyPeerAsync` dGPU→iGPU queued on the **destination** stream returned
all zeros **even with the producer kernel fully synchronized first** — so the
`project_peer_copy_stream_rule` corruption is a copy-path/mapping issue
(iGPU-side engine reading dGPU VRAM), not a fence-scope race. Source-stream
copies, iGPU→dGPU, and dGPU kernels dereferencing iGPU GTT memory directly all
verified correct. `hipDeviceCanAccessPeer` = 1 both ways.

Host-RAM sanity: `/sys/devices/system/memory` shows 47 × 2 GiB = 94 GiB
present + 2 GiB carve-out = 96 GiB physical; MemTotal 91.97 GiB. The "128 GiB
− 37 GiB carve-out" framing in the memory notes is stale.

## 3. What is on the iGPU and what to move (step 4)

Inventory at production config (UD-Q2_K_XL, K=8 budget 344 slots, dedup on,
2 lanes, B_MAX=1024) — total iGPU ≈ 81.37 GiB:

| family | count | total | notes |
|---|---|---|---|
| routed-expert slabs `IgpuLayerWeights.routed.{gate,up,down}.buffer` — one contiguous `DeviceBuffer<u8>` per (layer, tensor) | 129 | **80.99 GiB** | alloc `crates/v4flash-kernels/src/het/weights.rs:519` (`load_experts_packed`, 40 layers) and `crates/v4flash-kernels/src/weights.rs:124` (`load_to_device`, layers 0–2 with K=0). Size = `(256−K(l)) × bytes_per_expert`. Typical layer (K=8): gate 573.5, up 573.5, down 759.5 MiB. L26 (IQ3_XXS/MXFP4) 2.48 GiB, L42 down MXFP4 1.05 GiB |
| prefill scratch `BatchIgpuScratch` (`bi_a`,`bi_b`) | 2 × 14 | 390.7 MiB | `het/batch_scratch.rs:446-493`; biggest `q2k_partials` 96 MiB/lane, `d_mid_cat` 48 MiB/lane |
| decode scratch `IgpuScratch` | 1 | 98 KiB | `het/scratch.rs:349-376` |
| `hot_remap` | 40 | 40 KiB | `het/weights.rs:915` |
| KV cache, expert-stat counters, hot experts, attention, head | 0 | 0 | all dGPU (`het/state.rs:154-176`, `het/engine.rs:892-900`) |

Peer-copy endpoints on the iGPU (must NOT move): `bi.ffn_input_norm_recv`,
`bi.d_selected`, `bi.d_ew` (dst, `forward_prefill.rs:2335-2343`), `bi.ffn_moe`
(src, `:2963`), `IgpuScratch.ffn_input_norm_recv` / `sel_ew_pack` (dst,
`forward_layer.rs:1233,1319`), `IgpuScratch.ffn_moe` (src, `:202,:1490`).
hipMemset/D2H users (must not move): `bi.group_count` (`fill_zero`,
`:2507`), `bi.n_*work_items` (`fill_zero` + `copy_to_host`, `:2581-2611,2654,2675`).
No dGPU kernel dereferences any iGPU pointer (peer access is only used by
`hipMemcpyPeerAsync`).

**Scratch cannot reach the target**: all iGPU scratch is 391 MiB, and the
larger pieces that are not peer endpoints (`q2k_partials`, `d_mid_cat`,
`d_midq_cat`, `d_xq_q8k`: 325 MiB) are kernel-only but too small. The 1.5 GiB
has to be expert slabs.

**Choice: two `down` slabs from packed layers** (`load_experts_packed` path),
e.g. L15 (747.3 MiB) + L25 (750.3 MiB) = **1.46 GiB**, or any two typical
layers ≈ 1.48–1.53 GiB, leaving ≥ 190 MiB of the ~1.7 GiB pool as eviction
headroom. Why these:
- allocated once at load, never freed, never peer-copied, never memset, never
  read back; only `copy_from_host` per expert into `slice_view_mut` at
  `het/weights.rs:527-529` (becomes a memcpy into the BO's CPU mapping);
- kernels address them as `base + slot*bpe` with the stride passed as
  `dbpe`; a relocated slab is a different base pointer and nothing else
  (`forward_layer.rs:1446-1453`, `forward_prefill.rs:2840-2845`);
- one dtype family (IQ3_XXS down) → one kernel family to validate;
- relocation happens before the first decode, so the 43 captured
  `routed_moe` iGPU graphs (`het/graph_cache.rs`) bake the final pointers —
  no invalidation needed as long as the pool is static for the process.
Three `gate` slabs (1.68 GiB) also fit the constraints but leave ~30 MiB of
headroom — too tight given §2's eviction result. Avoid L0–2 (different
upload path), L26 and L42 (odd dtypes).

**Tradeoff vs. touching the hot-expert machinery:** the slab *contents* depend
on the derived placement (`hot_experts.txt`, per-layer K 0..12 → slab size
varies per restart), so the pool must pick slabs by size at load time (greedy:
largest `down` slabs that fit `budget − headroom`), not by fixed layer id.
The dedup remap (`encode_igpu_remap`, `het/weights.rs:677-693`) and the
placement allocator are untouched: they only produce slot indices. Moving a
slice of experts *within* a slab would instead require a second base pointer
per tensor in every MoE kernel — do not do that.

## 4. Engine change (only if R2 is pursued)

All in `crates/v4flash-hip` (device-memory layer), no new crates (std already
links libc; declare `ioctl`/`mmap`/`munmap`/`readlink` as `extern "C"`;
struct layouts and ioctl numbers mirrored from `amdgpu_drm.h`/`drm.h`, e.g.
`DRM_IOCTL_AMDGPU_GEM_CREATE = 0xC0206440`, `GEM_MMAP = 0xC0106441`,
`GEM_VA = 0x40286448`, `DRM_IOCTL_GEM_CLOSE = 0x40086409` — verify with a
`static_assert`-style unit test compiled by hipcc in `build.rs`).

1. `vram_pool.rs`: `VramPool::open(igpu: &Device, budget_mib) -> Option<VramPool>`
   - find the iGPU's render node by matching `pciBusID` from
     `hipGetDeviceProperties` against `/sys/class/drm/renderD*/device/uevent`
     (`PCI_SLOT_NAME`), then the **already-open** fd in `/proc/self/fd` that
     links to it (libhsakmt opens it at ROCr init; the engine has queried
     device properties long before weights load, so it exists — assert).
     Do not open a fresh fd: a new fd has its own VM.
   - budget = `min(env, vram_total − vram_used − HEADROOM(256 MiB))` from
     `mem_info_vram_*`; log it.
   - `alloc(bytes) -> Result<VramBacking>`: `GEM_CREATE(VRAM, align 2 MiB,
     flags 0)` → `hipMemAddressReserve(2 MiB aligned)` → `GEM_VA MAP
     (R|W|X)` → `GEM_MMAP` + `mmap` (kept for the process lifetime; WC) →
     run a 1-page touch/verify kernel before returning (guards against the
     PTE update racing the first use).
   - `Drop`: `GEM_VA UNMAP`, `hipMemAddressFree`, `munmap`, `GEM_CLOSE`.
2. `buffer.rs`: `DeviceBuffer` gains a `backing: Backing::{Hip, Vram(VramBacking)}`;
   `DeviceBuffer::new_pooled(pool: Option<&mut VramPool>, device_id, len)`
   tries the pool and **falls back to `hipMalloc`** (info log) when the pool
   is absent, full, or any ioctl fails. On `Backing::Vram`:
   `copy_from_host` = memcpy into the CPU mapping (+ `sfence`), `fill_zero`
   = memset via the mapping; `copy_to_host`, `copy_from_device`,
   `copy_to_peer_async`, `fill_zero_async` return `Err(unsupported)` so a
   misuse fails loudly instead of segfaulting. `raw()` unchanged → kernels
   unchanged.
3. `het/weights.rs:519`: `load_experts_packed` takes the pool; for
   `tensor == down` (policy: largest-first, pool-fit) use `new_pooled`. The
   per-expert upload at `:527-529` works unchanged through `slice_view_mut`
   + `copy_from_host`. Pool is created in `het/engine.rs` init right before
   `HetModelWeights::load_all`, after device selection
   (`deepstrix-server/src/engine_worker.rs:396-404`).
4. Fail-fast canary: keep `expected_vram_used` at load; every N requests (or
   the server health tick) read `mem_info_vram_used`; if it drops by ≥ the
   smallest pooled slab, log FATAL "iGPU VRAM pool evicted — weights are
   stale" and `exit(75)` (launcher restarts). Optionally also a 4 KiB
   pattern page at the end of each pooled slab checked by a tiny kernel once
   per forward. Cost: ~µs.
5. Enable: `DEEPSTRIX_IGPU_VRAM_POOL_MIB=<n>` (unset/0 = off, default). Prints
   at load: slabs placed (layer, tensor, MiB), pool used/total, and
   `mem_info_vram_used` / `mem_info_gtt_used` before/after.

## 5. Verification

- After load: `cat /sys/class/drm/card2/device/mem_info_vram_used` ≈ 162 MiB +
  placed bytes; `mem_info_gtt_used` lower by the same; `MemAvailable` up by
  ~1.5 GiB (GTT pages are host RAM). `DEEPSTRIX_ALLOC_TRACE=1` still lists the
  slabs (the tracer runs before the backing choice — extend the line with the
  backing kind).
- Correctness: all 29 oracles (they drive the routed MoE kernels through the
  same base+stride path); `vector-test` 13/13.
- Perf: `bench_decode` @4K and `bench_prefill_chunked` with `PIPELINE_LANES=2`,
  pool off vs on **back-to-back** (thermal drift ~2 tok/s); expectation is
  noise (240 vs 241 GB/s).
- Robustness: run `vram_pressure 1024 5` against a serving instance → the
  canary must fire within one tick. Then confirm a normal run never trips it.
- Nothing else may use iGPU VRAM while serving (no Vulkan/GL clients on
  renderD129 / card2, no compositor).

## 6. Risks

| risk | evidence / mitigation |
|---|---|
| Eviction → silent corruption | measured (§2). Headroom ≥ 190 MiB, canary + exit, single-tenant iGPU. Cannot be fixed from userspace: KFD's eviction-fence/restore machinery only covers KFD-owned BOs, and importing into KFD forces GTT. |
| HIP APIs blind to pool pointers | `hipMemcpy*`/`hipMemset*`/`hipMemcpyPeerAsync` fail or SIGSEGV; restricted by the `Backing` enum to slabs that only ever see `copy_from_host`. |
| Fine- vs coarse-grain | `GEM_VA` default flags give normal cached (coarse-style) VRAM mapping; BW equal to coarse GTT. No cross-device coherence needed (iGPU-only readers). |
| Peer copies | none on the chosen slabs. If ever needed: dma-buf + `GEM_VA` on the dGPU's libhsakmt fd works (measured), copy via kernel or `hsa_amd_memory_async_copy`, not `hipMemcpyPeerAsync`. |
| Kernel / ROCr version dependence | relies on: KFD acquiring the render fd's VM (stable ABI since 4.x), `GEM_VA` on a compute VM (what Mesa↔ROCm interop uses), libhsakmt keeping the fd open, `hipMemAddressReserve` VA living in the same VM, and `apu_prefer_gtt` semantics (6.10 "Let VRAM allocations go to GTT domain on small APUs", 6.16 "only if GTT is larger"). A future ROCm that exposes a real APU VRAM pool would obsolete this — the env flag makes it easy to retire. |
| Fragmentation | carve-out is a drm_buddy heap; two ~750 MiB non-contiguous BOs from a fresh 1.7 GiB heap allocate fine (1720 MiB single BO worked). Allocate pooled slabs first at load. |
| Multi-process | pool is per-process; a second engine instance gets whatever is left, falls back to GTT. |
| Crash cleanup | kernel frees GEM BOs on fd close (verified after a SIGSEGV). |
| Only ~1.5 GiB, not 1.8 | hard ceiling ≈ 1.7 GiB minus headroom. |

## 7. Effort

- R1 (carve-out → 512 MB): 10 minutes + one reboot; needs root:
  `echo 0 | sudo tee /sys/class/drm/card2/device/uma/carveout` (expect `0` on
  read-back; `EINVAL` means the firmware option lacks the Custom/Auto flag →
  R1 impossible), reboot, verify `mem_info_vram_total` = 536870912 and
  MemTotal ≈ +1.5 GiB. Reversible (write `2`).
- R2: ~2–3 days — `vram_pool.rs` + ioctl FFI (~300 lines), `Backing` enum in
  `buffer.rs` (~100), `load_experts_packed` policy + engine wiring (~60),
  canary (~40), oracle + bench A/B + eviction test (half a day). Ship
  default-off behind `DEEPSTRIX_IGPU_VRAM_POOL_MIB`.
