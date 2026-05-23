# V4 Flash Heterogeneous Inference Engine — Design Doc (v2)

## Phase 0 status: complete

All five Phase 0 gates passed; full measurement results in [`PHASE0.md`](PHASE0.md). The doc below has been updated inline to reflect measured numbers — search for **[P0 measured]** for the load-bearing reconciliations.

## 0. Changes from v1

This is a substantial rewrite. The following load-bearing claims from v1 were wrong and have been corrected:

- **GEMV decomposition into v_known + Δ does not save bandwidth.** Splitting the input vector doesn't halve the weight read; it doubles it unless the weights stay in cache. The streaming-compute story in v1 was incorrect. v2 uses sequential post-attention MoE compute with MALL prefetch as the only bandwidth-saving mechanism.
- **Cache strategy is inverted.** v1 cached W_down and bypassed W_up/W_gate. v2 caches W_up/W_gate (which is what gets re-read implicitly via the GEMV) and streams W_down from DRAM.
- **Speculative routing is required by Phase 3, not Phase 4.** v1 implicitly required it but listed it later. v2 makes exact-with-fallback speculation a load-bearing Phase 3 deliverable.
- **Shared expert runs sequentially after attention.** v1 claimed full parallelism with attention. v2 only overlaps the bus-return transfer with the shared expert kernel.
- **HSA_OVERRIDE_GFX_VERSION is process-global.** Single-process dual-device with override may not work. v2 makes this a Phase 0 hard gate with three branches.
- **Memory accounting was low.** Q8_0 is 1.0625 bytes/weight; v1 used 1.0. Routed experts total 78 GB not 70 GB. Sizes corrected throughout.
- **LM head was missing from the per-token budget.** Adds ~1 ms/token at Q8_0, ~6% of decode time.
- **MALL is hardware-managed, not programmer-managed.** v1 treated cache residency as invariant. v2 treats all MALL strategies as best-effort hints.
- **Performance targets revised.** v1's "75-85 tok/s likely" was too optimistic. v2 targets 55 tok/s default with 65 tok/s stretch.

## 1. Scope

### Goal
A high-throughput single-token decode engine for DeepSeek V4 Flash on AMD Radeon RX 9070 XT (eGPU, RDNA 4, gfx1201) over Oculink + AMD Ryzen AI Max+ 395 with Radeon 8060S (iGPU, RDNA 3.5, gfx1151) and 128 GB unified LPDDR5X. Target ≥55 tokens/sec at antirez's IQ2_XXS / Q2_K / Q8_0 mixed quantization, with exact-routing numerical equivalence to the Metal reference.

### Non-goals (initial)
- Prefill (architecture must permit efficient prefill later, but Phase 1 ships decode-only).
- Generality across models. This engine is V4 Flash-specific.
- Cross-platform. Linux + ROCm 7.2.3 only.
- Training, fine-tuning, gradient computation.

### Stretch goals
- MTP-based speculative decoding (1.3-1.6× over Phase 3 baseline; not 2× as v1 claimed).
- Persistent on-disk KV cache.
- Long-context inference (1M tokens).

## 2. Hardware

### 2.1 9070 XT — eGPU

| Property | Value |
|---|---|
| Architecture | RDNA 4 (gfx1201) |
| Compute Units | 64 CUs, 4096 stream processors |
| AI Accelerators | 128 (2 per CU, 2nd-gen WMMA) |
| Boost clock | 2.97 GHz |
| FP32 vector | 48.7 TFLOPS |
| FP16 vector (packed) | 97.4 TFLOPS |
| FP16/BF16 matrix (WMMA) | 194.6 / 389.3 TFLOPS (dense / sparse) |
| FP8 matrix (E5M2, E4M3) | 389 / 779 TFLOPS (dense / sparse) |
| INT8 matrix | 389 / 779 TOPS |
| INT4 matrix | 779 / 1557 TOPS |
| VRAM | 16 GB GDDR6 @ 20 Gbps, 256-bit |
| VRAM bandwidth | 644 GB/s theoretical, **~605 GB/s sustained on Q8_0 GEMV (94%) [P0 measured]** |
| Infinity Cache (3rd gen) | 64 MB |
| TDP | 304 W |
| ROCm target | gfx1201, officially supported in ROCm 7.0.2+ |

### 2.2 Strix Halo — host + iGPU

| Property | Value |
|---|---|
| CPU | 16 Zen 5 cores, up to 5.1 GHz |
| iGPU architecture | RDNA 3.5 (gfx1151) |
| iGPU Compute Units | 40 CUs, 2560 stream processors |
| iGPU boost clock | 2.9 GHz |
| FP32 vector (dual-issue) | ~29.7 TFLOPS |
| FP16/BF16 matrix (WMMA) | 59.4 TFLOPS theoretical, ~37 TFLOPS measured w/ hipBLASLt |
| INT8 matrix (WMMA) | ~118.8 TOPS |
| INT4 matrix (WMMA) | ~237.6 TOPS |
| FP8 matrix | NOT supported (RDNA 3.5 lacks FP8 WMMA) |
| Unified memory | 128 GB LPDDR5X-8000, 256-bit |
| LPDDR5X bandwidth | 256 GB/s theoretical, **~230 GB/s sustained on Q8_0 GEMV [P0 measured]** (doc's 212-215 was conservative) |
| Allocatable to iGPU | up to 96 GB (UMA setting) |
| Infinity Cache (MALL) | 32 MB, hardware-managed memory-side cache |
| CPU↔iGPU memory copy | ~84 GB/s measured |
| TDP | configurable 45-120W |
| ROCm target | **gfx1151 native, no override [P0 measured]**. ROCm 7.2.3 supports it directly; `HSA_OVERRIDE_GFX_VERSION=11.5.1` actively breaks the runtime |

### 2.3 Interconnect — Oculink (PCIe 4.0 x4)

| Property | Value |
|---|---|
| Theoretical bandwidth | 64 Gbps = 8 GB/s |
| Effective bandwidth | ~7 GB/s after PCIe overhead |
| Per-transfer setup latency | **~5 μs submission + ~5 μs HSA signal propagation (one-way) [P0 measured]** |
| Round-trip event sync | **~11 μs steady-state amortized [P0 measured]** (27-38 μs for one-shot RTT with host poll). Symmetric. |
| Cross-device peer-copy rule | **MUST queue on source device's stream [P0 measured]** — queuing on dst-stream silently returns zeros for dGPU→iGPU. |

### 2.4 Implications for design

- Single-token decode is bandwidth-bound on weight reads, not compute.
- Compute headroom enables parallel kernels, speculative work, prefill (batch >> 1).
- Cross-device sync overhead **does not exceed assumed budget [P0 measured]** — ~11 μs steady-state, well within the doc's 10-30 μs estimate. Minimize sync points anyway; each 11 μs adds up.
- FP8 only exists on 9070 XT. iGPU side stays on FP16/BF16/Q-quantized.
- Cache behavior is observable but not directly controllable. **MALL holds ~24 MB effective on Strix (not the 32 MB spec) [P0 measured]**; non-temporal bypass via `__builtin_nontemporal_load` confirmed working.

## 3. Target Model: DeepSeek V4 Flash

### 3.1 Architecture summary
- 43-layer all-MoE backbone (no dense MLPs)
- 256 routed experts + 1 shared expert per block; top-6 routing
- First 3 blocks: `DeepseekV4HashGate` deterministic routing via tid2eid lookup
- Remaining 40 blocks: learned router with sigmoid + auxiliary-loss-free bias
- Hybrid attention: SWA for first 2 layers; then alternating CSA / HCA
- CSA: compress KV 4×, Lightning Indexer selects top-1024 compressed blocks, sliding-window branch for local tokens
- HCA: compress KV 128×, dense attention over compressed sequence
- Manifold-Constrained Hyper-Connections (mHC) with `hc_mult=4` streams, Sinkhorn-Knopp normalized mixing
- 284B total params, 13B activated per token
- 1M token context

### 3.2 Per-block tensor inventory (representative CSA block, sizes computed at 1.0625 bytes/weight for Q8_0)

| Component | Tensor | Shape | Dtype | Size |
|---|---|---|---|---|
| Attention | attn_q_a / q_b | [4096,1024] / [1024,32768] | Q8_0 | 4.5 + 35.7 MB |
|   | attn_kv | [4096,512] | Q8_0 | 2.2 MB |
|   | attn_compressor_{kv,gate,ape} | various | F16 | 8 MB |
|   | attn_output_a / b | [4096,8192] / [8192,4096] | Q8_0 | 71.3 MB |
|   | norms, sinks, kv_a_norm | various | F32 | <100 KB |
| **Attn subtotal** | | | | **~122 MB / layer** |
| Routed MoE | ffn_gate_exps, ffn_up_exps | [4096,2048,256] | IQ2_XXS | 555 MB each |
|   | ffn_down_exps | [2048,4096,256] | Q2_K | 705 MB |
|   | ffn_gate_inp | [4096,256] | F16 | 2 MB |
|   | per-expert (×256): W_up + W_gate + W_down | | | **~7.1 MB / expert** |
| Shared | ffn_{gate,up,down}_shexp | [4096,2048] or [2048,4096] | Q8_0 | 8.91 MB each |
| **Shared subtotal** | | | | **~26.7 MB / layer** |
| HC | hc_attn_fn, hc_ffn_fn | [16384,24] | F16 | 768 KB each |
| **HC subtotal** | | | | **~1.5 MB / layer** |
| Active per token | attn + 6×routed + shared | | | **~191 MB / layer** |

Total model size at this quant mix: ~85 GB. Plus embeddings, LM head: ~86 GB. LM head specifically: vocab≈150K × hidden=4096 at Q8_0 = ~654 MB.

## 4. Design Overview

### 4.1 Device roles

**9070 XT (resident):**
- All 43 layers of attention weights (~5.2 GB)
- All 43 layers of shared expert weights (~1.15 GB)
- KV cache for all layers (~1-3 GB depending on context)
- HC mixing matrices replicated (~65 MB)
- Lightning Indexer scratch buffers, sparse gather temporaries (~500 MB)
- Stream replica for current layer (32 KB)
- Embedding + LM head (~1.3 GB)
- Optional: MTP draft model state (~2 GB if enabled)
- **Total: ~9-11 GB of 16 GB**

**Strix Halo (resident):**
- All 43 layers × 256 routed expert weights (~78 GB at IQ2 mix)
- Router weights, bias arrays, norms
- HC mixing matrices (canonical copy)
- Stream replica (canonical, 32 KB live)
- Distilled router prediction heads (~50 MB total)
- Persistent KV cache (on-disk staging, in-RAM hot pages)
- Tokenizer state, sampler state, control flow
- **Total: ~80 GB of 96 GB GPU-addressable**

### 4.2 Per-layer dataflow (non-hashgate CSA layer)

This is the load-bearing flow. The corrected version uses sequential MoE compute with MALL prefetch — no GEMV decomposition.

1. **t=0**: Strix has streams[t-1]. Distilled router head r_ℓ runs on Strix from previous block's output → predicted top-N (N≥8) expert IDs.
2. **t=0-40**: Strix→9070XT, send mHC_attn_pre input over Oculink (one async transfer; chunking deferred to Phase 3+ optimization).
3. **t=0-178**: Strix iGPU has no useful sequential work yet (MoE input depends on attn output). Use this window to **prefetch W_up + W_gate for top-N routed experts into MALL**. ~35 MB at 215 GB/s ≈ 162 μs, fits in attention window.
4. **t=40-218**: 9070XT runs attention (CSA: indexer + sparse gather + softmax + attn_output stages a then b). 122 MB at 644 GB/s ≈ 190 μs.
5. **t=218-258**: 9070XT→Strix sends attention output (40 μs bus + setup latency, unknown).
6. **t=218-268** (in parallel with step 5): 9070XT runs **shared expert sequentially**: read W_up + W_gate, apply to MoE input, SiLU(gate)*up = h_inter, then h_inter @ W_down. ~27 MB at 644 GB/s ≈ 42 μs read + ~8 μs compute. Overlapped with bus 2.
7. **t=258-263**: Strix runs **real router** using actual MoE input → true top-6.
8. **t=263-283**: Strix computes `u_moe @ W_up` and `u_moe @ W_gate` for true top-6. If true top-6 ⊆ predicted top-N: reads from MALL (cache hit), ~20 μs. Otherwise: synchronous LPDDR5X read for missing experts, +30 μs per missed expert.
9. **t=268-278**: 9070XT→Strix sends shared expert output (~10 μs).
10. **t=283-293**: Strix applies SiLU(gate)*up = h_intermediate.
11. **t=293-373**: Strix computes `h_intermediate @ W_down` for top-6 routed experts. W_down streams from LPDDR5X (no MALL role here). 16.6 MB at 215 GB/s ≈ 77 μs.
12. **t=373-383**: Strix combines routed outputs with shared output via router gates, computes mHC_ffn_post → updated streams.
13. **t=383-393**: Strix→9070XT sends updated streams for next layer.

**Per-layer total: ~393 μs steady state, assuming all caches behave.** Worst case (every layer has 1-2 mispredicted experts): ~430 μs.

Per-token: 43 × 393 μs = 16.9 ms + ~1 ms LM head = **~56 tok/s**. Worst case ~50 tok/s.

### 4.3 Cache strategy (best-effort, not invariant)

**Strix MALL: ~24 MB effective capacity [P0 measured]** (less than the 32 MB spec; the cache curve drops sharply between 24 and 32 MiB working sets).
- Target residency: W_up + W_gate for top-**N=6** predicted routed experts (≈ 20.9 MB, fits comfortably). Doc's original N=8 (~35 MB) and N=7 (~30.5 MB) both overflow effective MALL.
- W_down streams through LPDDR5X with **`__builtin_nontemporal_load`** non-temporal hints. **Confirmed bypassing MALL on Strix [P0 measured]** (511 GB/s → 207 GB/s, ~DRAM rate). Doc's bypass plan is viable.
- MALL **evicts under interleaved write traffic [P0 measured]** (511 GB/s drops to 121 GB/s after 32 MB pollution). Therefore the non-temporal hint for W_down is **load-bearing**, not optional; otherwise W_down reads will evict the W_up/W_gate residency we want.

**9070 XT Infinity Cache: ~64 MB effective [P0 measured]** (matches spec; holds 32 MiB happily at 1037 GB/s, 64 MiB at 1126 GB/s).
- Shared expert weights (~27 MB) DO benefit from IC across consecutive layers when there's no pollution [P0 measured]. Worth designing around explicitly rather than treating as opportunistic.
- IC also evicts under pollution (32 MiB pollute → 48 MiB read drops 998 → 448). Streaming attention weights while shared expert is supposed to stay resident is the risk; either time-separate them or accept partial residency.
- v1 claimed "attention weight reuse across consecutive layers." This was wrong; consecutive layers use different attention weights.

### 4.4 Speculative routing (load-bearing in Phase 3)

For learned-router layers, true top-6 depends on the post-attention MoE input, which doesn't exist at t=0. Naive design: wait for attention, run router, then read experts. This serializes MoE bandwidth after attention and tanks throughput.

Solution: predict top-N ⊇ top-6 at t=0 using a distilled router head trained on offline routing data, prefetch those N experts' W_up/W_gate into MALL during attention, then verify against the true router. **N=6 [P0-revised]** — see §4.3 for why (was N=8 in v2 but effective MALL is 24 MB, not 32 MB). N=6 means top-N coverage of top-6 is harder to achieve; the hit rate target needs re-evaluation against the actual routing distribution. Three outcomes:

| Outcome | Frequency target | Cost |
|---|---|---|
| True top-6 ⊆ predicted top-N | ≥95% | Cache hit on subsequent read, ~20 μs total |
| 1 expert mispredicted | ~4% | Synchronous DRAM read for 1 expert, ~30 μs |
| ≥2 experts mispredicted | <1% | Synchronous DRAM read for missing experts, ~60+ μs |

This is the only way to overlap routed expert bandwidth with attention compute *and* preserve exact-routing numerical match (because we run the real router and use its decisions; the predictor only determines which weights to prefetch).

The distilled router head is a small MLP per learned-router layer: ~1M params, ~50 MB total across 40 layers, trained offline against the true router on ~10M cached routing decisions.

### 4.5 Streaming RMSNorm

For RMSNorm, σ depends on the squared sum of the full normalized vector. The MoE input is v_moe = mHC_mix(streams) + s·attn. We can accumulate σ² streamingly across two phases:
- v_known phase (t=0): compute partial Σᵢ v_known[i]² 
- attn-delta phase (post-bus-return): finalize σ² using arrived attn vector

The cross term `2s Σᵢ v_known[i] · attn[i]` must be computed once attn arrives, but it's a cheap reduction over 4096 dims (~1 μs). σ² finalizes; γ scaling and division happen after.

This does NOT enable bandwidth-saving GEMV decomposition (that's the v1 mistake). It just means σ computation isn't a critical-path bottleneck.

## 5. Module structure

### 5.1 Repository layout

```
v4flash-engine/
├── Cargo.toml                  # workspace root
├── crates/
│   ├── v4flash-core/           # tensor types, GGUF loading, model graph
│   ├── v4flash-hip/            # safe wrappers around HIP runtime + cubecl-hip-sys
│   ├── v4flash-engine/         # main inference orchestration
│   ├── v4flash-server/         # OpenAI/Anthropic-compatible HTTP API
│   ├── v4flash-router-distill/ # offline distilled router head training
│   └── v4flash-cli/            # CLI and integration tests
├── kernels/                    # HIP C++ kernels, built by build.rs
│   ├── common/                 # shared headers, intrinsics
│   ├── dequant/                # IQ2_XXS, Q2_K, Q8_0 dequant + fused dequant-gemv
│   ├── attention/              # CSA, HCA, SWA, lightning indexer
│   ├── moe/                    # expert gemv (cached + uncached variants), mHC, router
│   ├── norm/                   # RMSNorm
│   └── activation/             # SiLU, fused SiLU+mul
├── tests/
│   ├── numerical/              # reference logit comparison
│   └── perf/                   # micro-benchmarks
└── build.rs                    # HIPCC orchestration
```

### 5.2 Host side (Rust)

- **v4flash-core**: GGUF parsing, tokenizer (`encoding_dsv4`), prompt rendering, hash-gate tid2eid loading, mHC matrix Sinkhorn factorization.
- **v4flash-hip**: typed wrappers around `cubecl-hip-sys`. `Device`, `Stream` (with priority), `Event`, `DeviceBuffer<T>` per-device, `KernelModule`, `HipError` enum.
- **v4flash-engine**: scheduler. Owns both `Device` handles, streams, events, the per-layer state machine. Implements speculative routing dispatch and verification.
- **v4flash-server**: OpenAI-compatible API + three-mode reasoning effort + streaming SSE.
- **v4flash-router-distill**: separate binary; collects routing decisions from a reference run, trains the small per-layer distilled heads, exports as serialized weights loaded by v4flash-engine.

Synchronization model:
- Each device has `compute` and `transfer` streams.
- Cross-device dependencies via HIP events — **verified working across dGPU↔iGPU [P0 measured]**, no host-bounce fallback needed.
- **Peer-copies must be queued on the source device's stream [P0 load-bearing rule]**. Queuing on dst-stream silently returns zeros for dGPU→iGPU. See `memory/project-peer-copy-stream-rule.md`. Hypothesis: HIP's default event-record fence scope is agent, not system, so PCIe peer reads see stale VRAM unless the source agent's DMA engine handles the copy.
- Steady-state cross-device sync cost is **~11 μs amortized [P0 measured]** (was estimated 10-30 μs).

### 5.3 Kernels (HIP C++)

Targets depend on Phase 0 outcome. Three possibilities, in order of preference:
1. Single-process: `--offload-arch=gfx1100,gfx1201` if HSA override works and gfx1100-binary-on-gfx1151 outperforms native.
2. Single-process: `--offload-arch=gfx1151,gfx1201` if HSA override breaks dual-device but native gfx1151 is workable.
3. Two-process: separate binaries per device, communicating via shared memory + Unix sockets. Use if neither single-process option works.

| Kernel family | Notes |
|---|---|
| **Dequant** | Lift from ggml-cuda. Provide both cache-friendly (normal loads) and cache-bypass (`__builtin_nontemporal_load`, `glc/slc/dlc` bits) variants. |
| **Attention** | Lift CSA/HCA/SWA/Lightning Indexer from ds4 rocm branch. Validate Q8_0 dequant performance on the two-stage attn_output structure. |
| **Expert GEMV** | Fused dequant-GEMV for IQ2_XXS, Q2_K. Two flavors: prefetch-only (touches LPDDR to warm MALL, discards results) and standard (reads + computes). |
| **MoE compute** | Sequential: read W_up, compute u_moe @ W_up; read W_gate, compute u_moe @ W_gate; SiLU+mul; read W_down, compute @ W_down. No streaming decomposition. |
| **mHC** | Sinkhorn-normalized mixing. |
| **Router** | Hash gate (tid2eid lookup), learned router (sigmoid + bias + top-k), distilled head (small MLP). |

Cache control:
- W_up/W_gate kernel variants: standard cached loads to warm MALL on prefetch; standard cached loads on actual compute (cache hit if predictor was right).
- W_down kernel: `__builtin_nontemporal_load` or per-instruction `slc=1` to bypass MALL.
- Phase 0 measures whether these hints actually affect MALL on gfx1151. If not, all cache strategies become "best-effort opportunistic."

## 6. Code we inherit from

### 6.1 ds4 rocm branch (https://github.com/antirez/ds4, branch `rocm`)
- V4 Flash architecture model graph and GGUF tensor name conventions
- Hash gate (`DeepseekV4HashGate`) lookup table loading
- CSA + HCA + Lightning Indexer algorithm (port to HIP from Metal/CUDA)
- mHC Sinkhorn formulation including `hc_split_sinkhorn` numerical recipe
- V4 Flash tokenizer / prompt encoding
- Three-mode reasoning effort dispatch
- Tool calling protocol
- Persistent on-disk KV cache layout
- HTTP server API

Note: rocm branch is community-maintained; antirez doesn't have AMD hardware. Don't rely on upstream review velocity.

### 6.2 ggml HIP backend
- Dequantization kernels for IQ2_XXS, Q2_K, Q8_0
- GEMM/GEMV templates and tiling strategies
- LDS staging patterns
- HIP/CUDA shim macros
- WMMA wrapper utilities

## 7. Build system

### 7.1 Toolchain
- ROCm 7.2.3 stable (via nixpkgs `rocmPackages`)
- HIPCC for kernel compilation
- Rust stable 1.83+ (in practice 1.95+ from nixpkgs)
- Hand-rolled FFI in `v4flash-hip` crate (subset surface; ~25 functions)
- Linux kernel ≥ 6.18.4
- **No HSA override [P0 measured]** — `HSA_OVERRIDE_GFX_VERSION=11.5.1` BREAKS the runtime (hipErrorLaunchFailure). ROCm 7.2.3 supports gfx1151 natively.

### 7.2 Per-target kernel compilation

For standalone code objects loadable via `hipModuleLoadData`, use `--genco` (NOT `-c`):

```bash
hipcc -O3 --genco --offload-arch=gfx1201 \
    kernels/foo.hip -o foo_gfx1201.hsaco

hipcc -O3 --genco --offload-arch=gfx1151 \
    kernels/foo.hip -o foo_gfx1151.hsaco
```

**[P0 measured]** `hipcc --genco` actually produces a CCOB (clang offload bundle), not a raw ELF. `hipModuleLoadData` accepts CCOB blobs directly — no unbundling needed.

**gfx1100 binary path dropped [P0 measured]**: Gate B confirmed gfx1100 on Strix iGPU is *identical* (1.00×) speed to native gfx1151 for our DRAM-bound Q8_0 GEMV workload. The "2-6×" claim from llm-tracker doesn't apply here (it was for compute-bound ops with poor gfx1151 codegen).

Embedded HSACO blobs via `include_bytes!` in Rust. Load at runtime via `hipModuleLoadData` to the appropriate device context.

### 7.3 Dependencies

```toml
[dependencies]
cubecl-hip-sys = "7.2"  # or hand-rolled bindgen
tokio = { version = "1", features = ["full"] }
bytes = "1"
serde = "1"
serde_json = "1"
axum = "0.7"
tracing = "0.1"
anyhow = "1"
thiserror = "1"
half = "2"

[build-dependencies]
cc = "1"
which = "6"
```

## 8. Implementation phases

**Phase 0 status: COMPLETE [P0 measured].** All five gates passed; full report in [`PHASE0.md`](PHASE0.md). Phase 1 is unblocked.

### Phase 0: Hardware viability gates (1 week)

Five required measurements / decisions:

**Gate A: HSA_OVERRIDE_GFX_VERSION compatibility with dual-device process.**
- Spawn a process targeting both gfx1201 (9070 XT) and gfx1151 (Strix iGPU).
- Set `HSA_OVERRIDE_GFX_VERSION=11.5.1`.
- Verify `hipGetDeviceProperties` returns correct ISA for each device.
- Launch trivial kernels on each device, check correctness.
- Decision:
  - All works → single-process design.
  - 9070 XT reports wrong ISA → drop override; use native gfx1151; lose the potential gfx1100-binary speedup.
  - Kernels misbehave on one device → two-process design.

**Gate B: gfx1100-on-gfx1151 performance characterization.**
- If Gate A allows override, build a representative MoE kernel (IQ2_XXS dequant-GEMV) for both gfx1151 and gfx1100. Run both on the Strix iGPU. Measure tokens/sec on the actual V4 Flash routed-expert workload.
- The "2-6×" claim from llm-tracker is for unspecified ops. Validate on our workload before committing.
- Decision: if gfx1100-via-override is meaningfully faster (>30%), use it. Otherwise native gfx1151.

**Gate C: Cross-device peer access and events.**
- `hipDeviceCanAccessPeer(egpu, igpu)` and reverse.
- `hipDeviceEnablePeerAccess`.
- `hipMemcpyPeerAsync` correctness and bandwidth between 9070 XT VRAM and Strix UMA.
- `hipStreamWaitEvent` for events recorded on the other device's stream.
- Cache coherency: does iGPU see writes done by dGPU after the event? Or are explicit cache flushes needed?
- Decision:
  - All works → direct peer transfers, ~10-30 μs sync overhead.
  - Anything fails → host-pinned bounce buffer design, ~50-100 μs sync overhead per transfer.

**Gate D: MALL behavior empirical characterization.**
- Write a kernel that reads a buffer of size X, then immediately re-reads it. Measure speedup as function of X. Identifies effective MALL capacity for our workload.
- Same test with intervening unrelated memory traffic (~16 MB write). Measures eviction sensitivity.
- Test cache-bypass hints: do `__builtin_nontemporal_load` and `slc=1` actually bypass MALL on gfx1151?
- Decision:
  - MALL holds ~30 MB resident under concurrent traffic → cache strategy in §4.3 works.
  - MALL evicts aggressively or hints don't work → fall back to "no cache assumption" design; performance target drops to ~45 tok/s.

**Gate E: Effective fused-dequant-GEMV bandwidth.**
- Measure actual sustained throughput on IQ2_XXS, Q2_K, Q8_0 batch-1 GEMV on both devices.
- Compare against theoretical (215 GB/s Strix, 644 GB/s 9070 XT).
- Scale all bandwidth-derived estimates by the efficiency ratio.
- Decision: if effective bandwidth <60% of theoretical, all perf targets shift down proportionally.

**Other Phase 0 work:**
- Verify `--genco` produces blobs that `hipModuleLoadData` accepts.
- Build hello-world ping-pong between devices, characterize actual cross-device latency.
- Replace soft numbers in §2 with measured numbers.

**Phase 0 exit criteria — MET [P0 measured]:**
- §2 bandwidth/latency numbers measured: ✓ (see PHASE0.md tables)
- §4.3 cache strategy: REVISED — MALL effective 24 MB (not 32), top-N = 6 (not 8), non-temporal bypass for W_down confirmed working
- §5.3 kernel target list locked: ✓ — gfx1201 + gfx1151 native only; gfx1100 path dropped (no speedup)
- Architecture decision: **single-process, no override**

### Phase 1: Single-device naive decode (1 week)
- Load V4 Flash GGUF on Strix iGPU only. Ignore the 9070 XT.
- Non-streaming kernels: separate up/gate/down_proj passes, separate attention with regular GEMM, naive routing (no speculation).
- Use ggml dequant kernels directly.
- Validate numerical correctness against reference logits (greedy match for first 50 tokens of a fixed prompt; KL <0.01 nats/token over first 100 token distributions).
- Target: correct tokens. Speed irrelevant.

### Phase 2: Heterogeneous split, no cache strategy (1-2 weeks)
- Move attention + shared expert to 9070 XT. Routed experts on Strix iGPU.
- Synchronous handoff: full hidden state, single transfer per direction.
- Two streams per device, events for cross-device deps.
- Stream replication on both devices.
- Sequential MoE compute (no speculation, no MALL prefetch).
- Validate numerical match continues.
- Measure tok/s. Expected: 25-35 tok/s.

### Phase 3: Speculative routing + cache strategy (3-4 weeks) ← **load-bearing phase**
- Train distilled router heads offline using v4flash-router-distill.
- Implement exact-with-fallback speculative routing: prefetch top-N (N=7 or 8), verify against real router, fall back to synchronous read on miss.
- Implement MALL prefetch for W_up/W_gate during attention window.
- Implement non-temporal/cache-bypass for W_down streaming.
- Numerical validation: KL <0.01 nats/token, top-5 set match 100%, greedy top-1 match ≥90% for first 50 tokens.
- Target: 55 tok/s default, 65 tok/s stretch.

### Phase 4: Profile, tune, fix what Phase 3 didn't deliver (1-2 weeks)
- Phase 4 in v1 was speculative routing, which is now in Phase 3.
- New Phase 4 is targeted optimization based on Phase 3 profiling: kernel fusion opportunities, scheduler tuning, addressing whatever bottleneck the profiler surfaces.
- Optional optimizations to evaluate: chunked hidden-state transfer (if bus latency is meaningful), partial-overlap shared expert (if shared-expert kernel can squeeze into the bus-return window), KV cache compression for HCA layers (orthogonal to bandwidth work).
- Target: 65 tok/s with hardened reliability.

### Phase 5: MTP-based speculative decoding (1-2 weeks)
- V4 Flash ships with MTP heads. Use them.
- Draft batch of K=3-4 tokens, verify in single forward pass with batch=K.
- Realistic speedup: 1.3-1.6× depending on acceptance rate (NOT 2× as v1 claimed).
- Target: 75-90 tok/s effective.

### Phase 6: Prefill (1-2 weeks)
- Reuse Phase 1-5 building blocks in batch>>1 configuration.
- Different optimal kernels: compute-bound, want WMMA, want large GEMMs (not GEMVs).
- Attention prefill needs full KV cache write path; reuse ds4's structure.
- Target: ≥500 tok/s prefill on representative prompts.

Total estimated effort: ~10-14 weeks of focused engineering, with Phase 0 and Phase 3 the highest-risk phases.

## 9. Performance targets

| Metric | Floor | Target | Stretch |
|---|---|---|---|
| Phase 0 viability | N/A | All 5 gates pass | gfx1100-via-override works |
| Phase 1 (naive) | tokens correct | tokens correct | N/A |
| Phase 2 (heterogeneous) | 20 tok/s | 30 tok/s | 35 tok/s |
| Phase 3 (cache + speculation) | 45 tok/s | 55 tok/s | 65 tok/s |
| Phase 5 (MTP) | 60 tok/s | 75 tok/s | 90 tok/s |
| Phase 6 (prefill) | 300 tok/s | 500 tok/s | 1000 tok/s |
| KL divergence vs reference | <0.05 | <0.01 | <0.005 |
| Top-5 set match (first 50 tokens) | 100% | 100% | 100% |
| Top-1 greedy match (first 50 tokens) | ≥85% | ≥90% | ≥95% |
| Time-to-first-token (4k prompt) | <5 s | <3 s | <1 s |
| KV cache @ 100k context | <4 GB | <3 GB | <2 GB |

Note on numerical match: exact greedy match for all 50 tokens (v1 target) is unrealistic due to FP16 accumulation order differences vs Metal reference. Top-5 set match catches material divergence; ≥90% greedy match catches numerical regressions.

## 10. Testing

### 10.1 Numerical correctness
- Reference: ds4 Metal output on M3 Max with identical quant. Generate logits for ~10 fixed prompts at multiple context lengths (1k, 16k, 128k).
- Per-phase gate: KL divergence <0.01 nats/token across first 100 tokens, top-5 set match 100%, greedy top-1 ≥90% first 50 tokens.
- Per-kernel unit tests: dequant kernels checked against CPU reference dequant; GEMV checked against rocBLAS reference.
- mHC Sinkhorn iteration: validate doubly-stochastic property.
- Speculative routing correctness invariant: when speculation succeeds AND when it fails, output must be identical (modulo FP accumulation order). Test by forcing artificial miss rates and verifying logit invariance.

### 10.2 Performance profiling
- HIP profiler (`rocprof` or in-kernel event timing) for per-stage wall clock.
- Track each phase of §4.2 dataflow; identify divergence from estimates.
- Hardware counters where available; sysfs scrape where `amdsmi` doesn't support gfx1151.
- Per-layer breakdown logged at debug level; aggregate latency histogram per session.

### 10.3 Integration
- Comparison against DeepSeek API on ~100 prompts spanning code, math, multi-turn, long-context.
- Claude Code-style agent loop test: small repo modification task, verify completion.
- Long-context regression: 500k token prompt, verify no quality cliff.

## 11. Open questions (investigated during Phases 0-3)

### Phase 0 resolves:
- HSA override behavior on dual-device process
- Cross-device peer access and event semantics
- MALL replacement policy under our access pattern
- Effective vs theoretical GEMV bandwidth on both devices
- gfx1100-via-override correctness for our kernel set

### Phase 3 resolves:
- Distilled router prediction hit rate on real workload (need ≥95% for cache strategy to pay off)
- Whether Sinkhorn router output streams empirically specialize (would enable additional optimizations; if not, mHC adds overhead with no payoff beyond stability)
- W_up/W_gate vs W_down cache choice (the analysis says W_up/W_gate but Phase 0 measurements may flip this)

### Later:
- Whether quantization recipe can improve (shared expert is on 9070 XT; reducing its precision saves 9070 XT VRAM bandwidth, not Strix LPDDR5X — clarification from v1 which had this inverted)
- Routed expert popularity: top-K experts handle disproportionate fraction of routing decisions. If hit rate is high enough, caching popular experts in MALL across layers might pay; needs measurement.
- Whether MTP draft model fits the remaining 9070 XT VRAM budget (~2-3 GB) usefully

## 12. Out of scope (explicitly)

- Multi-user concurrency
- Multi-instance / distributed inference
- Windows support
- ROCm < 7.2 or > 7.5 compatibility (pin toolchain)
- Quantization-aware training, fine-tuning
- Models other than V4 Flash (and V4 Pro by extension if quantization fits)
- Mobile or embedded targets
- Per-request KV cache eviction policies more sophisticated than LRU

## 13. References

- DeepSeek V4 technical report: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
- antirez/ds4 (main + rocm branch): https://github.com/antirez/ds4
- ggml HIP backend: https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda
- cubecl-hip-sys: https://github.com/tracel-ai/cubecl-hip-sys
- mHC paper: arXiv:2512.24880
- Hyper-Connections paper: arXiv:2409.19606
- Chipsandcheese Strix Halo Infinity Cache analysis: https://chipsandcheese.com/p/evaluating-the-infinity-cache-in
- Strix Halo ROCm setup guide: https://github.com/kyuz0/amd-strix-halo-toolboxes

