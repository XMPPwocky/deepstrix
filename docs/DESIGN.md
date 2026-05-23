# V4 Flash Heterogeneous Inference Engine — Design Doc

## 1. Scope

### Goal
A high-throughput single-token decode engine for DeepSeek V4 Flash on a specific heterogeneous AMD setup: AMD Radeon RX 9070 XT (eGPU, RDNA 4, gfx1201) over Oculink + AMD Ryzen AI Max+ 395 with Radeon 8060S (iGPU, RDNA 3.5, gfx1151) and 128 GB unified LPDDR5X. Target ≥60 tokens/sec on V4 Flash at antirez's IQ2_XXS / Q2_K / Q8_0 mixed quantization.

### Non-goals (initial)
- Prefill. Architecture must permit efficient prefill later, but Phase 1 ships decode-only.
- Generality across models. This engine is V4 Flash-specific. Architectural changes that would help arbitrary models but cost decode performance are out of scope.
- Cross-platform. Linux + ROCm 7.2.3 only.
- Training, fine-tuning, gradient computation.

### Stretch goals
- MTP-based speculative decoding (V4 Flash ships with MTP heads).
- Speculative expert routing via distilled router prediction heads.
- Persistent on-disk KV cache (V4 supports this and ds4 already does it).
- Long-context (1M token) inference with CSA/HCA's compressed KV.

## 2. Hardware

### 2.1 9070 XT — eGPU (attention + shared expert + KV)

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
| VRAM bandwidth | 644 GB/s |
| Infinity Cache (3rd gen) | 64 MB |
| TDP | 304 W |
| ROCm target | gfx1201, officially supported in ROCm 7.0.2+ |

### 2.2 Strix Halo — host + iGPU (routed experts + KV staging + CPU coordination)

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
| LPDDR5X bandwidth | 256 GB/s theoretical, ~212-215 GB/s measured |
| Allocatable to iGPU | up to 96 GB (UMA setting) |
| Infinity Cache (MALL) | 32 MB, GPU-side fills only |
| CPU↔iGPU memory copy | ~84 GB/s measured |
| TDP | configurable 45-120W |
| ROCm target | gfx1151 (NOT in official support matrix); use HSA_OVERRIDE_GFX_VERSION=11.5.1 with ROCm 7.2.3 |

Notes on Strix Halo ROCm support:
- Not officially in the support matrix; runs via override.
- Use ROCm 7.2.3 stable (not nightlies; nightlies cap allocation at 64 GB which is below our needs).
- Kernel must be ≥ 6.18.4 for stable amdgpu driver.
- `linux-firmware-20251125` is broken on Strix Halo — pin older.
- Empirically, gfx1100 kernels run 2-6× faster than gfx1151 kernels on this hardware. Compile primary path with `--offload-arch=gfx1100` and run via HSA override; keep gfx1151-native as fallback for ops where gfx1100 doesn't work.

### 2.3 Interconnect — Oculink (PCIe 4.0 x4)

| Property | Value |
|---|---|
| Theoretical bandwidth | 64 Gbps = 8 GB/s |
| Effective bandwidth | ~7 GB/s after PCIe overhead |
| Per-transfer setup latency | ~1-5 μs (no Thunderbolt protocol translation) |
| Round-trip event sync | ~10-15 μs (HIP event signaling, cross-device) |

### 2.4 Implications for design

- Single-token decode is **bandwidth-bound on weight reads**, not compute. The TFLOPS columns above matter much less than the GB/s columns. Both devices have far more compute than they can feed at batch=1.
- The compute headroom enables: parallel kernels (attention + shared expert on 9070 XT), speculative work (draft model, router prediction), prefill (batch >> 1).
- Cross-device synchronization adds ~10-15 μs per dependency edge. Minimize sync points; batch transfers where possible.
- FP8 only exists on 9070 XT. Any FP8-using kernels must live there. iGPU side stays on FP16/BF16/Q-quantized.
- Oculink is latency-bound, not throughput-bound, for the ~32 KB/layer of cross-device traffic we expect.

## 3. Target Model: DeepSeek V4 Flash

### 3.1 Architecture summary
- 43-layer all-MoE backbone (no dense MLPs)
- 256 routed experts + 1 shared expert per block; top-6 routing
- First 3 blocks: `DeepseekV4HashGate` deterministic routing via tid2eid lookup
- Remaining 40 blocks: learned router with sigmoid + auxiliary-loss-free bias
- Hybrid attention: SWA for first 2 layers; then alternating CSA / HCA
- CSA: compress KV 4×, Lightning Indexer selects top-1024 compressed blocks, sliding-window branch for local tokens
- HCA: compress KV 128×, dense attention over compressed sequence
- Manifold-Constrained Hyper-Connections (mHC) with `hc_mult=4` streams
- Sinkhorn-Knopp normalized mixing matrices
- 284B total params, 13B activated per token
- 1M token context (CSA/HCA give 7% of V3.2 KV at 1M)

### 3.2 Per-block tensor inventory (representative CSA block)
| Component | Tensor | Shape | Dtype | Size |
|---|---|---|---|---|
| Attention | attn_q_a / q_b | [4096,1024] / [1024,32768] | Q8_0 | 4 + 32 MB |
|   | attn_kv | [4096,512] | Q8_0 | 2 MB |
|   | attn_compressor_{kv,gate,ape} | various | F16 | 8 MB |
|   | attn_output_a / b | [4096,8192] / [8192,4096] | Q8_0 | 64 MB |
|   | norms, sinks, kv_a_norm | various | F32 | <100 KB |
| **Attn subtotal** | | | | **~114 MB / layer** |
| Routed MoE | ffn_gate_exps, ffn_up_exps | [4096,2048,256] | IQ2_XXS | 555 MB each |
|   | ffn_down_exps | [2048,4096,256] | Q2_K | 705 MB |
|   | ffn_gate_inp | [4096,256] | F16 | 2 MB |
|   | per-expert (×256): W_up + W_gate + W_down | | | **~7.1 MB / expert** |
| Shared | ffn_{gate,up,down}_shexp | [4096,2048] or [2048,4096] | Q8_0 | 8.4 MB each |
| **Shared subtotal** | | | | **~25 MB / layer** |
| HC | hc_attn_fn, hc_ffn_fn | [16384,24] | F16 | 768 KB each |
| **HC subtotal** | | | | **~1.5 MB / layer** |
| Active per token | attn + 6×routed + shared | | | **~182 MB / layer** |

Total model size (at this quant mix): ~83 GB. Plus embeddings, LM head, vocab tables: ~85 GB.

## 4. Design Overview

### 4.1 Device roles

**9070 XT (resident):**
- All 43 layers of attention weights (~4.9 GB)
- All 43 layers of shared expert weights (~1.1 GB)
- KV cache for all layers (1-3 GB depending on context)
- HC mixing matrices replicated (~65 MB)
- Lightning Indexer scratch buffers, sparse gather temporaries (~500 MB)
- Stream replica for current layer (32 KB)
- Embedding + LM head (~1 GB if tied)
- Optional: MTP draft model state (~2 GB if enabled)
- **Total: ~9-10 GB used of 16 GB**

**Strix Halo (resident):**
- All 43 layers × 256 routed expert weights (~70 GB at IQ2 mix)
- Router weights (`ffn_gate_inp`), bias arrays, norms
- HC mixing matrices (canonical copy)
- Stream replica (canonical, 32 KB live)
- Distilled router prediction heads (~50 MB total across 43 layers)
- Persistent KV cache (on-disk staging, in-RAM hot pages)
- Tokenizer state, sampler state, control flow
- **Total: ~75 GB used of 96 GB GPU-addressable**

### 4.2 Per-layer dataflow

For one non-hashgate CSA layer:

1. **t=0**: Strix has streams[t-1]. Distilled router head r_ℓ runs on Strix from previous block's hidden state → predicted top-6 expert IDs.
2. **t=0-40**: Strix→9070XT, send mHC_attn_pre input (4096-dim, chunked) over Oculink. 9070XT begins streaming QKV projection as chunks arrive.
3. **t=5-200**: 9070XT runs attention (CSA: indexer + sparse gather + softmax + attn_output stages a then b). In parallel, **shared expert** computation on 9070XT (Wup + Wgate + SiLU + Wdown for the shared FFN) overlapped via second compute queue.
4. **t=0-200**: Strix streams W_up, W_gate for predicted routed experts from LPDDR5X. Accumulates `v_known @ W_up_e` and `v_known @ W_gate_e` partial sums (mHC linearity decomposition).
5. **t=200-220**: 9070XT→Strix sends attention output AND shared expert output (two transfers, ~16 KB total).
6. **t=120-200** (concurrent with above): Strix has idle LPDDR5X bandwidth; **prefetch routed W_down (~16.6 MB) into Infinity Cache MALL**.
7. **t=220-240**: Strix computes attention delta against W_up/W_gate (small), finalizes σ (RMSNorm denominator), applies SiLU(gate)*up = h_intermediate.
8. **t=240-260**: Strix computes `h_intermediate @ W_down` for routed experts (W_down resident in MALL → fast read).
9. **t=260-275**: Strix combines routed outputs with received shared expert output via router gates. mHC_ffn_post updates streams.
10. **t=275-285**: Strix→9070XT sends updated streams for next layer's attention input.

**Per-layer total: ~285-310 μs steady state.** Per-token: ~12-13 ms across 43 layers ≈ **75-85 tok/s**.

### 4.3 Streaming compute key invariant

The MoE input u_moe is decomposable:
$$u_{moe} = \frac{\gamma}{\sigma} \cdot (v_{known} + s \cdot \text{attn})$$

Where `v_known` is computable at t=0 from streams[t-1] alone (no attention needed). The up_proj for any expert e becomes:
$$u_{moe} @ W_{up}^{(e)} = \frac{1}{\sigma} \left[ (v_{known} \odot \gamma) W_{up}^{(e)} + s (\text{attn} \odot \gamma) W_{up}^{(e)} \right]$$

The first term streams through expert weights during attention; the second is a small delta after bus return. This is the load-bearing optimization. RMSNorm's σ also accumulates streamingly.

### 4.4 Cache strategy

- 9070 XT VRAM: holds attention weights + shared expert + KV resident throughout inference.
- Strix LPDDR5X: holds routed expert weights resident.
- Strix Infinity Cache (32 MB MALL): used for **W_down prefetch only**. W_up/W_gate stream through with non-temporal hints to bypass MALL (via `hipMemAllocationTypeUncached` or equivalent). Routed W_down for 6 experts = ~16.6 MB fits with headroom; remaining ~15 MB free for opportunistic uses.
- 9070 XT Infinity Cache (64 MB): used for attention weight reuse across consecutive layers (much larger than per-layer attention weights, but at single-token decode each layer's weights are touched once per token regardless, so this is less critical).

## 5. Module structure

### 5.1 Repository layout

```
v4flash-engine/
├── Cargo.toml                  # workspace root
├── crates/
│   ├── v4flash-core/           # tensor types, GGUF loading, model graph
│   ├── v4flash-hip/            # safe wrappers around HIP runtime + cubecl-hip-sys
│   ├── v4flash-engine/         # main inference orchestration, scheduling, MoE coordination
│   ├── v4flash-server/         # OpenAI/Anthropic-compatible HTTP API
│   └── v4flash-cli/            # CLI and integration tests
├── kernels/                    # HIP C++ kernels, built by build.rs
│   ├── common/                 # shared headers, intrinsics
│   ├── dequant/                # IQ2_XXS, Q2_K, Q8_0 dequant + fused dequant-gemm
│   ├── attention/              # CSA, HCA, SWA, lightning indexer
│   ├── moe/                    # streaming up/gate, down_proj, mHC, router
│   ├── norm/                   # RMSNorm (streaming + standard)
│   └── activation/             # SiLU, fused SiLU+mul
├── tests/
│   ├── numerical/              # bit-level checks against reference logits
│   └── perf/                   # micro-benchmarks
└── build.rs                    # HIPCC orchestration
```

### 5.2 Host side (Rust)

Crate responsibilities:

- **v4flash-core**: GGUF parsing (V4-specific tensor names), tokenizer (`encoding_dsv4`), prompt rendering, hash-gate tid2eid loading, mHC matrix Sinkhorn factorization.
- **v4flash-hip**: typed wrappers around `cubecl-hip-sys`. `Device`, `Stream` (with priority), `Event`, `DeviceBuffer<T>` (newtypes per-device so the compiler catches cross-device pointer mistakes), `KernelModule` for loaded .hsaco blobs, `HipError` enum with the ~10 errors we actually see + `Unknown(hipError_t)`.
- **v4flash-engine**: the actual scheduler. Defines two `Device` handles (egpu, igpu) and the per-layer state machine. Owns the streams, events, and the coordination logic. Implements speculative routing (distilled head dispatch, LRT decision on partial logits).
- **v4flash-server**: OpenAI-compatible Chat Completions API + ds4-style three-mode reasoning effort + streaming SSE.
- **v4flash-cli**: harness for benchmarks, numerical validation, KV cache dump/restore.

Synchronization model:
- Each device gets two streams: `compute` and `transfer`.
- Cross-device dependencies are HIP events. Bus transfers use `transfer` stream; kernel launches use `compute` stream; both wait on events from the other device's stream as needed.
- Host-side coordination is async Rust (Tokio runtime) but kernel dispatch is synchronous within a layer — async exists for the HTTP server, not for inner loop.

### 5.3 Kernels (HIP C++)

Kernels are dispatched to specific gfx targets at build time. Use `--offload-arch=gfx1100,gfx1201` for cross-target HSACO; the gfx1100 binary runs on gfx1151 under HSA override and is empirically faster than gfx1151-native.

| Kernel family | Files | Notes |
|---|---|---|
| **Dequant** | `iq2_xxs.hip`, `q2_k.hip`, `q8_0.hip` | Lift directly from ggml-cuda. Fused-with-GEMM variants for streaming consume. |
| **Attention** | `csa.hip`, `hca.hip`, `swa.hip`, `lightning_indexer.hip` | Lift from ds4 rocm branch as base. Heavily modify CSA for chunked streaming QKV projection. |
| **MoE streaming** | `streaming_up_gate.hip`, `down_proj.hip`, `router.hip`, `mhc_mix.hip` | Mostly new code. Streaming kernels are the load-bearing custom work — see §4.3. |
| **RMSNorm** | `rmsnorm.hip`, `rmsnorm_streaming.hip` | Streaming variant accumulates σ as chunks arrive, finalizes on signal. |
| **Activation** | `silu_mul.hip` | Fused SiLU(gate) * up elementwise. |
| **Router** | `hash_gate.hip`, `learned_router.hip`, `distilled_head.hip` | Hash gate: tid2eid lookup, trivial. Learned router: sigmoid + bias + top-k. Distilled head: small MLP. |

Cache control intrinsics used (RDNA 3.5/4):
- `__builtin_amdgcn_global_load_lds_*` for direct LDS fills
- Inline asm with `glc`, `slc`, `dlc` bits for cache-bypass on W_up/W_gate streaming reads (RDNA 4 uses these slightly differently from RDNA 3; per-arch defines)
- `__builtin_nontemporal_store` for outputs that won't be re-read this layer

## 6. Code we inherit from

### 6.1 ds4 rocm branch (https://github.com/antirez/ds4, branch `rocm`)

Take as base:
- V4 Flash architecture model graph
- GGUF tensor name conventions and loader
- Hash gate (`DeepseekV4HashGate`) lookup table loading
- CSA + HCA + Lightning Indexer implementation (Metal-translated to HIP; the algorithmic structure ports cleanly even where the kernel code doesn't)
- mHC Sinkhorn formulation including the `hc_split_sinkhorn` numerical recipe from the released code
- Prompt encoding for V4 Flash tokenizer
- Three-mode reasoning effort dispatch (Non-think, Think, Think Max)
- Tool calling protocol
- Persistent on-disk KV cache layout
- HTTP server API (OpenAI + Anthropic compatible)

Don't take:
- Their dispatch loop (we're heterogeneous, they're single-device)
- Their KV cache memory layout (we have different cache hierarchies)
- Their MoE expert dispatch (we're doing streaming-and-compute, they're not)

The rocm branch is community-maintained because antirez doesn't have AMD hardware. Plan to upstream improvements as patches but don't expect upstream review velocity to be useful.

### 6.2 ggml HIP backend (`ggml/src/ggml-cuda` with HIP shims)

Take as base:
- Dequantization kernels for IQ2_XXS, Q2_K, Q8_0 (battle-tested, fast)
- GEMM templates and tiling strategies for HIP
- LDS staging patterns and bank-conflict-free indexing
- HIP/CUDA shim macros (`ggml-cuda/vendors/hip.h`)
- WMMA wrapper utilities

Don't take:
- The graph executor (we have a fixed model graph, no need for general dispatch)
- The memory allocator (we manage memory explicitly per-device)
- The KV cache implementation (V4 needs specialized compressed KV)

## 7. Build system

### 7.1 Toolchain
- ROCm 7.2.3 stable, installed to `/opt/rocm`. Verify with `rocminfo | grep -E 'gfx115|gfx120'`.
- HIPCC for kernel compilation.
- Rust stable (1.83+), `cargo` for host crates.
- `bindgen` for HIP runtime FFI (cubecl-hip-sys provides current bindings; regenerate from ROCm 7.2.3 headers if needed).
- Linux kernel 6.18.4 minimum.
- Set `HSA_OVERRIDE_GFX_VERSION=11.5.1` in env for Strix Halo iGPU.

### 7.2 Per-target kernel compilation

`build.rs` invokes HIPCC for each kernel file with:
```
hipcc -O3 -fgpu-rdc \
  --offload-arch=gfx1100 \
  --offload-arch=gfx1201 \
  -c kernel.hip -o kernel.hsaco
```

For gfx1151 (Strix iGPU) specifically: ship gfx1100 binary, run under HSA override (empirically 2-6× faster than native gfx1151). Provide gfx1151-native fallback path for any kernel where gfx1100 binary doesn't execute correctly.

Embedded HSACO blobs via `include_bytes!` in Rust modules; load at runtime via `hipModuleLoadData` to the appropriate device context.

### 7.3 Dependencies

Host:
```toml
cubecl-hip-sys = "7.2"  # or hand-rolled bindgen against ROCm 7.2.3
tokio = { version = "1", features = ["full"] }
bytes = "1"
serde = "1"
serde_json = "1"
axum = "0.7"          # HTTP server
tracing = "0.1"
anyhow = "1"
thiserror = "1"
half = "2"
```

Build:
```toml
[build-dependencies]
cc = "1"               # for non-HIP C/C++ glue
which = "6"            # locate hipcc
```

## 8. Implementation phases

Each phase is a checkpoint with working end-to-end inference at degraded performance. Don't build phase N before phase N-1 ships and validates numerically.

### Phase 0: Bring-up (1-2 days)
- Verify ROCm 7.2.3 works on both devices, `rocminfo` sees both, `rocm-bandwidth-test` confirms specs.
- Build hello-world HIP kernel, dispatch from Rust via cubecl-hip-sys, verify cross-device events fire correctly.
- Build cross-device round-trip ping-pong benchmark; measure actual Oculink latency and bandwidth.
- Measure actual LPDDR5X bandwidth on Strix iGPU (`rocm_bandwidth_test`), MALL hit/miss latency.
- Measure 9070 XT VRAM bandwidth.
- Replace soft numbers in §2 with measured numbers.

### Phase 1: Single-device naive decode (1 week)
- Load V4 Flash GGUF on Strix iGPU only. Ignore the 9070 XT.
- Implement non-streaming kernels: separate up/gate/down_proj passes, separate attention with regular GEMM, naive routing.
- Use ggml dequant kernels directly; no fused dequant-gemm yet.
- Validate numerical correctness against reference logits (ds4 generates these; check token-by-token match for the first 100 tokens of a fixed prompt).
- Target: anything that produces correct tokens. Speed is irrelevant in this phase.

### Phase 2: Heterogeneous split, no streaming (1-2 weeks)
- Move attention + shared expert to 9070 XT. Routed experts stay on iGPU.
- Synchronous handoff: full hidden state crosses bus, no chunking.
- Two streams per device, one event per cross-device dependency.
- Stream replication on both devices, with the "m" transfer (routed MoE output → 9070 XT) at end of each layer.
- Validate numerical match continues.
- Measure tok/s. Target: 20-30 tok/s. Identifies bottlenecks for Phase 3.

### Phase 3: Streaming compute (2-3 weeks) ← **load-bearing phase**
- Chunked hidden state transfer (4-8 KB chunks).
- Streaming QKV projection on 9070 XT (accumulate as chunks arrive).
- Streaming v_known @ W_up/W_gate on Strix during attention window.
- Streaming W_down with W_down prefetch into MALL.
- Cache control: non-temporal hints on W_up/W_gate, normal loads on W_down.
- Concurrent kernel scheduling on 9070 XT (attention + shared expert).
- Target: ≥60 tok/s.

### Phase 4: Speculative routing (1-2 weeks)
- Distilled router heads trained offline against full router outputs (per-layer, ~1M params each).
- Predict top-6 from previous-layer hidden state; start expert prefetch immediately.
- Optional LRT refinement against partial router logits as bus chunks arrive (skip first pass; only add if Phase 4 measurement justifies).
- Target: +10-15% over Phase 3.

### Phase 5: MTP-based speculative decoding (1-2 weeks)
- V4 Flash ships with MTP heads. Use them.
- Draft batch of K=4 next tokens, verify in single forward pass with batch=K.
- Schedule the verification pipeline-parallel across layers (batch>1 unlocks real pipelining).
- Target: ~2× over Phase 4 in average-case decoding.

### Phase 6: Prefill (1-2 weeks)
- Reuse the Phase 1-5 building blocks but in batch>>1 configuration.
- Different optimal kernel choice for prefill: compute-bound, want WMMA, want large GEMMs.
- Attention prefill needs full KV cache write path; reuse ds4's cache structure.
- Target: ≥500 tok/s prefill on representative prompts.

Total estimated effort: ~8-12 weeks of focused engineering, with Phase 3 being the most uncertain.

## 9. Performance targets

| Metric | Target | Stretch |
|---|---|---|
| Single-token decode (Phase 3) | 60 tok/s | 80 tok/s |
| Single-token decode (Phase 5 MTP) | 100 tok/s | 150 tok/s |
| Prefill throughput (Phase 6) | 500 tok/s | 1000 tok/s |
| Time-to-first-token (4k prompt) | <3 s | <1 s |
| Numerical: KL(ours ‖ ref) on first 100 tokens | <0.01 nats/token | <0.001 |
| Numerical: exact top-1 token match for first 50 greedy tokens | 100% | 100% |
| KV cache @ 100k context | <3 GB | <2 GB |

## 10. Testing

### 10.1 Numerical correctness
- Reference: ds4 Metal output on M3 Max with same quant. Generate logits for ~10 fixed prompts at multiple context lengths (1k, 16k, 128k).
- Each phase ships only when first 100 greedy tokens match exactly and KL divergence on full distribution stays under threshold.
- Per-kernel unit tests: dequant kernels checked against reference dequant (CPU implementation in Rust); GEMM checked against rocBLAS reference.
- mHC Sinkhorn iteration: validate doubly-stochastic property of resulting H matrix.

### 10.2 Performance
- Per-layer wall-clock breakdown via HIP profiler (`rocprof` or in-app event timing).
- Track each phase of the per-layer dataflow (§4.2) and identify where reality diverges from the design doc estimates.
- A/B test cache-bypass hints by running with and without; measure MALL pressure with hardware counters where available (note: amdsmi doesn't fully support gfx1151, so some metrics come from sysfs scraping).
- Microbenchmark Oculink under contention: bus throughput with concurrent compute on both devices.

### 10.3 Integration
- Compare output against DeepSeek API for V4 Flash on a curated test set (~100 prompts spanning code, math, multi-turn, long-context).
- Coding agent loop test: drop into a Claude Code-like harness, run a small repo modification task, verify it completes.
- Long-context regression: 500k token prompt, verify no quality cliff.

## 11. Open questions

To investigate during Phase 0/1:
- **Actual MALL bandwidth on Strix Halo.** Chipsandcheese measured fabric and DRAM bandwidth, not MALL hit bandwidth specifically. Need direct measurement before trusting the ~1.5 TB/s estimate that the W_down prefetch optimization depends on.
- **Whether non-temporal hints actually bypass MALL on gfx1151.** ROCm 7.0.2 added `hipMemAllocationTypeUncached` for allocation; less clear if per-load `glc/slc/dlc` bits propagate to MALL through the iGPU's memory hierarchy. May need experimentation.
- **Practical Oculink round-trip latency under concurrent traffic.** All measurements I've seen are quiet-bus. Real workload has 4+ transfers per layer.
- **gfx1100-binary-on-gfx1151 correctness.** The "2-6× faster" finding from llm-tracker is for specific kernels; need to verify our kernel set works correctly under HSA override. Some intrinsics may differ.
- **Whether the attn_output_a / attn_output_b two-stage structure is something we can fuse**, or if it has a non-obvious dependency we have to honor as-is.

To investigate during Phase 3:
- **Actual L2 hit rate on consecutive expert reads.** If experts are touched twice in a layer (rare but possible during MTP verification), L2 caching matters and changes the design.
- **Sinkhorn router output specialization.** If empirically the streams ARE specialized (mostly carrying different kinds of info), the speculative cross-stream parallelism becomes worth trying. Measure first.

To investigate later:
- **Whether antirez's quantization recipe can be improved.** The shared expert at Q8_0 contributes ~25 MB/layer; if Q6_K is acceptable quality-wise, that's 19 MB/layer; Q4_K = 13 MB/layer. Each tier saves ~25 μs/layer of LPDDR5X bandwidth.
- **Whether we can fit more on 9070 XT VRAM.** We have ~6 GB headroom; a draft model for MTP, a router prediction model, possibly even some popular routed experts (top 5% of experts by activation frequency).

## 12. Out of scope (explicitly)

- Multi-user concurrency. Single user, single conversation at a time.
- Multi-instance / distributed inference. One Strix box, one 9070 XT.
- Windows support. Linux only.
- ROCm < 7.2 or > 7.5 compatibility. Pin the toolchain version.
- Quantization-aware training, fine-tuning.
- Models other than V4 Flash (and V4 Pro by extension if quantization fits).
- Mobile or embedded targets.

## 13. References

- DeepSeek V4 technical report: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
- antirez/ds4 (main + rocm branch): https://github.com/antirez/ds4
- ggml HIP backend: https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda
- cubecl-hip-sys: https://github.com/tracel-ai/cubecl-hip-sys
- mHC paper: arXiv:2512.24880
- Hyper-Connections paper: arXiv:2409.19606
- Chipsandcheese Strix Halo Infinity Cache analysis: https://chipsandcheese.com/p/evaluating-the-infinity-cache-in
- Strix Halo ROCm setup guide: https://github.com/kyuz0/amd-strix-halo-toolboxes
