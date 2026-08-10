# Plan: Add Laguna-S-2.1 as deepstrix's second model

Status: **APPROVED v3** (architect review 3 rounds; all blockers/majors/must-fixes resolved). Ready for Phase 0.
Branch: `laguna-model-support`

---
## 0. OUTCOME (2026-07-23) — SHIPPED on `laguna-spike`, merged to `master`

Laguna-S-2.1 runs end-to-end (prefill + decode, het dGPU/iGPU split), oracle-parity
clean. **The Phase-1 spike stayed disciplined (purely additive: 44 new files, 0
edits to ds4 shared code — config.rs/attention.rs/het/* untouched), so it became the
product rather than being thrown away.** The plan's "delete the spike, reimplement in
Phase 2" is therefore superseded — merging the additive spike cannot regress ds4, so
it was merged as-is. The **Model-trait/ModelConfig refactor (Phase 2) is deferred as
optional cleanliness** (dedup the het pipeline across ds4+Laguna); it enables nothing
new and is not a correctness/merge gate.

**Performance (measured, this HW; SWA-on, B_MAX=512, pipelined dGPU∥iGPU):**
- Prefill: 490 @4K · 467 @16K · 443 @32K · 362 @64K · **298 @100K** tok/s.
- Decode: ~26.7 @4K, ~19.7 @32K tok/s.
- Prefill met the ≥400 target at ≤~24K context. **400 @100K NOT reached** — the 12
  dense global O(L²) attention layers are the wall; the WMMA kernel is at its
  structural floor (~5% of matrix peak, memory/LDS-bound, matrix core ~95% idle;
  occupancy lever measured dead). MoE floor ≥490 (from 490@4K), so 400@100K is
  physically reachable ONLY via an *algorithmic* change (sparse/strided global
  attention) that diverges from the dense reference — **user chose to bank the
  parity-exact 298 rather than trade quality.** Note: ds4's 400@100K uses MLA
  (compressed KV), a cheaper long-ctx attention — the target was apples-to-oranges.

**Key perf levers (all parity-exact, oracle token 22718):** SWA windowing (1.9×@16K +
correctness) · FA2 register-O (occupancy 1→2 WG/CU) · SWA ring-KV (KV 20→5.3 GB @100K,
killed the OOM, unlocks native 262K) · head-grouped global attn · register-O +
warp-parallel softmax. Session arc @100K: broken/OOM → 214 → 244 → **298**.

**Correctness:** batched==sequential==oracle greedy 22718; per-kernel GPU-parity tests
green (gqa 8/8, MoE, tokenizer exact). **SWA >512 validated 4 ways** (isolation vs
CPU-windowed ref; llama.cpp boundary formula q_pos−k_pos<512; smooth-across-512 e2e no
NaN/discontinuity; coherent varied-content continuation at ctx 523). Open (low-risk,
documented): full external teacher-forced argmax match at >512 — blocked by oracle
greedy-degeneration + llama.cpp not exposing per-position argmax.

**Deliverables:** `laguna-chat` (interactive REPL, chat template, sampling) +
`laguna-oracle-gen` (correctness harness) bins; kernels + forward + tokenizer under
`crates/v4flash-{core,kernels}`, all additive. Env A/B toggles documented in
`laguna_het.rs` (LAGUNA_ATTN_HG, _WMMA_LEGACY, _KVFIRST, _NAIVE, _FLASH, _SWA_OFF,
LAGUNA_PIPELINE, LAGUNA_PREFILL_HET, LAGUNA_HG_G).

The original phased plan (spike → extract seam → completeness/perf) follows below for
historical reference; §0 above is what actually shipped.

---

## 1. Goal & framing

deepstrix today runs exactly one model — DeepSeek V4-Flash ("ds4") with antirez's iq2
quant — with the architecture baked in at compile time. We want to add
**Laguna-S-2.1** (poolside, 117.6B/8.5B-active coding MoE, GGUF Q4_K_M) as a *second,
first-class* model, and introduce the minimum abstraction that lets a third model be
added without another archaeology dig.

Success criteria, in priority order:

1. **Correctness**: Laguna logits match reference within tolerance (see §7 for the
   dual-oracle strategy) on the same Q4_K_M GGUF.
2. **Cleanliness**: ds4 behavior is byte-identical after the refactor, and the
   model-specific surface for a *new* arch is a small, well-defined set of files.

Non-goals for v1: DFlash speculative decoding, FP8-KV tuning, 1M context, peak-perf
kernel tuning. Get it correct and resident first, then optimize vs the roofline
([[feedback_roofline_first]]).

### 1.1 Key correction from review round 1
The v1 plan assumed GQA would *reuse* ds4's attention WMMA kernels and refactored the
abstraction *before* learning from Laguna. Both were wrong (see §3, §5). This revision:
**spikes Laguna correctness first** (throwaway), then extracts the abstraction with two
real clients, and treats GQA/SWA + dense Q4_K/Q6_K as **new kernel work**, not reuse.

## 2. What Laguna is, and how it differs from ds4

From `config.json` + the `Laguna-S-2.1-GGUF` repo (arch string `"laguna"`, Q4_K_M 75 GB):

| Axis | Laguna-S-2.1 | ds4 (V4-Flash) | Verdict |
|---|---|---|---|
| Attention | **GQA**, head_dim 128, 8 KV heads; **per-layer Q heads: 48 on full layers, 72 on SWA layers** (GQA ratio 6 vs 9) | MLA (single 512-wide latent, MQA-like) | **NEW kernel family** — see §3; per-layer q/o width |
| Attn layout | **Mixed per-layer**: full on il%4==0 (0,4,…,44 = 12); **SWA window=512** on other 36 | all-global MLA | **NEW**: real 512 sliding window + per-layer dispatch |
| QK-norm | **RMSNorm on Q and K at head_dim(128)**, after proj, before RoPE (Qwen3-style) | none | NEW op — reuse rms_norm kernel |
| Attn gate | per-head **softplus** gate, computed from pre-attn normed input, applied to attn output **BEFORE o_proj** | none | NEW small op (softplus, NOT sigmoid — that's the router) |
| RoPE | **NEOX**, per-layer-type: full = θ500000, n_rot **64** (partial 0.5) + **YaRN** (factor 128, orig_ctx 8192); SWA = θ10000, n_rot **128** (full), plain no-YaRN | single rope | Two `RopeParams` configs via existing `rope_tail` (YaRN already implemented) — **PARAMETERIZE, not new** |
| MoE | 256 routed **top-10** + **1 shared** (always-on, 1024); **sigmoid** router + score-correction bias `exp_probs_b` (selection only) + sum-norm + **scale 2.5** | 256 routed + shared, top-6 | **REUSE** routing + batched expert GEMM (verify dims/scale) |
| Dense layer | **layer 0 is a dense SwiGLU FFN, width 12288** (`n_layer_dense_lead=1`); layers 1–47 MoE | — | NEW: dense FFN path (Q4_K/Q6_K dense matvecs) |
| Activation | **SwiGLU/SiLU**, parallel gate·up (experts, shared, dense) | — | reuse swiglu.hip |
| Norm | **RMSNorm** eps 1e-6, pre-norm, no post-norms; output_norm before LM head | RMSNorm | reuse |
| Tokenizer | vocab 100352, **custom Laguna BPE pre-tokenizer regex**, eos=[2,24] (tok 24 `</assistant>` = eot/control), clean_spaces=false, pad=9 | different | NEW pre-tokenizer + special tokens |
| Quant | **GGUF Q4_K_M** (Q4_K + some Q6_K) | iq2_xxs / q2_k / q8_0 | REUSE GGUF loader; but see §3 kernel gaps |
| KV cache | FP8 (`fp8_e4m3fn.hip` exists) | FP8 (compressed) | REUSE fp8 primitive |
| Dims | 48 layers / hidden 3072 / interm 12288 | 43 / 4096 | Runtime config |
| Vocab / ctx | 100352 / 262K (1M ckpt exists) | 129280 | NEW tokenizer + template |
| Draft | DFlash 2.2 GB BF16 (spec decode) | MTP | Phase-4 lever, reuse verify infra |

**ds4-only baggage** (must NOT leak into shared paths, must NOT be paid for by Laguna at
runtime — modules *and scratch*): mHC (`hc_*`, `mhc_pre_fused`, `broadcast_to_hc`),
compressor/indexer (`compressor_*`, `indexer_*`, `comp_kv_append`, `indexer_scores`
scratch = 49408 floats/head), MLA LoRA Q/KV, hash router.

**Genuinely shared**: MoE routing + **batched** expert GEMM (incl. `q4_k_pair_matvec_fused_swiglu_batch`),
fp8 KV primitive, het dGPU/iGPU **placement policy** (keys on per-layer expert counts,
not ds4 tensor names), rmsnorm, sampler, swiglu, rope core, GGUF parse, BPE algorithm,
and the het **runtime infra** (streams/events/graph cache/peer-copy/perfetto).

## 3. Kernel reality (grounded in the code — corrects v1)

The v1 "Q4_K exists → reuse" and "GQA reuses score/smwsum" claims are wrong:

- **ds4 attention kernels hard-assert MLA shape.** `attention.rs:402,466,485,531`:
  `head_dim != 512 → Err`, `SMWSUM_HEAD_TILE==16`, and the score kernels read one
  512-wide latent shared across all Q heads with **no KV-head grouping**. Laguna GQA
  (8 KV heads × 128, each shared by 6 Q heads) needs an entirely new
  **GQA score + softmax + wsum** kernel set. Start from the plain non-WMMA path;
  WMMA is a later perf pass.
- **Existing Q4_K kernel is MoE-batched-only** (`q4_k_matvec_par_batched`,
  `q4_k_pair_matvec_fused_swiglu_batch` — take `selected`/`n_used`/expert-stride +
  Q8_K-quantized activations). There is **no dense Q4_K matvec**; dense weight matvecs
  exist only for Q8_0. Laguna's attention Q/K/V/O projections and its 100352-row LM
  head need **new dense Q4_K (and Q6_K) matvecs**.
- **No Q6_K kernel exists** at all; Q4_K_M mixes Q6_K tensors → new kernel.
- **SWA cap mismatch**: ds4's SWA is a 128-row memmove-evicted window fused with the
  compressor (`ATTN_SWA_MAX_KV=128`, `SWA_WINDOW=128`). Laguna needs a real 512-token
  window, no compressor. 512 > 128 → separate kernel with its own compile-time cap.

**New kernels/ops required** (correctness-first, unoptimized OK):
1. Dense Q4_K matvec (attn q/k/v/o **with per-layer width 6144/9216**, dense-layer-0 FFN
   width 12288, shared-expert 1024). — *in progress, being written by a Fable subagent.*
2. Q6_K matvec (LM head, and any ffn_down/attn_v promoted to Q6_K — confirm §9/Phase 0).
3. GQA score + softmax + wsum (plain), full-attention first. Must handle **per-layer Q
   head count (48 vs 72)**, **QK-norm before RoPE**, **NEOX partial/full rope + YaRN**,
   and the **softplus gate applied before o_proj**.
4. SWA windowed variant (window 512 as per-arch `#define`) for the 36 SWA layers.
5. **QK-norm** (RMSNorm at head_dim 128 on Q and K) — reuse `rms_norm` kernel machinery.
6. Per-head **softplus** gate op (small; gate = softplus(W_g·rmsnorm(x)), multiply attn
   out per-head before o_proj).
**PARAMETERIZE (not new kernels — architect-confirmed):**
- **YaRN RoPE**: the existing `rope_tail_ext_inplace` (`rope.rs`) ALREADY implements YaRN
  (`ext_factor`, `mscale_eff`, `corr_dims` NTK-by-parts ramp); ds4 runs per-layer YaRN via
  the existing `Fn(i32)->RopeParams` closure. Drive it with per-layer `RopeParams` from the
  GGUF YaRN keys (full: θ500000/n_rot64/YaRN factor128/orig_ctx8192; SWA: θ10000/n_rot128/
  plain). Confirm tail-64 partial-rotary convention (= ds4 N_ROT=64) and populate
  `n_ctx_orig=8192` per-layer-type. **NOT new kernel work.**
- **QK-norm** reuses `rms_norm` / `RmsNormNoWeight` (two extra calls in forward).
- **Dense SwiGLU FFN (layer 0, 12288)** reuses the dense Q4_K/Q6_K matvec + `swiglu.hip`.
- **Shared expert** reuses ds4's always-on shared-expert path.
- **Router**: REUSE `router_topk_par` with two arch-gated changes (verified against the
  kernel): (a) gating `sqrt(softplus(logits))` → **`sigmoid(logits)`** for Laguna (add a
  gating-mode flag; ds4 keeps sqrt-softplus → byte-identical); (b) **`ROUTER_MAX_USED`
  8→≥10** (`router_topk_par.hip:15` + `router_topk.rs` const). The `exp_probs_b` bias
  (score = p+bias, selection only), sum-normalize, and `expert_weight_scale`=2.5 ALREADY
  match Laguna exactly. See task #5.

MoE expert GEMM = reuse batched Q4_K (verify 3D-stacked `[3072,1024,256]` layout, top-10);
add **batched Q6_K expert GEMM** iff `ffn_down_exps` is Q6_K (Phase 0).

Full tensor-name contract + GGUF `laguna.*` metadata keys captured in the arch spec
(saved to `docs/laguna/ARCH_SPEC.md`); the loader reads head_count as a **per-layer array**.
**Phase-3 WMMA note**: 72 heads is not %16 — GQA WMMA head-tiling must use a divisor of
gcd(48,72)=24 (e.g. 8), not ds4's `SMWSUM_HEAD_TILE=16`. Plain Phase-1 kernels unaffected.

## 4. Target architecture

Principle: **thin seam at the forward-pass boundary; fat shared primitive library +
shared runtime infra underneath.** No generic op-graph engine.

```
crates/
  v4flash-kernels/           (rename to deepstrix-kernels DEFERRED — see §5 note)
    kernels/                 HIP; attn split by arch: attn/{mla,gqa,swa} each with its
                             own #defines (LDS caps, window, head tile). quant: add
                             dense q4k, q6k. common: rmsnorm/rope/sampler/swiglu/fp8.
    src/                     kernel bundles (generic + per-arch) + GgufType dispatch
  deepstrix-models/          (new) Model trait, ModelConfig, per-arch impls, templates
    src/models/{common, deepseek_v4flash, laguna}/
  v4flash-core/              GGUF + BPE (unchanged)
  v4flash-hip/               HIP runtime (unchanged)
  deepstrix-cli/             template comes from the Model
  deepstrix-server/          ONE model per process; arch auto-detected from GGUF (see §4.3)
```

### 4.1 The `Model` trait (KV owned internally — revised per M2)

```rust
pub trait Model {
    fn config(&self) -> &ModelConfig;
    fn template(&self) -> &dyn ChatTemplate;

    // Forward: the Model OWNS its arch-specific KV cache; callers never see it.
    fn prefill(&mut self, tokens: &[i32]) -> Result<Logits>;
    fn decode(&mut self, token: i32, pos: u32) -> Result<Logits>;

    // Server-facing KV lifecycle ONLY (opaque; no arch layout leaks):
    fn reset(&mut self);                       // session boundary (== HetModelState::reset_in_place)
    fn kv_room(&self) -> KvRoom;               // used/capacity for admission control
    fn fingerprint(&self) -> ModelFingerprint; // arch+weights id; guards cross-model restore
    fn checkpoint(&self) -> Vec<u8>;           // opaque per-model KV blob (device→host copy — NOT cheap)
    fn restore(&mut self, blob: &[u8]);        // reload opaque blob into device buffers
}
```

`KvCache`/`KvLayout` are **removed** from the trait. ds4's KV bundle (monotonic-append
f16 window + main/indexer compressors, `state.rs`) and Laguna's GQA windowed cache are
each concrete types owned by their Model. Only lifecycle crosses the seam.

**The real snapshot seam (per review round 2, grounded in `deepstrix-server/src/snapshot.rs:575-754`).**
The server today does NOT treat KV as opaque — `save()`/`restore()` are free functions
that reach into `HetModelState` (`layer.kv_cache`, `n_raw.min(SWA_WINDOW)`,
`compressor.*`, `indexer_compressor.*`) and write a **typed multi-blob disk layout**
(`kv.bin`, `comp_kv.bin`, `comp_state.bin`, `index_comp_kv.bin`, …) plus a structured
ds4-specific `PerLayerMeta` (`coff`, `n_index_comp`, `ratio`). So the seam is:
- `checkpoint()` returns an **opaque per-model blob**; the server keeps owning
  `{tokens, ModelFingerprint, LRU, prefix-match}` — do NOT stuff those into the blob.
- The existing `save`/`restore`/`PerLayerMeta` + multi-blob format **move under the
  Model impl** (they currently take `&HetModelState` concretely). Laguna's blob shape
  (no compressor/indexer; GQA windowed) is entirely different — fine under opacity.
- The on-disk **format version** (`snapshot.rs:33`, already evicts mismatches at
  startup) gains a **per-arch axis**, and `ModelFingerprint` (`snapshot.rs:60`, already
  exists) must **key snapshots by arch** so a ds4 snapshot never restores into Laguna.
- `checkpoint()` is a full-buffer `copy_to_host` per layer (reads the whole oversized
  `kv_cache`, slices the live window — `snapshot.rs:621-633`) and flips
  `set_current()` between devices; the trait doc marks it as non-trivial cost, not a getter.

### 4.2 `ModelConfig` (runtime shape) vs per-arch `#define`s (kernel geometry)

Split the two things v1 conflated (per m2):
- **Model shape → runtime `ModelConfig`** built from GGUF metadata: `n_layer`, dims,
  `n_kv_head`/`head_dim`, expert counts/top-k, rms_eps, and `Vec<LayerKind>` where each
  **`LayerKind` carries the full per-layer descriptor** (architect-confirmed richer than the
  v2 sketch): `{ n_head (48 full / 72 SWA), kv_group, window: Option<u32> (None=full,
  Some(512)=SWA), rope: RopeParams (per-layer-type — YaRN on full, plain on SWA), ffn:
  FfnKind (Dense{12288} on layer 0 | Moe on 1..47) }`. ds4 packs its compress_ratio /
  hash-router flags in the same slot. This is DATA — built by the existing
  `Fn(i32)->RopeParams` per-layer mechanism. Host-side loop bounds & launch params (incl.
  **head_count as a runtime kernel arg**) — zero perf cost.
- **Kernel LDS/geometry caps → per-arch compile-time `#define`s**, resolved by the
  `attn/{mla,gqa,swa}` kernel split. These *cannot* be runtime: `ATTN_SWA_MAX_KV`,
  `SMWSUM_HEAD_TILE`, scores stride, LDS sizing. This is why the attn kernels are
  physically separate per arch, not one parameterized kernel.

### 4.3 Two models can't co-reside — ONE model per process (resolves B2)

ds4 ≈ 86 GiB + Laguna ≈ 75 GiB ≈ 161 GiB ≫ budget (~72+ GiB via
`no_system_mem_limit`, [[project_strix_memory_budget]]). Therefore:
- `deepstrix-server` binds to **exactly one model per process**, arch auto-detected
  from the GGUF `general.architecture` string at load. No co-residency.
- A "registry" is just arch → constructor selection at load time, NOT simultaneous
  residency. Multi-model serving = multiple processes / cold model swap (evict + load,
  documented cost) — **deferred**, explicitly out of v1 scope.
- Cheap to implement: `general.architecture` read + `ModelFingerprint` (`snapshot.rs:60`)
  and host-side embed from the gguf mmap (`weights.rs:114`, M57) already exist as hooks.
  The fingerprint must additionally **key snapshots by arch** (ties to §4.1).

### 4.4 Engine ownership (resolves M4)
Split today's `DeviceEngine` (which bundles ds4 kernels `mhc_pre_fused`/`compressor`/
`indexer` next to generic `rms_w`/`rope`/`q8`):
- **`HetRuntime`** (shared, injected into every Model): streams, events, graph cache,
  peer-copy/sync, perfetto/trace — arch-agnostic orchestration substrate.
- **Per-arch kernel bundle** (owned by the Model): only the kernels that arch uses.
- **Per-model scratch** (resolves M5): each Model allocates only its own scratch; the
  ds4-shaped `DgpuScratch`/`IgpuScratch` (mHC/indexer/compressor buffers) moves under
  `deepseek_v4flash` and is not allocated for Laguna.

### 4.5 KV budget vs hot-expert budget on the dGPU

Attention (and therefore the KV cache) runs on the dGPU, so KV competes with het-split
hot experts for the 16 GB VRAM. Laguna's attention layout makes this *easier* than ds4,
not harder.

**KV size (per token per layer, GQA + FP8):** K and V each = `n_kv_head·head_dim` =
8·128 = 1024 elems → 2048 bytes/token/layer (FP8), + small per-group scale overhead.

**Only 12 of 48 layers grow.** SWA layers (36) are bounded to the 512-token window;
global layers (12) grow linearly:
```
KV(L) = [12·L + 36·min(L,512)] · 2048 bytes         (FP8)
```

| Context L | KV (FP8) | dGPU room left for hot experts¹ | Max K² |
|---|---|---|---|
| 4K   | 0.13 GB | 13.7 GB | ~52 |
| 32K  | 0.80 GB | 13.0 GB | ~50 |
| 128K | 3.1 GB  | 10.7 GB | ~41 |
| 256K | 6.2 GB  | 7.6 GB  | ~29 |
| 512K | 12.3 GB | 1.5 GB  | ~5  |
| 1M   | 24.6 GB | <0 → KV must spill | — |

¹ dGPU fixed for Laguna ≈ **~2 GB** (GQA attn proj ~1.2 GB + shared expert ~0.26 GB +
LM head ~0.25 GB + scratch), vs ds4's **~9 GB** dGPU side (which carries mHC + compressor
+ indexer + MLA-LoRA). Laguna frees ~7 GB → 16 − ~2.2 ≈ **13.8 GB** for KV + hot experts.
² Per hot expert per layer (Q4_K, moe_interm 1024): ~5.4 MB → K costs ~259 MB × K across
48 layers.

**Conclusion for the user's question:** at serving contexts (≤128K) Laguna affords K in
the tens — far above ds4's `K≤6 @ 192K` cap ([[project_server_launch]]) — because SWA
caps most layers *and* the dGPU side is ~7 GB lighter. The squeeze only bites past
~512K, where the 12 *uncompressed* global layers dominate (unlike ds4, there's no MLA
compression to lean on — only GQA's 8 KV heads + FP8).

**Max context on this hardware (dGPU 16 GB is the binding constraint).** Decode requires
KV to be dGPU-resident (§ the offload note below), so the 16 GB VRAM bounds context, not
the 75 GB model (which lives across dGPU + Strix Halo system RAM; 75 GB is *easier* than
ds4's ~86 GB, [[project_strix_v4flash_memory_tightness]]). With ~2.2 GB dGPU fixed →
~13.8 GB for KV + hot experts (the KV numbers below assume zero-overhead FP8 =
2048 B/token/layer; real e4m3 + per-group scales add ~3–6%, verify in Phase 0):
- **FP8 KV**: KV ceiling (K=0) ≈ **~0.55M tokens**; at K=8 ≈ ~475K; at K=16 ≈ ~425K.
- **Full native 262K context fits comfortably** — KV there is only ~6.0 GB, leaving room
  for **K≈30** hot experts. (An all-global 48-layer model would need ~24 GB KV at 262K —
  infeasible; SWA's 4× saving is what makes 262K practical here.)
- **f16 KV** (spike, or if FP8 unshipped) halves all of the above → ~275K ceiling, right
  at the native limit — another reason FP8 KV matters for production.
- Bottom line: the **model's native 262K** is the practical ceiling, not the hardware;
  the hardware could go to ~0.5M if a longer-context checkpoint existed.

**Design levers this exposes (Laguna-specific):**
- **Layer-type-split KV placement**: SWA KV is tiny (36 MB total, constant) and hot
  every step → always dGPU. Global-layer KV is the only thing that grows → the natural
  knob to manage at extreme context.
- **Context-dependent K**, same shape as ds4's rule but with the crossover pushed way
  out; admission control can trade K for context.
- **FP8 KV matters for runway** (2× vs f16). The Phase-1 spike uses f16 KV → doubles the
  table above, but the spike runs at short context so it's fine; production uses FP8.
- **Do NOT offload growing KV to iGPU/host**: decode reads all KV every step; global KV
  at 256K ≈ 6.2 GB/token over the fabric is fatal to decode latency. Cold *experts* on
  the iGPU are fine (conditional reads, local compute); *KV* is not. At extreme context
  the honest answer is lower K or lower context, not KV offload.
- **Prefill scratch** for global-layer scores is O(L) per query tile and must be
  per-model-sized (ties into §4.4 per-model scratch).

## 5. Execution plan (re-sequenced: SPIKE FIRST — resolves M1)

Rationale: designing the `Model`/KV/scratch seam from ds4 alone yields the wrong shape
for GQA+SWA. Learn the seam from a real second client *before* abstracting.

### Phase 0 — Recon (no engine code)
1. Dump GGUF header: `general.architecture`, all `laguna.*` keys, expert count/top-k,
   shared-expert presence, rope scales, per-layer attention pattern.
2. Enumerate tensor names + dtypes → confirm Q4_K vs Q6_K split, GQA tensor layout,
   attn-gate tensor, LM-head dtype, and **per-expert gate/up/down dtype** (decides
   whether the MoE GEMM is pure reuse or also needs a batched Q6_K expert variant, §6).
3. Memory sizing: confirm Laguna-alone (75 GB) fits dGPU(16)+iGPU(host) with KV
   headroom; record the co-residency decision (§4.3).
4. Inspect poolside `llama.cpp` fork (branch `laguna`): resolve softplus-vs-sigmoid
   gate, rope scale values, and whether/how it can emit per-layer hidden states
   (instrumentation cost for §7). Build via `nix-shell` ([[feedback_nix_shell_for_tools]]).

### Phase 1 — Laguna correctness SPIKE (throwaway; deliverable = knowledge + parity)
On a sub-branch, hack freely (fork code, hardcode Laguna dims, single stream, f16 KV,
**global-attention only, no gate / no SWA / no FP8**). Goal is **layer-by-layer logit
parity**, not clean code.
1. New kernels (unoptimized): dense Q4_K matvec, Q6_K matvec, plain GQA
   score+softmax+wsum.
2. Laguna `WeightSpec` (tensor names), tokenizer special tokens + chat template.
3. Bring up embed → layer0 attn → layer0 MoE → … → logits; parity vs fp16 ref + fork.
4. **Output**: a written answer to "what is the right Model/KV/scratch seam for two
   archs?" that Phase 2 consumes.
5. **Run REAL per-layer `RopeParams` (YaRN on full layers) — do NOT stub to plain rope.**
   YaRN is not a no-op below `original_context_length` (the NTK ramp + mscale rescale even
   at ≤512 tokens), so stubbing would produce plausible-but-wrong logits on full layers and
   mask a rope bug. Cost is zero (the kernel already does YaRN).

**Anti-rot guardrails (per review round 2 — spike must not become the product):**
- **Scope cap**: the spike stops at global-attention greedy-decode logit parity (layer 0,
  then full model). That is *sufficient to learn the seam*. The **attention gate, SWA,
  per-layer-type rope, and FP8** are deliberately pushed to Phase 3 — they do NOT belong
  in the throwaway (adding them bloats sunk cost and gravity to keep the spike).
- **Additive-only**: the spike branch is **purely additive** — it must NOT edit shared
  code in place (`attention.rs`, `het/engine.rs`, `config.rs`). New spike kernels/files
  only. This keeps Phase 2's ds4-byte-identical gate rebasing onto an *unmodified* ds4
  baseline.
- **No perf machinery**: the spike uses single-stream, no graphs, no HetParallel, no
  het-split. By construction nothing in it is salvage-worthy, so "port cleanly" can't
  quietly become "keep and wrap."
- **Exit criterion (Phase 2)**: the spike branch is **deleted, not merged** — never
  fast-forwarded to main. Phase 2 re-implements Laguna cleanly under `models/laguna`
  against the extracted seam.

### Phase 2 — Extract the abstraction (two real clients: ds4 + Laguna)
Now that both forward passes exist, land the seam as reviewable PRs:
1. `config.rs` consts → runtime `ModelConfig`; `Model` trait (KV-owned, §4.1).
2. Split `DeviceEngine` → `HetRuntime` + per-arch kernel bundle; per-model scratch (§4.4).
3. Move ds4 `forward_layer`/`weights`/`state` under `deepseek_v4flash`; port the spike
   cleanly into `models/laguna`.
4. Per-model `WeightSpec`/loader; `ChatTemplate` from the Model (remove hardcoded token
   IDs from `chat.rs`).
5. **Gate**: ds4 logits oracle + prefill/decode tok/s unchanged
   ([[feedback_bench_ab_methodology]]); Laguna parity preserved.

### Phase 3 — Laguna completeness & perf
1. SWA (512, per-arch `#define`) + per-layer Global/Sliding dispatch.
2. FP8 KV (reuse `fp8_e4m3fn.hip`).
3. WMMA GQA attention (perf pass over the plain kernels from Phase 1).
4. het-split expert placement for Laguna; roofline pass.

### Phase 4 (deferred) — DFlash speculative decoding
Reuse MTP verify infra ([[project_mtp_verify_dp4a]]) with the 2.2 GB DFlash draft.

### Crate rename note (m1)
`v4flash-kernels → deepstrix-kernels` is **removed from the critical path**. It touches
every `use`, Cargo.toml, and `KERNEL_*` build-script env names — pure churn that would
bloat the Phase-2 byte-identical diff (the safety gate). Do it as a single isolated
mechanical commit at the very end, or not at all. (`v4flash-core`/`v4flash-hip` stay.)

## 6. New-kernel + reuse ledger
NEW: dense Q4_K matvec, Q6_K matvec, GQA score/softmax/wsum (plain → WMMA), SWA windowed
variant, softplus/sigmoid gate epilogue.
NEW (conditional, verify Phase 0): **batched Q6_K expert GEMM** — the existing batched
Q4_K expert kernel (`q4_k.rs:45`, `n_rows%8==0`, Q8_K activations) transfers only if
Laguna's expert gate/up/down are all Q4_K. Q4_K_M often puts `ffn_down` in Q6_K → then
the batched expert path needs a Q6_K variant too, not just the dense Q6_K matvec.
REUSE: batched Q4_K expert GEMM + swiglu (dims are args), fp8 KV, rmsnorm, rope core,
sampler, swiglu, GGUF parse, BPE, HetRuntime infra.
REUSE (policy only, per-model wiring): het placement **policy** (global-greedy by count)
transfers; the loader (`parse_hot_expert_file`, `weights.rs:515`, `N_LAYER`-hardcoded +
ds4 default file) is per-model.
PARAMETERIZE: rope per-layer scales (existing `Fn(i32)->RopeParams` closure already
supports per-layer `RopeParams`; confirm Laguna's `n_rot` — likely full-128 vs ds4's
partial `N_ROT=64`), ModelConfig-driven dims.

## 7. Verification (dual oracle — resolves M6)
1. **Quant-parity oracle**: poolside `llama.cpp` fork (branch `laguna`) on the same
   GGUF — but a forked reference can be non-canonical ([[reference_sglang_mtp]] warns of
   exactly this with ds4-ROCm). llama.cpp is **not** instrumented for per-layer dumps;
   patch the fork to emit hidden states (as `ds4.c` was patched for MTP) — tracked work.
2. **Canonical anchor**: a low-tolerance fp16 reference (HF transformers / vLLM) on a
   short prompt to catch "the fork itself is wrong."
3. Layer-by-layer bring-up to localize drift (the method that validated MTP).
4. **ds4 non-regression**: existing oracles + back-to-back tok/s A/B.
5. **Test strategy**: golden-file logits per model checked into the harness; a CI gate
   that runs both models' parity checks. (Previously unspecified.)

## 8. Open questions / risks
1. top-8 vs top-10 (config vs card) — GGUF metadata authoritative.
2. Shared expert present? — GGUF tensor names authoritative.
3. Q6_K + dense Q4_K + LM-head kernels are NEW (not reuse) — scoped in §3.
4. Attn gate softplus vs sigmoid — resolve from fork source, not model card. Small in
   code but **oracle-critical**: it's a per-head multiplicative factor on attention
   output, so a wrong gate drifts every downstream layer and is hard to localize (it's
   an epilogue, not a matvec). Treat it as a **first-class layer-parity checkpoint**.
5. Tokenizer pre-tokenizer/regex + special-token scheme differ per arch; `v4flash-core`
   BPE genericity beyond token tables is **unverified** — check in Phase 1.
6. Refactor blast radius: mitigated by spiking first; Phase 2 seam is designed from two
   clients, not one.

## 9. Environment
- Q4_K_M GGUF on disk (`/persist/lumi/models/laguna-s-2.1-int4`, user's HF task, ~1h) —
  I will NOT download weights.
- Network to build/inspect poolside's `llama.cpp` fork for the oracle.

## 10. Performance targets & roofline (perf-engineer review, phase0-grounded)

Hardware ceilings (**phase0-measured**, `crates/phase0/results/*.json`): dGPU (gfx1201)
**600 GB/s achievable** (95% of 644 theoretical) / ~97 TF16 (spec); iGPU (gfx1151)
**229 GB/s achievable** / ~30 TF/s effective; **peer fabric ~6 GB/s** (~90× slower than
local DRAM → KV MUST stay dGPU-resident, confirms §4.5). B=1 iGPU decode reads
effectively ~180 GB/s (scattered cold-expert access, latency-bound). Calibration anchors:
ds4 decode 29.4 tok/s @4K, prefill 500/471/417.

**The binding resource FLIPS vs ds4.** Laguna moves ~3.5 GB/token at decode (vs ds4's
~9.5 → ~2.7× less) because GQA attn proj is ~5× lighter than MLA. So:
- **Decode is iGPU-cold-expert-BW bound at ≤~32K** (the light dGPU attn leg no longer
  hides the cold-expert leg — opposite of ds4's dGPU-bound decode), **transitions ~130K**,
  and is **dGPU-global-KV-read bound at 262K** (KV alone ≈ 6.5 GB/token = ~11 ms).
- **Prefill stays iGPU-MoE-compute bound** across the whole ≤262K range; global-attn
  O(L²) on the dGPU stays hidden under it until it crosses over near ~250–360K (≈ native max).
- **Consequence: K (hot-expert count) is now a decode-THROUGHPUT lever**, not just the
  VRAM knob of §4.5 — every +1 to top-k adds ~0.25 GB/token of iGPU traffic unless a hot
  expert on the dGPU captures it. Cold-expert capture fraction is a Phase-3 metric to tune.

**Per-phase targets:**
- **Phase 0** — no perf target, but MUST resolve **top-k (8 vs 10)** and **whether
  `intermediate_size=12288` is a live dense FFN**: an unaccounted ~1–2B active params
  (card says 8.5B active; identifiable is ~6.5–7.4B) lands on the **dGPU** (dense FFN →
  partially un-flips decode toward dGPU-bound) or the **iGPU** (bigger experts/top-k →
  cold leg even more binding). This decides *which GPU binds decode* — a perf blocker,
  not a footnote.
- **Phase 1 (spike)** — correctness only, **NO perf target** (single-stream, f16 KV,
  unoptimized kernels will read ~2–3× slow; do not mistake for roofline).
- **Phase 2 (extract seam)** — ds4 non-regression gate: prefill/decode tok/s
  byte-identical to pre-refactor (back-to-back A/B, [[feedback_bench_ab_methodology]]).
- **Phase 3 (Laguna completeness + perf):**
  - decode (plain GQA + FP8 KV + het-split) **@4K ≥ 40 tok/s** (roofline ~90–110; realistic
    ~45–65), **@128K ≥ 30**, **@262K ≥ 25** (dGPU-KV-read bound).
  - prefill after Q4_K WMMA/dp4a expert-GEMM opt (the kwide-equivalent lever): **@4K ≥ 450**,
    **@32K ≥ 420**, **@96K ≥ 380** (roofline ~530–650; iGPU-MoE bound).
  - measure cold-expert capture fraction; tune K so the iGPU leg ≤ dGPU leg at ≤32K.
  - **Bottleneck acceptance test**: decode iGPU-cold-expert bound short / dGPU-KV bound
    long; prefill iGPU-MoE bound throughout. Any other profile = a bug (peer-copied KV, or
    a dense FFN nobody budgeted).
- **Phase 4 (DFlash spec-decode, deferred)** — target +30–60% effective decode.

**Kernel effort priority** (from the budget table): the **batched Q4_K expert GEMM** is
THE prefill lever (kwide/WMMA, like iq2 was) and dominates decode too; a **decode-shaped
(B=1, grid.z) Q4_K expert variant** is likely needed since the batched kernel is tuned for
B=512 (ds4 needed the same). Decode's non-GEMM lever is **fusion + graph capture** to cut
the dispatch floor (ds4's ×1.7 floor should be ~×1.4–1.5 for Laguna — no compressor/indexer
zoo). Dense Q4_K matvec and LM-head (Q6_K, ~0.39 ms/token fixed) are BW-bound; WMMA helps
prefill only. GQA WMMA score matters only at long ctx.
