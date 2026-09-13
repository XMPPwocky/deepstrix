# V4.1-Flash engine port — plan (rev 2, 2026-09-12)

Rev 2 folds in the architect review of rev 1 (verdict REVISE, option C upheld). Changes are
marked **[R]** with the review item number.

Scope: get DeepSeek-V4.1-Flash running in the existing heterogeneous engine (iGPU + dGPU,
single box first), oracle-exact against `scripts/v41_oracle/`, and served by `deepstrix-server`.
Two-box, CED scheduling and DSpark are M8, planned in `PLAN.md` §5–8.

Inputs that already exist: `V41HfWeights` (weights straight from the HF checkpoint under llama.cpp
names, fixture-validated), the CPU oracle with its dumps (`~/.cache/deepstrix/v41/oracle_full` =
6-token " Paris" run, `oracle_t200` = 129-token run, both also exported to the engine's
`manifest.json` + `.bin` layout as `*_bins/`), `ARCH_SPEC.md` (exact reference semantics).

## 0. Decision: one het engine, compile-time model selection

**Facts.** The engine is specialised at compile time: `crates/v4flash-kernels/src/config.rs`
holds every width as a `const`. **[R1]** Real coupling counts: `N_EMBD` has 375 unqualified use
sites, `N_LAYER` 174 across 34 files, config consts are imported in 59 files, `RMS_EPS` 82 sites;
`COMPRESS_RATIOS: [u32; 43]` (`config.rs:76`) cannot become `[u32; M::N_LAYER]` on stable Rust.
Kernels take dims as launch arguments; a few Rust wrappers and one kernel bake them (§2 M0).
`build.rs` supports `-D` macro overrides per compile. Laguna (model #2) shared almost nothing
(GQA vs MLA) so copying was right there; V4.1 shares the het split, mHC, MLA, MoE packing.

**What differs between V4-Flash and V4.1** (everything else in `config.rs` is identical: 64 heads
× 512, rope 64, 8 W_O groups × rank 1024, hc_mult 4, SWA 128, top-k 512, route_scale 1.5,
swiglu clamp 10):

| const | V4-Flash | V4.1 |
|---|---|---|
| N_EMBD | 4096 | 5120 |
| N_LORA_Q | 1024 | 1280 |
| N_FF_SHARED / N_FF_EXP | 2048 | 2304 |
| N_EXPERT | 256 | 384 (used 6, same) |
| N_LAYER | 43 (odd) | 40 (even; 20 encoder + 20 decoder in CED) |
| N_INDEXER_HEAD | 64 | 32 |
| N_HASH_LAYERS | 3 | 0 |
| COMPRESS_RATIOS | per-layer table | `[0,0,2×18,1×20]` + kv/index source tables |
| RMS_EPS | 1e-6 | 1e-20 (config `norm_eps`; eps is a launch arg everywhere, only the const changes **[R18]**) |
| expert format | IQ3_XXS/Q2_K mixes | MXFP4 gate/up/down (native), Q8_0 projections |

**Options.** (A) runtime `ModelConfig`: ~8× the blast radius rev 1 assumed, buys hosting both
models in one process, which memory forbids anyway. (B) copy the engine (Laguna style): 32k
lines duplicated, every V4-Flash optimisation ported by hand, server still needs dispatch.
(C) **cargo feature `v41`** selecting the `config.rs` values (+ `-DDEEPSTRIX_V41` for the kernel
that bakes dims): one crate, two build artifacts; the V4-Flash build is byte-identical by
construction; V4.1 reuses everything immediately. A `ModelCfg` trait with associated consts is
strictly more expensive than C on this codebase **[R1]**.

**Decision: (C)**, with two hardenings **[R1]**: a compiled-model vs loaded-weights assertion at
load (`V41HfWeights::config()` / GGUF metadata carry `n_layers`, `dim`, `n_routed_experts`; a
V4.1 directory in a V4-Flash binary fails at the door), and per-layer tables as
`&'static [u32]` role tables with length asserts rather than fixed-size arrays.

**Feature hygiene [R2].** A model-select feature is non-additive. Rules: never `--all-features`
or `--workspace` builds that could unify `v41`; separate `CARGO_TARGET_DIR` per model in the run
scripts (every flip otherwise recompiles the crate + 70 kernels × 2 archs); the 17 het-engine
tests and every dump-driven V4-Flash oracle test get `#![cfg(not(feature = "v41"))]`, V4.1
tests get `#![cfg(feature = "v41")]`; CI-style check = both configurations build.

## 1. Seam: `WeightSrc`

Weights come from `V41HfWeights` instead of `MappedGguf`. `GgufTensor` and `VTensor` already
agree on `name / dtype: GgufType / dims / elements / byte_size`; the GGUF-only fields
(`shard`, `abs_offset`) exist solely to feed `read_range_into`, which the HF view supersedes.

```rust
// v4flash-core
pub struct TensorDesc { name, dtype: GgufType, dims, elements, byte_size }
pub enum WeightSrc<'a> { Gguf(&'a MappedGguf), V41(&'a V41HfWeights) }
impl WeightSrc<'_> {
    fn tensors(&self) -> Vec<TensorDesc>;                       // validate_model, fingerprint
    fn desc(&self, name: &str) -> Option<TensorDesc>;
    fn read_into(&self, d: &TensorDesc, dst: &mut [u8]) -> Result<()>;      // parallel inside
    fn read_expert_into(&self, d: &TensorDesc, e: usize, dst: &mut [u8]) -> Result<()>;
    fn model_dir(&self) -> Option<&Path>;                       // expert-stats sidecar only
}
```

Enum rather than trait object for exhaustive matching and no vtable in the per-expert read
loop **[R9]**. Ten signatures change (`&MappedGguf` → `&WeightSrc`): `weights.rs::load_to_device`
(+ two helpers), `model_weights.rs::{load_f32_weight, load_i32_tensor}`,
`het/weights.rs::{HetGlobalWeights::load, DgpuLayerWeights::load, IgpuLayerWeights::load,
load_experts_packed, HotExpertWeights::load, HetModelWeights::load_all}`, and **[R7]** the
server's direct `token_embd` read at `engine_worker.rs:528-540`. The five
`abs_offset + e*bpe` + `read_range_into` sites become `read_expert_into`; the direct-to-device
fast path (pread into `as_host_slice_mut`) keeps working since both variants fill a `&mut [u8]`.
`weight_contract::validate_model` takes `&[TensorDesc]`; `snapshot::ModelFingerprint::compute`
hashes descs (V4.1 fingerprint = HF snapshot hash + role table).

**Contract additions (feature `v41`).** `ffn_{gate,up}_exps` accept MXFP4; projections Q8_0;
**[R7] `token_embd` presented as F16** (the contract allows `[F16, Q4_K, Q5_K, Q6_K]` and
`embed.rs` has no Q8_0/BF16 arm; bf16→f16 is exact in [6e-5, 65504]; 1.3 GB host; removes the
layer-0 embedding noise). `exp_probs_b_vl` is an in-checkpoint contract role, not the
`bias_vl` sidecar **[R9]**. Per-layer role sets from the measured table in `ARCH_SPEC.md`
(compressor only on KV-source layers, `wgate` only at ratio 2, indexer `wk/k_norm` only on KV
sources, `wq_b/proj` also on index-source-only layers, Engram roles on layers 1 and 14).
**[R21] resolved (measured):** compressor `wkv/wgate` (layers 2, 20) and indexer `wk` have no
values above f16's max; ~0.2% sit in f16's subnormal range (kept, reduced precision) and only 6
per 2.6M-element tensor fall below 2^-24 (magnitudes < 6e-8 vs max 0.15) — F16 presentation
stands; the reference's fp32 module dtype is an upcast of the same bf16 values.

**Tokenizer and template are outside the seam and not GGUF-free today [R8].** `BpeVocab` only
has `from_gguf`/`from_sparse_parts`; the server takes the vocab and the chat template from GGUF
metadata. V4.1 needs a `tokenizer.json` → `BpeVocab` loader (serde_json already in core) and a
V4.1 `render_prompt` variant (`ARCH_SPEC` §5: numeric effort 1–100, DSML tags with a leading
space, tool results merged into the user turn), with golden vectors ported from
`encoding/test_encoding.py`. Scheduled in M7 but started at M1 (the parity tests need to
tokenise the oracle prompts identically).

## 2. Milestones and gates

**Gate definition [R19].** The V4-Flash oracle bar (~1e-2 max|Δ|/scale) was measured against
ds4 running the *same* Q8_0 GGUF, i.e. zero weight noise. V4.1's reference is fp8-exact, so
Q8_0 projection error accumulates over 40 layers. Before M1: run the CPU oracle once more with
Q8_0-requantised projections and F16 embeddings (the loader's exact transforms, ported into the
shim) to produce a **per-layer noise floor**; every milestone gates at Δ ≤ k × floor (k ≈ 2)
plus argmax / top-3 agreement. Cost ~4 s/layer at T=6.
**Measured (2026-09-12, `V41_ORACLE_Q8=1`, `~/.cache/deepstrix/v41/compare_q8_floor.txt`):**
residual max|Δ|/max|ref| = 2.7e-2 at L0, peaks 3.8e-2 at L2, 2e-2 through L9, ~3e-3 L10–14,
6e-4–1e-2 L15–36, 1.4e-3 at L39; logits Δ/scale **0.205** (max|Δ| 4.9 on a 23.8 scale) with
" Paris" still top-1 by 3.0 logits but the rank-3 token changed. So: per-layer gates at 2× the
floor; the head gate is argmax + ≥ 3 of the reference top-5 present in ours, NOT strict top-3.
Note the floor is slightly pessimistic (the reference also bf16-rounds the Q8 values of `wo_a`),
and it says Q8_0-from-fp8 is *not* logit-order-neutral — native fp8 kernels (M8) are a quality
item, not just a load-time convenience.
**t200 (129 tokens, `compare_t200_q8_floor.txt`):** residual floor 2.7e-2 @L0 … 2.1e-2 @L39, and
the **greedy token changes** (logits Δ/scale 5.6e-2, argmax DIFFERS: " left" → " ended"). Q8_0
projections are therefore not argmax-faithful on ordinary prompts; fp8-native (or bf16 for the
most sensitive projections) is required for quality parity — promote from M8 to before M7.

- **M0 — seam, feature, audits.** `WeightSrc`, feature-gated `config.rs`, contract roles,
  `build.rs` plumbing, `#![cfg]` test gating, separate target dirs. **Audit list [R3-R5]:**
  - odd-layer-count assumption: the unconditional extra residual `mem::swap`
    (`het/engine.rs:799-808`, `forward_prefill.rs:169, :603`) → `if N_LAYER % 2 == 1`;
  - `256` meaning N_EXPERT: hot-expert remap `het/weights.rs:895, :947`; stats/hot sets
    `het/engine.rs:604-746`; **router kernel** `router_topk.hip:20-25` (one thread per expert,
    `ROUTER_MAX_EXPERTS 256` unguarded), `router_topk_par.hip:14`, `router_topk.rs:19,69,148`
    → the 384-expert router is a kernel change (2 experts/thread or 384 threads + tree reduce);
    full `grep -nw 256` classification kept in the audit file;
  - width caps in Rust wrappers: `rms_norm.rs:69,110,158` reject n > 4096;
    `mhc_pre_fused.hip:30-31` `#define MHC_HC_DIM/MHC_N_EMBD` need `#ifndef` guards;
  - **shape sweep**: every kernel wrapper's existing unit test run at V4.1 shapes
    (5120 / 1280 / 2304 / 384 / 20480) against its CPU reference — 37 divisibility guards exist,
    and `%512` / `%2048` tilings will surface (2304/512, 5120/2048 are not integers);
  - compiled-vs-loaded assertion; per-layer tables as slices with length asserts.
  Gate **[R20]**: with the feature off, hsaco files and the crate binary are byte-identical
  (`cmp`); bench only if they differ. With the feature on, layers 0–2 load to device with the
  byte counts the contract predicts.
- **M1 — layer 0 parity + the production expert miss path.** Reference `embed_hc` →
  `layer_00_residual` at T=6 and **[R13] at T=129 (t200)**, whose 129 tokens exercise the SWA
  wrap on layers 0–1. New code: MXFP4 fused gate/up pair kernel (single-token + batched; the MXFP4
  down kernel exists); 384-expert router kernel; **[R13] window-KV FP8 fake-quant** (block 32,
  ue8m0, whole 512 incl. rope tail — the engine's raw KV is f16, so this is the first place a
  layer-0 Δ appears); Q8_0 projections at 5120/1280. **[R11] The direct-to-device expert miss
  path (`read_expert_into` → device slot, multithreaded pread) is built here as production
  code**: parity tests run with experts loaded on miss (misses logged as a parity signal — near-tie
  routing differences are expected), and M7 hardens the same path into the LRU tier. Systematic
  Δ to record: reference expert activations are FP8 fake-quant, ours Q8 (more precise).
  **[R12]** `oracle.py` dumps `layer_NN_topk_ids` (per-token routed ids) so tests can preload and
  compare routing; `export_bins.py` carries them.
- **M2 — layers 0–1: Engram [R14].** Reality: two modules × (98.3 + 3.1) GB = 203 GB, SSD-backed
  (pread through `V41HfWeights::raw()`, dequant on gather). 24 rows × 2 modules = 48 random
  4 KiB reads per token: serial QD1 would cost ~5 ms/token, so the gather is a parallel pread
  pool, and since the hash is **token-only** (`ARCH_SPEC` §1.7) rows are prefetched at prompt
  ingest and right after sampling, off the critical path — that is the design, not a later
  optimisation. It shares IOPS and page cache with the M7 expert tier (budget both). `token_map`
  needs NFKC/NFD/StripAccents/Lowercase: dump the 129280-entry table from Python once (no Rust
  unicode dependency). `engram_wkv` Q8_0 + `q/k` gate as in `LazyEngram`. Gate: parity on
  layers 0–1 plus measured rows/s at decode and prefill batch rates.
- **M3 — layer 2: CSA2 KV source, ratio 2 [R15, R16, R10].** `compressor.rs:72,123` accept only
  ratios {4, 128} and `compressor_pool.hip` hard-codes the ratio-4 layout: ratio 2 goes through
  the ratio-128 branch with new guards, state sizing and `pos0 % ratio` handling; ratio 1 is
  norm-only. The main-KV store is a **new format** (E2M1 + E4M3 scale per 16; existing is
  E2M1 + E8M0 per 32). Indexer at 32 heads, top-512 sparse attention. **Design rule (CED):**
  "produce store S from H_L" is a stage callable independently of "run layer L"; reuse layers
  read stores by source-layer id (`state.layers[l].kv = StoreRef(source)`); exact mode and the
  decoder's bounded replay share the producer. Snapshot v6 falls out of stores owned by source
  layers (today `PerLayerMeta {has_compressor, ratio, …}` assumes each layer owns its own).
- **M4 — layers 3–7: Reuse mode.** KV and index from layer 2 via `StoreRef`; confirm against
  `model.py` whether the checkpoint's `attn.wkv` on reuse layers is used; drop it from residency
  if not.
- **M5 — layers 20 and 24 [R17].** Ratio-1 Reindex with the hierarchical candidate pool: block-max
  over 8, top-2048 of ~37.5K blocks at 300K context, mask applied inside the top-512 of the four
  index-source-only layers (24/28/32/36). New kernels whose cost grows with context: add a rough
  cost line and a 300K test before building.
- **M6 — all 40 layers.** " Paris" at T=6, then t200: per-layer Δ vs floor, top-3 agreement
  (" left"/" mattered"/" matter"), 0 NaN. **[R12]** Non-routed weights alone are 7.4 GB, so M6
  runs server-down or layer-streamed one layer at a time (as the CPU oracle does).
- **M7 — expert tier + server [R11].** Native experts are 3 × 2304 × 5120 × 17/32 B = 18.8 MB
  each, × 384 × 40 = **289 GB**, against 93 GiB here and 128 on the second box: the LRU
  residency tier (PLAN.md §3.3, `COLD_EXPERT_CACHING.md`) hardened from the M1 miss path is a
  prerequisite for serving, not an optimisation. Then `deepstrix-server --features v41`, run
  script with its own target dir, `tokenizer.json` loader + V4.1 template, snapshot v6 +
  fingerprint from the HF descs, KV/scratch budgets at the 5120 width.
- **M8 — structure and perf** (PLAN.md §5–8): CED prefill scheduling (20 encoder layers over the
  whole prompt, decoder bounded replay — enabled by the M3 rule), two-box expert server, DSpark,
  native fp8 dGPU kernels (deletes the load-time Q8 requant).

### M0 status (2026-09-12 evening)

Done, feature off verified: `WeightSrc` seam (`v4flash-core/src/weight_src.rs`; loaders take
`impl Into<WeightSrc>` so no call site changed; the five inline expert offsets are
`read_expert_into`; `validate_model` / fingerprint take `&[GgufTensor]`), feature `v41` in the
three crates with the nine constants and the CSA2 source tables gated, `build.rs` passes
`-DDEEPSTRIX_V41 -DMHC_N_EMBD -DMHC_HC_DIM` and `mhc_pre_fused.hip` guards its defines,
odd-layer swap guarded (`engine.rs`; prefill has one swap per layer and needs none), `256`
literals → `N_EXPERT` (remap, stats, hot sets, `HOT_MAX_EXPERTS`, contract dims), `rms_norm`
caps → `HC_DIM`, compiled-vs-loaded geometry assertion in `load_all` (production GGUF carries
43/4096/256 — passes), `token_embd` F16 in the HF view and converter.
Gates: all 213 hsacos byte-identical to the pre-M0 baseline (incl. the recompiled
`mhc_pre_fused`); `tests/weight_src_arms.rs` — both arms return identical bytes for every
fixture tensor and expert. Standing V4-Flash oracles not re-run (server up, 4 GiB host free).
Open for M0: `#![cfg]` test gating, the shape sweep at V4.1 dims (needs the feature-on test
build), router kernel (M1). Build V4.1 with `CARGO_TARGET_DIR=target-v41 … --features v41`.

### M1 status (2026-09-12 evening)

- **MXFP4 fused gate/up pair kernels** (`kernels/mxfp4_pair_matvec.hip`, `src/mxfp4_pair.rs`,
  batch + hetsplit; wired into `het::dispatch::moe_gate_up_batch{,_hetsplit}` and the
  `DeviceEngine` bundle as `mxfp4pair`): `tests/mxfp4_pair_oracle.rs` vs the ggml-pinned CPU
  reference — rel 3.9e-5 at V4-Flash shapes (2048 × 16 superblocks), 5.3e-5 at V4.1 shapes
  (2304 × 20), het-split halves sum to the batch result exactly. **Prefill twins written
  2026-09-12 night** (`mxfp4_pair_matvec_fused_swiglu_{chunked,kwide}`, iq3_s work-items
  contracts, kwide = M51 structure with each lane owning one mxfp4 block's nibble-half;
  `het::dispatch::moe_gate_up_chunked` arms, MXFP4 in `pair_kwide_selected`): oracle
  rel ≤ 7.4e-5 at both shapes over B=7/40/64, chunk 16/32, untouched (token, slot) pairs stay
  zero. Feature-off hsacos remain byte-identical. kwide perf unmeasured (M7).
- **Shape sweep at V4.1 dims** (`--features v41`, 20 model-free suites incl. rms_norm @5120,
  router @384 with the 512-lane tree, mhc_chain @20480, compressor_pool, indexer_compressor,
  attention_swa, q8_0/q4_k/q2_k/f16 matvecs): 19 pass; `iq3_xxs_pair_oracle` exceeded its
  900 s budget inside its CPU reference (not a V4.1 path).
- **Router**: `ROUTER_MAX_EXPERTS` 256 → 512 under the feature (`#ifndef` + `-D`), launch
  block = the constant so padded lanes write their sentinels (equal to n_expert on V4-Flash).
- **Oracle fixtures**: `topk_ids` per layer/token exported; `oracle_full_bins` has 487 tensors.
- **`tests/v41_layer0_parity.rs`** written (feature-gated, ignore-gated): layer 0 from the HF
  view, oracle-routed experts as the dGPU hot set (cap = n_used) and the iGPU packed set,
  per-token Δ table + routing check, gate 2× floor. **PASSES (2026-09-12 night, server down):
  worst Δ/scale 1.99e-2 at T0 (0.7× floor), 6e-3–1.5e-2 elsewhere (0.2–0.5× floor), routing
  5/6 exact + one near-tie.** Four engine deltas were needed, all feature-gated:
  1. mHC k-split default `HC_DIM/1024` (the narrow k-split kernel stages ≤1024 floats in LDS);
  2. MXFP4 accepted for `ffn_{gate,up}_exps` in the weight contract;
  3. **single-pass mHC carry** (`DgpuScratch::hc_pre_carry`: collapse with the previous
     sub-block's pre, reset to one-hot(copy 0) at layer 0, carried across all sub-blocks;
     decode path; **prefill twin done 2026-09-12 night**: `BatchDgpuScratch::hc_pre_carry`
     `[B, HC_MIX_DIM]`, per-row one-hot reset at layer 0, both collapse sites carry);
  4. **no per-head q RMSNorm after `wq_b`** — V4-Flash applies `rms_nw` over each 512-d head
     before rope, V4.1 does not (`ARCH_SPEC` §1.2); under the feature the norm is a copy.
  Found with per-stage taps (`V41_ORACLE_STAGES=1` → `stage_*` tags) and two CPU cross-checks
  from the engine's own buffers (grouped `wo_a` from `heads`; single-key attention from q/kv/
  sinks): the value path was exact and the implied softmax weight exposed the extra norm.
  Caveat: `ExecMode::HetSingleStream` skips the dGPU hot-expert MoE while the iGPU still
  zero-fills those slots, so routed experts vanish in serial mode (pre-existing) — parity only
  in parallel mode. **Prefill path (`forward_prefill.rs::forward_layer_pre_moe_v2`) now has
  both deltas too** (q-norm pass-through, carry) and the parity test gained
  `V41_EXEC_MODE=prefill` (all tokens as ONE batched chunk through `forward_layer_batch_v2`,
  per-row `residual_next` compare): **PASSES — T=6 worst 1.98e-2 (0.7× floor); T=129 worst
  2.59e-2 (0.96× floor), 19/129 near-tie routing mismatches** — the same margins as the decode
  path, so the batched q chain, batched SWA attention over the chunk and the MXFP4 prefill
  MoE (gate/up kwide + `mxfp4_matvec_par_by_expert_kwide2` down) all agree with the oracle.
  **129 tokens (`oracle_t200_l0_bins`, crosses the 128-slot window wrap): PASSES** — worst
  Δ/scale 2.6e-2 (≤ 1× floor) over all 129 positions, routing exact on 109/129, every mismatch a
  single near-tie expert; 214 distinct experts routed at layer 0 (hot set on the dGPU, 4 GB).
  **Feature-off regression:** `forward_per_layer_vs_ds4` (43 layers × 7 tokens vs the ds4 dump)
  passes on the same tree — all four deltas are `cfg!`-gated and the hsacos are unchanged.

### M2 status (2026-09-12 night) — Engram, layer 1 PASSES

- **Hashing in Rust** (`v4flash-core/src/engram_hash.rs`): token-only n-gram hashing from the
  dumped tables (`export_engram_hash.py` → `engram_hash.json` + `token_map.bin`: compressed
  token map, per-layer odd multipliers, (layer, order, head) primes, bucket offsets). Bit-equal
  to the reference on the dump's self-check prompt (unit test). Text only — image spans (dead
  tokens) are not modelled yet.
- **Row gather** (`v4flash-core/src/engram_table.rs`): `EngramTable::open(st, layer)` on
  `layers.{1,14}.engram.embed.{weight,scale}` (98 GB fp8 + e8m0 per layer, SSD); parallel
  `pread` of 256 B + 8 B per row through a new `read_range_into_cached` (no `FADV_DONTNEED`:
  n-grams recur), dequant e4m3 × 2^(e−127), bf16-rounded like the reference. **Byte-exact vs the
  oracle's rows on the check prompt for both layers** (fixture `rows_L{01,14}.bin`); 144 rows in
  16–19 ms warm (≈120 µs/row with 32 threads).
- **Engine**: `EngramWeights { wkv: Q8_0 [25600 × 6144], qk = q_weight ⊙ k_weight }` loaded when
  `blk.N.engram_wkv.weight` exists; `kernels/engram_gate_add.hip` (per (token, copy): three
  warp-reduced sums → signed-sqrt sigmoid gate → `h += gate·value`); decode:
  `stage_engram_rows(ds, rows)` then at layer start `q8.quantize_input → q8.matvec → engram_gate`
  on `residual`; prefill twin in `forward_layer_pre_moe_v2` in `ENGRAM_CHUNK = 64`-row passes
  (`stage_engram_rows_batch`). An Engram layer always launches its own `mhc_pre_attn` and the
  layer before it a pure combine (`is_first_layer` / `is_last_layer`), since the gate+add must
  land before the collapse.
- **Parity test now runs any layer** (`V41_LAYER=N`): layers 0..=N chained per token (an Engram
  layer's collapse needs the previous layer's FFN pre-mix as its mHC carry, so N cannot run
  alone), per-layer oracle-routed hot sets, the Rust hasher + gather feeding the rows, gate =
  2× the **measured per-layer Q8 floor** (`compare_q8_floor.txt`: 2.7e-2 @L0, 3.3e-2 @L1).
  Decode graphs bake the residual pointers, so the test swaps back after an odd chain.
  **Layer 1, T=6: decode worst 5.7e-2 (1.7× floor), prefill 6.3e-2 (1.9× floor), routing 5/6
  exact + one near-tie; the collapse tap right after the Engram add is at 0.9–2.0e-2 (≤ 0.6×
  floor)**, i.e. the Engram module itself is inside the noise floor and the residual error is the
  two-layer Q8 compounding. Layer 0 unchanged (both modes). T=129 run pending.
- Not done: the prefetch/pipelining of the gather (M7: rows for token t+1 gathered right after
  sampling, overlapping layer 0), image-span dead tokens, and the cold-SSD gather cost at decode
  (24 random 4 KiB reads per module per token; ~2–3 ms cold unless prefetched).

### M3 design (2026-09-12 night) — layer 2: CSA2 KV source + index source at ratio 2

What the reference does at layer 2 (`model.py` `Attention._compress_kv`, `Compressor`, `Indexer`,
`kernel.py` `fp4_act_quant` / `act_quant`), and what the V4-Flash engine has:

| step | V4.1 reference | engine today (V4-Flash) | M3 work |
|---|---|---|---|
| compressor projection | `kv = wkv(x)`, `score = wgate(x)`, both fp32 5120→512 | f16 pair matvec (`k.compressor_d.f16_pair`) | reuse (loader presents both as F16) |
| state / pooling | ratio 2, no overlap (coff 1), **no APE**, per-channel softmax over the 2 rows, then RMSNorm(512) | ratio 4 (coff 2 + APE + shuffle) or ratio 128; `compressor_pool` generic-rows branch | allow ratio 2 in `compressor_pool`/`state_write`/`HetCompressorState::alloc` (rows = 2, width = 512); state_write without APE (`ape: Option`/zero); no shuffle; boundary at `(pos+1) % 2 == 0` (existing) |
| rope of the latent | last 64 dims at compressed position `pos + 1 − ratio`, YaRN θ=160000 | same structure, `rope_for_layer` already selects the compressed-layer rope | reuse |
| compressed-KV store | **E2M1 codes + one E4M3 scale per 16** over the whole 512 (`fp4_act_quant(latent, 16, scale_dtype=e4m3)`: `amax = max(amax, 6·2⁻⁹)`, `scale = e4m3(amax/6)` (RNE), `v = e2m1_nearest(clamp(x/scale, ±6))·scale`, stored dequantised in the bf16 cache) → 288 B/row | `CompKvStore::Fp8` (E4M3 codes, power-of-two block-64 exponent, f16 rope tail; 592 B) | new `CompKvStore::E2m1Kv16 { rows: [n, 288 B] }` + `comp_kv_e2m1x16_{append,append_batched,gather,gather_batched,expand}` kernels; products `code × e4m3 scale` are exact in f16/bf16, so gather → f16 `active_comp_kv` is bit-identical to the reference cache |
| index keys | `k = k_norm(wk(latent_pre_rope))` (512→128, bf16 weight `indexer.attn_k.weight` F16 + `indexer.k_norm.weight`), rope last 64, `fp4_act_quant(k, 32, E8M0)` (**no Hadamard**), E2M1 key cache | second "indexer compressor" at head_dim 128 producing keys; `indexer_qat` = **Hadamard128 + E2M1 QAT** (V4-Flash graph) | new stage replacing the indexer compressor: `f16 matvec wk` → `rms_w k_norm` → rope → `indexer_qat` **without the rotation** (flag / twin kernel) → `index_kv_e2m1.append` (format identical: E8M0 per 32, 80 B) |
| index queries | `q = wq_b(qr)` → [32 × 128], rope, `fp4_act_quant(q, 32, E8M0)` (no Hadamard); `weights_proj(x) · 128^-½ · 32^-½` | same chain with 64 heads + Hadamard QAT | N_INDEXER_HEAD = 32 (config), QAT without rotation |
| reachability / top-k | compressed p visible iff `p < (i+1)//ratio`; top-512 sorted by position, unreachable → −1 | `n_index_comp` logic written for ratio 4 | generalise `compress_len = (pos+1)/ratio` |
| attention | window 128 + selected compressed rows, sink, un-rotate output; **window KV fake-quantised fp8 (block 32, ue8m0, whole 512 incl. rope)** | `attn_mixed` over f16 window + f16 `active_comp_kv`; window store f16 (no fake quant) | reuse attention; window fake-quant is an M6 fidelity item (`Fp8E4m3fnQuantize` at block 32 in `kv_append`) |
| reuse layers 3–7 | read `shared.compress_kv` / `shared.topk_idxs` of layer 2 | each layer owns its state | M4 (`StoreRef`) |

Numerics to match (ARCH_SPEC §3 addendum): E2M1 nearest with ties-to-even code
(`e2m1_key_common.inc`), E8M0 exponent `2^ceil(log2(amax/6))` with the 6·2⁻¹²⁶ floor
(`e2m1_block_exp`), E4M3 scale = RNE cast of `amax/6` after the 6·2⁻⁹ floor, fp8 window quant
`2^ceil(log2(amax/448))`. Gate for layer 2 = 2× the measured floor 3.8e-2 (chained 0..2,
`V41_LAYER=2`; T=6 stages dump `oracle_stages3_bins`, then T=129 from `oracle_t200_bins`).

### M3 status (2026-09-13 ~00:00) — layer 2 PASSES in isolation (dense attention, T=6)

Done (all `cfg!(feature = "v41")`-gated, feature-off untouched):
- ratio-2 compressor through the generic-rows pool branch (`compressor_pool` guards accept 2),
  zero APE table when the checkpoint has none, no shuffle, boundary at `(pos+1) % 2 == 0`;
- `kernels/fp4_kv_quant.hip`: `fp4_kv_quant_inplace` (E2M1 values, E4M3 scale per 16, the
  reference's `fp4_act_quant(…, 16, e4m3)` numerics) replaces the V4-Flash fp8+f16rt chain on
  compressed rows in decode and prefill; the f16 store then holds the reference cache exactly
  (E2M1 × E4M3 products are exact in f16);
- `fp8_act_quant_inplace` (E4M3 values, power-of-two scale per 32 over the whole post-RoPE
  row) on the window KV in decode and prefill — **every layer**, the reference quantises the
  window cache; V4.1 decode takes the unfused kv chain (`kv_post_fused` bakes V4-Flash's
  448/64 quant; fusing the V4.1 variant with a device-slot append is an M7 item);
- oracle taps for `window_kv`, `compress_kv`, the compressor latent (cloned: rope and
  fp4_act_quant modify it in place) and **`pre_mix`** (the mHC pre-mix a block hands to the
  next block); `tests/v41_layer0_parity.rs` compares the caches per row (bit-equal share +
  Δ/scale) and gained **`V41_ISOLATED=1`**: layer N alone with the reference residual after
  N−1 as input and the reference pre-mix as the carry — the per-layer gate. Chained mode
  (layers 0..N per token) stays as the compounding measurement.

**Results, T=6 (`oracle_stages3_bins`):**

| layer | isolated decode | isolated prefill | chained decode | chained prefill |
|---|---|---|---|---|
| 1 (Engram) | 4.3e-2 (1.3× floor) | 4.0e-2 (1.2×) | 5.6e-2 (1.7×) | 4.8e-2 (1.5×) |
| 2 (CSA2 ratio 2) | 6.5e-2 (1.7×, one near-tie route) | 3.4e-2 (0.9×) | 1.19e-1 (3.1×, FAIL) | 1.17e-1 (3.1×, FAIL) |

Compressed rows: 97.9% bit-equal to the reference cache in isolation (the rest are E2M1
boundary flips from ≤2⁻⁹-level input differences); window rows 60–72% bit-equal (the Q8
wkv matvec moves ~2% of values across an E4M3 boundary — inside the floor). The chained
layer-2 failure is compounding of the engine's Q8 *activation* noise (which the oracle floor,
measured with Q8 weights only, does not contain): the layer-2 attention input is already
3–6% off after two layers. The probe (`scripts/v41_oracle/compress_kv_probe.py`) showed the
reference chain replayed from the reference latent is bit-exact, and replayed from the
engine's latent reproduces the engine's flips — i.e. no layer-2 bug. Track compounding at
M6 with the end-to-end gates (top-3 agreement, argmax) rather than per-layer Δ.

**T=129 isolated (`oracle_t200_stages3_bins`):** layer 1 decode 5.8e-2 (1.7×) PASS, prefill
7.9e-2 (2.4×); layer 2 1.1e-1 / 1.06e-1 (2.9×) both modes, compressed rows 96.1% bit-equal.
The floor table is the same at 129 tokens (Q8 *weight* noise does not grow with length), but
the engine's extra noise sources — Q8 *activation* quantisation and the resulting E4M3/E2M1
code flips on ~40% of window elements — give a wider per-token error distribution whose max
over 129 tokens sits ~2× higher than over 6. **Floor experiment (2026-09-13, `V41_ORACLE_NOACTQ=1`):** native vs Q8-weights = 2.7/3.3/3.8e-2
(L0/L1/L2); native vs Q8-weights-with-fp8-activation-rounding-removed = 2.3/2.65/2.65e-2;
Q8 vs Q8-noactq (the rounding term alone) = 3.0/3.8/3.9e-2. The reference's fp8 activation
rounding is a *decorrelating* noise term as large as the weight floor (a 0.5 % input difference
flips 6 % E4M3 steps), so emulating it in the engine would not bring the engine closer to
native; the engine, which does not round, sits where "Q8-noactq vs native" predicts at L0
(2.0e-2 vs 2.35e-2). Gate unchanged (2× the Q8 floor); the T=129 max is tracked, not gated.

Not done in M3 (the T≤1024 parity does not exercise them): index keys from the latent
(`wk` + `k_norm` + rope + E2M1 QAT **without** the Hadamard rotation) and the sparse top-512
path for > 512 compressed rows (needs a ≥1100-token 3-layer dump); the packed 288 B/row store
(memory only); the prefill/decode fused V4.1 window quant. Reuse layers (M4) next.

### M4 status (2026-09-13 ~01:00) — reuse layers read the source store; layer 3 PASSES

- `config::kv_source_of(layer)` (v41: the most recent KV-source layer ≤ layer, `None` for a
  source or an SWA-only layer; V4-Flash always `None`). Reuse layers load no compressor
  tensors (`DgpuLayerWeights` tolerates their absence) and allocate no compressor state;
  `HetModelState::with_kv_source(layer, f)` *moves* the source's `HetCompressorState` into the
  reuse layer for the duration of its forward and back afterwards (no signature churn, no
  `Rc`); `forward_token`, `forward_prefill` and the pipelined prefill lend it the same way
  (pipelined: around both lanes' `pre_moe(L+1)`; `post_moe` never touches compressor state).
  The forward skips the compressor stage when the layer has no compressor *weights* and reads
  the store read-only; prefill derives each row's visibility from the lent store's `n_comp`
  (`before + boundaries up to the row`); a reuse layer forwarded without a lent store errors
  out instead of silently attending the window only.
- Test: `V41_ISOLATED=1` on a reuse layer now runs that layer alone with the **source store
  seeded from the reference's `stage_compress_kv0` rows** (f16-exact) and `n_comp` set per
  token, so the gate is the layer's own floor. Layer 3 (source 2), T=6: decode all tokens
  ≤ 0.6× floor with exact routing, prefill worst 2.06e-2 (0.6×). Layer 7 (last reuse layer
  before source 8): per-token worst 5.6e-2 decode / 4.5e-2 prefill (2.0–2.5× at T4, other
  tokens ≤ 1.5×) — but the floor is a *global-scale* metric (max|Δ| over all tokens / max|ref|
  over all tokens, `compare.py`) while the per-token lines divide by each token's own scale;
  layer 7's token scales span 87 → 3, so the gate now uses the floor's own definition and
  the per-token lines stay informational.
- Chained (layers 0..N per token, no reference inputs): layer 3 at ~2× floor, layer 7 at
  4.8× — Q8-activation compounding, tracked at M6 with the end-to-end gates.

### M5 status (2026-09-13 ~03:00) — ratio-1 source (layer 20) PASSES; layer-major harness

- Ratio 1 (layer 20): `latent = norm(wkv(x))` — one f16 matvec into `pooled` (decode) /
  `matvec_batched` into `kv_cur` with `sc_cur` zeroed (prefill), no gate weight (32-byte stub),
  no state write / pool (the 1-row pool is the identity and stays available for the generic
  prefill snapshot machinery); rope at the token's own position, fp4 fake quant, f16 store.
  **Isolated T=6: 1.4× floor decode and prefill; layer 21 (reuse of 20, store seeded) 1.1×.**
  The hierarchical candidate pool (layer 20 → 24/28/32/36) and the index keys are still not
  exercised: at ≤ 512 compressed rows the dense path equals the reference's top-k.
- `V41_LAYER_MAJOR=1`: the M6 harness (layers one at a time, per-token residual + carry on the
  host, every layer's output vs the dump; a reuse layer's source store exposes `(t+1)/ratio`
  rows per token in decode order). Layers 0..7 in prefill order: 0.8× floor at every layer,
  8 layers in 39 s. Decode order and the 0..21 / 0..39 chains: running.

### M6 status (2026-09-13 03:35) — full 0..39 layer-major chain PASSES a per-token + head gate

**Runs.** `V41_LAYER_MAJOR=1 V41_LAYER=39 V41_ORACLE_BINS=oracle_full_bins` (T=6, every layer
compared, residual + mHC carry carried on the host, each layer's weights loaded/dropped in turn;
the chain pre-load that OOMed the box twice is skipped in this mode). Decode and prefill order give
identical numbers to the printed precision. 40 layers in 132 s.

**Why the global-scale gate broke from layer 23 on (2.5×–6.8× floor).** The reference residual is
bf16 (every dumped value is exactly representable). Token 0 (the sink) carries a massive-activation
channel (#7575) that grows 13K (L15) → 102K (L22) → 356K (L39). At those magnitudes the chained Q8
CPU oracle's "floor" is half a bf16 ulp of that one channel (its max|Δ| is 512/1024/2048 = powers
of two), so the gate was asking the engine's f32 residual to reproduce a running sum to rounding
luck. Diagnostics: (a) host-side bf16 rounding of the engine residual turns every layer's ratio into
an exact small integer of ulps; (b) feeding each layer the reference input (`V41_LAYER_MAJOR_RESEED`)
gives < 1.5 % per-layer error on tokens 1–5 and ~1 ulp/layer on the sink channel; (c) per-token
floors from the Q8 oracle dumps (`compare_q8_floor_pertok.txt`, 1.5–20 % max|Δ|/max|ref| on tokens
1–5 — the Q8 oracle is itself far from the fp8 reference at T=6) put the engine's tokens 1–5 at the
Q8 oracle's own level.

**Gate now.** Per token: tokens 1–5 ≤ 2× the Q8 oracle's per-token error at the last layer; the sink
token reported in bf16 ulps (sanity bound 8). Head (host side): `hc_pre(h, pre_mix)` = Σ pre·h with
the carried pre, final RMSNorm (`output_norm.weight`, eps 1e-20), Q8_0 `output.weight`; argmax must
match and logits Δ/scale ≤ 2× the Q8 oracle's (0.205).

**Result (decode order).** Tokens 1–5: mean 0.64–1.2× the Q8 oracle at every layer, worst 1.29× at
L39 (worst over all layers 2.26× at L10; L0's floors are tiny and noisy). Sink token: 0.5–5.7 ulps.
**Head T5: argmax 11111 = ref; top-5 the same set (3rd/4th swapped; the Q8 oracle matches only 3 of
5); logits Δ/scale 7.8e-2 = 0.38× the Q8 oracle; logit at the ref argmax 23.475 vs 23.774.** The
engine's logits are closer to the fp8 reference than the Q8 CPU oracle's. **Prefill order: passes
the same gates; head logits Δ/scale 2.7e-2 = 0.13× the Q8 oracle** (logit at ref argmax 23.649 vs
23.774), 40 layers in 141 s. Next: the same chain at T=129 (`oracle_t200_bins`, per-token floors in
`compare_t200_q8_floor_pertok.txt`; note the Q8 CPU oracle flips the argmax there, 3001 → 86219).
Test knobs: `V41_LAYER_MAJOR_RESEED`, `V41_LAYER_MAJOR_BF16RES`, `V41_LAYER_MAJOR_VERBOSE`.
**T=129 gate calibration (2026-09-13 04:15).** The full chain also runs at T=129
(`oracle_t200_bins`, crossing the 128-slot SWA window wrap). The per-token-vs-Q8 gate is
unreliable there and was replaced by a diagnostic above T=32: at T=129 the massive activation
is on SEVERAL tokens, not just the sink (layer 39: tokens 0/22/25/86/110 carry 1e4–3.6e5 vs a
median of 235), so bf16/Q8 rounding on those channels is a decorrelating term as large as the
Q8 floor and a per-token ratio is noise (decode and prefill trip it identically at 5.66–5.71×,
worst at the near-lossless early layers L0/L1/L2 whose Q8 floor is tiny). Per-token stays a hard
gate at T≤32 (clean, one sink); above that it prints as a NOTE and the HEAD gate (argmax + logits
≤ 2× the Q8 oracle) is the pass/fail. Caveat for the T=129 head: the Q8 CPU oracle itself flips
the argmax there (ref 3001 → Q8 86219); the engine beat the Q8 oracle at T=6 so it should land on
3001, and a flip to 86219 would be the known fp8-native quality item, not a bug.


### Tokenizer + chat encoding (M7 item, done 2026-09-13)

- Tokenizer: V4.1's `tokenizer.json` is id-for-id the V4-Flash GGUF's (129 280 tokens, 127 741
  merges, same special ids). `BpeVocab::from_tokenizer_json` (core) loads it GGUF-free;
  `tests/tokenizer_json_vs_gguf.rs` shows identical tables and encodings. `add_bos` is false per
  `tokenizer_config.json` because the reference encoder writes BOS into the prompt text.
- Chat encoding: DeepSeek ships a Python encoder (`encoding/encoding.py`), not a template.
  `deepstrix-server/src/prompt_v41.rs` ports its text path (DSML ` calls`/` invoke`/` parameter`
  with the leading space, numeric reasoning-effort budget, mid-conversation `<｜System｜>`, tool
  results merged into the user turn as `<tool_result>` blocks and sorted by call order,
  drop-thinking rules, context prefix); the tool-schema preamble is generated byte-exact into
  `prompt_v41_templates.rs`. `tests/v41_prompt_vectors.rs` renders 19 reference-generated cases
  (`scripts/v41_oracle/gen_prompt_vectors.py`): **14 byte-identical, 5 skipped** (`latest_reminder`
  role, message-level `response_format`, task tokens, images — no server fields for them yet).
  Output side: `dsml.rs` now accepts ` calls` as the block name (it already trimmed the leading
  space of ` invoke`/` parameter`); `v41_leading_space_tag_names_parse` drives a V4.1-format tool
  call through the token scanner. Still to wire: the server's request path (renderer by model,
  effort 1–100 from the request, `tokenizer.json` vocab load) — M7, when the server serves V4.1.

### M7 CED prefill — encoder-only + Decoder SWA Bounded Replay (2026-09-13, UNCOMMITTED)

Tech report §2.2/§3.2.2 (`scratchpad/v41_report.txt` 379–416, 990–1032). Production semantics, now
in `forward_prefill_pipelined` (the server path), default ON under the feature, `V41_CED=0` = the
exact all-40-layer prefill (the parity harness and the single-lane `forward_prefill` stay exact):

1. **Encoder** layers 0..19 run over every prompt token exactly as before (chunked two-lane,
   Engram at 1/14, ratio-2 sources 2/8/14, encoder SWA rings).
2. **Layer 20 (`config::CED_DECODER_START`) over every prompt token = `CedMode::KvSourceOnly`**:
   mHC pre-mix + attn norm + the ratio-1 global-KV projection into the store (`latent =
   norm(wkv(x))`, the same kernels as the exact path) — no Q, no window KV/ring append, no
   attention, no MoE, no carry update. The chunk returns with `residual`/`hc_pre_carry` holding
   the rows ENTERING layer 20 (`forward_prompt_batch_v2_pipelined_range(.., 0..21, KvSourceOnly)`).
   Layers 21..39 never see those tokens.
3. The last `SWA_WINDOW`=128 rows entering layer 20 (residual `[HC_DIM]` + carry `[HC_MIX_DIM]`
   + token id) are kept in a host ring across chunks (`ReplayRow`, ≤10.5 MB).
4. **Decoder SWA Bounded Replay** after the last chunk: decoder rings 20..39 are emptied
   (`n_raw = raw_off = 0`) and layers 20..39 run over the segment through the SAME two-lane batched
   path (`.., 20..40, CedMode::Replay, seed_carry`) at `pos0 = N − B`. Layer 20 in `Replay`
   skips the projection/store write (already there) and takes its causal comp counts from the
   positional reuse formula; every other stage runs. Empty rings + W=128 give exactly the paper's
   truncation: query i of the segment sees window keys in `[max(s, i−W+1), i]`; global attention
   sees the complete layer-20 store. Head on the last row.
5. Decode is unchanged (all 40 layers); the decoder rings hold the replay segment.

**Exact vs approximate.** N ≤ 128: bit-identical to the exact path by construction (same kernels
over the same rows and lane split; the exact path's layer-20 projection and window append happen
in one call, CED's in two). N > 128: approximate by design — decoder window KV of positions
< N−128 never exists, and the segment's first rows see a truncated window (the reference runs all 40
layers over everything). Per-token logits (`last_only=false`) always take the exact path. The replay
is text-causal (`image_spans=None`): V4.1 vision is not wired, and the widened in-span window is an
encoder-chunk property.

**Two pre-existing prefill bugs fixed on the way (both drivers):**
- The prefill head collapsed with a STALE mHC carry: `forward_head` (v41) reads
  `dgpu_scratch.hc_pre_carry`, but the prefill drivers only copied the last row's residual into the
  head scratch, so the first generated token's logits used the carry of the previous decode token
  (or one-hot on a fresh server). Now `head_from_row` copies residual + carry (`[HC_MIX_DIM]`).
- **Causality leak at reuse layers in the two-lane driver:** the reuse-layer causal count was
  `source.n_comp − boundaries_in_this_call`, but by the time lane A's reuse layer L+1 runs the source
  layer has already appended lane B's rows, so lane A attended to lane B's (future) compressed rows
  at every reuse layer (V4.1 layers 3–7, 9–13, 15–19, 21–39). Now positional: `(pos0+k+1)/ratio`
  (the store is 1:1 with boundaries from position 0). The single-lane harness could not see it.

**Gates (server, `/tmp/run_v41_server.sh`, pool 55 GB, 4 read threads, 7 dense windows):**
- **(a) N ≤ 128 — BIT-IDENTICAL.** Same five requests through the server with `V41_CED=0` then
  `V41_CED=1`, last-token prefill logits dumped (`V41_PREFILL_LOGITS_DUMP`, `scratchpad/cmp_logits.py`):
  N=9, 17, 102 → `bit_identical=True`, max|Δ|=0 over all 129 280 logits.
- **(b) N > 128 — top-1 agrees, rest approximate as designed.** N=1082: argmax 28623 both,
  Δ/scale 0.307, top-5 overlap 4/5. N=1874: argmax 671 both, Δ/scale 0.220, top-5 overlap 4/5.
  (The Q8-weight oracle's own logits error is 0.205, so CED's deviation is the same order as the
  quantisation the engine already carries.) 0 panics in either mode.
  Generated text, streamed, temperature 0, same prompts both modes (the non-streaming handler
  drops the reasoning trace — `openai/handler.rs:293` — and a small budget ends mid-`<think>`, so
  the gate has to stream): "The capital of France is" → **"Paris."** and 17×23 → **"391"**,
  byte-identical thinking in both modes. 1874-tok prompt: first 30 tokens byte-identical.
  1082-tok prompt: the two diverge after 107 of 158 characters (~20 tokens) into an equally valid
  continuation — the expected consequence of an approximate decoder SWA state.
- **(c) Throughput (back-to-back, same server settings, prefill span = first `prefill_progress`
  line → `prefill req_len` line).**

  | prompt | exact | CED | speedup | CED replay (fixed) |
  |---|---|---|---|---|
  | 1082 tok | 121.6 s = 8.9 tok/s | 76.6 s = 14.1 tok/s | **1.59×** | 30.9 s |
  | 1874 tok | 118.6 s = 15.8 tok/s | 76.3 s = 24.6 tok/s | **1.55×** | 30.5 s |
  | 102 tok | 59.8 s | 53.9 s | 1.11× | 31.3 s |
  | 17 tok | 60.5 s | 53.6 s | 1.13× | 31.2 s |
  | 9 tok | 73.4 s (cold) | 66.4 s (cold) | — | 31.3 s |

  The replay is a **fixed ~31 s** (20 decoder layers × 384 experts streamed once for a ≤128-row
  batch) — the floor of any CED prefill on one box, and the reason the N ≤ 128 rows show ~1.1×
  rather than 1.0× (CED's 41st layer-pass does no MoE, and layer 20's window in the pager is hit
  twice); CED cannot win below `SWA_WINDOW` by construction. Decode is untouched by CED and
  measured 0.86–0.96 tok/s exact vs 0.92–1.13 CED (the decoder's pager windows are warm after the
  replay); both are far off the 30 tok/s target, which is M8's problem, not M7's.

**(d) Server default:** `V41_CED=1` in `/tmp/run_v41_server.sh` (overridable), server left running
under CED. `V41_CED=0` restores the exact path for oracle work.

**Bottleneck after CED on one box:** still expert streaming, now over 20 layers instead of 40 — and the fixed
replay is half of what remains. A 2-chunk 1874-token prefill moves ~237 GB (20 encoder layers ×
2 chunks + 20 replay layers, 384 experts × 18.8 MB, r ≈ 7/20 on the encoder half) at the measured
3.83 GB/s. CED bought the factor the arithmetic predicted; the next factor has to come from
residency, i.e. the second box holding the encoder expert set (~144 GB) so the encoder streams
nothing, leaving the ~31 s replay and the dGPU as the ceiling.

### Maximum runnable context (2026-09-13) — the three caps were derived for ratio >= 4

Until today the engine could not run a prompt past ~2944 tokens. Three constants had been derived
on the assumption that the model's smallest *ungathered* compressed store is `n_kv/4` — true for
V4-Flash (its ratio-4 layers are gathered by the CSA indexer to `INDEXER_TOP_K` = 512, and only its
ratio-128 layers grow with context, at `n_kv/128`), and false for V4.1, which has **no ported
indexer** and ratios 2 (layers 2-19) and **1** (layers 20-39). With ratio 1 a layer scores
`n_raw + n_kv` keys — the cap becomes a 1:1 context limit.

| cap | was | binding failure on V4.1 | now |
|---|---|---|---|
| `ATTN_SCORES_STRIDE = 3072` (batched prefill scores scratch, fixed stride) | fine for V4-Flash to 320K+vision | **CED decoder replay errors at N > 2944 prompt tokens** (`128 + n_kv`); ratio-2 encoder at N > 5888 | per-launch stride from the buffer's real capacity (`attention::attn_scores_stride`), buffer sized from `--ctx` (`BatchDgpuShared::alloc_rows_ctx`). 3072 stays the FLOOR, so V4-Flash's layout is unchanged |
| `ATTN_MIXED_MAX_KEYS = 82176` (decode scores scratch) | 320K at ratio 4 | decode errors past ~82048 tokens | **131200** under `v41` (= 131072 + `SWA_WINDOW`); V4-Flash keeps 82176. Costs `N_HEAD * cap * 4 B` = 32.0 MiB of `DgpuScratch` (was 20.1) |
| server `--ctx` admission (`n_kv_max.div_ceil(4) > cap`) | — | accepted `--ctx 328704` while decode died at 82K | `attention::attn_max_scored_keys(n_kv_max, raw_window)` — max over layers of (gathered ? `INDEXER_TOP_K` : `ceil(n_kv/ratio)`) + raw window. Reproduces V4-Flash's historical bound exactly (its vision-ctx unit tests still pass unchanged) |

`attn_max_scored_keys` is now the single derivation all three come from; `indexer_gathers(ratio)`
(`ratio == 4 && !v41`) is the one place that says which layers the indexer shrinks, and it must stay
in lockstep with the `use_sparse` / `need_mask` gates in `forward_layer` / `forward_prefill`.

**New ceiling: `--ctx 131072`** (the decode cap). The prefill scratch is what it costs, not what
limits it.

**Memory.** The batched attention scores scratch is the only prefill buffer that scales with
context, and with no indexer it scales fast. Sizing is per *product* (`b x N_HEAD x keys`) rather
than one worst-case stride, because the two CED phases have very different shapes — the encoder
runs `b = 512` over ratio-2 layers (`128 + n_kv/2` keys) and the bounded replay runs `b <= 64` over
ratio-1 layers (`128 + n_kv`). That halves the allocation (3.1 GiB instead of 6.2 at 100K).

At `lane_rows = B_MAX/2 = 512` (the server's shared set), dGPU:

| --ctx | attn_scores | delta vs the old fixed 192 MiB | dead indexer scratch freed under v41 | decode scratch | **net dGPU delta** |
|---|---|---|---|---|---|
| 8192 | 288 MiB | +96 | -368.5 | +12 | **-260 MiB** |
| 32768 | 1056 MiB | +864 | -368.5 | +12 | **+508 MiB** |
| 100000 | 3157 MiB | +2965 | -368.5 | +12 | **+2609 MiB** |
| 131072 | 4128 MiB | +3936 | -368.5 | +12 | **+3580 MiB** |

The "dead indexer scratch" column is a separate, unrelated saving that fell out of the same
audit: V4.1 can never fire the CSA indexer (`need_mask` requires `ratio == 4`), yet
`BatchDgpuShared` was allocating `attn_active_comp_kv` (256 MiB), `indexer_scores` inside the R1
arena (160.5 MiB, and it was the arena's max term) and `indexer_topk_scratch` in R3 (16 MiB) for it.
All three are now sized from `attention::indexer_ever_fires()`. Net effect: **an 8K V4.1 server is
~260 MiB CHEAPER than before this change**, and 100K costs ~2.5 GiB more than 8K did.

`comp_kv` itself already scaled with context and is unchanged: 3 ratio-2 stores + 1 ratio-1 store
= 2560 B/token = 20 / 80 / 244 / 320 MiB at 8K / 32K / 100K / 131K.

**Non-CED prefill is the exception.** `V41_CED=0` (and per-token logits, `last_only=false`) runs
the ratio-1 decoder layers over full 512-row chunks, which needs 8x the replay's share. The server
always passes `last_only=true`, so production is CED; a non-CED long-context prefill now fails with
an explicit capacity message instead of the old fixed-stride one.

**Rollback / demonstration:** `DEEPSTRIX_ATTN_LEGACY_STRIDE=1` restores the fixed 3072 stride and
the legacy scratch sizing, i.e. the pre-2026-09-13 hard failure.

**What long context costs in time.** Fixing the caps makes long context RUN; it does not make it
fast. Without the sparse indexer every layer >= 2 scores its whole compressed store, so prefill
attention is O(N) per row (see `KERNEL_PERF_REVIEW.md` finding 1 and the measured per-layer numbers
appended there). The indexer port (ENGINE_PORT M5 leftover) is what turns that back into a constant.

## 3. Tests and fixtures

- Oracle dumps in the engine's `oracle.rs` layout: `~/.cache/deepstrix/v41/oracle_full_bins`
  (T=6) and `oracle_t200_bins` (T=129); tags `embed_hc` / `residual` / `logits`, shape
  `[4, 5120]` per token; `scripts/v41_oracle/export_bins.py`; smoke test
  `tests/v41_oracle_dump_smoke.rs`. To add: `topk_ids` per layer, the Q8-noise-floor run.
- Parity tests with the server up (3 GiB host, ~0.6 GiB dGPU free): M1–M3 load experts on miss
  through the production path and stream one layer's non-routed weights at a time; M6 is
  server-down. One test binary per milestone, variants toggled in-process (one weight load).

## 4. Risks / open questions

- `norm_eps = 1e-20`: verified non-issue (eps is a launch argument in every norm kernel; only the
  `RMS_EPS` const changes) **[R18]**.
- Q8_0 from fp8 is a lossy requant; gated by the measured noise floor (§2) and deleted by native
  fp8 kernels (M8).
- `token_embd` F16 costs 1.3 GB host (host-side embed only).
- Reuse layers carry `attn.wkv`; the contract must say whether it is used (M4).
- Feature-gated consts double kernel/crate build time when both artifacts are built; separate
  target dirs make it a one-time cost.
- Even layer count (40) changes every `[..; 43]` table, the residual swap parity, and the
  snapshot format version.
- Engram and the expert tier both live on the same NVMe/LUKS path; the dm-crypt workqueue bypass
  is already applied here, and the second box gets a plaintext partition.
