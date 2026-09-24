# DeepStrix architecture and quality review

*Scope: every crate in the workspace at `f38679e`. Read-only. Built from eight subsystem maps, merged by an architect pass, then checked claim by claim by an adversarial reviewer against the code. Corrections from that check are folded in; where it changed a verdict the text says so. Items marked **(verified)** were re-read by both the architect and the reviewer. Line numbers refer to `f38679e`.*

---

## 1. Verdict

It is partly slop, but the slop is in specific places. The kernels, the numerics and the invariant reasoning are good. The kernels have been measured against roofline, the box-2 wire carries Q8_K so remote partials are bit-identical, and state rollback and arena admit check their preconditions. The KLD, oracle and determinism culture is real. The prompt layer's `Seg::Ours/Client` trust split and its golden tests are also careful work.

Everything above the kernels was built up by accretion, not designed:

- Experiments were added next to each other instead of replacing each other. There are two full layer implementations, four arena decode drivers, two chunked-CED prompt drivers, two expert-miss readers, three samplers and three Engram gathers.
- About 300 distinct env-var names are read directly in `src/`, with four boolean grammars and no owner. The production configuration is split across three places, none of them in the repo: `~/run_v41_server.sh`, an inline env prefix of about 15 more vars that exists only in a memory note, and `~/b2_run_expertd.sh` on box 2.
- The `v4flash-kernels` crate (55K lines) holds the kernel wrappers, the whole hub engine, the box-2 daemon and a second model (Laguna).
- There is no CI. The V4-Flash server build no longer compiles, and nothing ever runs the 187 GPU/weights tests in `v4flash-kernels/tests`, which are `#[ignore]` by convention.

That structure is now causing real bugs, not just slowing work down. A KNOWN_BUGS #28 fix landed in one copy of a duplicated driver and not the other. Production doesn't reach the unfixed copy, but a bare launch or a DSpark launch does. Multistream checks a restored snapshot by token ids where the serial path checks bytes. The daemon's pool claims a slot before the read and never rolls it back. A cfg attribute drifted onto the wrong item. The fix is mostly mechanical and can be verified, but step 0 has to be a real fidelity gate, because today's gates are too thin to protect a refactor.

---

## 2. Ontology

### 2.1 Canonical glossary

| Term (canonical) | Meaning | Aliases to retire | Where it lives now → should live |
|---|---|---|---|
| **hub** | Box 1: dGPU + iGPU, scheduler, HTTP | box1, b1, local, "het engine" | everywhere; knob prefixes `V41_B1_*`, `V41_LOCAL_*` |
| **expert server** (`expertd`) | Box 2: pages and computes routed experts | remote, b2, box2, daemon, T2, shard, "she" (forward_prefill.rs:6440) | `het/remote_experts.rs` → `deepstrix-expertd` + `deepstrix-experts` |
| **device split** | Dividing one token's picks between the dGPU hot tier and the iGPU (kernel remap) | het-split, hetsplit, M63 de-dup, `hot_cold_split` (Laguna) | ~12 kernels, `dispatch.rs:70` |
| **box split** | Dividing picks between hub and expert server | remote split, T2 catch-all, partition, "small-B catch-all" | `forward_layer.rs:2499-2630`, `forward_prefill.rs:6342-6600` → `placement.rs` |
| **placement** | The per-pick decision: which box and device computes it | owns, owns_remote, owns_eff, partition, box1_owns | same as above |
| **advertised set** | Static bitmap in HELLO | `ShardInfo::owns`, `RemoteExpertClient::owns` | `remote_experts.rs:452, 5089` |
| **accepts** | "Executor will compute this" (all-true once paged) | `ExpertShard::owns`, `LayerShard.owned` | `remote_experts.rs:2422, 1186` |
| **lane** | One sub-batch of a multi-lane step, with its own buffers, control state and event set | t1/t2 (`sync_events_t1/_t2`), bd_a/bd_b | `engine.rs:495-501`, `batch_scratch.rs` |
| **seq** | One live decode sequence (serial or arena) | slot (arena), stream (`StreamKv`), row, live session | `kv_arena.rs`, `multistream.rs:44`, `engine_worker.rs:867` |
| **slot** | **Only** an expert pool slot on a device | also KV append row, arena stream, MTP source, pick index 0..5 | pager, ShardPool |
| **append row** | Raw-KV row a token writes | `kv_slot_dev`, `slot_per`, `slot_per_dec` | `scratch.rs:201`, `kv_arena.rs:175` |
| **stream** | **Only** a HIP stream | arena sequence, SSE phase | — |
| **raw window** | SWA raw-KV region `[off, off+n)` in an oversized ring | decoder rings, ring, cache, `kv_cache`, region | open-coded in ≥6 places → `RawWindow` type |
| **batched layer** | The B-row layer body (prompt, verify and arena decode all use it) | prefill, `_v2`, layer-major | `forward_prefill.rs:2943-8360` |
| **prompt prefill** | Ingesting prompt tokens | prefill (also used for decode) | `PrefillJob`, `forward_prefill_pipelined` |
| **arena step** | One multi-seq decode step over the KV arena | "prefill-shaped" path | `forward_step_arena*` |
| **decode kernel / prefill kernel** | Kernel regime: per-token vs grouped by expert | `_batch` (means n_used slots!), `_bxn`, `_b512`, chunked, kwide, kwide2, staged, tile8 | quant wrappers |
| **gate_up** | Fused gate+up MoE matvec with SwiGLU | pair (`Iq2SPairMatvec`, `mxfp4_pair.rs`) | wrappers and .hip |
| **two_col** | One weight, two activation columns | pair (`q8_0.rs:352 matvec_pair`) | `q8_0.rs` |
| **dgpu_slot / encoded_slot** | Value of `remap[e]` | `dense` (in every hetsplit kernel) | kernels |
| **Remap** (newtype) | Per-layer `i32[N_EXPERT]`: `-(slot)-1` = here, `≥0` = other device | remap_dev/hosts/stage; sentinels `NO_PICK` vs `SENTINEL_EXPERT` | open-coded in 15 sites across 4 files |
| **ExpertKey** | Packed `(layer<<16)\|expert` | words, key; `(i32,u32)` vs `(u32,u32)` | `expert_pager.rs:44`, `remote_experts.rs:1408` |
| **evict-protect** | Picks not evictable this request | pinned, parked_pins, cur_pins | `ExpertShard.pinned` |
| **pinned memory** | `hipHostMalloc` host buffers | — | `PinnedBuffer` |
| **merge** | Box 2 combining two lanes' same-layer requests | coalesce, partner | `remote_experts.rs:4110, 4544` |
| **span read** | Reading all 3 roles of an expert in two preads | coalesce (V41_B2_COALESCE, V41_PAGER_COALESCE) | two copies of `read_miss_into` |
| **speculative read** | Look-ahead guess | prefetch, hint, look-ahead | REQ_FLAG_PREFETCH |
| **early page** | Demand read started early for queued work | prefetch(`certain=true`), park read | `remote_experts.rs:4273, 4331` |
| **RequestShape** | `Decode1 / Verify(b) / Prefill(b)` | `b<=4`, `b>16`, `b==1`, `b>1` literals | `remote_experts.rs:3352, 3378, 3389, 4876, 5188` |
| **drafter** | V4.1 DSpark 3-layer block-5 drafter | MTP, Mtp*, dspark (MTP was V4-Flash's different head) | `mtp.rs`, `weights.rs:195-335` |
| **residual capture** | Drafter's capture of main-model residuals | capture, `mtp_capture_rows` | `mtp.rs:1670` |
| **graph capture** | HIP graph capture | — | `graph_cache.rs` |
| **comp snapshot** | Compressor state mark for rollback | CompStateMark::capture | `state.rs:681` |
| **snapshot** | On-disk full KV state keyed by blake3 of decoded bytes plus image hashes | disk hit | `snapshot.rs` |
| **checkpoint** | Snapshot saved mid-prefill with **empty decoder rings** (must be marked in meta) | snapshot | `multistream.rs:556, 612` |
| **lineage** | Client session id | session, sid, sessionId hint | `snapshot.rs:291-324` |
| **S2 index source** | Key for reusing a gathered indexer selection = `index_source_of(layer)` | "store", `last_idx_gather_src`, `indexer_saved_store` (docs still say kv_source) | `engine.rs:417`, `batch_scratch.rs:240` |
| **IndexerKind** | `V4FlashCompressor` vs `V41IndexK` | `ratio == 4`, `is_index_source_layer` (two predicates) | `forward_prefill.rs:3968…4913`, `forward_layer.rs:85` |
| **HC ops** | Hyper-connection mix ops | "head" (`head.rs`) | `head.rs`, `mhc_pre_fused.rs` → `hc.rs` |
| **hot tier** | dGPU-resident routed experts (M56/M61/M63) | hot, DGPU_HOT_* | `het/weights.rs` |
| **hub affinity set** | Hub-side live ranking of which experts box 1 keeps | hot_set | `expert_pager.rs:405-446` |
| **knob** | A setting resolved once at startup (or explicitly reloadable) | env flag, knobs file | nowhere → `knobs.rs` |
| **model path** | Checkpoint location (GGUF file or HF dir) | `--gguf`, `gguf_path`, V41_MODEL, V41_HF_DIR, DEEPSTRIX_GGUF | `main.rs:28`, `engine_worker.rs:676` |
| **TensorDesc** | Tensor descriptor with `loc: Gguf{..} \| Synth` | GgufTensor (fake for HF), VTensor, StTensor | `gguf.rs:301`, `hf_v41.rs:56` |
| **scratch** | Transient device buffers **only** | — (control state moves to `LaneCtl`) | `batch_scratch.rs:214-376`, `scratch.rs:183` |

### 2.2 Worst collisions, with evidence

1. **"prefill" means "the batched layer".** Production multi-stream *decode* runs `forward_step_arena*` in `forward_prefill.rs:2322-2940` and obeys `V41_REMOTE_SPLIT` and `V41_PREFILL_PRESUBMIT`. The knobs named `_DECODE` (`V41_REMOTE_SPLIT_DECODE` at `forward_layer.rs:2311`, `V41_DECODE_PRESUBMIT` at `forward_layer.rs:160`) only reach the legacy per-token path. An A/B of a `_DECODE` knob on the multistream server therefore measures nothing. This is the most expensive naming bug in the repo.
2. **"slot" has five meanings.** In `kv_arena.rs::tables()`, `slot` (a stream) and `slot_per` (a raw row) sit on adjacent lines. `n_slots` means number of streams in `KvArena:258` and number of resident experts in `RoutedExpertWeights` (`model_weights.rs:47`). Kernels mix pick index and pool slot in the same expressions.
3. **"owns" has five predicates.** The advertised bitmap diverges from real routing once box 2 runs `--paged` (`remote_experts.rs:2486-2498`). That divergence is why the masked `submit` dropped 77% of experts (documented at `remote_experts.rs:5118-5127`).
4. **"pair" has opposite meanings.** `Iq2SPairMatvec`/`Mxfp4PairMatvec` mean two weights and one activation. `Q8_0Matvec::matvec_pair` (`q8_0.rs:352`) means one weight and two activations. The name is the same and the semantics are opposite.
5. **"dense" has four meanings.** One of them is a kernel variable called `dense` that holds a dGPU pool slot (`iq2_s_pair_matvec.hip:234-262` and 11 other kernels).
6. **"ratio == 4" stands in for "has a V4-Flash indexer".** It appears in 6 branches of `forward_prefill.rs` (3968, 3986, 4187, 4508, 4825, 4913), in `snapshot.rs:1066/1128`, `compressor.rs:79/147` and `state.rs:191`. Under v41 the ratios are `{0,1,2}` (`config.rs:138-141`), so the `ratio == 4` *arms* are dead in the production build. Some sites are ternaries whose other arm is live (`forward_prefill.rs:3968`), so replace them with an `IndexerKind` predicate; don't delete them.
7. **"MTP" vs "DSpark" vs "drafter".** Counts: MTP_ 240, Mtp 68, dspark 125, drafter 144. "MTP" was a *different* V4-Flash mechanism.
8. **Type names say GGUF when it isn't.** `GgufType::MXFP4` names two incompatible byte layouts: ggml v1 on disk, and the engine's v2 super-block produced by `hf_v41.rs:1140`. `GgufTensor` gets fake offsets for HF tensors. `MappedGguf` does not mmap. `--gguf` takes an HF directory. `MmprojHost` holds V4.1's HF tower.
9. **Stage numbers disagree.** "Stage 10" is the iGPU MoE in `forward_layer.rs:3-17` and the shared expert in `forward_prefill.rs:3223-8064`.
10. **Mirror-image knob names.** `DECODE_INDEXER` (off disables sparse, `forward_layer.rs:1462`) and `INDEXER_DECODE` (selects a kernel, `:1534`) are read 70 lines apart.
11. **`is_first_layer` / `is_last_layer`** (`forward_layer.rs:539`) are also true for Engram layers and in standalone mode.
12. **`v2` suffix with no v1.** `forward_prompt_batch_v2`, `forward_layer_batch_v2`, `forward_layer_pre_moe_v2`, `forward_layer_post_moe_v2`: no v1 exists anywhere.

### 2.3 Crate and module naming

| Now | Problem | Target |
|---|---|---|
| `v4flash-hip` | Product is deepstrix; not model-specific | `deepstrix-hip` |
| `v4flash-core` | Doc says "no model knowledge"; contains the V4.1 tensor table, Engram, Laguna tokenizer, glibc trim, box-2 drive policy | `deepstrix-formats` (GGUF, safetensors, numfmt, tokenizer); V4.1/Engram go to the engine |
| `v4flash-kernels` | Wrappers + hub engine + box-2 daemon + Laguna | `deepstrix-kernels` (wrappers only), `deepstrix-experts`, `deepstrix-engine`, `deepstrix-laguna` |
| `v4flash-vision` | fine apart from the prefix | `deepstrix-vision` |
| `het/` | Means dGPU+iGPU, local/remote split and Laguna's two-GPU path; also hosts generic `sync.rs`/`graph_cache.rs` | `engine/` (with `batched/`, `decode/`, `kv/`, `placement/`); generic utilities go to `deepstrix-hip` |
| `forward_prefill.rs` | Holds decode drivers | `engine/batched/{layer,prompt,arena_step,stages/*}.rs` |
| `mtp.rs` | Drafter | `engine/drafter.rs` |
| `head.rs` | HC ops | `kernels/hc.rs` |
| `phase0`, `phase1` | May bring-up crates, no dependents | delete from workspace |
| `embed.rs` (server) | Mostly a GPT-2 byte decoder | `BpeVocab::decode_token_bytes` in formats |

---

## 3. Architecture: current

```
                       HTTP (axum)
                           |
 +-------------------------v--------------------------------------------+
 | deepstrix-server (16.5K)                                             |
 |  openai/handler.rs --#[cfg(v41)] template choice--> prompt.rs /      |
 |      ^   (X1) imports accumulate from engine_worker   prompt_v41.rs  |
 |      |                                                               |
 |  engine_worker.rs (5.6K) <--(X1)--> openai::handler::MIN_TOP_P,      |
 |      | serial loop, finish_decode 2K lines,           dsml, ToolCall |
 |      | DSpark harness, Engram gather #1, sampler #2,                 |
 |      | request-end heap trim (never runs in production)              |
 |      | env::set_var("DGPU_HOT_EXPERTS_FILE") -----(X2)----+          |
 |  multistream.rs (1.3K, production) -- Engram #2,#3,   |          |
 |      | sampler #3, re-derives kernel lane splits (X3),    |          |
 |      | flips forward_prefill::LH_FORCE, set_small_b_*(X4) |          |
 |  snapshot.rs -- serialises HetModelState internals (X5)|          |
 +--------+-----------------------------------------------|----------+
          |  pub fields, 20-arg drivers, statics          | env
 +--------v-----------------------------------------------v----------+
 | v4flash-kernels (55K)                                             |
 |  het/  (~33K) hub engine                                          |
 |    engine.rs  HeterogeneousEngine: kernels + sync fabric + graphs |
 |               + S2 model state + telemetry + TCP client (X6)      |
 |    forward_layer.rs  (per-token layer, 2.7K-line fn)   } two layer|
 |    forward_prefill.rs (batched layer, 8.8K; 4 arena    } impls    |
 |                        drivers, 2 CED drivers, 29 statics)        |
 |    expert_pager.rs  pool + ROUTING POLICY GLOBALS (X7)            |
 |    remote_experts.rs (5.6K) proto + client + DAEMON serve loop,   |
 |                     ShardPool, signal handlers, knobs file (X8)   |
 |    state.rs --imports--> mtp::MTP_BLOCK, batch_scratch::B_MAX (X9)|
 |  ~57 kernel wrappers (~22K) --imports--> het::image_spans (X10)   |
 |  attention.rs: ctx-admission policy, V41_INDEX_K parse #1,#2 (X11)|
 |  laguna*.rs (~5K) --imports--> het::graph_cache, het::sync (X10)  |
 |  dense_gemm.rs --loads--> laguna_moe_tiled.hip  (X10, 2-way dep)  |
 +--------+----------------------------------------------------------+
          |
 +--------v-----------+  +------------------+  +----------------------+
 | v4flash-core (7.3K)|  | v4flash-vision   |  | deepstrix-expertd    |
 |  gguf, safetensors |  |  leaves iGPU     |  |  236-line arg parser |
 |  hf_v41 + box-2 IO |  |  current (X13)   |  |  links ALL of kernels|
 |  knobs & global    |  +------------------+  |  incl. hub engine and|
 |  counters (X12)    |                        |  Laguna (X8)         |
 |  heap.rs (server)  |                        +----------------------+
 +--------+-----------+
 +--------v-----------+
 | v4flash-hip (2K)   |  safe fns that can UAF: slice_view,
 |                    |  Graph::from_raw, copy_from_host_async,
 |                    |  PinnedBuffer<T: any>  (X14)
 +--------------------+
```

**Violations:**

- **X1.** Two-way import between the engine worker and the OpenAI layer (`engine_worker.rs:2995, 5335-5440` ↔ `handler.rs:34`). They are modules in one crate, so this is a layering smell (low), not a build cycle.
- **X2.** The process environment is used as a parameter channel between crates (`engine_worker.rs:959`; also `laguna_chat.rs:331`). This is unsound once threads exist.
- **X3.** The server re-derives the drivers' row split: `b/3 + (i < b%3)` at `multistream.rs:857`, `div_ceil(2)` at `:875, :882`. If a driver changes, logits are silently mapped to the wrong seqs.
- **X4.** Kernel-crate globals are part of the server API: `set_small_b_catchall_max`/`set_single_lane_max` (`engine_worker.rs:3337-3342`), `LH_*`, `take_layer_host_timing`.
- **X5.** Snapshot format is coupled to engine internals. Format v6 exists only because `index_k` was silently not persisted (`snapshot.rs:50-65`).
- **X6.** The engine opens TCP from env and panics on failure (`engine.rs:461-491`).
- **X7.** Placement policy is process-global: `partition_box2`, `hot_set` COUNTS/OWN, `BOX2_MISSED`, `RESIDENCY_HINTS`, `PREFETCH_WORDS`.
- **X8.** The daemon lives in the hub's kernel crate. The linker drops unused hub code, so the cost is rebuild time and version lockstep, not an automatic redeploy.
- **X9.** Core KV state depends on drafter and scratch constants.
- **X10.** Wrappers import upward from the orchestrator, and Laguna and DeepSeek depend on each other.
- **X11.** Model policy lives in a kernel wrapper, and `V41_INDEX_K` is parsed in 4 places: `attention.rs:161, :207`, `forward_layer.rs:155`, `forward_prefill.rs:103`.
- **X12.** Deployment policy (which drive, what split fraction) lives in a format crate as mutable statics (`hf_v41.rs:176-203`).
- **X13.** Ambient device state leaks across crates. The server has a hand-written restore at `engine_worker.rs:5052-5061`.
- **X14.** Soundness is left to callers by the lowest layer.

---

## 4. Architecture: target

```
 L4  apps     deepstrix-server        deepstrix-expertd        deepstrix-cli
              http/openai, prompt/,   serve loop, merge/park,  REPLs, tools
              scheduler (policy only),ExpertShard pool policy,
              SnapshotIndex (keys,    reload of RuntimeKnobs,
              lineage, LRU)           signals
                    |                        |
 L3  engine   deepstrix-engine  --------------+----------+
              Engine { kernels, fabric, knobs, telemetry }|
              Session/Seq API: prefill(), step(), save/load_blobs()
              batched layer (THE layer impl) + stages/*   |
              model/{v4.rs, v41.rs}  (cfg at module edge) |
              kv/{RawWindow, KvView, Rollback, arena}     |
              drafter/, engram/, sampler (host ref)       |
              knobs.rs (typed, resolved once, logged)     |
                    |                                     |
 L2  experts  deepstrix-experts  <------------------------+
              proto (frames, named offsets, caps+layout_version)
              ExpertKey, Remap, RequestShape
              residency core (slot map, tick LRU, victim search: shared
                 by hub pager and ShardPool, GPU-free, unit-tested)
              expert_io (miss read, span read, repack; knobs passed in)
              placement::decide(layer, picks, residency, mode) -> Placement
              MoeExecutor (shared kernel contract), RemoteExpertClient
                    |
 L1  kernels  deepstrix-kernels            deepstrix-formats      deepstrix-laguna
              wrappers only; one MoeGateUp/ GGUF, safetensors      (engine + its
              MoeDown per format table;     (OpenOptions), numfmt,  kernels; depends
              generated kernel table;       tokenizer (PreTok enum) on L0/L1 only)
              common.inc / hetsplit.inc;
              KernelKnobs injected
                    |
 L0  runtime  deepstrix-hip: FFI, RAII, DeviceSlice<'a>, Pod bound,
              DeviceGuard everywhere, GraphCache (capture guard),
              peer_push<T>, Module::load_for_device
```

**What each layer owns.**

- **L0** owns device soundness. Nothing above it calls `set_current` directly.
- **L1** turns a kernel launch into a typed Rust call. It has no env reads and no model policy.
- **L2** is the single definition of the two-box contract: wire format, residency rules and placement. It is shared by both binaries, so the hub and the daemon cannot drift apart.
- **L3** owns all model math: layer, KV, Engram, sampling and state serialisation. The API is shaped around a *sequence*, so the server never touches `HetModelState` fields or picks drivers.
- **L4** owns policy about *requests*: admission, scheduling, snapshot keys and lineage retention, and HTTP.

**Model split.**

- **V4 vs V4.1.** Stop splitting per statement: engine_worker.rs has 77 `cfg(v41)` sites, config.rs 34, forward_layer.rs 15. Put each model behind one module boundary: `engine/model/{v4.rs,v41.rs}`. Each provides `GEOMETRY`, `indexer_kind(layer)`, `stage_indexer`, `stage_compressor`, drafter support, and the Engram and paging switches, selected by one `#[cfg] pub use model::v41 as model;`. Keep the cargo feature, because performance needs monomorphised constants. Enforce both configs in CI.
- **Retire V4-Flash?** The non-v41 server does not compile today (see §5 #4), so this is decision #1 in §8. If V4-Flash is retired, deleting its arms is the largest simplification available.
- **Laguna** becomes its own crate on L0/L1. Its ablation kernels go behind a `bench` feature.
- **GGUF vs HF** is a *format* axis: `WeightSrc::{Gguf, HfSafetensors}`. It must never double as model identity.

**Configuration.**

- `engine::knobs::Knobs` is one typed struct built once before any thread spawns, from defaults, then `deploy/*.env`, then env. It has one boolean grammar (0/1/on/off/true/false, anything else panics) and canonical names `DEEPSTRIX_<AREA>_<NAME>`, with old names accepted as logged aliases.
- Every knob is classified as `Tuning`, `Diagnostic` or `SemanticsAltering`. Semantics-altering knobs log at WARN and are recorded in snapshot meta.
- The full resolved table is printed at startup.
- `RuntimeKnobs` (atomics) is the *explicit* reloadable subset, used by hub_knobs SIGUSR2 and the expertd knobs file. Precedence is file > env > default, printed on reload.
- The v41 defaults become the production values, so a bare launch equals production.
- Scratch sizing takes its inputs as constructor arguments. It never reads a flag (this is the "a bound must not read a runtime flag" rule, already in memory).

**Two-box protocol home.** `deepstrix-experts::proto`, with named offsets (no more `HDR_LEN + 20` at `remote_experts.rs:4434/4481/4687`) and error status constants. HELLO carries `layout_version` (from `mxfp4_tables::MXFP4_LAYOUT_VERSION`) and a `caps` word. The client refuses on mismatch. `docs/v41/REMOTE_EXPERTS.md §2` (still "version 1") is regenerated from the proto constants.

---

## 5. Top problems (ranked)

**1. Duplicated drivers, and fixes land in one copy only. KNOWN_BUGS #28 is still live on the serial path (verified).**
- `prefill_job_finish` has the #28 fix at `forward_prefill.rs:699`: `if job.pos0 == 0 || t > b_seg {`.
- Its twin inside `forward_prefill_pipelined`, at `forward_prefill.rs:2089-2090`, still reads `// Fresh prompt only … if pos0 == 0 {`.
- The serial path reaches the twin through `prefill_suffix` → `forward_prefill_pipelined(last_only=true, pos0=loaded)` (`engine_worker.rs:5270`) after a snapshot restore.
- **Reachability (corrected by the reviewer):** your production launch (`V41_MULTISTREAM=1`, `V41_DSPARK=0`) never reaches the twin. Two things do reach it:
  - Any launch without `V41_MULTISTREAM=1`, because `multistream::enabled()` defaults OFF (`multistream.rs:36-37`). A bare launch therefore runs the serial driver.
  - `V41_DSPARK=1`, because `multistream.rs:333` routes every request to the serial handler whenever the drafter is loaded.
- So this is latent in dev and DSpark runs, not live in production. It still has to be fixed, and it shows the pattern.
- The same duplication appears in the row-capture loops (`613-625` vs `1960-1971`), the replay drains, four arena drivers (`2322/2424/2613/2763`), four Engram-staging copies, seven residual-seeding loops and three perfetto emitters.
- *Why it matters:* this is a fidelity bug, and the pattern will keep producing them.
- *Fix:* rewrite the `last_only && ced` branch of `forward_prefill_pipelined` as `PrefillJob::new` + a `prefill_job_chunk` loop + `prefill_job_finish`, with cancel and callbacks in the wrapper. Then collapse the arena drivers into `arena_step_prepare/finish` plus a `LaneSchedule` enum.
- *Gate:* for `pos0 == 0`, `V41_PREFILL_LOGITS_DUMP` must be bit-identical on 100/1K/10K-token prompts. For restored suffixes longer than 128 rows the output *intentionally* changes, so gate with the MS_DIAG=restore KL test pointed at the serial path.

**2. Two complete layer implementations with diverging policy.**
- Per-token: `forward_layer_impl_inner`, `forward_layer.rs:510-3193` (2,684 lines, three mode flags, four public wrappers and a pass-through shim).
- Batched: `pre_moe_chain` 3025-6220 (3,196 lines), plus route, prep, launch and post (about 5K lines).
- Production multistream decode uses the batched path. The per-token path serves the serial worker, DSpark/MTP and tests. **At b=1 the per-token path is currently the fast one:** a 1-row arena step measured p50 106.6 ms against 58.5 ms legacy (2026-09-23).
- Knob divergence:
  - Box split: `V41_REMOTE_SPLIT` has 5 modes (`forward_prefill.rs:74-137`); `V41_REMOTE_SPLIT_DECODE` has 2 (`forward_layer.rs:2311`).
  - Presubmit defaults to ON for decode (`forward_layer.rs:160`) and OFF for batched (`forward_prefill.rs:351`).
  - mHC variants differ by name set.
- The code admits it: `forward_prefill.rs:327-347` says "the two paths therefore computed different numbers BY CONSTRUCTION".
- *Fix:* don't declare a winner yet.
  - First extract the stages both bodies share (mHC-pre, attention, placement, the knob reads) into common functions, so the two copies stop drifting.
  - Stop adding knobs to either body.
  - Retire the per-token body only when b=1 batched matches its wall time back-to-back **and** passes the pinned-routing golden gate (step 0).

**3. No configuration layer.**
- About 300 distinct env names are read as direct literals in `crates/*/src` (304 unique `env::var("…")` literals, plus helper-mediated ones; roughly 490 across the tree including tests).
- Four boolean grammars: `.is_ok()` at 27 sites (so `X=0` turns the flag *on*: V41_PAGER_NOGRAPH, V41_REMOTE_DBG, ATTN_FUSED), `!= "0"` at 33, `== "1"` at 58, `1|on` at 18.
- Uncached reads per layer: `forward_layer.rs:2795/2827/2852/3067`, `:592/:2033`; `forward_prefill.rs:6374`. `IQ2_VARIANT` (7314) and `Q2K_VARIANT` (7798) allocate a String per lane-layer. The prior audit counted ~500-1200 getenv calls per step and they are still there.
- The production config lives in three places, none of them in the repo:
  - `~/run_v41_server.sh` exports V41_PAGED_EXPERTS, V41_REMOTE_SPLIT(_DECODE), V41_T2_CATCHALL, V41_T2_PARTITION, V41_INDEX_K, V41_PREFILL_UNIFIED_POOL, V41_PAGER_* and V41_REMOTE_ADDR.
  - An inline prefix of about 15 more vars (V41_MULTISTREAM=1, V41_CANDIDATE_POOL=1, V41_SMALL_B_CATCHALL_MAX=8, V41_B1_*, V41_PARTITION_BOX1_SHARE, V41_PAGER_MISS_PAR, ...) exists only in a memory note.
  - `~/b2_run_expertd.sh` on box 2.
- A bare V4.1 launch OOMs (`engine_worker.rs:973-976` only warns), and without V41_MULTISTREAM it silently takes a different driver.
- The script's own comments are wrong in places. `V41_PAGER_WINDOWS=0` is described as "derive from stride and pool", but `expert_pager.rs:1220-1221` maps 0 to exactly one dense window.
- Semantics-altering modes ship unguarded: `V41_REMOTE_SPLIT=3/4`, `LAGUNA_SWA_OFF`, `V41_REMOTE_NOMASK`.
- *Fix:* the §4 knob registry, resolved lazily per knob (the existing `LazyLock` pattern). Resolve-before-threads breaks the in-process A/B tests and the runtime `set_var` of DGPU_HOT_EXPERTS_FILE. *Gate:* the startup knob-table dump is identical under the production env, and the golden gate is unchanged.

**4. No gate harness and no CI. The V4-Flash server build is broken (verified).**
- At `engine_worker.rs:5457`, `#[cfg(feature = "v41")]` now sits on `static REQ_TOUCHED` (docs were inserted between it and its function). `reset_drafter_ring` at `:5588` is unconditional and touches `state.mtp`, while `:5594-5595` defines it again under `not(v41)`. `REQ_TOUCHED` is used unconditionally at `:2054`. That makes three independent compile errors in the non-v41 build. There are 14 `not(v41)` arms in that file and 43 workspace-wide, and none of them are verified.
- 187 of 205 tests in `v4flash-kernels/tests` are `#[ignore]` because they need a GPU and weights. That is normal. The defect is that nothing ever runs `cargo test -- --ignored`.
- `tests/hf_v41_fixture.rs:24-27` returns early when its env var is unset.
- `layer0_parity` is now strict by default. `V41_ALLOW_MISSING_FLOORS` is a loud opt-in, and the gate script must guarantee it is unset. (An older memory note says it "silently relaxes"; it no longer does.)
- 60+ test files assume V4-Flash shapes without a cfg guard.
- *Fix:* Roadmap step 0.

**5. Box-2 pool corrupts residency on a read error, and prefetch threads hold raw pointers (verified).**
- `ensure_layer_inner` claims the victim first (`owner_of`/`slot_of`/`held` at ≈`remote_experts.rs:2733-2739`). It writes `remap_hosts[layer][e]` only after the read, repack and copy succeed (≈`:2800`), and there are `?` exits in between (≈`:2788, 2793, 2802`).
- On an error, `slot_of` says the expert is resident while its remap is 0 ("other device"). `serve()` keeps the shard across connections, so the next request for that expert hits and is computed by *nobody*, with no error.
- Separately, `prefetch_words_ex` passes `OwnerPtr(&self.owner)` and staging pointers to up to 16 detached threads (`:2170-2182`), with no `Drop` that joins them. `OwnerPtr` points into ExpertShard's own by-value field (`:1436`), and `B2Prefetch` (`:1258`) owns the pinned staging those threads pread into. Dropping or moving the shard mid-read is therefore a **use-after-free**, not just a stale lifetime comment. The SAFETY note ("daemon lifetime") is false in the loopback test and the bench.
- *Fix:* a transactional claim (a reserved set, committed on success, rolled back on error). Hold the weights as `Arc<V41HfWeights>`. Add `impl Drop for B2Prefetch` that joins the threads.
- *Gate:* an injected-failure unit test on an extracted `ShardPool`. The happy path must keep the same sha (`T2_CATCHALL=2`, `POOL_FLOOR=0`).

**6. Multistream accepts a restored snapshot on token-id equality (verified).**
- The check is `multistream.rs:489`: `tokens[..r.tokens.len()] == r.tokens`. Synthetic image ids encode only the layout. `r.image_spans` (the content hashes) is never compared, and the session-hint path skips the byte-hash walk.
- A same-session request that swaps in a different image of the same size therefore restores the old image's KV.
- The checkpoint saves at `:571, :636, :661` use `spans_in_range(..).unwrap_or_default()`, which the serial path explicitly forbids (`engine_worker.rs:2626-2631`), because a straddling cut drops every span hash from the key.
- *Fix:* one `verify_restored` (the byte-aligned LCP from `engine_worker.rs:1371`) shared by both paths, and snap checkpoints back to span boundaries. *Gate:* a unit test with two same-layout images under one session. Text-only behaviour is identical.

**7. The DSML scanner entity-decodes tool arguments that the model never escaped (verified via the code's own watch-item).**
- `dsml_attr_decode` (`dsml.rs:1150-1186`) runs on every string parameter.
- The V4.1 reference parser does not unescape. `prompt_v41.rs:343-356` escapes nothing, and `dsml.rs:324-334` admits the prompt no longer teaches the escape.
- A Write tool call containing `&lt;` reaches the client as `<` and silently corrupts files.
- *Fix:* `Dialect { entity_decode }`. V4.1 gets none; V4 decodes only the one sequence the renderer produces. *Gate:* render → tokenize → scan round-trip tests per entity, per build.

**8. Placement policy has no owner.**
- There are two inline if-chains: `forward_layer.rs:2499-2630` and `forward_prefill.rs:6342-6600`. They read globals in `expert_pager.rs` (`partition_box2`, `hot_set`, `BOX2_MISSED`, `RESIDENCY_HINTS`) and `PREFETCH_WORDS` in `remote_experts.rs`.
- The two have drifted. `mark_box2_miss` is only called at `forward_layer.rs:3039`, so the arena path never feeds the victim cache. Look-ahead words from lane A ride on whichever lane submits next (`remote_experts.rs:5239-5244`).
- The masked/unmasked submit choice is separate again (`submit_dispatch`, `:5153`), and inverting it drops or double-counts experts (`:5140-5151`).
- `verify_routing_exactly_once` ignores the dGPU hot tier.
- *Fix:* `placement::decide(...) -> Placement`, with the remote set authoritative and submits always unmasked with that set. Hints and prefetch become explicit `SubmitExtras`. *Gate:* the `T2_CATCHALL=0/1/2` temp-0 sha on both drivers, plus an exactly-once assertion on the wire `sel`.

**9. God files.**

| File | Lines | Worst function |
|---|---|---|
| `forward_prefill.rs` | 8,764 | `pre_moe_chain`, 3,196 lines; 29 statics; ~57 knobs; ~34% comments |
| `remote_experts.rs` | 5,636 | about 14 responsibilities; `serve_connection` ≈700 lines, decoding the same frame 3-4 times (4100/4122/4135/4180) |
| `engine_worker.rs` | 5,595 | `finish_decode` 2941-5009, about 2,070 lines, with a ~250-line plain decode loop buried inside ~1,500 lines of DSpark harness |
| `engine.rs` | — | `forward_token_impl` 922-1618, ~60 lines of loop and ~250 lines of env diagnostics |
| `expert_pager.rs` | — | about 90 fields and 4 residency modes |

*Fix:* extract-method along the existing stage banners into a `LayerCtx`, with no reordering of launches. *Gate:* bit-identical logits plus an identical rocprofv3 kernel sequence for one chunk.

**10. Lane-aliasing hazards are still structural (verified).**
- `engine.rs:495-501` maps every lane ≥2 to `sync_events_t2`. `forward_step_arena_lanes` and `_ready_first` check only `n < 2` (`forward_prefill.rs:2629, 2779`). Four lanes would share events without error; the only thing preventing it is the `V41_MS_LANES≤3` convention.
- `route_probe.rs:63` does `lane.min(1)`.
- The `stage_cap` graph key `layer | b<<8 | lane_hash<<16` (`forward_prefill.rs:821-822`) collides when b ≥ 256, and on a hash collision.
- The root cause is that "scratch" carries cross-phase control state (`batch_scratch.rs:214-376`: remote_ticket, indexer_saved_store, mtp_captured, …). Three past alias bugs came from that.
- *Fix:* `lanes: [HetSyncEvents; MAX_LANES]` with a bail on overflow, `LaneBufs` + `LaneCtl`, and an explicit lane index in graph keys. *Gate:* bit-identical for n ≤ 3, plus a unit test that n = 4 returns an error.

**11. The kernel crate is also the engine, the daemon and a second model.**
- Wrappers import upward (`attention.rs:113` → `het::image_spans`; `laguna.rs:41` → `het::graph_cache`).
- `deepstrix-expertd/Cargo.toml` pulls in all 55K lines.
- About 70 hand-copied arch ladders and 153 `include_bytes!(env!(..))` constants. `DEEPSTRIX_GFX_TARGETS` is a fake knob, because the wrappers hard-code both arch names.
- *Fix:* the crate split in §4, done as pure moves. *Gate:* sha256 of the hsaco files in OUT_DIR is unchanged, and so is the determinism sha.

**12. The server does engine work, and the production path skips housekeeping.**
- Three samplers: the device sampler; an f64 host copy (`engine_worker.rs:3940-3972`); an f32 approximation (`multistream.rs:1078-1127`). They use different top_p clamps (`:2995` vs `multistream.rs:716`).
- Three Engram gathers (`engine_worker.rs:121-193`, `multistream.rs:1196-1215, 1235-1265`).
- Input rows built twice.
- Snapshot code hand-serialises `HetModelState` and re-derives compressor geometry (`snapshot.rs:1066`).
- The request-end block (`engine_worker.rs:1707-1910`) is the **only** caller of `heap::trim_and_stats()` and of `pg.take_miss_shape`. Multistream returns before it at `:1658`, so production never trims the glibc heap, and it pays for `V41_MISS_HIST` accounting that nothing reports. (The old 0.44 → 9.2 GiB RSS figure predates `GLIBC_TUNABLES` arena_max=2, so the current impact is unmeasured.)
- *Fix:* a `Seq` API in the engine, `sample_host` gated against the device sampler, `HetModelState::save_blobs/load_blobs` (format stays v6), and `telemetry::on_request_end` called from both paths.

**13. v4flash-hip has safe functions that are unsound, and three conventions for the current device.**
- `slice_view(&self)` returns an unlifetimed owning-typed alias (`buffer.rs:102-123`, about 559 call sites including the `_mut` and typed variants).
- `PinnedBuffer<T>` accepts any T and hands out `&[T]` over zeroed memory (`:420-450`).
- `Graph::from_raw` is safe and destroys on Drop (`graph.rs:49`).
- `copy_from_host_async` is safe while the DMA outlives the borrow (`:236`).
- The current device is set three ways: about 352 bare `set_current` calls, `DeviceGuard`, and `set_current_cached`. `Device::synchronize` never restores it (`device.rs:108`), and the vision tower leaves the iGPU current, which forced the server patch at `engine_worker.rs:5052`.
- *Fix:*
  - Now: rename `slice_view` to `slice_view_unchecked` or make it `unsafe`. That is grep-able and carries zero risk. Also add an `unsafe trait Pod`, `unsafe fn from_raw`, an unsafe or pinned-only async copy, and guards everywhere.
  - Later: introduce `DeviceSlice<'a,T>` in new code and hot spots only. Views are stored in long-lived scratch structs, so a blanket lifetime rewrite would be structural, not mechanical.

**14. `GgufType::MXFP4` names two layouts, and V4-Flash is probably silently wrong because of it (reviewer upgraded this from PLAUSIBLE).**
- The kernels read layout v2, and v1 is "NOT interchangeable" (`mxfp4_tables.rs:11-28`). The only v1→v2 conversion is on the HF path. The V4-Flash GGUF loader passes `ffn_down_exps` through untouched (`het/weights.rs:895-903`), and the unsloth UD V4-Flash GGUFs carry MXFP4 downs at blk.26 and blk.42.
- **By reading, V4-Flash on a UD GGUF now feeds v1 bytes to v2 kernels on those layers.** That is not run-verified, but it is a strong input to §8 Q1.
- `hf_v41.rs:1140-1158` emits the v2 super-block. The module doc (`:12-16`) and `scripts/v41_convert/to_gguf.py:59-63` still describe v1.
- `hf_v41_fixture.rs` compares bytes and has not changed since `857bca6` (before v2), so it cannot pass for `*_exps`. It hides this by skipping.
- An MXFP4 tensor from a real GGUF would reach the v2 kernels with no layout check (`dispatch.rs:61/144`, `forward_prefill.rs:7183`).
- *Fix:* carry the layout with the tensor, and repack or refuse on the GGUF arm. Fix the fixture; `fixture_or_skip` should panic when `V41_REQUIRE_FIXTURES=1`.

**15. Wrapper and kernel duplication with a latent LDS-overrun hazard.**
- Five formats × four variants of near-identical MoE launchers (`iq2_s.rs:310`, `iq3_s.rs:231`, `iq3_xxs_pair.rs:174`, `iq2_xs.rs:172`, `mxfp4_pair.rs:217`), plus the hand-written vtable `dispatch.rs`.
- The hetsplit decode is pasted into about 12 kernels, and the copies have drifted.
- `MXFP4_KW_MAX_CHUNK = 32` is hard-coded in Rust (`mxfp4_pair.rs:16`) while the kernel honours `#ifndef` (`mxfp4_pair_matvec.hip:366`). A CFLAGS override shrinks LDS while the Rust guard still admits 32. `build.rs:59` exports this only for IQ2S and IQ3S. `LAGUNA_HG_G` sizes a grid at runtime against a compile-time kernel constant.
- *Fix:* a `MoeWorkItems` struct, one `MoeGateUp`/`MoeDown` per format table, `hetsplit_common.inc`, and export of every `#ifndef` geometry macro. *Gate:* a debug launch recorder diff of `(symbol, LaunchConfig, arg bytes)` plus an objdump diff.

**16. Documentation is detached from the code it describes.**
- Doc comments attached to the wrong item: `forward_prefill.rs:62-69, 81-83, 204-206, 284-341, 731-737, 2261-2268, 8577`; `forward_layer.rs:44-84, 268-275`; `remote_experts.rs:1637-1746` (110 contradictory lines on `coalesce_check`), `1277, 1289, 1754, 1822, 3302, 4726`; `expert_pager.rs:282-330`; `engine.rs:417, 502, 696`; `state.rs:272` (says rollback does not restore the compressor, but the code does), `362, 436`; `engine_worker.rs:196, 1300, 1932, 2862, 5002, 5451`; `hf_v41.rs:138, 882, 1210`; `buffer.rs:79`.
- The S2 key docs (`engine.rs:417`, `batch_scratch.rs:240`) describe the *bug* keying.
- Stale defaults: b2_pool_floor documented as 0.90 when the code uses 0.0 (`remote_experts.rs:1545/1589`); `IQ2_VARIANT` "staged (default)" at `7495` when the default is kwide.
- No README and no current architecture doc. `DESIGN.md` describes the single-box May engine. `KNOWN_BUGS.md` lists FIXED items under "## Open". `AUDIT_2026-09-22.md` ends in a pasted JSON dump.
- *Fix:* a mechanical doc pass plus `clippy::empty_line_after_doc_comments`, and move lab-notebook narratives to `docs/journal/`. Codegen is unchanged, so the gate is a build artefact hash.

---

## 6. Delete list

Verification: for each item the subsystem readers ran `grep -rnw NAME` across `crates/`, including `tests/`, and excluded the definition. I re-ran that check for the items marked (✓). Generic method names can hide overloads, so every deletion must be followed by `cargo check --workspace --tests` in **both** target dirs.

**Engine (kernels/het)**
- `forward_layer.rs:3258` `softplus_stable`, `:3269` `topk_desc` and `:3286` `_unused_imports_warn_suppressor` (✓: the remaining uses are local copies in tests and routing.rs).
- `forward_layer_impl` (the shim at `forward_layer.rs:481`).
- `forward_prefill.rs:132` `remote_split_dryrun` (✓, 1 ref) and `:8339` `READY_FIRST_SPINS` (✓, written and never read).
- `route_probe.rs` plus `route_probe_after_layer`: a probe for the layer-20 predictor, which memory declares dead (2026-09-21).
- `forward_prompt_batch_v2`, and `forward_prefill` in its single-lane form. Port their 3 tests to the pipelined driver first; they cannot run V4.1.
- `engine.rs:1104-1195`, the `force_standalone = false && …` branch (✓, unreachable).
- `DeviceEngine.indexer_bitpack`, `.indexer_topk`, `.swiglu_cw`, plus `IndexerBitpack` and `indexer_bitpack.hip` (✓, 0 launch calls).
- `maybe_dump_subtensor_f32_view` (identity wrapper). `engine.rs:1931` `f16_bits_to_f32` (duplicate).
- `BatchScratch` → move to `tests/common`.
- `HetCompressorState::alloc` igpu parameter; `IgpuLayerWeights.rope_params`.
- **dGPU hot tier (settled by the reviewer):** `het/weights.rs:1451-1458` makes `dgpu_hot_experts()` return 0 whenever paged experts are on, which production sets. The server still computes and `set_var`s `DGPU_HOT_EXPERTS_FILE` (`engine_worker.rs:955-959`) for nothing. The M56/M61/M63 hot-tier code, the placement-file writer and `DGPU_HOT_*` are therefore dead in the production configuration, unless non-paged v41 is still supported.
- Pending owner confirmation: the `V41_COMP_ROLLBACK=0` arm (selects documented-buggy behaviour); the `DECODE_PREISSUE` path with `forward_layer_preissued_moe`/`issue_igpu_moe` (default off since M54, "not composed" with het-split); `MHC_FUSED` with `mhc_pre_fused.rs` (a kept negative result); `V41_MS_LANES=3` (measured as a loss, `multistream.rs:826-831`); `V41_PAGER_DECODE_FRAC` (unreachable under the production `V41_PAGER_WINDOWS=0`).
- Under v41, cfg-gate rather than delete the `ratio == 4` indexer-compressor branches (`forward_prefill.rs:4508-4686`, etc.).

**Two-box**
- `RemoteExpertClient::head_ready` (✓: the only other reference is a stale comment), `head_seq`, and `clock_offset_ns` (✓).
- **Merge, not delete:** `submit_flags_unmasked` is byte-identical to `submit_unmasked_flags`, but `deepstrix-expertd/src/bin/bench.rs:107` calls it. Update that caller.
- `MoeExecutor::run`, `xq_host_mut`; `ExpertShard::ensure_layer`, `has_layer`; `ExpertdTracer::machine`; `HINTS_APPLIED` (✓, write-only); the always-0 return of `encode_request`; the unused `let cap` at `:2435`.
- The hint *evict* list: encoded and parsed, never read. Send `n_evict=0`; no version bump needed.
- `ExpertPager::sync_remap_blocking` (✓), `clear`, `dtype`.
- `V41_B2_DECODE_DOWN` diagnostic chain, after checking launch scripts.

**Kernels**
- `q2_k` and `router_topk` `for_arch_serial` plus their `.hip`; the `HcSinkhorn` serial branch (N_HC is const 4); `f16_matvec_narrow_ksplit_reduce`.
- `iq2_xxs` `launch_fused_swiglu`, `_b512`, `_bxn`, `_by_expert`, `_chunked_{inline,lds,padded,zeroidx,nodot}`; `matvec_grouped_pair`; `launch_slotdev`; `launch_inverse_pdev`; the attention bench variants (`_ldsv_db`, `_regv_db`, `_htiled_wmma_f16s`, …). Delete them, or move to `kernels/experimental` behind a `bench-kernels` feature.
- `iq2_xs` wmma modes 2-5, which the code itself says produce garbage.
- `wmma_probe.rs`, `wmma_wsum.rs`, `device_ceilings.rs`, `oracle.rs` and `bin/mtp_probe.rs` → move behind a feature or into dev crates.
- `LagunaModel`: keep only as a test-gated parity reference.

**Server and prompt**
- `snapshot::restore` (✓, 0 callers; rename `restore_vl` to take its place).
- `tokens::is_think_marker` (✓: wire it into the two inline copies, or delete it). `DSPARK_DESYNC`, `Stream.send_failures`, `_session_id`, `let _ = start_pos`, `snapshot::hex` (duplicate).
- `dsml::tool_calls_from_events` (✓), `render_tools_prompt`, `impl Default for DsmlScanner` (hard-codes 128825), `vision_prompt::is_image_start` (✓), `sse::encode_done`, and the `const _` lint silencers at `lib.rs:69-70` and `engine_worker.rs:5446`.
- Research knobs: `V41_SINGLE_LANE_AB`, `V41_SMALL_B_CATCHALL_AB`, `V41_XCHECK_POISON`, plus their statics.

**Core, HIP and vision**
- `WeightSrc::as_gguf`, `hf_layout_direct_capacity`; `expert_run_lens` (✓); `shard_name`; `advise_random` (move its measured negative result to docs); `MappedGguf::drop_page_cache`; `v4flash_hip::device_synchronize`; `stream_priority_range`; `Graph::new`/`Default`; the unused `sys.rs` constants; the `#[ignore]` placeholder test in `hip/lib.rs`.
- `v4flash_vision::TEXT_N_LAYERS`/`N_ROUTED_EXPERTS` (these V4-Flash values are wrong for V4.1). `Tower::load_v41_from` (✓, 0 callers): *use* it instead of deleting it, and drop the second SafetensorsDir.

**Workspace**
- `crates/phase0` and `crates/phase1`: no reverse dependencies; last commits 2026-05-23.
- `scripts/`: the source patchers `e1-4.py`, `patch*.py` (already applied); the duplicates `sched_sim2.py`, `step_misses.py`, `trace_analysis.py` (identical to `v41_sched/*`); the 7 tracked `.pyc`; move the ~110 other one-off analyses from `f9b6ccc` out of git.
- `scratch/` → move to `v4flash-hip/examples` or delete. `artifacts/` → move into the CLI crate's data.
- `docs/v41/AUDIT_2026-09-22.md`: truncate the JSON dump from line 345 and fix the mojibake. Mark `DESIGN.md` historical. Fix or remove the dangling references to `docs/strix-halo-memory.md` and `docs/v41/KERNEL_ROOFLINE.md`.
- Extra target dirs `target-v41-ss`, `target-ooo`, `target-b2`: leave them alone until production stops running out of `target-v41` (see §8).

---

## 7. Roadmap

Ordered by value divided by risk. Every step is a separate PR. "Gate" means the step must not change output unless it says otherwise.

**Step 0: a golden-reference fidelity gate (about 3-5 days including capture; prerequisite for everything else).**

The anchor is the CPU reference, not the engine's own past output. Comparing the engine with itself is how the prefill f16-vs-f32 drift and the submit-mask drop went unnoticed. `scripts/v41_oracle/` already runs DeepSeek's unmodified `inference/model.py` on CPU, bit-deterministically, and `export_bins.py` already converts its dumps into the `oracle.rs` manifest format. What's missing is a frozen corpus and a gate that runs every engine path against it.

*0a. Freeze a golden corpus.*
- **Capture per case:** `embed_hc`, per-layer residuals, last-position logits (plus per-position top-k logits for decode steps), and per-layer **routing**:
  - top-k expert ids (already dumped as `layer_NN_topk_ids.pt` at `oracle.py:195`)
  - top-k weights (**new**)
  - the full pre-top-k router scores (**new**; these give each selection's margin to the (k+1)-th expert)
  - the indexer's selected KV blocks (**new**)
- **Cases:**
  - short (T≈6)
  - past the 128-row SWA ring with compressor pooling (T≈200, extending the existing `oracle_t200` run)
  - a decode continuation of N teacher-forced steps
  - a restored-snapshot continuation with a suffix longer than 128 (#28's shape)
  - an image prompt
  - one long-context case past 16384 (the candidate pool's regime), if the reference can run it in acceptable time
- **Storage:** reference runs cost minutes per layer, so capture once and never regenerate casually. Store the corpus outside git, under a fixed path, with a checked-in manifest giving the sha256 of every tensor, the reference commit, and the model checkpoint hash. A missing or mismatched fixture is a **hard failure**, never a skip or a loosened threshold.

*0b. Two gate modes per engine path.*

Run each case through serial prefill, serial decode, multistream rows at S=1/2/4/8, and restore-then-continue.
- **Pinned routing (numerical fidelity).** Overwrite the engine's routing with the reference's selections, then compare residuals layer by layer and logits by KLD. Thresholds are tight, and any growth is a real kernel or precision regression. Details:
  - Experts are pinned after the router top-k and **before** placement, dispatch and submit masks, so the box split, paging and remote path all run under reference routing.
  - The indexer's top-k is pinned too. It has the same discontinuity, and it's where the >16384 trouble lives.
  - By default only the ids are pinned. Weights come from the engine's own router scores for those ids, so weight drift stays visible and continuous. An option pins the weights too.
  - The override is a typed field on a test/probe config, not another env var read on the hot path.
- **Free routing (end-to-end fidelity).** Compare mean and **max-token** KLD against the reference; a mean hides a cliff. Report every selection flip together with the reference's score margin:
  - A flip where the reference margin is within the pinned run's measured score error is expected.
  - A flip with a large margin is a bug, whatever the KLD says.
- Pinned mode separates "the math drifted" from "the math drifted slightly and a near-tie amplified it", which is exactly what today's single-number gates can't do.

*0c. Tiers in `scripts/gates.sh`.*
- **G-compile:** `cargo check --workspace --tests` for both configs, in separate target dirs.
- **G-unit:** non-GPU tests.
- **G-golden:** 0b above. This is *the* fidelity gate.
- **G-self:** byte-for-byte self-regression on `V41_PREFILL_LOGITS_DUMP`/`V41_DECODE_LOGITS_DUMP`, used only for refactors that claim to be bit-identical. **Prove it deterministic before blessing**: run it twice from cold with the pager geometry pinned (`V41_PAGER_POOL_GB/STRIDE/WINDOWS`, `T2_CATCHALL=2`, `POOL_FLOOR=0`, box 2 restarted per run) and require identical shas. Compare serial against multistream with KLD, not bytes, because their samplers break argmax ties differently (host first-max at `multistream.rs:1079` versus the device kernel).
- **G-ms:** `multistream_step` G1-G5e at S=1..8, 2 lanes.
- **G-launch (advisory):** a debug-only launch recorder in `v4flash-hip` logging `(symbol, grid, block, args)`. Canonicalise device pointers to `(allocation id, offset)` so reordered allocations don't fail it. Log graph-launch keys separately, because captured stages are only seen at capture time.
- **G-host:** host-time counters (`LH_*`/`ms.stage`) per step, no regression allowed. Refactors that add indirection on the per-layer host path (placement, dispatch tables) must pass this, because several paths are already host-bound.

*0d. Plumbing.*
- `fixture_or_skip()` panics when `V41_REQUIRE_FIXTURES=1`, and the gate script guarantees `V41_ALLOW_MISSING_FLOORS` is unset.
- A git pre-push hook runs G-compile and G-unit, since the GPUs are local.
- Fix the cfg attribute at `engine_worker.rs:5457/5588` so G-compile passes, or decide to retire V4-Flash first (§8 Q1).

*Unblocks:* everything.

**Step 1: correctness fixes the review surfaced (small; intentional behaviour changes, each gated separately).**
- (a) #28 on the serial path: the `forward_prefill_pipelined` → PrefillJob wrapper. Bit-identical at `pos0=0`; the restore KL test at `pos0>0`.
- (b) Transactional ShardPool claim and joined prefetch threads, with an injected-failure test. Happy-path sha unchanged.
- (c) Multistream `verify_restored` plus span-safe checkpoints. Text sha unchanged; a new image-aliasing test.
- (d) DSML dialect decoding: round-trip tests.
- (e) Lane bound checks and a full-width graph key (a no-op for n ≤ 3).
- (f) Run `on_request_end` telemetry and heap trim from multistream; check RSS on a long agent session.
- (g) readyz uses the resolved hang deadline.
- (h) V4.1 tool-result image order (`sort_tool_results` vs `handler.rs:244`), and validation of non-function tools.
- The snapshot cache keys on prompt bytes, so prompt-fidelity changes (effort default 50 vs 75, tools without a system message) are **separate, announced** changes, not refactors.

**Step 2: knob registry (2-3 days).**
- Create `knobs.rs` containing every current knob with its current default and current parse semantics, *including* the `.is_ok()` quirks, with each quirk flagged. Replace call sites file by file.
- *Gate:* identical startup table under production env and bare env, plus G-self.
- Then, as separate commits:
  - Check in `deploy/v41-hub.env` and `deploy/v41-box2.env` as the **single source** of the production config, including the inline vars that exist today only in a memory note. Warn at startup when no deploy profile is loaded.
  - **Don't** flip the compiled defaults to production values. Production defaults need box 2 (the connect panics on refusal) and change temp-0 output through pager geometry, which would invalidate every blessed file and break single-box dev runs.
  - The unified bool grammar, keeping the old spellings as logged aliases for one release. A bad value logs an ERROR in a pre-load validation phase that prints the whole table; it does not panic mid-restart.
  - Add the `RemoteSplitMode` and `T2Mode` enums with a truth-table unit test.
  - Stop using `env::set_var`; pass the placement path explicitly. Do this *before* moving any knob to resolve-once.
  - Give in-process A/B tests a `cfg(test)` override API, so they keep their one weight load.
- *Unblocks:* every refactor below (no more drift from four parsers), and trustworthy A/Bs.

**Step 3: mechanical hygiene (1-2 days, no codegen change).**
- The doc-reattachment pass, `KNOWN_BUGS` restructure, `ARCHITECTURE.md`, `OPERATIONS.md`, `KNOBS.md` (generated from the registry), and the move to `docs/journal/`.
- Execute the delete list, after porting the tests that depend on the single-lane drivers.
- *Gate:* G-compile, G-launch unchanged, build artefact hash where only docs changed.

**Step 4: extract pure CPU pieces with unit tests (3-5 days).**
- `RawWindow`, replacing the 6 open-coded copies. Property-test it against the old formulas.
- `KvMark::advanced_by` counter math.
- `proto` with named offsets and trailing-block property tests.
- A GPU-free `ShardPool` plus a shared residency core. *Gate:* replay a recorded pick trace through the old and new pools and require identical victim sequences.
- `ExpertKey` and `Remap` newtypes.
- `RequestShape`.
- Numeric codecs into `formats::numfmt`, with an exhaustive 2^32 / 2^16 equality test before the copies are deleted.
- `BpeVocab::decode_token_bytes`, with an exhaustive per-vocab test.
- `rope_for_layer` into config, with a field-by-field test.
- *Gate:* G-self, G-ms.

**Step 5: split the batched layer (1-2 weeks, highest payoff for day-to-day work).**
- (a) First land or abandon the in-flight branches (`b2-ooo-park` and the builds in `target-ooo` and `target-b2`). A mass move would make rebasing them a hand-merge, which is the failure mode that produced #28. Then do the rename-only commit: `git mv` into `engine/batched/{layer,prompt,arena_step}.rs`, drop the `_v2` suffixes, `prefill_hot_active` → `hot_tier_active`. Gate: G-compile, G-launch.
- (b) Collapse the four arena drivers into `arena_step_prepare/finish` plus `LaneSchedule`. Gate: G-ms at 2 and 3 lanes, bit-identical.
- (c) Extract stages from `pre_moe_chain` one at a time into `LayerCtx` methods. Gate after each: G-launch identical plus G-self.
- (d) `PreMoeCarry` → typestate `ChainOut → RouteOut → PrepOut`, with `RouteOpts` replacing the mutable carry flags.
- (e) `SPECULATIVE_APPEND` becomes a `PassOpts` field. The `LH_*` counters move into a `LayerHostCounters` struct.
- (f) Per-model `#[cfg]` indexer and compressor functions behind `config::indexer_kind`.

**Step 6: placement plus the experts crate (1 week).**
- `placement::decide`, with each branch ported verbatim from both paths. `SubmitExtras` replaces the global queues. The client exposes `take_stats()` instead of the `HOP_*` globals. Collapse the client to one `submit(..., SubmitOpts)`. `expert_io` is shared by both boxes.
- Then move the daemon (serve loop, merge/park, knobs, signals, ShardPool policy) into `deepstrix-expertd`. Add `layout_version` and `caps` to HELLO, a one-time bump to v5; until then, a const assert pinning `VERSION == 4 && MXFP4_LAYOUT_VERSION == 2`.
- `decide` writes into caller-owned reusable buffers, and dispatch uses enums/generics, not `dyn`. This runs per layer per lane on a host path that is already host-bound.
- *Gate:* G-golden in pinned mode (placement must not change any expert's result), the `T2_CATCHALL=0/1/2` sha on both drivers, G-host, the loopback test (un-ignored on the hub), a byte-compare of recorded request frames, and `nm` showing the hub no longer links `serve_connection`.

**Step 7: server over a Seq API (1-2 weeks).**
- Move into the engine: `HetModelState::save_blobs/load_blobs` (v6 bytes unchanged; existing on-disk snapshots must restore to byte-identical continuations); `Engram::rows` (one implementation); `input_row`; `sample_host` gated against the device sampler; `engine.decode_step(StepConfig, …) -> logits`, which owns the driver choice and the row split.
- In the server: one `SessionPlanner::plan` shared by both lifecycles; `SnapshotIndex::save/restore` owning the fingerprint; `TurnAssembler` in `openai/`, with an equivalence test across stream and non-stream folds; `prompt/{mod,v4,v41}.rs` with goldens byte-identical.
- Split `finish_decode` into `decode_loop_plain` plus `dspark/` (gate: temp-0 A/B with DSpark unset and set to shadow).

**Step 8: converge the two layer bodies (measurement-driven; possibly never fully).**
- Today b=1 batched is about 1.8x *slower* than the per-token path (106.6 vs 58.5 ms p50, 2026-09-23). So the first move is to share stages (mHC-pre, attention, placement) between the two bodies, each gated with G-golden in pinned mode.
- Once shared stages have closed the b=1 gap, put `forward_token` on b=1 batched behind a flag, then run G-golden plus a back-to-back tok/s A/B (the noise floor is about 8%). Delete `forward_layer_impl_inner` only if both pass.
- If b=1 batched stays slower, record why, and keep the per-token path, as a thin driver over the shared stages.

**Step 9: crate split and kernel-layer consolidation (1-2 weeks, can run in parallel after step 3).**
- HIP soundness: `slice_view` → `slice_view_unchecked` (or `unsafe`), `Pod`, unsafe `from_raw`, guards everywhere, then delete the server's device restore. `DeviceSlice<'a,T>` goes into new code only. Gate: G-compile plus G-golden on text-after-image.
- A generated kernel table and `Module::load_for_device`; `common.inc` and `hetsplit_common.inc` (objdump diff); `MoeGateUp`/`MoeDown` tables replacing `dispatch.rs` (G-launch); export all geometry macros from build.rs; parallel hipcc.
- Move Laguna into its own crate; split `v4flash-core` into formats plus engine-owned V4.1/Engram; move `heap.rs` to the server.
- Rename the crates `v4flash-*` → `deepstrix-*`, with `use … as` shims.
- Gate: hsaco sha256 identical, G-self.

---

## 8. Open questions for the owner

1. **Is V4-Flash still a product?** The non-v41 server does not compile (§5 #4), `run_deepstrix.sh` was last touched 2026-09-10, and 43 `cfg(not(v41))` arms plus ~60 V4-only tests are unverified. By reading, V4-Flash on UD GGUFs is also numerically wrong today (MXFP4 v1 bytes reach v2 kernels, §5 #14). Retiring it removes the largest amount of code. Keeping it requires G-compile for both configs from now on.
2. **Is the per-token decode layer allowed to die** once b=1 batched passes G-golden, even at a small tok/s cost on the serial and DSpark path? (It is 1.8x faster at b=1 today, so this is a later question.)
3. **DSpark accept** is documented as corrupting output, and loading the drafter sends *all* multistream traffic to the serial path (`multistream.rs:333`). Should the DSpark harness (about 1,500 lines, ~40 statics) move behind a cargo feature, or stay in the production binary?
4. **Production config.** Can the hub script, the inline launch vars and the box-2 script be checked into `deploy/` as the single source (compiled defaults stay single-box-safe)? Should production stop exec'ing out of the dev `target-v41/` dir and use versioned installs instead?
5. **Semantics-altering knobs** (`V41_REMOTE_SPLIT=2/3/4`, `V41_REMOTE_NOMASK`, `LAGUNA_SWA_OFF`, `V41_INDEXER_FORCE`, `DECODE_INDEXER=off`): delete them, or gate them behind `DEEPSTRIX_UNSAFE_DIAGNOSTICS=1`?
6. **V4.1 prompt defaults.** The server maps an absent reasoning effort to budget 50, while the reference and the goldens use 75. Tools with no system message render no schemas. Both are fidelity versus behaviour-continuity decisions, and changing either invalidates the snapshot cache.
7. **Is non-paged V4.1 still supported?** Under `V41_PAGED_EXPERTS` the dGPU hot tier is provably off (`het/weights.rs:1451-1458`). If non-paged V4.1 is not supported, the M56/M61/M63 hot-tier code, the placement-file writer and the `DGPU_HOT_*` knobs can go from the v41 build.
8. **Fp8→Q8_0 requant.** Memory records a 2026-09-20 decision to stop it, but `hf_v41.rs:18` and `push_q8` (`:323`) still requant every fp8 projection. Is that decision pending, or was it reversed?
9. **Laguna.** Is it still active enough to justify its own crate and gates, or should it be archived?
10. **Renames.** Is a workspace-wide rename acceptable (crates to `deepstrix-*`, `het/` to `engine/`, `mtp` to `drafter`, `--gguf` to `--model` with an alias)? It is behaviour-free but touches most of the tree and will conflict with any in-flight branches.
11. **Lane ceiling.** Should `MAX_LANES` be 3, matching the multistream cap, or should the fabric be sized dynamically?
12. **Scripts and journals.** Can the ~110 one-off analysis scripts and the dated docs leave `main` (for a `journal` branch or an external store), or do you want them kept in-tree under `docs/journal/` and `scripts/journal/`?