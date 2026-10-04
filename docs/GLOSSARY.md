# DeepStrix glossary

**Status: CANONICAL (owner sign-off 2026-10-04).** This glossary comes from `docs/ONTOLOGY_REVIEW_2026-10-04.md`; the owner's answers are recorded in its §10. It replaces the glossary in `docs/ARCHITECTURE_REVIEW_2026-09-24.md` §2.1.

Its machine-readable half is `scripts/ontology_terms.tsv`. `scripts/ontology_census.py` reports adoption. With `--ratchet`, it fails when a retired name mints a new identifier; turn that on at the `worktree-ms-dspark2` merge (review §9).

## How a name is chosen

Take the first rule that gives an unambiguous name:

1. **The model's own name** in the reference (`inference/model.py`, summarised in `docs/v41/ARCH_SPEC.md`), for anything the model defines.
2. **The owner's working vocabulary**, for engine and deployment concepts. This is the vocabulary of commit messages, measurements and design docs: stream, box 2, hub, phase, lane, pin, drafter.
3. **The spelling that already dominates** the code.
4. **A new word.** Use one only when rules 1–3 are ambiguous.

A platform or external API keeps its own name inside its own boundary: HIP's `Stream`, OpenAI's `stream` field, the checkpoint's `mtp.*` tensor names.

The 09-24 glossary reversed this order. It invented `seq`, "raw window", "expert server" and "stream = HIP stream only". None of them took, and the alias counts grew instead.

**One word, one axis.** Every name answers exactly one of the questions below. Most confusion in this tree is a word of one axis reused on another. "Prefill" is a schedule word used as the name of a code path. "Hot" is a residency word used for placement. "Slot" is a residency word used for a stream.

| Axis | Question it answers |
|---|---|
| **Model** | What math is computed, and in which number format? |
| **Work** | Over which rows, for which request? |
| **Placement** | Which box and device compute it? |
| **Residency** | Where do the expert weights live? |
| **Schedule** | When does it run, and in what order? |
| **Transport** | How do the two boxes talk? |
| **Lifetime** | How long does a piece of state live? |

## Model

These names follow the reference. Engine code uses them as they are.

| Term | Means | In code | Not this / retired |
|---|---|---|---|
| layer | One of 40 backbone blocks (`Block`) | `layer`, `il` | — |
| encoder layers / decoder layers | CED split at `CED_DECODER_START`. At prefill, encoder layers run every prompt row; decoder layers run only the bounded replay window. | `CED_DECODER_START`, `ced_*` | Always "decoder **layer**". A bare "decoder" collides with *decode* (a phase), the DSML decoder and the byte decoder. |
| window | The SWA window: `window_size` = 128 positions | `SWA_WINDOW` | — |
| raw region / raw window | Engine storage for the window. The **raw region** is one stream's per-layer allocation (`ARENA_RAW_ROWS`). The **raw window** is its live span `[raw_off, raw_off + n_raw)`, which in the arena *slides* through slack (`kv_arena.rs:118-131`) rather than wrapping. | `raw_off`, `n_raw`, `_dec` twins; `RawWindow` (missing today) | **"decoder ring(s)" for the arena's windows.** KB #25, #28 and #42 were raw-window counter bugs filed under that name. The legacy single-sequence path is a real ring and may keep the word. `raw_window` in `engine_worker.rs` is the V4 image window; it goes with the V4 code. |
| append row | The raw row a token writes | `slot_per`, `slot_per_dec`, `kv_slot_dev` → rename | "slot" |
| compressor / compressed KV | Reference names. Storage: **comp store**, **comp region**. | `CompStore`, `CompRegion` | "compressed" alone: Engram's hashed ids are also "compressed" (KB #23) |
| indexer / index keys / selection | The indexer scores compressed positions. Index keys are its per-stream state. The selection is its top-k. | `indexer_*`, `index_k`, knobs `V41_INDEXER_*` | knob prefixes `V41_INDEX_*`, `V41_IDX_*`; `idx` except as a generic index |
| kv_source / index_source layer | Reference names | `index_source_of`, `is_index_source_layer` | "S2", "store" |
| router | Computes each row's top-6 picks. The reference class is `Gate`; the engine keeps `gate` for the SwiGLU gate projection. | `router_*`, `d_router` | — |
| routed expert / shared expert | Reference names | — | — |
| pick | One routing choice: (row, rank k ∈ 0..5, expert, weight) | `sel`, `ew` (wire), `pick` | **"slot"** for the rank or a pick position |
| Engram | Reference name (layers 1 and 14) | `engram_*` | — |
| mHC | Hyper-connection mixing (`hc_pre`, `hc_post`, mixes, sinkhorn) | `hc_*`, `mhc_*` | — |
| head | The LM head (`ParallelHead`, including the HC collapse) | `head.rs`, `forward_head` | "hot head" (say **top experts**) |
| **drafter** | The engine's drafting role: the three `DSparkBlock`s plus the Markov and confidence heads. The reference has two names here: `mtp` for the checkpoint namespace and container (`self.mtp`, `n_mtp_layers`) and `DSpark*` for the classes. "Drafter" is the owner's word for the role. | `drafter`, `draft_*`; `DSpark*` for model parts | **`mtp`, `Mtp*`, `MTP_*` for runtime state.** The string `"mtp."` stays in the loader. The `V41_MTP_*` env names wait for registry aliases. Owner's choice (review §10 Q2: option a). |
| drafted block / K | Up to `block_size` drafted tokens / how many of them are verified this step | `Drafted`, `k_for` | — |
| weight format | The number format of a stored tensor (MXFP4 v2, Q8_0, IQ2_S, …) | `GgufType` → `QuantType` | `GgufType` for a safetensors model; "GGUF" for the HF checkpoint path (`--gguf <HF dir>`) |

## Work

| Term | Means | In code | Not this / retired |
|---|---|---|---|
| request | One `/v1/chat/completions` call | **`RequestId`** = the `chatcmpl-…` id, carried to the worker, logs and evtrace (missing today) | — |
| turn | One user→assistant exchange | — | — |
| conversation | Turns that share a prompt prefix | — | — |
| lineage | Prefix-trie ancestry (`KV_PREFIX_STORE_DESIGN.md` §8.2) | — | lineage meaning "session id" |
| session hint | The optional client field `session_id` | `session_id` | "session" for resident KV (`LiveSession`) |
| **stream** | One live sequence in the KV arena, from reservation to release. The owner's word (multistream, lone stream, two-stream). | `Stream`, `StreamKv`; **`StreamId`** newtype (missing today) | **`slot`** for a stream id; **`seq`**, which is the box-2 RPC sequence number. The HIP type keeps its name inside the HIP crate and is imported as **`HipStream`** elsewhere. "SSE" for the response stream. |
| position | A token's KV index | `pos`, `pos0` | "rows" (`V41_MS_CTX_ROWS` is a position budget) |
| row | One (stream, position) input to one forward pass. Its role is prompt, decode or verify. | `b` = row count (kernel parameter); `StepRows` (root / parent / chain) | — |
| step | One forward pass over every live stream's rows (decode or verify) | `forward_step_arena*`, `hub_step` | — |
| tick | One iteration of the scheduler loop | — | "step" |
| **prefill** | Ingesting a prompt. Only that. | `PrefillJob` | the batched layer (`forward_prefill.rs` is a misnomer); "prefill-shaped" for a row count |
| job / chunk / LM window / layer group / unit / replay | The units of prefill. A **job** is one request's prompt suffix. A **chunk** is ≤ `chunk_rows` positions through the encoder layers. An **LM window** (layer-major) is ≤ `V41_LM_ROWS` positions run group by group. A **layer group** is the layers between source layers. A **unit** is one group × sub-chunk forward. The **replay** is the CED decoder-layer pass over the last window. | `PrefillJob`, `LmWindow`, `lm_*` | "layer-major" for any batched forward (it means LM windows only); `prefill_job_chunk` for a unit |
| batched layer | The B-row layer body that prompt chunks, arena steps and verify all run | today in `forward_prefill.rs` | "prefill", `_v2` |
| request shape | How many rows one box-2 request carries | **`RequestShape`** (missing today) | literal `b > 16`, "prefill-shaped", "decode" for b ≤ 4 or b == 1 |

## Placement

There is one process per box, so `b1_`/`b2_` name the machine *and* the state of the process that runs on it, as the other box sees it (`b2_view`, `b2_merge`, `V41_B2_*`). **hub** and **expertd** name the processes in prose, crate names and process-local code.

| Term | Means | In code | Not this / retired |
|---|---|---|---|
| box 1 / box 2 | The machines and, by the rule above, their process's state | `b1_`, `b2_` | `b1` for batch size 1 (use `bs1`); `t2`/`T2` for box 2 |
| hub | The box-1 process: HTTP, scheduler, dense layers, expert client | `hub_*` | `het` / "het engine", `local` |
| expertd | The box-2 process: pool, readers, expert execution, serve loop | `deepstrix-expertd` | `remote` (correct only inside the hub's client), `shard` for the machine, `peer`, `T2`, "expert server", pronouns |
| dGPU / iGPU | Box-1 devices | `d_`/`i_` fields, `dgpu.`/`igpu.` stages | — |
| home | Where an expert is preferred to live: box 1, box 2, or **both** when replicated (the hot-split simulator's "top 10 on both boxes"). Slow, per expert. The **box-1 home set** is the experts homed on box 1. | today `hot_set`, `box1_owns`, `partition_box2`, `V41_T2_PARTITION` | `owns`, `partition`, "T2", "hub affinity set" |
| placement | Per pick and per lane-layer: which box and device compute it. One value, **`PlacedPicks { box1, box2 }`**. The wire and the exclusion remap are both built from it. | (missing today: four sites compute it) | `owns_eff`, `owns_remote`, `extra_remote`. "Placement file" means the ranked-experts file; rename it **rank file**. |
| miss fallback | The policy that sends a box-1-home pick to box 2 when box 1 lacks it | `V41_T2_CATCHALL`, small-B catch-all | "catch-all tier" for the pool |
| advertised set | Box 2's static HELLO bitmap | `ShardInfo::owns` → `advertises` | `owns` |
| accepts | Box 2 will compute this pick (all-true once paged) | `ExpertShard::owns` → `accepts` | `owns` |
| exclusion remap | Per-layer `i32[N_EXPERT]`: `Here(pool slot)` or `Elsewhere` | **`Remap`** newtype (missing today; KB #4 open) | `hetsplit` remap described as "the dGPU takes it" |
| ExpertKey | (layer, expert). The wire calls packed keys "words". | **`ExpertKey`** (missing today) | open-coded `layer << 16 \| e`; `(i32,u32)` vs `(u32,u32)` |

## Residency

| Term | Means | In code | Not this / retired |
|---|---|---|---|
| pool | Device memory that holds expert weights: the **box-1 pool** (`ExpertPager`) and the **box-2 pool** (`ShardPool`) | — | "global pool" / "unified pool" (two knobs, two boxes), "cold / victim / catch-all tier" |
| **pool slot** | One expert-sized region of a pool. **The only meaning of "slot".** | `slot`, `n_slots` (pool code), evtrace `b2_read.slot` | stream id, append row, pick rank, drafter source index, accumulator block |
| resident / pinned / held | **Resident**: in a pool. **Pinned**: box 2 promised not to evict it. **Held**: the hub relies on that promise. Invariant: `held ⊆ pinned ⊆ resident`. | `PinBook`, `PinLedger` | — |
| pin / release | The hub↔box-2 pin protocol. Only this. | `REQ_FLAG_PIN/RELEASE` | **Evict-protect sets inside box 2 → protect.** That covers `parked_pins`, `ExtraPins`, `cur_pins`, and `ExpertShard::pinned` (a manual rename; KB #37). Host memory is always "pinned memory" / `PinnedBuffer`. "Pinned windows" on box 1 → reserved windows. |
| miss / demand read | A pick whose expert is not resident; the read that serves it | `ensure`, `n_miss` | — |
| read class | **demand**; **certain** (early page, park read); **speculative** (look-ahead, admission, restore) | (missing today; introduce with the `remote_experts.rs` split) | "prefetch" for every class (box 2's `REQ_FLAG_PREFETCH` words also carry admissions and restores) |
| background readers | Box 2's read thread pool | `B2Prefetch` → rename at the split | "prefetch readers" |
| span read | Reading an expert's three roles in two preads | `coalesce` (Rust names); `V41_*_COALESCE` knobs wait for registry aliases | "coalesce" (which reads as *merge*) |
| primary drive / mirror drive | Box 2's two NVMe copies of the checkpoint | `V41_EXPERT_MIRROR_*` | "mirror" for anything else |
| drive route | Which drive serves a read (`split` / `urgency`) | `ExpertRoute` → `DriveRoute`, `V41_B2_ROUTE` | "route" alone (the router routes) |
| box-2 view | The hub's copy of box 2's residency map | `b2_mirror` → **`b2_view`** when `b2_mirror.rs` is split (review S4) | "mirror" |
| substitution / cache prior | Swapping a missed pick for a resident one at route time | `V41_SUB*` | — |
| admission | Hub-side admission of displaced wants. The TinyLFU check is the admission filter. | `V41_SUB_ADMIT*` | — |
| victim | An expert chosen for eviction. Not a tier. | `pick_victim` | "victim cache" / "victim tier", which describe opposite directions (`expert_pager.rs:767` vs `:866`) |
| eviction class | Mode-evict's ordering of victims | today "tiers 2–5" | "tier" |
| pick stats | Expert popularity, with named decayed views | today four counters with three decays → one `PickStats` | — |
| dGPU hot tier | dGPU-resident routed experts. **Inert** under paged experts, which production runs (`het/weights.rs:1451`). | `DGPU_HOT_*`, M56/M61/M63 | delete it, or confine it to Laguna |

## Schedule

| Term | Means | In code | Not this / retired |
|---|---|---|---|
| phase | The hub's alternation between prefill and decode. The hub **declares** it to box 2 on every request (`REQ_FLAG_DECODE`). | `Phase`, `hub_phase`, `REQ_FLAG_DECODE` (documented today as a busy-poll hint only) | lane "phases" (→ waits), `PreMoePhase` (→ stage), link busy-poll "phase" (→ link mode); a row count standing in for phase |
| smoothed phase | Mode-evict's 16-request hysteresis over the declared phase | `prefill_phase` | "inferred" (box 2 is told the phase; it does not guess it) |
| burst | One contiguous run of a phase | `V41_MS_*_BURST_MS` | — |
| lane | One sub-batch of a step's or chunk's rows, with its own buffers, control state and events. Index 0..`MAX_LANES`. The lane cuts and schedule for one step form the **`LanePlan`**. | `lane`, `MAX_LANES` | `bd_a/b/c`, `sync_events_t1/_t2`, `stagger2` |
| stage | A numbered section of the layer body (Stages 1–12) and its timing range | `ms.stage`, `dgpu.*`/`igpu.*` | DSpark's "stage-1 gate" → **draft gate** |
| merge | Box 2 combining two lanes' same-layer requests | `REQ_FLAG_PARTNER`, `b2_merge` | "coalesce" |
| park | Box 2 deferring a paging request (`REQ_FLAG_OOO`) | `park_*` | the scheduler's parked prefills → "waiting for KV room" |
| fast chain | Box 2's direct small-batch execution path | `B2FastChain` | — |

## Transport

| Term | Means | In code | Not this / retired |
|---|---|---|---|
| wire / proto | The hub↔expertd protocol: frames (HELLO, REQUEST, RESPONSE, ERROR), flags, `VERSION` (4) | `proto::*`, `REQ_FLAG_*` | "version 1" (`REMOTE_EXPERTS.md:36`) |
| seq | The RPC sequence number of a request frame. **The only meaning of `seq`.** | `seq` in `hub_req` / `b2_req` | token history (`Stream.seq` → `tokens`) |
| ticket | The hub's handle on one submitted request | `Ticket` | — |
| partial | Box 2's MoE output for one request | `RemotePartial` | — |
| hop timing | The hub↔box-2 latency breakdown | `hop_*`, `t1`..`t4`, `t2_b2`/`t3_b2` | `t2` for anything but a timestamp |

## Lifetime

| Term | Means | In code | Not this / retired |
|---|---|---|---|
| process / stream / request / forward / lane | The lifetimes state can have. A struct holds one lifetime. | — | `Stream` mixes request and stream; `HetModelState` mixes buffers and counters; `MtpCtx` mixes weights, scratch and stats |
| scratch | Reusable device buffers. **Buffers only.** | `BatchScratch`, `DgpuScratch` | control state (KB #22, #27) → **`LaneCtl`**, reset at forward start |
| KV buffers / counters | Device memory vs per-stream position counters | `HetModelState` → split | — |
| carve / commit / rollback | Arena operations: allocate a stream's regions; keep the accepted rows; return to a mark | today `KvArena::admit`, `KvArena::accept(keep)` | "admit" (the scheduler admits *requests*), "accept" (DSpark *accepts* drafts) |
| mark | An in-memory rollback point | `KvMark`, `CompMark`, `CompStateMark` | "comp snapshot" |
| snapshot | On-disk KV, with an explicit kind: `Prompt`, `Checkpoint` or `Switch` | `snapshot::save` | checkpoint *inferred* from empty decoder-layer windows; "partial snapshot" in logs |
| restore | Disk → GPU. Only this. | `snapshot::restore*` | arena rollback, "pin restore", "delta restore" |

## Configuration and telemetry

- **Knob.** A registry entry (`knobs!`), read through the registry only.
  - `V41_` is the deepstrix knob namespace. Its etymology is stale but it is unambiguous, so keep it.
  - After `V41_`, the area names the process that **reads** the knob. A `V41_B2_*` knob that only the hub reads is misfiled, and a misfiled knob silently does nothing (`remote_experts.rs:2089-2093` records one).
  - There is one boolean grammar: the registry's.
  - **An env name is an interface.** Rename it only once the knob is in the registry *with an env-name alias*. Today `.legacy()` aliases a one-value file and `.alias()` aliases a knob-file key; neither aliases an env name.
  - The conversion list is `docs/v41/KNOB_AUDIT_2026-10-04.md`, which lives on `worktree-ms-dspark2` until it merges.
- **Telemetry.** A field uses the glossary's word for what it measures, and one metric has one name across `ms.stage`, `hub_step` and `hub_phase`. A field name is an interface too: rename it only as a schema bump, with `scripts/evtrace.py` and the other readers changed in the same commit. Today, a decode step reports a `prefill_*` stage (`d_prefill_indexer`).
- **History.** No milestone code (`M56`, `S2`, `T2`, `G5g`, `SF-B`, `Tier B`) may be the only explanation in a name or a comment. No `_v2` without a v1.

## Words to qualify or avoid

These words carry several meanings in this tree. Never use them bare; use the qualified term.

| Word | Use instead |
|---|---|
| tier | the box, the pool, the eviction class, or the trace thread (evtrace "Tier A/B" → the writer thread / the device-timing thread) |
| region | arena region (raw, comp) vs pool range (box 2's per-layer range) |
| dense | non-MoE (matvec, FFN) vs reserved windows (box 1's prefill windows) |
| window | SWA window, raw window, LM window, drafter window, merge wait |
| chunk | prefill chunk, store chunk, read chunk, SSE chunk |
| block | drafted block, accumulator block, candidate block, quant block |
| group | layer group, compressor group, expert group |
| split | placement, device placement, drive route, replication, CED split |
| hot | home set, resident, top experts |
