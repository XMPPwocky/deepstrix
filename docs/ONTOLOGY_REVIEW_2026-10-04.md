# DeepStrix ontology review

*Scope: every crate at `origin/main` `128f7e6`, which is 227 commits after `f38679e`, the commit the 09-24 architecture review covered. The unmerged branches `worktree-ms-dspark2` (which contains `worktree-lm-prefill-prod`), `worktree-kv-prefix-store` and `worktree-architecture-review` were also checked for the vocabulary they add. Nothing was built, run or renamed.*

*Method: two subsystem maps (request → step → KV state; expert residency → placement → wire) plus the model, knob, crate and telemetry vocabulary, all checked against the code. Every count in this document was produced by `scripts/ontology_census.py` (comments stripped, string literals kept) or by a grep quoted next to it. Line numbers refer to `128f7e6`. The proposed vocabulary is `docs/GLOSSARY.md`.*

*Question asked: is the terminology consistent across the whole codebase, and do the terminology and the architecture carve nature at its joints?*

---

## 1. Verdict

**The model layer is carved well.** The engine, scheduling and residency layers are not, and the problem has grown since 09-24.

- **The model layer follows the reference.** Names like compressor, indexer, `kv_source`/`index_source`, Engram, the `hc_*` ops, the CED encoder/decoder layers and `window_size` come straight from `inference/model.py`. Where the engine stays close to those names it is consistent, and its boundaries match the model's (§8).
- **Above the model, names record the order things were built in, not what they are.**
  - The batched layer is called "prefill" because prompt ingestion used it first.
  - The hub engine is `het`, after the first dGPU+iGPU split.
  - Box 2 is "remote" because the code started as a client.
  - The crates say `v4flash` because that model came first.
  - The drafter is `mtp` because V4-Flash had an MTP head.

  Each of those names was right once. Each has since acquired a second referent, and the code now uses one word for two things or two words for one thing. That happens in exactly the places where bugs have clustered (§4, §5).
- **The 09-24 glossary changed nothing.**
  - Every alias it retired gained uses.
  - None of the seven types it proposed exists: `StreamId`/seq, `RawWindow`, `ExpertKey`, `Remap`, `LaneCtl`, `RequestShape`, `Placement` all have 0 uses.
  - The V4-Flash removal decided that day is still sitting on an unmerged branch.

  There were two causes. Nothing measured drift. And several of its picks fought both the reference and the owner's own vocabulary, so the code went the other way (§2).
- **What to change is mostly not renaming.** The worst confusions sit on structural mis-cuts: placement computed in four places, one 10K-line file serving both processes, one live stream split across four structs by a bare `u32`. Fix the cut and most names become right on their own. "Remote", for example, is correct inside a hub-only client and wrong only because the file also holds the daemon. Renaming without the cut just moves the confusion.

---

## 2. What happened to the 09-24 glossary

Counts are occurrences / distinct identifiers in code, `f38679e` → `128f7e6`. `scripts/ontology_baseline.tsv` holds the full table.

| Retired name (09-24 row) | f38679e | 128f7e6 | What happened |
|---|---|---|---|
| `mtp` (→ drafter) | 551 / 56 | 828 / 67 | The new DSpark code imports `mtp::{MTP_BLOCK, …}` and names its own types `MsDspark`/`SlotDraft`. Both vocabularies now coexist. |
| `slot` as a stream id, in arena/scheduler files | 127 / 6 | 406 / 9 | `kv_arena.rs` went from 81 to 195 slot lines and `ms_dspark.rs` from 0 to 63. The `StepRows` branch adds 132 more. |
| `owns` (→ advertised / accepts / placement) | 77 / 10 | 98 / 13 | It gained `owns_remote_some`, `remote_owns_layer` and `partition_masks_by_owns_eff`. |
| decoder ring (→ raw window) | 3 / 1 | 20 / 2 | It gained `decoder_rings_empty`. KB #42 is a decoder-ring bug. |
| `partition_box2`, `T2` | 16 / 4 | 35 / 4 | — |
| raw `env::var("V41_…")` knob reads | 238 / 175 | 333 / 246 | The registry arrived on 10-01 with 77 knobs. In the same period, 71 new knob names were added *outside* it. |
| `pin` (all senses) | 136 / 15 | 1068 / 158 | The pin protocol coined 140 identifiers beside `PinnedBuffer` and box 2's evict-protect `pinned`. |
| `mirror` (all senses) | 36 / 10 | 194 / 24 | `b2_mirror` arrived beside the mirror *drive*. |
| `StreamId`/seq, `RawWindow`, `ExpertKey`, `Remap`, `LaneCtl`, `RequestShape`, `PlacedPicks` | 0 | 0 | Never built, on any branch. |

**Why it didn't take.**

1. **No measurement.** Nobody could see the counts move, and nothing failed when they did. The census and ratchet in this review fix that (§9, Phase 0).
2. **It invented names where the code already had better ones.**
   - **`seq` for a live sequence.** `seq` already meant the box-2 RPC sequence number: 128 of its 227 uses are in `remote_experts.rs`, and it appears in 5 evtrace kinds. In `multistream.rs`, `Stream.seq` is the token history. Meanwhile the owner and every design doc say **stream**: "multistream", "lone stream", "two-stream DSpark", with 80 uses of "stream(s)" in `MULTISTREAM_DECODE_PLAN.md` and 114 in `DSPARK_ARENA_PLAN.md`.
   - **"Stream" for HIP streams only.** That fights the product name.
   - **"Expert server".** It merged a machine with a role, and nobody says it.
   - **"Comp snapshot".** The code's `KvMark`/`CompMark` is better.
   - **"Lineage" as the session id.** `session_id` is never sent: 0 of 318 snapshots carry one (`KV_PREFIX_STORE_DESIGN.md:108`). The KV-store design has since given "lineage" its real meaning, prefix-trie ancestry.

   `docs/GLOSSARY.md` therefore picks names by precedence: the reference first, then the owner's vocabulary, then dominant usage, and invents only when all three are ambiguous.
3. **The cleanup it depended on was never merged.** `worktree-architecture-review` holds 36 commits: the V4-Flash removal, the `v41` cfg collapse, `deploy/`, and the removal of non-paged mode and the dGPU hot tier. Its base is `f38679e`, 227 commits behind main. Main still has 171 `cfg(feature = "v41")` lines, the `ratio == 4` arms and the V4 GGUF paths. Removing V4-Flash is the single largest removal of wrong vocabulary available (§9, Phase 3).

---

## 3. The joints

Every name in the engine answers one of six questions. A word belongs to one of them.

| Axis | Question | The real entities |
|---|---|---|
| **Model** | What math? | layer (encoder/decoder under CED), attention = SWA window + compressed KV + indexer, router → picks, routed and shared experts, Engram, mHC, head, drafter (DSpark blocks + heads) |
| **Work** | Which rows, for whom? | request → turn → conversation; **stream** (a KV sequence); **row** = (stream, position) with a role of prompt, decode or verify; **step** = one forward pass over all rows; prefill job → chunk / LM window → unit; request shape |
| **Placement** | Who computes it? | box 1 / box 2 (machines), hub / expertd (processes), dGPU / iGPU; **home** (slow, per expert), **placement** (per pick), miss fallback (policy), advertised set, accepts |
| **Residency** | Where do the weights live? | pools and **pool slots**, resident / pinned / held, misses and read classes, drives and drive route, the hub's box-2 view, victims, pick stats |
| **Schedule** | When? | **phase** (prefill/decode alternation) and bursts, tick, step, lane and `LanePlan`, stage, merge, park |
| **Lifetime** | How long does state live? | process (weights, pools, knobs), stream (KV, drafter ring, rng), request (SSE, stop, budget), forward and lane (control), scratch (buffers only), mark vs snapshot |

The biggest naming failures are one axis's word used on another:
- "prefill" (Schedule) names a code path (Work).
- "slot" (Residency) names a stream (Work).
- "hot" (Residency) names a placement policy.
- "phase" (Schedule) names lane waits and a link mode.
- "decode" (Schedule) names a row count on box 2 (Work).
- "pin" (a Residency protocol) also names host memory (Lifetime).

The biggest structural failures are one *type* or *file* holding state from two axes or two lifetimes (§4).

---

## 4. Where the architecture cuts in the wrong place

Each item gives the real joint, where the code cuts instead, the evidence, and the right cut. They are ordered by bug history.

**S1. Placement has no single home. Four sites compute it, under one overloaded word.**
- **The joint.** Four different things:
  - **home**: which box an expert is assigned to (slow, per expert);
  - **residency**: what a pool holds (cache state);
  - **placement**: who computes each pick (per pick, per lane-layer);
  - **miss fallback**: the policy that maps home and residency to placement.
- **The code** spells all four as `owns`: 13 distinct identifiers and six predicates.
  - `ShardInfo::owns` (`remote_experts.rs:602`) and `RemoteExpertClient::owns` (`:8249`) are the HELLO bitmap.
  - `ExpertShard::owns` (`:4615`) is "accepts".
  - `hot_set::box1_owns` (`expert_pager.rs:553`) is the box-1 home.
  - `owns_eff` is the final placement.
  - `owns_remote` is the HELLO bitmap in the batched driver (`forward_prefill.rs:8844`) and the *computed decode placement* in the per-token driver (`forward_layer.rs:2519`).
- **Two drivers run different mode ladders.** The per-token driver has six (`forward_layer.rs:2518-2640`); the batched driver has its own (`forward_prefill.rs:8844-9100`). `remote_sel_override` (`:114-168`, new since 09-24) patches the mismatch between them.
- **The client re-masks through a `mask: bool`** across six `submit*` entry points (`remote_experts.rs:8288-8385`).
- **The exactly-once audit checks the hub's `owns_eff`, not the wire `sel`.**
- **Bugs:** KB #0c (every box-1 pick double-counted), #29, #46, and **#5, still OPEN** ("remote wire mask is applied against the static HELLO bitmap"). Each one was "the wire was masked by one `owns` while the hub computed by another".
- **Cut:** one pure `place(…) -> PlacedPicks { box1, box2 }`. The exclusion remap and the wire are both built from it. `submit` takes `box2` with no mask flag. The audit checks `box1 ⊎ box2` against the router's picks. Rename `owns` to `advertises` / `accepts` / `home_of`. This changes the wire masks, so it goes behind the golden gate (Phase 2).

**S2. The batched layer is named after one of its callers.**
- **The joint.** *A forward pass over rows* is one thing; *the role of those rows* (prompt, decode, verify) is another.
- **The code.**
  - `forward_prefill.rs` (11,745 lines) holds the layer body that prompt chunks, arena decode steps (`forward_step_arena*`, `:3662-4316`) and DSpark verify all run.
  - Every multistream decode step therefore reports a `dgpu.prefill_indexer` stage (`:6605`, evtrace `d_prefill_indexer`).
  - The `_DECODE` knob trap from 09-24 §2.2 is still live: `V41_DECODE_PRESUBMIT` (`forward_layer.rs:162`) and `V41_REMOTE_SPLIT_DECODE` (`:2328`) reach only the legacy per-token path. The 10-04 knob audit on `worktree-lm-prefill-prod` confirms they are "inert on the multistream path". An A/B of them on production measures nothing.
  - `_v2` names with no v1 are unchanged: 15 distinct.
  - "Layer-major" has since gained a precise meaning, LM windows (`V41_LM_PREFILL`). Yet `forward_prefill.rs:1967` still uses it for any batched forward.
- **Cut:** the layer goes into its own module, `engine/layer`. Drivers are named by caller: prompt (`PrefillJob`), arena step, verify. A row carries its role. "Prefill" means prompt ingestion only. This is a pure move plus the `_v2` drop; the delete-the-per-token-path question (09-24 Q2) stays separate.

**S3. One 10,402-line file is the whole two-box contract and is compiled into both processes.**
- **The joint.** Three parts: the wire, which both processes need; expertd (pool, readers, execution, serve loop); and the hub's client.
- **The code.** `remote_experts.rs` holds:
  - sockets and clock sync (`79-1239`)
  - the `proto` (`321-978`)
  - the pool, pin, mode-evict and reader machinery (`1359-3790`)
  - `ExpertShard` (`3791-5550`)
  - `MoeExecutor` (`5556-6525`)
  - the serve loop (`6622-7841`)
  - the client (`7842-8680`)

  It grew 5,636 → 10,402 lines since 09-24. `deepstrix-expertd/src/main.rs` is 246 lines.
- **The naming consequence.** "Remote" is deictic: it means "the other box". Half of this file *is* the other box, so in the daemon code "remote experts" are local experts. The same file contains 98 `remote*` identifiers, `shard` (both the box-2 pool and checkpoint files, `:2029`) and `peer` (both the TCP peer and the dGPU→iGPU push).
- **Cut:** a shared `experts-wire` crate (proto, clock, socket), `expertd::{pool, readers, exec, serve}`, and a hub `expert_client`. After the cut, "remote" survives only in the client, where it is correct, and no rename is needed. The cut is the fix.

**S4. `b2_mirror.rs` (new, 2,603 lines) is five things.**
- **The joint.** The hub's **view** of box 2's residency is state. Several policies read that view.
- **The code** puts all of these in one module:
  - the residency view
  - miss substitution and the cache prior (`V41_SUB_*`)
  - the INCOMING overlay
  - the hub half of the pin protocol (`V41_B2_PIN_*`)
  - the TinyLFU admission filter. Its knob is filed as `V41_SUB_ADMIT_GATE`, yet it is a no-op unless pins are active (`:1247`).
- **"Mirror" has two other meanings nearby:** the second NVMe copy (`V41_EXPERT_MIRROR_*`, `hf_v41.rs:175`) and the host mirror of `remap_dev` (`remote_experts.rs:2228`).
- **Cut:** `box2_view`, `pin_ledger`, `substitution`. "Mirror" then means only the drive.

**S5. One live stream is split across four structs, joined by a bare `u32` called `slot`.**
- **The joint.** A **request** is one HTTP call. A **stream** is one KV sequence, and it may outlive a request. The planned RESIDENT streams keep a finished stream's arena slot and serve the next turn in place (`KV_PREFIX_STORE_DESIGN.md:227`). **Draft state** is per stream.
- **The code:**
  - `Stream` sits in `Sched.streams: Vec` and is found by search (`multistream.rs:86`).
  - `StreamKv` sits in `KvArena.streams` and is indexed by slot (`kv_arena.rs:480`).
  - `SlotDraft` sits in `MsDspark.slots` (`ms_dspark.rs:89`).
  - `Pending` / `Prefill` / the parked tuple cover the earlier states.
- **Their lifetimes differ.** The slot is reserved at prefill start, the drafter ring is seeded at prefill finish, and `Stream` exists only from admission.
- **`Stream` mixes request fields with stream fields.** Request fields: `tx`, `cancel`, `max_new`, `sample_mode`. Stream fields: `seq` (the tokens), `compressed`, `next`.
- **The request has no identity past the handler.** The `chatcmpl-…` id is minted at `openai/handler.rs:352` but never reaches the worker: `GenerateReq` has no id field. Scheduler logs name a stream by a slot that is reused across requests. A client report can only be joined to a step by timestamp.
- **Cut:** a `RequestId` carried into the scheduler, logs and evtrace; a `StreamId` newtype; and one table of `StreamEntry { state: Queued | Prefilling | WaitingForRoom | Decoding | Stalled, kv, draft }`. Split request fields from stream fields **before** RESIDENT streams land.

**S6. `HetModelState` is buffers plus counters, used in three roles.**
- **The roles:** the legacy serial sequence, prefill scratch (`spare_states`), and the arena's buffer holder. In the arena role its counters "are meaningless … and stay 0" (`kv_arena.rs:25-27`). `RowLayout::Arena` (`forward_prefill.rs:297-313`) exists to tell the layer to ignore them.
- **Cut:** `KvBuffers` (device memory) plus per-stream counters.

**S7. Scratch holds control state.**
- **The joint.** Reusable buffers live for the process. Control state lives for one forward on one lane.
- **The code.** `BatchDgpuScratch` carries `indexer_saved_store`, `engram_rows_ready`, `mtp_captured*`, `mtp_lane_cut`, `remote_ticket`, `remote_upload_pending` and more (`batch_scratch.rs:240-430`).
- **Bugs:** KB #22 (a long prompt's indexer selection leaked into later forwards) and #27 (lane A masked with lane B's blocks).
- **Cut:** `LaneCtl`, reset at forward start. It has 0 uses today.

**S8. The prefill and checkpoint API is named for its old implementation.**
- **`Prefill.prefix` changes meaning mid-life.** It holds the restored prefix, then becomes the whole prompt (`multistream.rs:1217`, `pf.prefix.extend_from_slice(&suffix)`).
  - **Bug:** KB #42. Checkpoints were keyed as prefix + done rows, so none was ever restorable: 15.2 GB of unrestorable files.
  - The fix (`checkpoint_tokens`) went around the name, which is still `prefix`.
- **`prefill_job_chunk` runs one LM *unit*** and returns 0 rows for most calls (`forward_prefill.rs:1335-1339`).
- **`checkpoint_ok()` is used as "a window is open"** (`multistream.rs:1232`).
- **A checkpoint is not a recorded kind.** It is inferred from empty decoder-layer windows (`snapshot.rs:1482`), and the log calls it a "partial snapshot".
- **Cut:**
  - `Prefill.prefix` → `tokens` + `restored`
  - `prefill_job_chunk` → `run_unit`
  - `checkpoint_ok` → `window_open`
  - an explicit `SnapshotKind` in the meta. The KV-store branch's `TailKind`/`Provenance` is the natural home for it.

**S9. Box 2 has five definitions of "decode", and box 1 books decode as prefill.**
- **The joint.** **Request shape** (rows in a request) is not **phase** (the hub's state). Box 2 can only *infer* the phase.
- **Box 2's definitions of "decode":**
  - `b == 1` (`remote_experts.rs:8003`)
  - `b <= decode_max_b` = 4 (`:5951`)
  - `b <= 16` via three independent constants (`PIN_DECODE_MAX_ROWS`, `PARK_MAX_ROWS`, `FAST_CHAIN_MAX_B`)
  - `REQ_FLAG_DECODE`
  - mode-evict's hysteresis
- **`ensure_layer_phased` is named for phase but takes a shape** (`:4726`), and `:6130` mixes the two.
- **On box 1, `count_as_prefill` is keyed on `!speculative_append()`.** Every non-verify arena decode miss is therefore booked as prefill (`expert_pager.rs:250-259`), and the `decode_hit` / `decode_misses` summary fields measure nothing. The comment explaining this is itself stale: it says DSpark is off, but it has been live since 10-01.
- **Cut:** `RequestShape` and `inferred_phase` as types; one named threshold; box-1 accounting keyed on row role.

**S10. One statistic, four implementations.**
- **The joint.** Expert popularity is one statistic, viewed with different decays.
- **The four implementations:**
  - `hot_set` halves at each refresh.
  - `PinLedger` halves every 256 steps, rank-weighted.
  - M62 `expert_stats` halves at 10M tokens.
  - `V41_PICK_TRACE` counts offline.
- **They are already coupled by convention:** "rank by the ROUTER's picks for the same reason" (`b2_mirror.rs:457`).
- **Cut:** one `PickStats` with named views.

**S11. Lane choice is spread over 8 knobs, 4 drivers and 7 local flags.**
- **The pieces:**
  - knobs `V41_MS_PIPELINE`, `_LANES`, `_LANES3_MIN_ROWS`, `_STAGGER`, `_PIPELINE_MIN_ROWS`, `_SPEC_LANES`, `_LANES_LEARNED`, `_LANES_MEMORY`
  - four `forward_step_arena*` drivers
  - local flags at `multistream.rs:1764-1800`
- **The lane split is re-derived** for heads, captures and stats. `lane_rows` warns that the derivations "must agree … a head reading a different split reads a stale row, silently" (`forward_prefill.rs:243-247`).
- **Trap:** `stagger2` means "two lanes, staggered *or* ready-first", while `V41_MS_STAGGER=2` means ready-first.
- **Cut:** one `LanePlan { cuts, schedule: Lockstep | RoundRobin | ReadyFirst }`, built once per step and passed everywhere.

**S12. Knobs are filed by subject, not by the process that reads them.**
- `V41_B2_*` is read by the hub (`B2_PIN*`, `B2_ASSERT_NO_SURPRISE`) *and* by expertd (`B2_MERGE`, `B2_PARK`, `B2_MODE_EVICT`, …).
- The two processes read different knob files. A misfiled name therefore does nothing, silently; `remote_experts.rs:2089-2093` records one such incident.
- Raw `env::var` reads still cover 246 distinct `V41_` names, against 77 registry knobs. 20 names are tested with `.is_ok()`, so `X=0` turns them *on*.
- **Cut:** per-process unknown-knob rejection in the registry (cheap), or process-named areas (`V41_EXPERTD_*`). Conversion order is in `docs/v41/KNOB_AUDIT_2026-10-04.md`.

**S13. Model identity and file format are mixed into type names.**
- `GgufType` (427 uses) is the dtype enum of a model that is loaded from safetensors. Production passes `--gguf <HF dir>`.
- Four crates say `v4flash` (1,569 uses). `het` (620 uses) names the whole hub engine, the box-2 daemon included.
- **Cut:** format is not dtype (`WeightFormat`/`QuantType`), and the crates are renamed *as they are split* (09-24 §4). Renaming a crate while it still holds the engine, the daemon and Laguna keeps the wrong joint under a new name.

---

## 5. Homonyms

A homonym is a word that names two or more glossary entries. The table gives the canonical owner of each word and what its other senses become. Census counts are occurrences / distinct identifiers.

| Word | Senses found | Canonical sense | Others become |
|---|---|---|---|
| **slot** (2197 / 100) | 1. pool slot; 2. arena stream id (`kv_arena.rs:542`); 3. raw append row (`slot_per`, five lines below sense 2 at `:315`); 4. pick rank ("sel slot i", `remote_experts.rs:352`); 5. drafter source index (`mtp.rs:792`); 6. accumulator block base (`slot_block`) | pool slot | stream id, append row, rank, source index, block base. KB #4 (OPEN) is three slot spaces in one buffer. |
| **stream** (4230 / 74) | 1. live sequence; 2. HIP queue, sometimes in the same signature (`KvArena::accept(slot, keep, stream: &Stream)`, `kv_arena.rs:1247`); 3. SSE; 4. box 2's "FIFO request stream"; 5. RNG stream | live sequence | `hip_stream` (or its role name) where both meet; SSE |
| **seq** (227) | 1. box-2 RPC sequence number (128 uses); 2. token history (`Stream.seq`); 3. the 09-24 glossary's live sequence (0 uses) | RPC seq | token history → `tokens` |
| **prefill** (1191 / 199) | 1. prompt ingestion; 2. the batched layer; 3. `Phase::Prefill`; 4. box 2's prefill class / pin band; 5. "prefill-shaped" = b > 16 | prompt ingestion (and phase) | batched layer; request shape |
| **owns** | six predicates (S1) | — | advertises / accepts / home / placement |
| **pin** (1068 / 158) | 1. the pin protocol; 2. box 2's evict-protect `pinned` / `parked_pins` in the **same struct** as the protocol's `pins` (`remote_experts.rs:2150-2156`); 3. `PinnedBuffer` host memory; 4. "pinned dense windows" on box 1; 5. "pinned (non-paged) shard" | the protocol | protect (KB #37); pinned memory; reserved windows; static shard |
| **ring** (389 / 59) | 1. a decoder layer's raw window ("decoder rings"); 2. the drafter ring; 3. the evtrace ring buffer | ring buffers (2, 3) | decoder-layer raw window (KB #25, #28, #42) |
| **mirror** (194 / 24) | 1. the mirror drive; 2. the hub's box-2 view; 3. the host copy of `remap_dev`; 4. "mirror onto the dGPU" | the drive | box-2 view; host copy |
| **split** (755 / 127) | 1. remote split (`V41_REMOTE_SPLIT` modes; KB #6 OPEN); 2. het-split (device); 3. drive split (`route=split`); 4. "hot split" (simulator); 5. `let split = CED_DECODER_START` | — | placement; device placement; drive route; replication; CED split |
| **hot** (615 / 93) | 1. dGPU hot tier (inert); 2. box-1 home set (`hot_set`); 3. box 2's LRU contents; 4. "hot head" (top experts) | — | home set; resident; top experts |
| **prefetch** (250 / 58) | 1. box-2 look-ahead words, which also carry admissions and restores; 2. box 2's background readers, which also run *certain* reads; 3. box 1's deferred demand fill (`V41_B1_PREFETCH`, not a guess); 4. residency hints | speculative read | read classes (glossary) |
| **victim** | 1. an eviction victim; 2. "victim cache" = box 1 caching box-2 misses (`expert_pager.rs:767`); 3. "victim tier" = box 2 behind box 1 (`:866`). Senses 2 and 3 point in opposite directions. | eviction victim | describe the direction |
| **phase** (320 / 27) | 1. scheduler prefill/decode; 2. ready-first lane waits; 3. `PreMoePhase`; 4. link busy-poll; 5. box 2's inferred phase | scheduler | waits; stage; link mode; `inferred_phase` |
| **admit** (173 / 42) | 1. `KvArena::admit` = carve regions; 2. the scheduler admits a request; 3. box-2 landing; 4. hub admission words; 5. the TinyLFU admission filter | scheduler | carve; land; admission (expert) |
| **accept** | 1. `KvArena::accept(keep)` = commit rows; 2. DSpark accepts drafts; 3. `V41_MS_DSPARK=accept`; 4. box 2 "accepts" a pick | DSpark | commit; accepts (placement) |
| **restore** (297 / 54) | 1. snapshot restore; 2. arena rollback; 3. hub "pin restore"; 4. box-2 "delta restore": two mechanisms on opposite boxes for the same displacement problem | snapshot restore | rollback; and pick one of the two re-admit mechanisms |
| **window / chunk / block / group** | Each has 5–6 senses: SWA, LM, drafter and pager windows; prefill, SSE, KV-store and read chunks; drafted, accumulator, candidate and quant blocks; layer, compressor and expert groups | always qualify | — |
| **b1 / t2** | `b1` = box 1 *and* batch size 1 (`PAGE_US_B1`, `remote_batched_b1`); `t2` = tier 2 *and* the box-2 receive timestamp (`t2_b2`) *and* lane 2 (`sync_events_t2`) | box 1; timestamp | `bs1`; lane index |
| **compressed** | Engram's hashed token ids vs compressed KV. KB #23: multistream hashed double-compressed ids. | compressed KV | hashed ids |
| **residual** | the model's residual vs the rejection-sampling residual `max(0, p − q)` (`spec_sample.rs:31`) | model | "rejection residual" |
| **placement** | the per-pick decision vs the ranked-experts *placement file* (`DGPU_HOT_EXPERTS_FILE`) | the decision | rank file |

The census reports each homonym row so that the counts can be tracked. Only aliases are ratcheted.

---

## 6. Names that record history or mechanism

These identifiers carry a date, a milestone or an implementation detail instead of a meaning. Each needs one line of glossary to decode.

- **Model history.** `v4flash_*` crates, `GgufType`, `het`, `mtp`, the `ratio == 4` arms (95 uses, dead under V4.1), and `raw_window` (the V4 image window, which blocks the SWA term).
- **Milestone codes.** None appears inside an identifier; the census ratchet keeps it that way. Many comments, though, use a code *as* the explanation: M50 ×35, M63 ×23, S2 ×19, M61 ×19, T2 ×18, M56 ×16, G5f/G5g ×11 each, "Tier B", "SF-B", "R1/R4". `T2` itself is never defined; the nearest is "M7 expert tier" (`expert_pager.rs:1`).
- **Version suffixes.** `_v2` with no v1 (15 identifiers, unchanged). `REQ_FLAG_HINTS "(v2: …)"`. `PIN_WANTS=0` = "the 2026-09-27 ledger exactly".
- **Names describing the mechanism, not the role.**
  - `B2Prefetch` is the background reader pool.
  - `coalesce` is a span read.
  - `EarlyPaged` is picks not yet landed at frame arrival.
  - `hetsplit` kernels now implement box-1/box-2 exclusion. `REMOTE_EXPERTS.md:137` still says "the dGPU takes it" on box 2, which has no dGPU.
  - `ENCODER_VICTIMS_FIRST` "predates the sweep rank" (its own comment).
  - `stagger` is a lane schedule.
- **Doc comments attached to the wrong item.** These are the definitions an ontology relies on, and they are misattached (KB #13 class, unchanged since 09-24 §5 #16):
  - `remote_experts.rs:8268-8272`: the masked `submit` doc sits on `submit_unmasked`.
  - `:2018-2030`: the `route=urgency` definition sits on `prefill_budget()`.
  - `:4702-4706`: the `ensure_layer` doc sits on `ensure_layer_reporting`.
  - `:7846`: the `Ticket` doc sits on `HOP_SUBMIT_TO_WRITE_NS`, and still says "responses arrive in submission order", which predates `REQ_FLAG_OOO`.
  - `expert_pager.rs:866-878`: the `V41_B1_PREFETCH` doc sits on `b1_page_misses()`.
  - `:846-850` and `:453-459`: similar.
- **Docs that contradict the code.**
  - `deepstrix-server/src/knobs.rs:24`: `V41_MS_CTX_ROWS` is "arena rows per stream". The code uses it as a position budget across all streams (`multistream.rs:262`).
  - `forward_prefill.rs:297-313, 3746`: "K=1 per stream", but a stream may run 8 rows.
  - `snapshot.rs:3`: keys are over token ids, but they have been over decoded bytes since format v2.
  - `REMOTE_EXPERTS.md:36, 41`: protocol version 1; it is 4.
  - `hub_step.pos_min/max` is `seq.len()`, one more than the position (`multistream.rs:1688` vs `:1710`).

---

## 7. Observability vocabulary

The owner reads `hub_step`, `hub_phase`, `b2_*` and `ms.stage` daily, so their names matter more than most identifiers.

- **Mostly well carved.** `d_`/`i_` for the device, `b1_`/`b2_` for the box, `lh_` for layer host time, `sub_`, `pf_`, `pin_`, `hop_`. One prefix per axis.
- **Wrong axis.** `d_prefill_indexer` and `d_prefill_indexer_reuse` appear on decode steps (S2). `d_mtp`/`i_mtp` should be `drafter`. `b2_ensure.prefill_shaped` is a request shape (S9). `seq` in `hub_req`/`b2_req` is the RPC seq, which is fine once "seq" means only that.
- **One metric, two names.** `decode_rows` writes the same numbers under `ms.stage` names and `hub_step` fields through translation tables (`multistream.rs:1969-2100`): `host.remote_wait` → `remote_wait_ms`, `box2.paged_x1e6` → `b2_paged_replies`, `prefetch.queued` → `b1_pf_queued`. `ms.phase` logs `hold_units` and `hold_ms`; `hub_phase` has no hold fields.
- **Unidentifiable records.** There is no request id (S5). The "stream admitted/done" logs name a reused slot.

---

## 8. What is carved well (keep)

- **Model-native names**, and the CED encoder/decoder split, which the residency code also respects ("encoder victims first").
- **The pin protocol's three-way distinction**, `held ⊆ pinned ⊆ resident`, with a stated invariant (`remote_experts.rs:2462-2518`). It needs only to stop sharing the word "pin" with evict-protect and host memory.
- **merge vs span read.** The 09-24 split was adopted: "coalesce" no longer means merge, and `partner` went 24 → 72.
- **`KvMark` / `CompMark`** for rollback points.
- **The DSpark arena vocabulary** (lone stream, plain step, drafted block, K, cost cells) is consistent with itself. **`StepRows`** on `worktree-ms-dspark2` is a real joint: it makes the per-row dependency explicit (`crosses(cut)` replaces a positional rule). It should say *stream* where it says "slot (stream)".
- **The KV-store design** (chunk, tail, waypoint, anchor, trie, lineage) is coherent. Its "chunk" is deliberately the prefill chunk's size; qualify it as *store chunk*.
- **`hub`** (94 → 228 uses), the box-2 fast chain, the partial, and the ticket.
- **The `V41_` knob prefix.** The etymology is stale but the prefix is unambiguous. Renaming 330 env names that live in deploy scripts buys nothing.

---

## 9. Plan

The order follows the in-flight branches. `worktree-ms-dspark2` (which contains `worktree-lm-prefill-prod`) rewrites `ms_dspark.rs` (+670), `multistream.rs` (+220), `forward_prefill.rs` (+160), `kv_arena.rs` (+54), `remote_experts.rs` (+183) and `b2_mirror.rs` (+190). Every rename in those files should wait until it merges, and then land as one mechanical commit per entry, each checked by `cargo check --workspace --tests` and the census.

**Phase 0: now (no conflicts).**
1. Adopt `docs/GLOSSARY.md` once the owner answers §10. Wire the ratchet (`scripts/ontology_census.py --ratchet scripts/ontology_baseline.tsv`) into the pre-push hook or `gates.sh`. It fails only on *new* identifiers minted from a retired name, so normal work isn't blocked.
2. **Align `worktree-kv-prefix-store` before it merges.** It is 8,505 new lines in `kvstore/*` with no overlap with main, so this is the cheapest point. Use "decoder-layer raw window", not "ring" (23 uses in its design), "store chunk", and lineage vs session hint per the glossary.
3. **`RequestId`.** Carry it from `handler.rs` through `GenerateReq` into `Pending`/`Stream`, the logs and evtrace. `engine_worker.rs` and `handler.rs` are untouched by the in-flight branches.

**Phase 1: right after `ms-dspark2` merges (mechanical, bit-identical).** Gate each with G-compile, the census, and a rebuilt binary's temp-0 sha.
1. Fix the misattached and contradicting docs (§6).
2. `Prefill.prefix` → `tokens` + `restored`; `prefill_job_chunk` → `run_unit`; `checkpoint_ok` → `window_open` (S8).
3. `StreamId`, and stream id instead of `slot` in `KvArena`, `Sched`, `MsDspark` and `StepRows`. About 350 lines.
4. `KvArena::admit` → `carve`; `KvArena::accept` → `commit`.
5. "Decoder rings" → decoder-layer raw window.
6. `mtp`/`Mtp*`/`MTP_*` → `drafter`/`Drafter*`/`DRAFT_*`. About 830 occurrences, one commit. Keep `"mtp."` in the loader.
7. Glossary renames with env aliases (the registry's `.legacy()`):
   - `b2_mirror` → `box2_view`
   - `ExpertRoute` → `DriveRoute`
   - `coalesce` → span read
   - `B2Prefetch` → background readers
   - `V41_T2_PARTITION` → `V41_HOME_SPLIT`
   - `V41_T2_CATCHALL` → `V41_MISS_FALLBACK`
   - `V41_INDEX_*` / `V41_IDX_*` → `V41_INDEXER_*`
   - batch-size `*_B1` → `*_BS1`
   - `parked_pins` / `ExtraPins` → protect

**Phase 2: behaviour-touching (behind the golden gate's pinned mode).**
1. **S1: placement.** `PlacedPicks`, kill the masked submit, audit the wire. Closes KB #5.
2. **S9.** `RequestShape` and `inferred_phase`; box-1 decode/prefill accounting keyed on row role.
3. **S7.** `LaneCtl` out of scratch. It conflicts with `worktree-architecture-review`, so do it after Phase 3.1.
4. **S11.** `LanePlan`.
5. **S10.** `PickStats`.
6. **S12.** Per-process unknown-knob rejection, then the `KNOB_AUDIT` conversion list.

**Phase 3: structural.** These are the 09-24 roadmap steps 5, 6 and 9, re-ordered.
1. **Resolve `worktree-architecture-review`.** Rebase the V4-Flash removal onto main, or redo it. It is 36 commits on a base 227 behind. It removes the `ratio == 4` arms, the V4 image window (which frees "raw window"), the V4 GGUF paths and the inert dGPU hot tier.
2. **S3.** Split `remote_experts.rs` into wire / expertd / client.
3. **S4.** Split `b2_mirror.rs` into view / pin ledger / substitution.
4. **S2.** Move the batched layer to `engine/layer`; name the drivers by caller.
5. **S5, S6.** `StreamEntry`; `KvBuffers` + counters, before RESIDENT streams.
6. **S13.** Crate split *and* rename in the same move; `GgufType` → `QuantType`.

**Not to rename:**
- the `V41_` prefix;
- the OpenAI `stream` field;
- the reference's names (`gate` as the SwiGLU projection, `head_dim`, `compressor`, `indexer`);
- the wire constant `REQ_FLAG_PREFETCH`, until a protocol bump (rename the Rust constant with v5);
- `remote` inside the hub's client after S3;
- `hub`.

---

## 10. Questions for the owner

1. **The four glossary picks that differ from 09-24:**
   - **stream** for a live sequence, not `seq` (and qualify HIP streams where the two meet);
   - **box 1 / box 2** for the machines, plus **hub / expertd** for the processes;
   - **phase** for the scheduler's prefill/decode alternation, with **burst** for one run of it;
   - **drafter** (with `DSpark*` for model parts), and `mtp` only in the loader.

   Approve these, or name the word you'd rather use.
2. **Order.** Merge `ms-dspark2` first, then the Phase 1 renames as one commit each, straight after? Or should the renames ride on the branch before it merges?
3. **`worktree-architecture-review`.** Rebase it onto main, or redo the V4-Flash removal from scratch on main?
4. **The ratchet.** Put it in a pre-push hook, in `gates.sh`, or keep it advisory?
5. **Knob filing.** Per-process unknown-knob rejection (no renames), or process-named areas (`V41_EXPERTD_*`, with aliases)?
6. **"Restore".** Keep one re-admit mechanism and drop the other: the hub's pin restore (`V41_B2_PIN_RESTORE`) or box 2's delta restore (`V41_B2_RESTORE`)?
7. **The staging band.** Same question for the box-2 staging band (`V41_B2_PREFILL_STAGE`) vs the hub's pin band (`V41_B2_PIN_PREFILL_BAND`). They are mutually exclusive (`b2_mirror.rs:479-481`), and the live configuration decides which one is dead.

---

## Appendix: the census

`scripts/ontology_census.py` reads `scripts/ontology_terms.tsv`. It has one row per name, giving the glossary entry, kind, ratchet metric, an optional path scope and a regex.

- **Default output:** occurrences and distinct identifiers for every row, over `crates/**/*.{rs,hip,h,inc,cpp}`, with comments stripped.
- **`--show <entry>`:** lists the matched identifiers.
- **`--ratchet <baseline>`:** exits 1 if a retired alias rose.
  - For identifier aliases it compares **distinct** identifiers, so new code may still *use* `MTP_BLOCK` until the rename lands, but it may not mint a new `mtp_*` name.
  - For literals such as `b > 16` it compares occurrences.
- **`scripts/ontology_baseline.tsv`** holds the `128f7e6` counts. The `f38679e` column in §2 came from running the census with `--tree` pointed at a checkout of `f38679e`.
- **Updating the baseline.** After a rename lands, lower it with `--save scripts/ontology_baseline.tsv`, and commit the new baseline in the same commit as the rename.
