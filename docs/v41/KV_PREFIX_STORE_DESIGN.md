# KV prefix store: chunked, content-addressed (design)

Status: DESIGN, revision 7 (2026-10-01). Revision 6 was APPROVED in review round 5; revision 7
records what the M1 implementation (`crates/deepstrix-server/src/kvstore/`) changed or measured,
marked "(M1)" in the text and listed in Appendix B. It replaces the whole-prompt snapshot store
(`crates/deepstrix-server/src/snapshot.rs`, format v6) on the multistream path. Appendix A lists
each design review finding and what changed.

- **MEASURED:** hub logs for 2026-09-28 16:00 .. 10-01 09:00 (65 h; files in 1.0), or the
  snapshot root `~/.cache/deepstrix/snapshots-v41`.
- **ESTIMATE:** a derived number; the derivation is given next to it.
- **Link generation:** every number that involves the link names the generation it assumes.
  Since the 09-28 reboot, the dGPU OCuLink root port `00:03.1` runs at 2.5 GT/s x4 (PCIe Gen1).

## 0. Decisions

1. **Chunks.** The positional KV is stored in chunks of C = 1024 positions (2.83 MB): the
   compressed rows and E2M1 index keys of the four KV-source stores. Each chunk is one file,
   written once and shared by every prompt that contains that prefix.
2. **Keys.** A chained blake3 over the **token ids**, not the decoded bytes (this changes the
   proposal; 4.3).
   - Each image's (offset, length, content hash) is folded in at its IMAGE_START position.
     A cut inside an image block is therefore allowed and still keyed by its image.
   - The seed is a namespace hash over semantics and ABI only: model, tower, Engram, store
     layout, C, the window, CED, keys on or off. Numerics knobs and the build id go in file
     headers, not the namespace. A low-bit change therefore does not cold-start every
     conversation. Stale namespaces are garbage-collected (4.4).
3. **Tails** hold the state that is not positional, needed to resume at a position T.
   - A **full tail**, written at every prompt end, holds the encoder and decoder windows, the
     accumulators and the DSpark ring: 5.68 MB, plus the open rows of the last partial chunk.
   - An **encoder tail** holds the encoder windows and the accumulators: 2.65 MB. One is written
     every K = 8192 positions during prefill, and on cancel.
   - The exact system-prefix anchor gets a **full** tail. It is written on the prefix's
     **second sight**, at the cost of one side replay per prefix that actually recurs.
4. **CED rule.** An encoder tail is used only when the suffix after it is longer than 128
   rows. In exactly that case `prefill_job_finish` discards the decoder windows anyway (#28).
   So no 128-row decoder replay is needed at tail points (a replay is 2.6 s p50, MEASURED).
5. **Lookup.** A walk of about 1 ms over the request finds the deepest usable tail. There is
   no boundary-token probing and no session hint.
6. **Exactness** (7):
   - The store round trip is bitwise by construction.
   - A resume at a matched chunk partition is bitwise, in a configuration that does not depend
     on residency.
   - An unaligned resume is **not** bitwise by construction: kernel choices depend on the batch
     size. It is gated against a measured null, the noise between two cold prefills that use
     different chunk partitions.
7. **Eviction.**
   - Tails are the unit. A chunk is reference-counted by the tails below it.
   - Lineage comes from the trie, not from `session_id` (which no request has carried).
   - A superseded full tail is demoted to an encoder tail.
   - Staleness propagates up a conversation's path. One global 100 GiB cap covers every
     namespace.
8. **Writes.** The only work on the scheduler thread is a device-to-host copy, about 5 ms per
   chunk at Gen1. A writer thread does temp file and rename, checksums, and applies
   backpressure with a deadline. ESTIMATE: about 1.3 GB/h, against 21.3 GB/h today.
9. **Checkpoints become encoder tails.** Both of today's checkpoint bugs disappear by
   construction.
10. **Rollout.** `V41_KV_STORE=off|shadow|on`. Shadow runs in two phases, the second with a
    small cap so that eviction actually runs. The old cache is dropped at the end.

**Where the store sits.** The disk store is the persistence tier. An arena-resident tier for
finished streams would serve active conversations without any restore at all (2.3). The two
complement each other; this doc designs the disk tier.

**Expected effect, stated plainly.** On the replay of the survivor-biased corpus, the share of
tokens restored barely moves: 89.96% to 90.57% (1.2). The wins:

- 16x fewer bytes written.
- At the same cap, the store holds about 3 days of writes instead of about 5 h.
- No more whole-snapshot cliffs: the 455 s retry becomes about 20 s.
- Checkpoints that can be restored.
- At Gen1, about 460 ms less scheduler stall per admission for writes: about 39 min per 65 h
  becomes about 2.5 min.
- Warm DSpark rings for short continuations.

Cold prompts caused by how the client lays them out (most of the 6.78 h) are not fixed (12).

## 1. Problem

### 1.0 Sources

- **Logs.** `scratch-ms/b2_hang_2026-09-27/hub_logs/v41-server.log.pre-8cap-2354`
  (09-28 16:00-23:54, +467 admissions) and the `~/logs/v41-server.log*` rotations up to
  10-01 09:00. Some rotations have since moved; `~/logs` alone now covers 57 h, 4,510
  admissions and 448M tokens.
- **Scripts.** They must be committed under `scripts/kvstore_analysis/` with the log files as
  arguments, before M2:
  - the owner's `cache_stats.py`, `cache_lcp2.py`, `cache_head.py` and `snap_diverge.py`
    (in the job tmp dir);
  - this doc's `sim.py`, `logstats*.py` (in `/tmp/kvps`) and the reviewer's
    `ck.py`/`ck2.py`/`sess.py`/`bpt.py` (in `/tmp/kvreview`).

### 1.1 Findings (65 h, 4,944 admissions)

- **Hit rate and prefill cost.**
  - 96.8% of the 500.9M prompt tokens were restored. The warm suffix was p50 1,280 and p90
    4,085 tokens.
  - 146 admissions restored nothing and cost 6.78 h of the 36 h of prefill. The 69 of them
    over 20K tokens cost 5.77 h: a mean of about 300 s, and 600-1,024 s for the largest
    (130K-330K tokens).
  - Fits: cold prefill takes 42 s + 3.04 ms/token, warm prefill 14.1 s + 4.10 ms per suffix
    token. Only the slope is a per-token cost.
- **Every admission saves its whole prompt.** ESTIMATE (prompt tokens × 2,760 B): 1,382 GB
  written in 65 h, 21.3 GB/h. That is corroborated by the 1,428 GB that 5,148 evictions freed
  (MEASURED).
  - The cap turns over every ~5 h of writes. The oldest entry still on disk is 11.8 h old
    (10-01 10:00), kept by LRU touches.
  - 318 entries hold 107 GB, and 90.2% of the stored tokens duplicate an older entry's
    prefix: about 10 GiB of distinct KV.
  - **The per-lineage cap (R1) has never run.** `session_id` is a non-standard request field
    (openai/types.rs). 0 of 318 entries carry one, and every eviction in the logs is
    "global cap".
- **A snapshot restores only whole, and only at a turn-boundary token.** A retry shared
  148,918 of 148,921 tokens with a 148,920-token snapshot. It restored 0 and re-prefilled for
  455 s.
- **50 checkpoints (15.2 GB) cannot be restored.**
  - (a) In `prefill_tick`'s cancel path and `prefill_job_tick`'s periodic checkpoint, the
    tokens are `pf.prefix + job.tokens()[..done]`. But `start_prefill` has already extended
    `pf.prefix` to the whole prompt. All 146 checkpoint log lines show
    tokens = pos0 + total + done.
  - (b) Even with the right key, they end at positions the lookup never probes.
- **Scheduler-thread stalls.** Every decode stream waits while these run.

| | Gen1 (since 09-28) | Gen4 (before) |
|---|---|---|
| D2H rate, from checkpoint saves recomputed over true positions (pos0 + done) | 0.57-0.59 GB/s p50 (p10 0.46-0.53, p90 0.62) | ~1.6 GB/s |
| restore rate, end to end | 0.44 GB/s | ~0.8 GB/s (host-bound by the per-element conversion loops) |
| restore time | p50 573 / p90 1,150 / mean 621 ms; 49.8 min per 65 h | |
| prompt-end save, 279 MB average | ~0.48 s per admission, ~39 min per 65 h | ~0.17 s |
| periodic checkpoints | p50 335 ms each | |

- **DSpark.** 3.6% of warm admissions (175 of 4,798) have a suffix of 128 tokens or fewer.
  Their ring is seeded from the suffix only (DSPARK_ARENA_PLAN 4.6).

### 1.2 Corpus replay

**The run.** `sim.py` at 10-01 ~09:20: 273 surviving prompt snapshots (32,775,755 tokens),
replayed in creation order with this design's rules and no eviction.

**A second run agrees.** The reviewer's rerun later that morning, on a corpus that had changed
to 271 prompts and 32.87M tokens, matches every share to within 0.01 point and every byte
figure to within 0.2 GB. The doc uses only the first run.

**Survivor bias.** The corpus holds what the LRU kept, so it misses far more than the live
store does: 89.96% restored here against 96.8% live. Use it only to compare the rules with each
other.

| rule | restored | bytes written |
|---|---|---|
| upper bound (longest common token prefix with any older prompt) | 90.75% | n/a |
| today (an older snapshot that is a complete prefix) | 89.96% | 91.9 GB |
| chunked, K = 4096 / 8192 / 16384 / 32768 | 90.65 / 90.57 / 90.56 / 90.46% | 12.2 / 11.2 / 10.7 / 10.5 GB |

## 2. Prior art: antirez/ds4 (`0aaea5a`, 2026-09-20)

### 2.1 What ds4 does

Sources: docs/SERVER.md "Disk KV cache", ds4_kvstore.{h,c}, ds4_server.c "KV Cache".

- **Tiers.** First, the live slot's token prefix. Then disk: "disk is the additional
  persistence layer".
- **What a file holds.** One file per whole-session checkpoint: a 48 B header ("KVC", version,
  quant, reason, ext flags, model id, tokens, hits, ctx, payload-ABI byte, created,
  last_used, payload bytes), the text length and the text itself, the model payload, and an
  optional trailer.
  - The V4.1 payload holds: the tokens; the last logits; each layer's raw window
    (min(pos, 128) × 512 f32); the four compressed stores and their index caches (f32); and
    the open ratio-2 pair only when pos is odd ("only owner caches, windows and unfinished
    compression pairs persist").
  - That is 6,400 B per position, against our 2,760.
- **Key.** SHA1 of the rendered byte prefix. For protocols that hide reasoning, it is the
  client-visible transcript instead. The payload keeps the exact token ids.
- **When it stores.**
  - **cold** (no hit, prompt of 512-30,000 tokens): at the chat anchor, i.e. the last
    `<User>` before the first `<Assistant>`, if that is ≥ 512 tokens in. Otherwise at
    (len − 32) rounded down to a multiple of 2048. The prefill stops there, stores, then
    continues.
  - **continued:** every 10,240 positions, from the prefill progress callback.
  - **evict:** the live state is stored before a disk load replaces it.
  - **shutdown.**
  - ds4 never rolls a session back to make an entry, and stores no session that holds vision
    state.
- **Boundary rules.** It trims 32 tokens because the tokenizer merges across the boundary. It
  aligns to 2048 to "match the backend prefill chunk schedule".
- **Lookup.** A linear scan. For each compatible entry it takes the SHA1 of the prompt's first
  `text_bytes` bytes, and the longest match wins. It verifies the text, reads the payload
  (no mmap), and rewrites the prompt as the stored tokens plus a tokenization of the rest of
  the text.
- **Eviction.** The budget defaults to 4 GiB, with 1% headroom. Score = (hits decayed with a
  6 h half-life + 1) × tokens / file_size; the lowest score goes first:
  - anchors (cold/evict/shutdown) score ×2;
  - a "continued" entry that is a strict prefix of the incoming one scores ×(0.05 + 0.45·h).
- **Hooks.** Per model family, `ds4_session_{stage,save,load}_payload`, versioned by the ABI
  byte. Trailer hooks carry the tool-id → exact-DSML map.
- **Crash safety.** Write `<sha>.kv.tmp.<pid>`, flush and close, re-check the size, rename.
  There is no fsync. The scan validates the header and the file length. Three cases unlink a
  file: a token-count mismatch after load, a prefill that fails after a restore, and an
  incompatible file under the same name. A plain load failure is only logged.
- **rax.c** indexes the tool-memory maps, not the KV cache.

### 2.2 What we adopt, adapt or reject

| ds4 | here | why |
|---|---|---|
| live-slot tier first | **adopt as a separate tier** (2.3), outside this doc's scope | removes restores of active conversations |
| SHA1 of the text + re-tokenize the suffix | **reject** for chunks; **possible fallback** for M4 turn-end tails | 4.3 |
| keys from the visible transcript | not now | our clients replay thinking |
| one whole-session file | **reject**: chunks + tails | 90.2% duplication, 21 GB/h |
| cold store at the chat anchor | **adopt**, at the exact anchor (5.2) | shared system + tools |
| continued every 10,240 | **adapt**: waypoints every K = 8192 | 1.2 |
| align stores to the chunk schedule | **adapt**: snapping (5.5); aligned-only resume stays a knob, default off (7) | |
| evict/shutdown stores | **adapt** in M4 / with the resident tier | |
| logits in the payload | **reject** | we always prefill ≥ 1 row |
| open pair only at odd pos | **adapt**: always store the accumulators (28 KB) | bitwise with any kernel that reads stale bytes |
| decayed-hit eviction score | **adapt** (8.3) | |
| tmp + rename, no fsync, unlink on failure | **adopt**, plus checksums and token-id checks | |
| model id + ABI byte | **adopt** as the namespace, plus garbage collection (4.4) | |
| exact DSML tool-call replay | **prerequisite of M4** | 4.3 |
| no vision sessions | **reject** | content-hash keying (4.3) |
| linear scan | **reject**: an O(L/C) walk | |

### 2.3 Tiers: where this store sits

**The tier this doc does not design: arena-resident finished streams.** The arena averages 2.77
of 8 slots live (MEASURED 09-27). Today a finished stream is released. The next turn of the
same conversation then restores its prefix from disk over the link right after its KV left
the GPU: p50 573 ms at Gen1, which is about 250 MB, a p50 prefix of ~91K tokens.

A resident tier would keep a finished stream's slot and rows until a reservation needs them.
A request whose tokens extend that stream's `seq` would continue in place: no restore, no link
traffic. This is the multistream plan's RESIDENT streams (M2); ds4's first tier does the same.

**The disk store (this doc)** covers everything else: conversations whose slot was reclaimed,
server restarts, retries and branches, and new conversations that share a prefix.

**How the two interact:**

- The prompt-end full tail is still written, for persistence.
- With a resident tier, the turn-end tail (M4) becomes the write made when a resident stream
  leaves the arena (ds4's "evict" reason).
- The restore latency M3 targets matters less once the tier exists, so weigh M3 against
  building the tier.

## 3. What must be persisted (V4.1)

| state | code | positional? | bytes | goes to |
|---|---|---|---|---|
| comp rows L2/L8/L14 (ratio 2, f16 × 512) | `HetCompressorState.comp_kv` | yes | 512 B/pos each | chunk |
| comp rows L20 (ratio 1) | same | yes | 1,024 B/pos | chunk |
| E2M1 index keys, 80 B per comp row | `.index_k` | yes | 40/40/40/80 B/pos | chunk |
| `n_comp`, `n_index_comp` | counters | derived: ⌊T/ratio⌋ | | header (keys must cover every row, otherwise refused: the v6 rule) |
| accumulators (`state_kv` + `state_score`) | `.state_kv/.state_score` | the state at T | 28 KiB | every tail, byte for byte |
| encoder windows L0-19 | `HetLayerState.kv_cache` | positional, but 20 KiB/pos | 2.62 MB | every tail |
| decoder windows L20-39 | same | **no**: they depend on where the replay or decode started | 2.62 MB | full tail |
| DSpark ring (3 × 133 × 512 f16), `ring_writes`, `last_ring_pos` | `ms_dspark::SlotDraft` | no | 0.41 MB | full tail |

**Not persisted:**

- Logits.
- Engram `compressed`, which is rebuilt from the tokens.
- The `PrefillJob` replay buffer: its rows come from the suffix (5.3).
- The DSpark `hidden`/gain.
- Per-call indexer selections (#22).
- The arena `StreamKv`, which `fill_reserved` rebuilds.
- Anything V4-Flash-only (`indexer_compressor`, the FP8 head), which is refused.

**Per position in chunks:** 3 × (512 + 40) + (1,024 + 80) = 2,760 B (VERIFIED: a 327,680-position
checkpoint has 904.4 MB of rows and keys). Encoder windows are not chunked: storing them per
position would cost 20 KiB/pos (7.4x the chunk payload). Waypoints every 8,192 positions cost
0.32 KiB/pos.

## 4. Storage: chunks and keys

### 4.1 A chunk

Chunk k covers positions [kC, (k+1)C). It holds, store by store, the rows
[kC/ratio, (k+1)C/ratio) and then their keys.

- Payload: 2,826,240 B. Header (256 B) and token ids (4 KiB) bring the file to 2,830,592 B.
- The store-major layout lets a restore stage each store's rows with one large copy.

### 4.2 Choosing C

| C | chunk | files at 100 GiB | open rows per full tail (avg / max) | hashes per walk at 330K |
|---|---|---|---|---|
| 256 | 0.71 MB | ~152K | 0.35 / 0.71 MB | 1,290 |
| **1024** | **2.83 MB** | **~38K** | **1.41 / 2.83 MB** | **323** |
| 4096 | 11.3 MB | ~9.5K | 5.65 / 11.3 MB | 81 |
| 16384 | 45.2 MB | ~2.4K | 22.6 / 45.2 MB | 21 |

Why 1024:

- It equals the production prefill chunk (`V41_MS_CHUNK_ROWS` = `B_MAX`), so cold prefills
  already end their chunks at multiples of C.
- The open rows cost 25% of a full tail.
- Restore granularity comes from K, not C. C = 4096 restores the same share (90.57%) and only
  adds open-row bytes.
- C is fixed at compile time and is part of the namespace.

### 4.3 Keys

Separate blake3 `derive_key` contexts keep the three kinds of hash apart:

```
ns          = derive_key("deepstrix kvstore v1 namespace", namespace inputs (4.4))
chain_0     = derive_key("deepstrix kvstore v1 chunk", ns)
chain_{k+1} = derive_key("deepstrix kvstore v1 chunk", chain_k ‖ le32(tok[kC..(k+1)C]) ‖ img(kC, (k+1)C))
tail(T)     = derive_key("deepstrix kvstore v1 tail", chain_⌊T/C⌋ ‖ le32(T − ⌊T/C⌋C) ‖ le32(tok[⌊T/C⌋C..T]) ‖ img(⌊T/C⌋C, T))
img(a, b)   = for each image whose IMAGE_START p is in [a, b): le32(p − a) ‖ le32(len) ‖ content_hash
```

**Why token ids, not bytes.** The proposal keyed on bytes. Token ids are better for four
reasons:

1. **KV is a function of the token ids.** Equal bytes split into different ids give different
   KV. Byte keys also make special tokens indistinguishable from their literal text.
2. **Bytes buy nothing here.** Multistream refuses a token mismatch (`is_prefix`), and 0 of
   4,811 restores (MEASURED, reproduced) hit one.
3. **Bridging does not fit chunks.** Chunk boundaries are token positions.
4. **Cheaper.** No vocab decode; about 1 ms for 330K tokens.

**Bytes do matter for M4.** Turn-end tails contain sampled tokens, which a client's re-encoded
history may split differently. If `kv.turn_lcp` (11) shows that happens, the fallback is a
byte-keyed side index over full tails only: ds4's rewrite of the request to the stored tokens
plus a tokenization of the rest. M4 also needs exact tool-call replay (ds4's DSML map, or an
equivalent), so that tool-call JSON re-renders to the sampled tokens. Chunks stay token-keyed
either way.

**Images (decision).**

- V4.1 image rows attend **causally**:
  - the reference `model.py` has no image-visibility mask;
  - the legacy `prefill_suffix` passes `image_spans = None` under `v41`;
  - VISION_PORT.md, "CED interaction", says the same.
- So a chunk end or a tail point **may fall inside an image block**. Rows before the cut
  depend only on the tokens and tower rows before it, so it is numerically fine.
- The keying requirement is that a cut inside a block still carries that image's identity. The
  synthetic ids inside the block encode only the layout, so two pictures of the same size give
  the same ids.
- **Chosen rule:** fold each image's record at its IMAGE_START position. Any cut after
  IMAGE_START, including one inside the block, then has the content hash of every image whose
  block starts before the cut. No snapping is needed for keying.
- The whole-image hash is stricter than causality needs: a different picture with identical
  leading rows misses, which is the safe direction.
- This replaces `checkpoint_spans`, which refused cuts that straddle a block only because v6's
  `image_spans` metadata cannot describe a partial span. Our per-file image records can.
- A restore that lands inside a block works because `chunk_inputs` splices tower rows by
  absolute index (`pf.vl.row_at(p0 + a + k)`).

**Guard.** Each file also stores its token ids, and the restore compares them with the
request.

### 4.4 Namespace and garbage collection

**The namespace holds semantics and ABI only.** A new namespace cold-starts every
conversation. At the 101K-token average prompt, that costs ~42 s + 3.04 ms × 101K ≈ 350 s per
conversation. Those prefills run one at a time (`V41_MS_PREFILL_JOBS=1`) against client
timeouts of ~15 min. The design already accepts low-bit chunk variants in production (E2, E5).
So separating on low-bit numerics would cost a cold start and buy nothing the null does not
already tolerate. 106 commits since 09-17 touched the prefill kernels, several of them
bit-changing (0193166, 1c5972f, cba1b00).

**Namespace inputs:**

- The model fingerprint (v6's).
- The **vision tower fingerprint**: tower rows depend on its weights, and the key carries only
  the pixel hash.
- The **Engram identity**: a hash of the hasher config and of the tables' tensor directory.
- The store layout: per store (layer, ratio, encoding, row and key bytes), C, `SWA_WINDOW`,
  `CED_DECODER_START`.
- The meaning of the stored state: `V41_CED`, which decides what a full tail's decoder windows
  mean, and index keys on or off (`V41_INDEX_K`).
- `KV_EPOCH` (4.4.1).
- The file format version (M1). A format bump then starts a new namespace: neither the new
  binary's scan nor a rolled-back one unlinks the other's files as a bad version, and the old
  namespace is kept as the inactive one (and evicted first).

**Headers only, not the namespace:**

- The writer's build id (git sha + dirty flag).
- A **knob hash** over the values of every env knob that can change prefill numerics:
  - `V41_PREFILL_F32_MATVEC`, `V41_REPLAY_F16X`;
  - the `V41_MHC_*` knobs, including `V41_MHC_GEMM_NARROW` (read in `f16.rs`);
  - `V41_ENGRAM_CHUNK128`, `DEEPSTRIX_COMP_GATHER`, `V41_SWA_MIXED`;
  - the small-batch dense dp4a knobs in `dispatch.rs`;
  - the MoE small-b arms, and so on.

  **Only the knobs that are set are hashed** (name, length, value; sorted by name), so
  classifying a new knob changes no hash until someone sets it (M1). Two kinds of knobs escape a
  hash of the launch env: knobs re-read at run time (`V41_MS_LM_FILE` re-reads its file every
  2 s per job; `V41_B2_KNOBS` is the path of box 2's runtime knob file), and knobs read by the
  box-2 daemon itself. M2 closes both: a write carries the knob hash of the values its job
  resolved, and box 2 reports its own (e.g. in HELLO).
- The **knob classification test** covers the whole prefill call graph, not two files.
  - **It scans string literals of knob names,** not only `std::env::var` calls. Some reads go
    through wrappers whose key is a variable: `env_f` / `env_u` in `expert_pager.rs`,
    `env_usize` in `multistream.rs`.
  - **Scope:** every literal matching `V41_*`, `DEEPSTRIX_*` or `VIT_*` in
    `crates/v4flash-kernels`, `v4flash-core`, `v4flash-vision` and `deepstrix-server`.
    `v4flash-vision` reads `VIT_GEMM` (WMMA against scalar GEMM), which changes tower rows and
    so the KV of image rows.
  - **It fails on any literal not classified** as numerics-relevant or not. Whole crates are
    scanned because the reads hide in leaf helpers. It also fails on a classified name that no
    longer occurs.
  - **The rule** (M1): a knob is numerics-relevant iff it can change what a prefill row computes
    (kernel arm, precision, fusion, the batch / lane / chunk geometry arms are picked by, Engram,
    the tower) or which device computes it (box-1 ownership and residency: hot set, box-1 pool
    geometry, catch-all, partition). Box-2 residency, prefetch and IO pacing are not: box 2
    computes what the hub's split assigns it, whatever is resident there.

**Purges are explicit operator actions, never routine retention.**

- `V41_KV_STORE_PURGE_BUILD=<sha>` (or `=knob:<hash>`, `=gen:<n>`, `=v6`) deletes the files
  written by that build, knob setting or numerics generation, plus every tail beneath a purged
  chunk.
- Chunks are written once by their first writer and never rewritten. So the oldest chunks of
  every conversation, and the shared system-prefix chunks with the anchor above them, carry the
  oldest build ids.
- A purge that reaches those chunks cascades through every conversation built on them. For
  those prefixes it is a store wipe, so it runs only on an explicit epoch-class or bug-fix
  decision (4.4.1).
- **Routine retention is eviction (8.3) and nothing else.** No build-, deploy- or age-based
  deletion exists.
- Round 4 removed `KEEP_BUILDS`, which counted deploys: 10-01 alone shipped 5 production
  binaries, none changing prefill numerics. It would have wiped every conversation through a
  shared prefix on the third deploy after that prefix was written (Appendix A, N16).

**Numerics generation.** `KV_NUMERICS_GEN` is a code constant, bumped in the same commit as any
change that alters prefill bits.

- It is recorded in every header next to the build id and knob hash. It is not part of the
  namespace, so a bump cold-starts nothing.
- Deploys that leave prefill numerics alone, most of them, leave it unchanged.
- **The effective generation is the pair** (`KV_NUMERICS_GEN`, knob hash).
  - A knob changed only in the launch env changes no commit. It changes the knob hash under the
    same constant, so the constant alone would hide it.
  - `kv.store gens=` reports pairs, and the oldest-present release check (4.4.1) works on
    pairs.
  - At startup, a pair the store has not seen before is logged (`kv.gen_new`), the same way as
    an unmeasured generation.
- **Release rule:** a diff that touches the prefill crates without bumping it carries a
  one-line "not bit-changing" justification in its commit message.

#### 4.4.1 When to bump `KV_EPOCH`

The release checklist bumps it, which starts a new namespace, on an explicit decision in
exactly two cases:

- **(a) A bug fix that corrected what stored KV contains,** the #22 / #23 / #28 class. #23 baked
  wrong Engram rows into snapshots. The stored rows are then wrong, not merely different in
  low bits. This always bumps, measured or not.
- **(b) A numerics generation whose effect exceeds the M0 null.**
  - It is measured G7-style: chunks of the **oldest effective generation actually present in
    the store**, i.e. the oldest (`KV_NUMERICS_GEN`, knob hash) pair the headers list, plus a continuation on the new build, against the
    new build's cold prefill, compared with the null p99 (7.2).
  - **Within the null:** the change ships with only its generation bump. Old chunks are then
    the low-bit variants E5 already accepts.
  - **Beyond the null:** an epoch-class event. The release note records the explicit choice:
    bump `KV_EPOCH`, or `PURGE_BUILD=gen:<n>` of the offending generation.

**The (b) check: who runs it and what it costs.**

- **Who:** the developer who bumps `KV_NUMERICS_GEN`, as part of the fidelity gate they already
  run with the server down (the multistream harness, one weight load).
- **One harness run, using fixtures keyed by generation.** The check that introduces generation
  n writes fixtures to `~/.cache/deepstrix/goldens/kvstore/gen-<n>/`: the chunks and tails of
  3 fixed prompts (8K, 20K, 40K, with cut classes from 7.2).
  - The new build restores the fixtures of the oldest generation present in the store,
    continues, and compares against its own cold prefill.
  - A generation introduced without the check has no fixtures. The check then uses the oldest
    present generation that does have them, and the release note records the gap.
- **Reduced matrix:** 3 prompts × 1 cut each, last-row KL plus the 32-step continuation KL.
  ESTIMATE: ~68K cold tokens × 3.04 ms ≈ 3.5 min of prefill at production speed, ~10-15 min
  with the harness's slower local pager.
- **When a change was not measured:**
  - its generation is still bumped (same commit);
  - no epoch bump, and no deletion;
  - the release note marks the generation "unmeasured";
  - the next measured check compares against the oldest generation then present that has
    fixtures.

**The DSpark drafter fingerprint is per tail, not in the namespace.** A ring whose drafter does
not match is dropped (a cold ring), because a drafter swap must not invalidate the KV.

**Garbage collection** (blocking finding 1):

- All namespaces live under `~/.cache/deepstrix/kvstore-v1/<ns16>/` and count against ONE
  global cap.
- At startup the store keeps the active namespace and the most recently active other one (for
  rollback). It **renames** all older ones into `kvstore-v1/trash/`: one rename each, so
  startup never waits on deletion. The IO thread unlinks the trash in the background.
  - A full 100 GiB namespace is ~37K files. MEASURED (M1 bench, 10-01, btrfs over dm-crypt):
    16K unlinks/s of 2.76 MB files, so ~2.3 s of background unlinks.
  - The kept inactive namespace counts against the cap with its persisted byte total
    (`<ns16>/bytes`, written by the IO thread at shutdown and hourly), not by statting its
    files (M1).
- **One process per root** (M1): opening takes `flock(LOCK_EX | LOCK_NB)` on
  `kvstore-v1/.lock` before any rename. If that fails, the store stays off and logs why: a
  second process would otherwise rename a live store's namespace or `tmp/` into the trash.
- Under pressure, the inactive namespace is evicted first, whole, the same way.
- Now that only semantics/ABI changes or an epoch bump create a namespace, this is rare: a
  model, tower or Engram swap, a store-layout change, or a 4.4.1 bump. Each strands at most
  one namespace until the next such change or the next pressure.
- Today's fingerprint-stale v6 directories are deleted at M5 along with the store.
- The root has mode 0700: the files contain prompts.

### 4.5 Files

- **`chunks/<h2>/<key>.kvc`.** Header:
  - magic `DSKVC`, format 1, ns, key, parent, k;
  - per store: rows, row bytes, key bytes;
  - provenance `prefill|decode|mixed|backfill-v6`, build id (`v6` for backfill),
    `KV_NUMERICS_GEN`, knob hash, created;
  - payload length and blake3;
  - a 16 B header checksum (M1). 256 B in all.

  Then the token ids, the image records and the payload.
- **`tails/<h2>/<key>.kvt`.** Header:
  - magic `DSKVT`, ns, key, base chain, T, kind `full|enc`, flag `anchor` (8.2);
  - flag `demoted` and an `origin` byte (prompt end, waypoint, anchor, cancel, turn end) (M1):
    thinning (8.2) applies only to demoted prompt ends, never to waypoints or cancel tails;
  - `n_raw`, `n_raw_dec`, drafter fingerprint, `session_id` if present (telemetry only);
  - hits, build id, `KV_NUMERICS_GEN`, knob hash;
  - **two** payload sections, each with its own length and blake3;
  - a 16 B header checksum over every header byte except the 4 B `hits`, so a touch stays one
    pwrite; `hits` is clamped on read so a damaged value cannot make a tail immortal (M1).
    512 B in all.

  Then the open tokens and images, and the two sections:
  - **section E:** the open rows and keys and the accumulators of each store, then the encoder
    windows;
  - **section D (full tails only):** the decoder windows (`n_raw_dec` rows), then, iff the
    drafter fingerprint is not zero, the ring, `ring_writes` and `last_ring_pos`. M2 defines its
    exact layout; M1 stores it as opaque bytes with its length and blake3. Asking for section D
    of an encoder tail is an error, never an empty section (the #25 signature).

  **Every mutation of an existing file runs on the IO thread:** the hits pwrite, the mtime
  update, the truncate and the header rewrite. The scheduler thread keeps its view (hits,
  last_used, kind) in memory and sends mutation messages. One thread applies them in order,
  so a demotion's header rewrite can never lose a concurrent hits update or tear the header.
  The rewrite carries the in-memory hits.

  **Demotion (8.2)** works because section D is last. It is two steps: `ftruncate` first, then
  rewrite the header. The startup scan compares the section lengths with the file size:
  - **a crash after the truncate:** the header still claims section D but the file ends after
    section E. The tail is treated as demoted and its header rewritten. Until then, a read that
    does not want section D is served from such a file (M1).
  - **a file longer than its sections:** it is truncated.
- **`tmp/`** holds writes in flight.

## 5. Tails

### 5.1 Kinds

| kind | contents | bytes (T ≥ 128) | restorable when |
|---|---|---|---|
| full | E + D | 5.68 MB + 2,760 B × (T mod C) | suffix t ≥ 1 |
| enc | E only | 2.65 MB + 2,760 B × (T mod C) | suffix t > 128 |

A full tail and an encoder tail at the same T share a key. The full tail replaces the encoder
tail.

### 5.2 Where tails are written

| point | kind | trigger | ESTIMATED volume (65 h) |
|---|---|---|---|
| prompt end L | full | `prefill_job_tick` finish, after the replay, the DSpark seed and `restore_compressor_lending` (where `snapshot::save` runs today). Skipped for marker-only prefills and when L < 512. | 4,944 × 7.1 MB = 35 GB |
| each multiple of K = 8192 crossed by a prefill ("waypoint") | enc (no open rows) | after the chunk that ends there (5.5) | 15.9M / 8192 × 2.65 MB = 5.2 GB |
| the **exact** chat anchor A (the last `<User>` before the first `<Assistant>`), on **second sight**: the job's walk matched chunks through ⌊A/C⌋C (another job already prefilled this prefix), found no tail at A, and A ≥ 1024, A − pos0 ≥ 128 | **full** (with open rows) | after the chunk snapped to end at A (5.5), via a side replay (below) | one 2.6 s replay + 5.7-8.5 MB per system prefix **that recurs** |
| cancel with fewer than 1,024 rows left | full | finish the job (≤ 1 chunk + a 2.6 s replay), write the tail, discard the logits | rare |
| other cancels, done ≥ 1,024 past the last tail | enc (with open rows) | the cancel path | small |
| turn end (M4) | full | stream finish or resident-tier eviction (2.3) | gated by `kv.turn_lcp` (11) |

**K = 8192.** The replay restores 90.57% at 8K, against 90.65% at 4K. Waypoint bytes are +11.7%
of chunk bytes at 8K and +23% at 4K. A restore between waypoints pays K/2 + 128 rows on
average, about 15 s, against a 455 s cold prefill.

**The anchor is exact, not rounded down to a chunk boundary.**

- Rounding down would lose up to 1,023 rows, 3-4 s at the warm slope, for **every** later
  conversation that restores it.
- The exact point costs the job that writes it at most two extra short chunks: one snap at A
  and one at ⌈A/C⌉C to realign (both in that job's snap set, 5.5), about 1-2 s once.

**The anchor is written on second sight, not on first sight.**

- **Most system prefixes never recur.** Snapshot corpus (`/tmp/kvreview/anchor2.py`;
  survivor-biased, ~11 h): the 16 first turns with A ≥ 1024 have 13 distinct anchor prefixes.
  11 of the 13 occur once; the other two occur 3 and 2 times.
- **That fits 12.1:** sub-agents embed unique paths in their system prompts. So the claim that
  sub-agents reuse an anchor many times is not measured.
- **What a first-sight write would waste.** Every unique-prefix spawn would pay a 2.6 s replay
  and ~7 MB for a tail nobody restores.
- **The rule.** The first job to prefill a prefix writes its chunks as usual and nothing extra.
  It does not snap at A and does not replay.
- **The second job** whose walk matches those chunks through ⌊A/C⌋C but finds no tail at A
  pays one miss: it prefills through A from its own restore point (a waypoint, or cold if
  A < K). It snaps at A, writes the anchor, and every later job restores from it.
- **The cost** is one miss per recurring prefix. The 11 unique prefixes above pay nothing.

**The anchor tail is full, not encoder-only.**

- **Why an encoder tail is not enough.** It serves only a first message longer than 128
  tokens.
- **The measurement** (`/tmp/kvreview/anchor.py`, 26 distinct first turns): the first-turn
  suffix after A (`<User>` … `<Assistant>`) is ≤ 128 tokens for 19%, with p25 216 and p50
  5,329.
- **What an encoder anchor would cost.** The median anchor sits at 5,282, below K, so such a
  conversation finds no waypoint and prefills its whole system prompt cold.
- **How it is written.** When the chunk ending at A completes, the job's replay buffer holds
  the residuals and carries of rows [A−128, A) (guaranteed by A − pos0 ≥ 128).
  - A new `prefill_job_side_replay` runs the CED decoder replay (layers 20-39, `Replay`) over a
    **copy** of that buffer, onto emptied decoder rings, and skips the head. It does not drain
    the buffer.
  - It captures section D. If DSpark is on, the ring is seeded from the side replay's captures
    into the job's reserved slot ring, which finish resets and re-seeds anyway.
  - The job's own finish then empties the decoder rings again, because t = L − pos0 > 128
    (`pos0 == 0 || t > b_seg`). So the job's result is unchanged.
- **Why the waypoint reason (5.3) does not apply.** This is one 2.6 s replay per system prefix
  that recurs, written once, not one per 8K positions.
- **It carries the header flag `anchor`,** which exempts it from demotion and thinning (8.2).
  Otherwise the first conversation built on it would demote it.
- **Gate G8 (7.3) checks the side replay,** because it is new code that mutates a running job.

**Cancel near the end.** An encoder tail with fewer than 129 rows left cannot serve a retry of
the same prompt. Finishing the job instead (one replay) lets the retry restore at L with t = 1,
rather than from a waypoint up to 8K + 128 rows back.

### 5.3 The CED problem

**Mid-prefill, the decoder windows do not exist.** The chunks run layers 0-20 only (L20 as
`KvSourceOnly`). That is why `clear_decoder_rings_for_checkpoint` empties those windows today.
Three ways to get exact decoder windows in a tail:

1. **A 128-row decoder replay at each tail.** 2.6 s p50 / 3.9 s p90 (MEASURED). A 330K cold
   prefill would pay it 40 times, about 104 s or +10%. Rejected.
2. **Store the replay inputs** (64 KiB per row): 8.4 MB per tail. Rejected.
3. **Use the decoder windows only where they are read.** `prefill_job_finish` keeps the rings
   only when `pos0 > 0 && t ≤ b_seg`, i.e. t ≤ 128 (#25). For t > 128 it empties them and
   replays the suffix's last 128 rows (#28); the replay buffer then comes entirely from the
   suffix. **Adopted:** an encoder tail is selected only when t > 128.

**What option 3 costs.** A request whose only candidate tail sits 128 rows or fewer before its
end falls back to an earlier tail. Retries are covered by the cancel-finish rule (5.2).

**Defensive check.** `prefill_job_finish` errors out, instead of replaying onto short rings,
when it keeps rings that hold fewer than min(pos0, 128) rows. That is the #25 signature.

**With `V41_CED=0`,** the decoder windows exist at every chunk end, so mid-prefill tails are
full. CED is part of the namespace.

### 5.4 DSpark ring

- **What a full tail stores:** the slot's ring after `MsDspark::seed`, plus `ring_writes`,
  `last_ring_pos` and the drafter fingerprint. Ring rows are roped at absolute positions.
- **Restore with t ≤ 128:** copy the ring into the reserved slot, then a new `seed_continue`
  appends the suffix rows without `reset`. It requires `last_ring_pos == pos0 − 1` and a
  matching drafter; otherwise it falls back to today's `reset` + `seed`.
- **Restore with t > 128:** section D is not read at all (6.6).
- **Expected value:** this closes plan 4.6. Plan 4.5 measured E 1.035 with a cold ring and
  1.649 seeded.

### 5.5 Chunk snapping

**Snap points** are the multiples of K. A job that will write the anchor (second sight, 5.2)
also snaps at A and at the realignment point ⌈A/C⌉C. They are enforced inside `image_spans::plan_chunk`, which caps
`chunk_end` at the next snap point.

- `plan_chunk` is the single point that both `PrefillJob::next_chunk_range` (lazy inputs,
  `chunk_inputs`) and `prefill_job_chunk` call. So the input rows and the chunk always agree.
- `PrefillJob::new` takes the snap points.
- Image blocks impose no snapping (4.3).

**Cost:**

- **Cold prefills** (chunk_rows divides K) cost nothing, except in the one job per recurring
  prefix that writes its anchor (5.2).
- **A warm prefill from an unaligned pos0** pays at most one extra short chunk when its suffix
  crosses a K-point (about 16% of warm admissions at S = 1,280). After that the chunks stay
  aligned.

## 6. Lookup and restore

### 6.1 The walk

Over R (L tokens), the spans and the marker flag:

- For each k with (k+1)C ≤ L, compute chain_{k+1}. Stop at the first key that is missing or
  pending deletion.
- At each matched node b (b = 0 included), test the tails attached there: a tail matches if
  its `tail(T)` over R[bC..T] is equal.
- **The writer's completion messages are processed before every walk** (9.4). Otherwise a
  retry that arrives right after the cancel that queued its tail would miss it.
- **Cost at 330K:** 323 blake3 calls over 4 KiB, about 1 ms, whatever the number of entries.

### 6.2 Selection

The candidate T is usable if:

- it is a full tail with T ≤ L − 1, or T = L with a trailing marker; or
- it is an encoder tail with L − T > 128;
- and T ≥ 64.

**Choice:** the deepest usable T wins; a full tail wins a tie.

**Logging:** the plan `(T, tail, path)` is logged together with the matched depth.

**On failure:** a file that fails its checksum or token-id check is evicted together with the
tails that depend on it, and selection re-runs without it.

### 6.3 Images

- Keying: 4.3.
- `encode_request_images` encodes **every** image in the request, prefix included. It is
  memoized by content hash, so a replayed image costs only the splice.
- `chunk_inputs` splices rows only for suffix positions.
- Image rows also get dead Engram ids (`EngramHash::compress`) and `bias_vl` routing from the
  synthetic ids. Both derive from token ids, so they are covered by the key.

### 6.4 Session hint

Dropped from lookup. The walk is exact and cheap, and `session_id` has never been sent (1.1).
When present, it is logged.

### 6.5 The admission path (`multistream.rs`)

- **The reservation** (`reservation`, `KvArena::reserve`) is sized from the prompt length before
  the lookup, unchanged.
- **`start_prefill`:**
  - `reset_in_place`, process the writer's completions, walk, then restore into the scratch
    state (compressors at rest);
  - counters: encoder `n_raw = min(T,128)` with `raw_off = 0`; decoder `n_raw` from section D,
    or 0 when t > 128; `n_comp = n_index_comp = ⌊T/ratio⌋`;
  - `prefix = R[..T]`, `pos0 = T`;
  - Engram `compressed` is rebuilt as today.
- **`PrefillJob::new(.., pos0 = T, snap points)`.**
- **`prefill_job_finish`** is unchanged, apart from the check in 5.3.
- **DSpark:** 5.4.
- **`fill_reserved` / `try_admit`** are unchanged.
- **Pins** are PATH pins (M1): pinning a chunk also pins its ancestors, otherwise an in-flight
  job's restored prefix could cascade away under the chunks it is writing, and its next tail
  would find a hole. Refs and pins are then both monotone up a path.
  - the restore plan's tail and chunk path, until the restore completes, owned by the job (and
    released at its end at the latest);
  - an in-flight job's written chunks, until the completion message of the next tail on its
    path (9.4).
- **Optional (M3):** walk at enqueue and order the queue by the real suffix length (SJF on the
  suffix, plan 5.3).

### 6.6 Restore IO (by link generation)

**Today:** per-element u16 conversion loops on the scheduler thread. 911 MB (a 330K prefix)
takes 2.1 s at Gen1 (0.44 GB/s) and 1.1 s at Gen4 (~0.8 GB/s).

**M2: the same bytes, a better path.**

- **Fixed host buffer pool.** A ring of 32 × 2.83 MB staging buffers is allocated once; no
  allocation per restore, given the hub's history of heap growth.
- **Pass 1: verify every file before the first host-to-device copy.** Pass 1 reads and checks
  the checksums of every chunk and tail section from the page cache.
  - blake3 MEASURED (M1 bench, this box, one thread): 7.2 GB/s, 911 MB in 126 ms. It is
    counted in the restore budget; parallel hashing is not needed.
  - A per-file "verified since startup" bit stops an active conversation's ~900 MB of chunks
    from being re-hashed every turn.
- **Pass 2: copy.** Read each chunk straight into staging (the bytes as they are) and do one
  large host-to-device copy per store run.
- **When t > 128,** section D (decoder windows + ring, 3.03 MB) is not read.
- **Target (ESTIMATE):** about 1.4 s at Gen1 (link-bound, ~0.65 GB/s on a 1 GB/s raw link)
  and about 0.45 s at Gen4 (≥ 2 GB/s). Gen1 is the binding case.

**M3: move the restore off decode time; keep link traffic out of decode bursts.**

- **Today** `start_prefill`, and so the restore, runs whenever a scratch state is free, in
  either phase.
- **M3** does two things while the request waits in the queue: it warms the page cache (reads
  the plan's files) and runs the verify pass. The staging ring (32 × 2.83 MB ≈ 90 MB) cannot
  hold a 911 MB plan, so nothing is staged ahead. The copy pass runs in the job's first prefill
  tick, in the prefill phase when no decode step runs. It streams from the page cache through
  the ring.
- **Not async host-to-device during decode bursts.** At Gen1 the link also carries every decode
  step's peer pushes and the 4.1 MB logits copy, so the stall would move into step time.

## 7. Exactness and gates

### 7.1 What is equal to what

**E1, store round trip (bitwise, required).** Restored bytes equal captured bytes, and
continuing from a restored state is bit-identical to continuing from the captured state. It
holds by construction.

**E2, resume at a matched partition (bitwise, in a configuration that does not depend on
residency).** An encoder tail at a K-point T plus t > 128 equals the uninterrupted prefill
whose chunks also end at T. Snapping makes that true for every cold prefill.

- **It holds in the harness:** local pager, no remote; or `V41_T2_CATCHALL=2`.
- **It does not hold in production.** Production runs `V41_T2_CATCHALL=1` with
  `V41_T2_PARTITION=1`, where the box-1/box-2 split of the MoE depends on residency (the
  determinism memo requires `=2`). So:
  - production chunks of one key are low-bit variants of each other;
  - a restore can combine chunks from writer A with a tail from writer B. B recomputed chunks
    that A had already stored, deduplicated its own write, and kept its own version in its
    state. That covers at most the K + 128 positions between B's restore point and its tail.
  - The writer logs `kv.dedup_mismatch` when a recomputed chunk's payload hash differs from the
    stored one, to measure this live.
    - **Cost:** hashing the job's own version needs a device-to-host copy of a chunk that would
      otherwise not be written, ~5 ms at Gen1. A job recomputes at most ~(K + 128)/C ≈ 9
      existing chunks.
    - **Sampling:** every deduplicated chunk in shadow phase A, 1 in 8 after `on`. That is
      ≤ ~6 ms per affected job on average.
- **Substitution and the cache prior do not touch prefill.** They apply to `RowLayout::Arena`
  (decode) rows only (VERIFIED).

**E3, unaligned resume with t > 128 (the usual next turn): NOT bitwise, by construction.** Kernel
choices depend on the number of rows in the batch. The thresholds apply **per lane**:
`plan_chunk` gives each lane `cb.div_ceil(2)` rows of a cb-row chunk.

- `prefill_f32_matvec(b)`: f32 matvec at lane b ≤ 64 (chunk ≤ 128), f16-activation WMMA above
  it (the compressor kv/score projections);
- the ratio-2 index-key projection, chosen by `n_boundaries` ≤ 64 per lane (chunk ≤ 256);
- `prefill_f32_matvec_qb_wo`: lane ≤ 16 (chunk ≤ 32);
- `mhc_pre_scaled_for`: lane ≤ 8 (chunk ≤ 16);
- `mhc_narrow_fallback_for`: lane ≤ 64 (chunk ≤ 128).

A prompt-end tail's open rows and windows come from the final, often short, chunk. A cold
prefill computes the same positions in a 512-row lane. The earlier claim that the evidence
"points both ways" was wrong:

- `MS_DIAG=job`'s "bit-identical" is a code comment; the mode prints diffs and asserts nothing.
- #22 compares one prompt before and after a long one, not two partitions.
- #28 itself says chunk size alone moves KL by 0.01-0.04.
- The 0.05 bar already failed once (0.068).

Today's whole-snapshot restore has the same property, so this design is no worse. The gate is
G3, against the null measured in 7.2.

**`V41_KV_RESUME_ALIGNED=1` stays a knob, default off.** It would restore t > 128 only from
aligned tails: C-snapping everywhere, plus an aligned encoder tail at ⌊L/C⌋C at every prompt end.

- **Cost (ESTIMATE):** about 512 extra rows per warm admission, i.e. ~2.1 s, ~2.8 h per 65 h,
  ~+8% of prefill hours, plus 2.65 MB per admission.
- **Benefit:** it buys bitwise resumes only in the harness, because production is not bitwise
  anyway (E2).
- It is enabled only if M0 shows unaligned resumes fall outside the cold-vs-cold partition
  noise.

**E4, short continuation (t ≤ 128, from a full tail).**

- Not equal to a cold prefill, by design: the rings carry the earlier replay's rows (#25).
- Bitwise equal to today's restore of the same point.
- KL against cold is reported (0.010 at S=60).

**E5, content addressing.** A key names a token prefix, not a computation. Chunk versions differ
in low bits because of E2 (production), E3, and decode provenance (M4). The first writer wins
and the header records the provenance. The fidelity contract is the KL gates set against the
null.

**E6, DSpark ring.** Bitwise through G1; acceptance is measured live.

### 7.2 M0: measure the null before setting bars

All of it runs in the harness (no remote, so residency plays no part).

**What can be measured.** This path yields last-row logits only:
`prefill_job_finish` → `head_from_row(last_idx)`. The per-token mode,
`forward_prefill_pipelined(last_only = false)`, turns CED off
(`ced = ced_enabled() && last_only`), so it is a different computation and cannot be the
reference. "Per-position KL" therefore does not exist here. The null is defined on what does
exist:

- **Metric 1:** KL of the last row, at the end of the prefill.
- **Metric 2:** the mean and max KL over an N = 32-step teacher-forced continuation. Both arms
  decode the same 32 tokens (taken from the cold arm's argmax) through the arena path, so the
  decode exposes the restored KV across many positions.
- **Diagnostics only, not gated:** per-store KV difference statistics (rows that differ, max
  |Δ| per store and per window).

**The items:**

1. **Matched-partition control (the proof of the restore path).**
   - P = a multiple of `MS_RESTORE_CHUNK` (P=1024, S=500; P=2048, S=1500), against cold with
     the same chunk size.
   - Both metrics **must be bitwise**. This item, not item 3, is what proves the restore and
     continuation path.
2. **Partition null.**
   - Cold(chunk=1024) against cold(chunk=X), X ∈ {256, 600, 1000}, and against cut placements
     at varied offsets in the chunk grid.
   - Over ≥ 20 real prompts (agent transcripts, 8K-60K tokens).
   - Classed by where the cut falls against the grid, and by the length of the last chunk.
     The classes in chunk rows are ≤ 16 / 17-32 / 33-128 / 129-256 / > 256. They follow the
     per-lane switch thresholds in E3 (lanes ≤ 8 / ≤ 16 / ≤ 64 rows, and ≤ 64 ratio-2
     boundaries per lane).
   - The G3 and G7 bars are the null's p99 in the same class, for both metrics.
3. **Attribute the 0.068 (attribution only).**
   - Re-run P=600/S=500 with the batch-size choices forced to their **exact** arms:
     `V41_PREFILL_F32_MATVEC=1`, `V41_MHC_PRE_SCALED=1`, `V41_MHC_NARROW=1`.
   - First check each forced arm against its per-row twin at these batch sizes.
   - **Not `=0`.** `V41_PREFILL_F32_MATVEC=0` routes lanes under 64 rows to
     `gemm_batched_wmma` (64-row tiles). That kernel has been measured writing zero rows at
     B=5 (`tests/f16_gemm_batched_small_b.rs`, on the iGPU; untested on gfx1201 at these
     sizes). The cold P+S = 1,100 run's second chunk is 76 rows: 38-row lanes with 19 ratio-2
     boundaries each, right in that regime.
   - **What the result means.**
     - If the 0.068 drops to the null, it was these switches.
     - If it does not, the remainder is batch dependence these three knobs do not cover (the
       small-batch dp4a dense arms in `dispatch.rs`, the MoE small-b arms from 1c5972f). That
       is documented, and is not a restore-path verdict: item 1 is the restore-path proof.

The fixed 0.05 bar is retired.

**Cost, honestly, and the split into windows.**

- **The null is the expensive item.** 20 prompts of 8K-60K (~30K average) × 4 cold arms is
  ~2.4M prefilled tokens. That is ~2 h at production's two-box 3.04 ms/token, and more with
  the harness's single-box local pager. Measure its rate on the first prompt and re-plan.
- So M0 is split:
  - **M0a, one ~1 h window:** item 1 (the control), item 3 (attribution), and G1/G2/G4 (7.3).
    These decide whether the restore path is right.
  - **M0b, one or more longer windows** (overnight if possible): item 2, the null. It can
    shrink to 12 prompts × 3 arms if windows are scarce. G3 and G7 run after M0b, since their
    bars come from it.

### 7.3 Gates

All harness gates run in one process (one weight load).

| gate | what | pass |
|---|---|---|
| G1 `MS_DIAG=store:P,S` (new) | **Prefill:** P via `PrefillJob`, written through the real writer to a temp store.<br>**Restore:** into (a) a FRESH state and (b) a scratch state DIRTIED by a longer prompt, which catches over-reads past the counters (KNOWN_BUGS #8; production reuses scratch states).<br>**Compare exactly:** per layer, raw window rows [0, n_raw) plus `n_raw`/`raw_off`; per store, comp rows [0, n_comp), keys [0, n_index_comp), every `state_kv` and `state_score` float, `n_comp`, `n_index_comp`; all 133 ring rows of each layer, `ring_writes`, `last_ring_pos`.<br>**Continue S from both;** compare logits.<br>**Matrix:** P ∈ {100, 1024, 3001 (odd: open ratio-2 group), 9000 (crosses K), 20000 (> 16,384: candidate mask live on L24-36)} × S ∈ {1, 60, 128, 129, 500}; a cancel tail at odd T; the marker case (T = L, t = 1); a DSpark t ≤ 128 continuation; a cut inside an image block with synthetic tower rows. | everything bitwise |
| G1-host (server crate, no GPU) | `chunk_inputs` with pos0 inside an image block equals the full-prompt splice | equal |
| G2 `MS_DIAG=store-resume` | cold P+S against a resume from the K-point tail inside it | bitwise |
| G3 | a resume from the unaligned prompt-end tail, against cold | last-row and 32-step continuation KL ≤ null p99 of the same class (7.2) |
| G4 `MS_DIAG=restore` (#28) | old and new restore paths | identical numbers |
| G5 unit tests | walk keys = write keys for every prefix length (golden vector); bug (a) pinned; refcounts under random insert/demote/evict; torn, short and corrupt files read as a miss; the selection table (t = 128 never picks an encoder tail); the knob-classification test (4.4) | pass |
| G6 live shadow | 11.1 | |
| G7 mixed writers | the tail of partition B over chunks of partition A, against cold. The same harness, run as old-build chunks + a new-build continuation against a new-build cold prefill, is the release-checklist step for `KV_EPOCH` (4.4.1). | both KL metrics ≤ null p99 |
| G8 anchor side replay (new code that mutates a running job) | (i) the same job run with and without `prefill_job_side_replay` at A: final state (every buffer and counter, as G1) and logits.<br>(ii) the anchor tail written at A, against the full tail that a cold prompt ending exactly at A writes at its own finish. The job under test is itself **cold (pos0 = 0)**, so both share the chunk partition before A: the job snaps at A. A second-sight job restored at an unaligned pos0 has a different partition and matches only within the null (E3). | (i) bitwise; (ii) bitwise, sections E and D and the ring, for pos0 = 0 |

## 8. Eviction and capacity

### 8.1 Model

- **Tails are the unit of eviction.** A chunk keeps `tail_refs`, the number of tails whose
  path contains it, at O(T/C) per insert or removal. It is deleted when that reaches 0,
  unless pinned.
- **Removing a tail frees its own bytes.** The chunk bytes go only with the last tail beneath
  them.

### 8.2 Lineage from the trie (replaces the session cap that never ran)

**Ancestors.** When a full tail is inserted at T, the full tails on its ancestor path are
already known: the job's walk matched them (tails at T' < T whose tokens prefix R).

**Demotion.** The newest N = 2 on the path stay full. Two cover "regenerate the last turn",
whose request is the previous prompt (t = 1). Older ones are **demoted** to encoder tails:
`ftruncate` section D, which frees 3.03 MB each (4.5). "Newest" on one path means deepest T: a
conversation's prompt ends grow with its turns. Anchors are exempt and do not count toward N
(M1).

**Thinning.** The path keeps at most one demoted tail per K-window [jK, (j+1)K), the deepest.
Waypoints are kept separately. This is ds4's continued-prefix discount, made structural.

**Pins block demotion and thinning, not only deletion.** The M3 prefetch and a restore in
flight read section D of pinned tails. A demotion or thinning that hits a pinned tail is
deferred until the unpin. The truncate-then-header order, and how the startup scan recovers a
crash in between, are in 4.5.

**Anchor tails are exempt from demotion and thinning** (header flag `anchor`, 4.5).

- **Why they need the exemption.** The anchor full tail at A (5.2) is an ancestor of every
  conversation that shares the system prefix, and each of their walks matches it. Without the
  exemption, the first such conversation would demote it to encoder-only at its third prompt
  end. Encoder-only is exactly what N4 fixed: 19% of first messages are ≤ 128 tokens after A.
- **Thinning would also remove it.** The median anchor (5,282) and that conversation's first
  prompt end usually share the first K-window, so thinning would then delete it.
- **No later write repairs it.** The anchor rule writes only when no tail exists at A.
- **So an anchor tail leaves only by score eviction (8.3).** Every new conversation that
  restores it adds a hit and keeps it young.
- **The cost is bounded:** one 5.7-8.5 MB tail per recurring prefix.

**Bound** for a 150K conversation with many turns (ESTIMATE): chunks 414 MB + 2 full tails
14 MB + ≤ 18 demoted × ~4.1 MB + 18 waypoints × 2.65 MB ≈ 550 MB.

### 8.3 Order

1. **Orphan chunks:** refs 0 and unpinned, left by killed jobs. Oldest first. A failed job's
   orphans are deleted at once (9.4).
2. **Tails by lowest score.**
   - score = `path_last_used + 6 h × log2(1 + hits)`.
   - `path_last_used` is the maximum `last_used` over the tail and every tail below it on the
     same path. Restoring or inserting a tail touches every ancestor tail the walk matched, so
     a live conversation's old waypoints (its branch points) stay young.
   - **Propagated touches are in memory only.** They never increment `hits` and are never
     persisted. At startup, `path_last_used` is recomputed from the leaves' persisted
     `last_used` (the file mtimes). `hits` counts only restores of the tail itself.
   - **Tie-break:** the larger bytes freed (cascade included) first. Within a dead
     conversation, leaves go first, so its chunks cascade.
3. **Inactive namespaces** go before all of the above (4.4).

**Global cap:** `V41_KV_STORE_CAP_GIB` = 100, with 1% headroom.

**Cost of a touch** (the restored or inserted tail only): an in-memory update, plus a message
to the IO thread, which updates the mtime (`utimensat`) and pwrites the 4 B `hits` (all file
mutations are on the IO thread, 4.5). Today every hit rewrites `meta.json` on the scheduler
thread.

**Removal is split between two threads.** Eviction, cascades, thinning, namespace GC and
`PURGE_BUILD` take index entries out on the scheduler thread, which is cheap and immediate:
lookups stop seeing them. The unlinks go to the IO thread, on a deletion queue served behind
pending writes.

- The cap is enforced on the index, so removed entries leave the count at once.
- The unlink backlog is reported (`kv.store trash_bytes`). In the worst case, a whole 100 GiB
  namespace, it lasts ~2.3 s (MEASURED in M1: 16K unlinks/s at 2.76 MB, 4.4). The filesystem
  keeps ≥ 380 GB of slack (8.5) for it.
- **An unlink never deletes a newer file** (M1). A later write to a path cancels a queued unlink
  of it; and an unlink of a key with a write in flight is skipped, because that write already
  ran or will run and the file at the path is the new one. If that write then fails, the stale
  file goes when nothing is in flight for the key.
- Neither the backlog nor the unlinks ever block the scheduler.

### 8.4 Capacity (ESTIMATE)

- **The 107 GB on disk today** would fit in about 12 GB.
- **Writes** drop from 21.3 to about 1.3 GB/h, so 100 GiB holds about 3 days of writes,
  against about 5 h.
- **How many of today's cold misses come from eviction** is unknown: the logs carry no
  session ids. Shadow measures it (11).

### 8.5 Disk budget

**Disk:** `/home` is btrfs on dm-crypt, 1.9 TB, 487 GB free. Logs and evtrace share the
filesystem.

| phase | v6 store | new store | free after |
|---|---|---|---|
| shadow A | 107 GB | ≤ 100 GiB | ~380 GB |
| shadow B | 107 GB | 15 GiB | ~470 GB |
| on (until M5) | 107 GB | ≤ 100 GiB | ~380 GB |
| after M5 | 0 | ≤ 100 GiB | ~490 GB |

### 8.6 Telemetry

**Log lines** (key=value, ANSI-free):

- `kv.plan prompt matched tail_kind restore lost suffix walk_ms reason` (in shadow, `reason` explains a plan shallower than the old store's restore: 11.1)
- `kv.restore pos chunks bytes verify_ms h2d_ms`
- `kv.write kind(chunk|enc|full|anchor|backfill) k bytes d2h_ms queue_mb wait_ms`
- `kv.write_dropped`
- `kv.dedup_mismatch`
- `kv.demote`, `kv.suspect` (9.4)
- `kv.evict kind T freed cascaded age_h hits why=cap|orphan|failed|corrupt|namespace`
- `kv.turn_lcp prev_prompt gen lcp new_len` (11)
- `kv.store`, every 10 min: bytes by kind and namespace, files, write and evict GB/h, oldest,
  restored share, `trash_bytes` (the unlink backlog)
- `kv.miss_diag`: on a cold or shallow plan, the first divergent position against the deepest
  matching lineage and 64 decoded bytes on each side. It replaces `diag_largest_divergence`
  and is the tool for the client changes in section 12.

**evtrace:** the kinds `KV_PLAN`, `KV_RESTORE`, `KV_WRITE` and `KV_EVICT` (added to
`evtrace_kinds::ALL` and `scripts/evtrace.py`).

## 9. Writes

### 9.1 When

| # | what | when |
|---|---|---|
| w1 | each chunk with (k+1)C ≤ pos0 + done, if not indexed or pending | after each `prefill_job_chunk` |
| w2 | an encoder tail at waypoints; a full tail at the anchor (a side replay, 5.2) | after the chunk that ends there |
| w3 | a full tail at L | at finish (5.2) |
| w4 | a cancel tail (5.2) | cancel |
| w5 | the chunks of decoded positions + a full tail | M4 |

Tokens always come from `p.req.tokens[..pos0 + done]` (+ the marker), never from `pf.prefix`.

### 9.2 Volume (65 h)

| | today | this design (ESTIMATE) |
|---|---|---|
| per admission | the whole prompt: 101K avg × 2,760 B = 279 MB | the new chunks + a 7.1 MB tail |
| total | 1,382 GB (21.3 GB/h) | chunks 44 + full tails 35 + waypoints 5.2 ≈ 85 GB (1.3 GB/h): 16x less. The corpus replay gives 8.2x. |

### 9.3 Cost on the scheduler thread (by link generation)

The only cost is a sync device-to-host copy of contiguous rows from the idle scratch state.

| | Gen1 x4 (0.57 GB/s) | Gen4 x4 (~1.6 GB/s) |
|---|---|---|
| chunk (2.83 MB) | ~5 ms | ~1.8 ms |
| encoder tail (2.65 MB) | ~4.6 ms | ~1.7 ms |
| full tail (7.1 MB avg, 8.5 MB max) | ~12 ms (≤ 15) | ~4.4 ms |
| warm admission (~1.25 chunks + full tail) | ~20 ms | ~7 ms |
| cold 330K (323 chunks + 40 waypoints) | ~1.8 s over a ~1,045 s prefill | ~0.65 s |
| per 65 h (85 GB) | ~2.5 min | ~0.9 min |
| today's whole-prompt save, per admission and per 65 h | ~0.48 s / ~39 min | ~0.17 s / ~14 min |

### 9.4 IO thread

- **One writer, FIFO**, bounded by `V41_KV_WRITE_QUEUE_MB` (512).
- **Backpressure with a deadline.** When the queue is full, the scheduler waits up to
  `V41_KV_WRITE_WAIT_MS` (200 per tick) for room, and only then drops the write
  (`kv.write_dropped`). Dropping is the last resort: a dropped chunk is a hole in the chain.
- **Broken applies per key, not per path.** A dropped chunk k blocks only the tails whose path
  includes k.
  - A job that skipped k because another job's write was pending subscribes to that write's
    completion.
  - If that write is dropped, the job re-enqueues k from its own scratch state, which still
    holds rows [0, done). If it no longer can, it marks k broken for itself.
- **Chunks may land before their parent** (M1): a dropped chunk k is re-enqueued behind k+1.
  The index holds such a chunk detached, unreachable for walks until its parent lands. A tail
  whose chunk path is incomplete when it lands is dropped (`kv.write_dropped why=broken_path`)
  and its file removed; the startup scan drops chunks that do not reach the root.
- **Writes in flight are queued per key, oldest first** (M1): an encoder tail and then the full
  tail of the same key (a waypoint at L % K = 0, an anchor on a K multiple, two jobs) both land,
  in order; a second write of the same kind is `Pending`.
- **Pins are released on the tail's completion message,** not at enqueue. A tail dropped as
  `broken_path` releases them too.
- **A job may end with writes in flight** (M1): M2 ends it in the tick that queues its last
  writes. Its late completions land unowned (no pins); a failed job's late chunks go at once if
  nothing references them; the store forgets the job when its writes drain. The invariant
  checker fails if a finished job still owns a pin.
- **Only data-attributable failures evict** (M1): a format verdict, a short or missing file. A
  read that fails transiently (EIO, EMFILE, ENOMEM, EACCES) returns an error, logs
  `kv.suspect` and evicts nothing; so does a demotion that fails that way.
- **Completion messages are processed at the top of every scheduler tick and before every
  walk.**
- **Shutdown drains the queue,** bounded by the queue size: ≤ 512 MB, under 1 s at disk speed.
- **A failed job** (an error, not a cancel):
  - its chunks that no tail references are deleted at once. A retry recomputes those positions
    anyway, to rebuild the encoder windows, so this costs nothing.
  - **The restored tail is kept,** unless the error is data-attributable.
    - **Evict it on:** a checksum, token-id, shape or ABI validation failure. Those already
      fire before the first copy to the GPU (6.6) and evict through 6.2.
    - **Never evict it on device or RPC errors.**
    - **What the logs show.** There are no prefill-path failures at all: "prefill failed"
      appears 0 times in every log. The device faults seen were step-level.
      - One decode-step failure on 09-28 06:45:38 (HIP error 999 in `hipMemcpy(HtoD)`) was fanned
        out by `abort_all` to every live stream, giving 16 log lines. It falls outside the 65 h
        window.
      - The ~10 "engine wedged" watchdog aborts are also from 09-28.
    - **No mechanism is known** by which a tail that passed checksum, token-id and shape
      validation causes a device fault.
    - **Why not ds4's rule.** ds4 evicts on a prefill failure after a load, which fits ds4,
      where such a failure implicates the payload. Here a persistent fault, or an intermittent
      one such as a flapping box-2 link, would fail a conversation's retries too. Evicting
      would turn an infrastructure fault into cold re-prefills.
    - **`kv.suspect`** logs the tail key, the error and the job, for diagnosis only. Nothing
      acts on it, and nothing persists it.

### 9.5 Atomicity and crash safety

- **Writes:** `tmp/` then rename, no fsync: this is a cache. The FIFO order puts tails after
  their chunks, but without fsync the filesystem may reorder them. MEASURED (M1): create + fsync
  costs 2.1-3.5 ms per file here; the checksums and the startup scan already turn an unsynced
  file into a miss, so no fsync stays. A new file's mtime is set to its header's `created`: a
  tail's mtime is its persisted `last_used` (8.3).
- **Startup:**
  - take the root lock (4.4);
  - clear `tmp/` (one rename into `trash/`);
  - rename stale namespaces into `trash/` (4.4); the unlinks run in the background;
  - scan the headers (~38K files, ~2 s, ESTIMATE);
  - check each tail's section lengths against its file size (4.5);
  - rebuild the index, the refcounts and `path_last_used`;
  - remove from the index the tails that have missing ancestors and the chunks that do not reach
    the root, and queue their unlinks. If any header read failed transiently, leave those on
    disk unindexed instead (the missing link may be the unreadable file) and judge them at the
    next startup (M1).
- **Startup time** is the header scan plus renames: a few seconds. It never waits on unlinks.
- **On every read:** checksums per section and a token-id comparison (6.6).
- **Invariant checker:** in debug builds and in shadow, every hour, a full refcount rebuild is
  compared with the incremental state.

### 9.6 Concurrency

- Production runs `V41_MS_PREFILL_JOBS=1`: the scratch state is read only between chunks, on
  the scheduler thread.
- Duplicate writes are deduplicated through the pending set, with completion subscriptions
  (9.4).
- M4 and the resident tier read arena slots between steps.

### 9.7 Code layout

- `crates/deepstrix-server/src/kvstore/` (M1): `keys.rs` (namespace, chain, the golden key
  vector), `format.rs` (headers, readers), `index.rs` (trie nodes, tails, refcounts, pins, the
  eviction order, demotion, thinning, walk, selection, the invariant checker), `io.rs` (the IO
  thread), `scan.rs` (root lock, namespace GC, startup scan), `knobs.rs` (the classification and
  the knob hash), `mod.rs` (the `Store` facade the scheduler thread uses), `tests.rs`.
- GPU side, `het/kv_capture.rs` next to `KvArena::export_to_state`: `capture_rows`,
  `capture_tail`, `restore_into`, plus the arena-slot variants for M4.

## 10. Checkpoints become tails

- **Periodic checkpoints.** `V41_MS_CHECKPOINT_EVERY` (32,768; a whole-prefix save each time,
  668 MB at 242K) is replaced by waypoints every 8,192 positions (2.65 MB each). The chunks
  are already on disk.
- **Cancel checkpoints.** They become the cancel rules in 5.2.
- **Killed prefills.** A kill loses at most K + 1 chunk. On 09-22 one lost 98K rows of a 262K
  prompt.
- **Retries** restore at the cancel tail, or at L when the job was finished on cancel.
- **Bug (a)** cannot happen: keys come from the request tokens, and G5 pins it.
- **Bug (b)** cannot happen: the walk does not depend on boundaries.
- **Retired** when on: `V41_MS_CHECKPOINT_EVERY` and `V41_MS_CHECKPOINT_MIN_ROWS`.

## 11. Migration and rollout

- **New format, new directory; the old cache is dropped.** `snapshots-v41` stays untouched
  until M5.
- **`V41_KV_STORE=off`** is the default until M3.
- **`shadow`:**
  - The old store serves every restore exactly as today.
  - The new store does every write, demotion, eviction and walk, but restores nothing. It
    logs its plan next to the actual restore.
  - It costs about 20 ms of scheduler time per admission at Gen1 and about 1.3 GB/h of IO, and
    never changes GPU state.
  - It also logs `kv.turn_lcp`. Per session key, either the `session_id` if sent or the
    lineage found by the walk, it compares the last finished stream's prompt + generated
    tokens with the next prompt. That measures how often the re-encoded response equals the
    sampled tokens, which gates M4.
- **`on`:** the new store serves, and the old one is neither read nor written. Rollback is
  `V41_KV_STORE=off` plus a hub-only restart (box 2 is not touched).
- **Legacy serial path:** it keeps the old store until M5 and serves no production traffic.

### 11.1 The two shadow phases and their gates

**Phase A.** Cap 100 GiB, at least 48 h. It measures planning, volume and stalls. It will not
fill the cap: 1.3 GB/h × 48 h ≈ 62 GB.

- **Backfill.** The store starts empty. After an old-store restore, the new store lacks the
  restored prefix, and w1 would write it all in the first tick: 146 chunks for 150K, ~0.7 s
  at Gen1, on top of the old store's own save. So:
  - these writes are tagged `kind=backfill` and kept out of the per-chunk-tick stall metric,
    with their own report;
  - each admission gets a budget, `V41_KV_BACKFILL_MS` (default 250 ms, ~50 chunks at Gen1),
    and the rest waits for the lineage's later turns, which again restore from the old store
    and backfill the next stretch;
  - a 330K lineage (323 chunks) therefore needs ~7 turns to backfill fully;
  - backfilled chunks carry provenance `backfill-v6`, build id `v6` and the generation `v6`
    (4.5). Their bits come from v6 snapshots written by older builds, not from the writer's
    build. Nothing deletes them routinely: `PURGE_BUILD=v6` is an explicit action (4.4), and
    the release check treats `v6` as the oldest generation present;
  - backfill exists only in shadow. Under `on`, restores come from the new store.
- **Plan against actual, with reason codes.** Every `kv.plan` whose restore point is shallower
  than the old store's actual one carries a reason:
  - `bootstrap`: defined from **writer-side** state, never from where the walk stopped.
    - **The writer's frontier, defined precisely:** the position f of the deepest tail
      whose **whole chunk path was completed**, written and renamed rather than just
      enqueued, by the time that tail's own completion message arrived.
      - Positions merely written past a gap do not count. A backfilling job prefills
        [T_old, L) while its backfilled chunks reach only ~50K. That job's frontier is its
        deepest tail with a gap-free path, so the gap cannot raise a false alarm.
    - **The entry:** an 8-byte prefix of a plain blake3 over `tokens[0..f]`, plus f. That hash
      is independent of the chunk-chain derivation.
    - **Storage:** a 10K-entry LRU, memory-only, which resets at restart. A key-derivation change
      across a deploy is therefore caught by G5's golden key vector, not by this gate.
    - **Lookup:** one incremental blake3 pass over R, cloning the hasher at the sorted distinct
      f values of the entries (~1 ms, like the walk).
    - **Only entries usable for this request count.** An entry at f is a tail that selection
      (6.2) may reject for this request: an encoder tail with L − f ≤ 128, or one below the
      T ≥ 64 floor. Such entries are ignored, so a plan that legitimately stops below them is
      not `unexplained`.
    - **`bootstrap`** means no usable frontier entry for this request's tokens reaches the old
      store's restore point.
    - **Before anything falls to `unexplained`,** the plan is tested, in order, for `dropped`,
      `evicted` (phase B), `thinned` and `demoted`. Each test uses the store's own record of
      that event for the tail or chunk where the walk stopped.
    - **`unexplained` (treated as a key-derivation bug, risk 2, or lost files)** means the walk
      stopped **below** a usable recorded frontier for the same tokens and none of those events
      explains it. A key bug can therefore never pass as bootstrap.
  - `demoted`: a t ≤ 128 request at an older prompt end that the old store still holds as a
    snapshot, while the new store has demoted it (8.2);
  - `thinned`: a branch inside a thinned K-window;
  - `dropped`: a write was dropped (9.4);
  - `evicted`: the cap, phase B only;
  - `unexplained`: everything else.

  Only `unexplained` is gated.

**Phase B.** Cap 15 GiB, at least 24 h after the first eviction. It exercises eviction,
cascades, demotion, thinning and refcount decrements, the riskiest new code.

- **It starts from phase A's store, deliberately.** The restart with the smaller cap triggers an
  immediate eviction storm of ~45 GB down to 15 GiB. That is a stress test of cascades and of
  the split removal (8.3). Its duration and the unlink backlog are reported.

| metric | today | needed |
|---|---|---|
| plan shallower than actual (A, `unexplained` only) | n/a | 0 admissions (otherwise a bug) |
| restored share, in aggregate (A) | 96.8% | ≥ 96.8%, computed over **non-bootstrap** admissions and reported over the last 24 h of phase A. Over the whole phase it fails for benign reasons: backfill takes ~7 turns for a 330K lineage. |
| write volume (A) | 21.3 GB/h (ESTIMATE) | ≤ 3 GB/h |
| write stall per chunk tick, p99 (A, backfill excluded) | n/a | ≤ 20 ms at Gen1 (≤ 8 ms at Gen4). The anchor's side replay (2.6 s, once per recurring prefix) is prefill compute, reported separately. |
| write stall per finish, p99 (A) | ~480 ms per save at Gen1 | ≤ 25 ms at Gen1 (≤ 10 ms at Gen4) |
| token-id or checksum failures, `kv.write_dropped` (A + B) | n/a | 0 |
| invariant checker (B) | n/a | clean every hour, including during the eviction storm |
| age at eviction (B) | ~5 h | within 20% of cap / write-rate. Extrapolated to 100 GiB: ≥ 24 h. |
| G1-G5, G7, G8 | | pass |

**After on, track:**

- restore ms p50/p90 against today at the same link generation;
- the cold-prefill hours of prompts that share ≥ 8K tokens with a stored prefix (expect ~0);
- DSpark E at t ≤ 128.

## 12. What this does not fix: client prompt layout

### 12.1 The measured causes

Among the stored prompts over 20K tokens, 24 have no reusable older prefix. The best common
prefix they share with any older prompt: 30 of 329,743 tokens, 393 of 241,666, 241 of
214,267, 459 of 208,954, 29 of 189,437, 0 of 119,006. The divergences:

- the 330K "compaction" request: a different system prompt ("You are a context summarization
  assistant…") in front of the same conversation;
- a "security reviewer" auxiliary prompt;
- sub-agents whose system prompts embed unique session paths, working directories or parent
  names in the first ~250-500 tokens.

KV depends on every earlier row, so a change at token 300 invalidates everything after it.
Cold cost is ~42 s + 3.04 ms/token: about 17 min for a 330K compaction, against a few seconds
of suffix if the prefix were reused.

### 12.2 Guidance for clients (letta, letta-code, the coding agents)

**Order of the prompt.** The stable prefix first; per-session details and task instructions
last:

1. **The system prompt,** byte-identical across sessions and sub-agents: no dates, ids, paths
   or user names.
2. **Tool definitions:** fixed set, order and JSON formatting.
3. **Slow-changing context** (memory blocks, project docs). An edit invalidates everything
   after it, so put the volatile blocks last.
4. **The per-session environment** (cwd, session path, parent agent, date), in the first user
   message.
5. **The task or role instruction.**
6. **The conversation, append-only.**

**Patterns:**

- **Compaction and summarization:** keep 1-6 and append the instruction as the last user
  message. Do not put a summarizer system prompt in front.
- **Auxiliary reviewers:** the same.
- **Sub-agents:** copy the parent's system prompt and tools verbatim, and put the role, paths
  and task in the first user message. A 15K shared prefix then saves ~45-60 s per spawn.
- **History:** never rewrite earlier messages (tool results, ids, JSON, thinking), and never
  insert timestamps into it.
- **Check it** with `kv.plan` and `kv.miss_diag`.

## 13. Milestones

| | scope | exit |
|---|---|---|
| M0a | Harness, server down about 1 h, one weight load: the matched-partition control, attribution of the 0.068 (the exact arms checked against their per-row twins first), and G1, G2, G4 against a prototype store inside the test. Commit the analysis scripts (1.0). | the control is bitwise; the 0.068 attributed or documented |
| M0b | The partition null (7.2 item 2), ~2 h+ of prefill over one or more longer windows (or the reduced 12 × 3 matrix); then G3 and G7; the first generation fixtures (`gen-<n>`, 4.4.1). | the bars set per class; the default for `V41_KV_RESUME_ALIGNED` chosen |
| M1 | The `kvstore` module without GPU: format (two sections, `anchor` flag), chain, namespace + GC, index, refcounts, demotion and thinning with the anchor exemption, eviction, IO thread, startup scan, invariant checker, the knob-literal scan; G5. Measure blake3 throughput and unlink rate. | `cargo test` green; property tests |
| M2 | Integration: w1-w4, snapping in `plan_chunk` (with ⌈A/C⌉C), the anchor side replay with its `anchor` flag, restore into the scratch state (two passes, buffer pool), the DSpark ring, `kv.suspect` logging, the backfill budget and `backfill-v6` provenance, the writer frontier and reason codes, file mutations on the IO thread, `KV_NUMERICS_GEN` in headers, telemetry, shadow. | G1-G4, G7, G8 and G1-host on the real code; shadow A then B (11.1) |
| M3 | `on`; old store off; restore in the prefill phase with disk prefetch; optional SJF on the suffix. Weigh against the resident tier (2.3). | after-on metrics; one week clean |
| M4 | Turn-end tails (w5), if `kv.turn_lcp` shows the sampled tokens are replayed; needs exact tool-call replay; byte-keyed side index if needed (4.3). Pairs with the resident tier. | warm suffixes shrink by the completion length; KL gate for decode-provenance chunks |
| M5 | Delete `snapshot.rs`'s store, the checkpoint knobs, and `snapshots-v41` together with its stale fingerprint dirs. | done |

## 14. Risks

1. **The restore path can be silently wrong.** Past cases: v6 `index_k`, #22, #25, #28.
   Mitigations: G1 compares every buffer and counter exactly, into a fresh and a dirtied state;
   per-section checksums; token-id checks; the v6 key rule; the finish check (5.3).
2. **A key-derivation bug orphans the whole store.** Mitigation: one function shared by the
   walk and the writer, plus a golden vector.
3. **Refcount or demotion drift** leaks bytes or deletes live chunks. Mitigation: a rebuild at
   startup, the hourly checker, and shadow B under eviction pressure.
4. **Production is not bitwise** (E2, E3): chunk versions mix. Mitigation: the null-based KL
   bars, G7, and `kv.dedup_mismatch` measured live.
5. **The writer falls behind on a disk stall.** Mitigation: backpressure with a deadline;
   breakage per key; re-enqueue from the job's own state.
6. **Disk fills from stranded namespaces.** Mitigation: one global cap, keep at most active +
   one previous, and evict inactive namespaces first (4.4).
7. **A numerics change beyond the null ships without an epoch bump**, or a fix that corrected
   stored KV ships without one. Mitigations:
   - the release-checklist step (4.4.1: G7 old chunks + new continuation against the null);
   - `KV_NUMERICS_GEN`, the knob hash and the build id in every header, and the commit rule for
     diffs that do not bump the generation (4.4);
   - purges are explicit decisions only, never routine retention (4.4);
   - the classification test over whole crates.
8. **The opposite failure: bumping the epoch too often** cold-starts every conversation,
   ~350 s each, serial. Mitigation: the namespace is semantics/ABI only, and the bump has
   exactly two triggers (4.4.1).
9. **Restore stalls move into decode steps** if host-to-device copies overlap decode at Gen1.
   Mitigation: restores only in the prefill phase (6.6).
10. **Transient device faults turn into cache cliffs.** Mitigation: only data-attributable
    errors evict. Device and RPC errors never evict; they are only logged as `kv.suspect`
    (9.4).
11. **Privacy:** the files contain prompt token ids. Mitigation: mode 0700.

## 15. Open questions

1. What is the 0.068 at P=600/S=500? Is it the three forced switches, or batch dependence they
   do not cover? The restore path itself is proven or refuted by the matched-partition control.
   (M0a, 7.2)
2. How often does a re-encoded response equal the sampled tokens? That decides M4 and the byte
   side index. (`kv.turn_lcp`)
3. How many cold misses come from eviction? (shadow: the plan restores where the old store
   restored 0)
4. K = 8192 or 4096? 0.08 points apart in the replay. Live branches may prefer 4096.
5. Should a disconnect in the middle of a long prefill keep prefilling further than the
   cancel-finish rule (5.2)?
6. Resident tier first, or M3 restore latency first? (2.3)
7. Keep the full tails of live lineages in pinned host RAM, or rely on the page cache?

## Appendix A: review dispositions

### Review round 1

| # | finding | resolution |
|---|---|---|
| 1 | blocking: namespace GC | fixed. One global cap over every namespace; keep active + one previous; evict inactive first (4.4, 8.3, 8.5). |
| 2 | the per-lineage cap never fires | fixed. Lineage from the trie; demotion (newest 2 stay full); thinning; staleness propagated up the path; ties go to the larger bytes freed (8.2, 8.3). R1's non-use stated in 1.1. |
| 3 | E3 evidence mischaracterized | fixed. E3 is not bitwise by construction (the kernel switches are listed); E2 holds only without residency dependence; M0 null (7.2); G3/G7 bars at the null; aligned-resume cost stated, default off; G7 added. |
| 4 | bandwidth numbers | fixed. Gen1 and Gen4 throughout (1.1, 6.6, 9.3, 11.1); today's save is ~0.48 s per admission at Gen1; M2 target re-derived (~1.4 s at Gen1); M3 restores move to the prefill phase, with no async host-to-device in decode bursts. |
| 5 | images (as corrected by the coordinator) | fixed. False claims removed; image rows are causal; cuts may fall inside blocks; the hash is folded at IMAGE_START, so it covers mid-block cuts (4.3); no prerequisite; G1 image case and G1-host added. |
| 6 | G1 comparator | fixed. Every buffer and counter compared exactly, ring included; fresh and dirtied states; matrix extended; one process (7.3). |
| 7 | shadow never evicts | fixed. Phase B at 15 GiB; bootstrap exclusion; gate on the invariant checker under pressure (11.1). |
| 8 | provenance | fixed. Files listed; scripts to be committed (1.0); "ESTIMATE" labels; the per-prompt mean added; the survivor bias reversed (1.2); functions cited, not line numbers. |
| 9 | namespace incomplete | fixed. Tower, Engram, knob hash with a classification test, drafter fingerprint per tail, build id + knob hash per header, targeted purge, derive_key contexts (4.3, 4.4). |
| 10 | keying, M4 | fixed. Special-token point added; byte-keyed side index as M4's fallback; tool-call replay is an M4 prerequisite; M4 gated by `kv.turn_lcp` (4.3, 5.2, 13). |
| 11 | write path | fixed. Backpressure with a deadline; breakage per key with re-enqueue; pins released on completion; drain on shutdown; completions before walks; a failed job's orphans deleted (9.4). |
| 12 | stall metric | fixed. Per chunk tick and per finish, by link (11.1). |
| 13 | anchor, cancel | fixed. Exact anchor, with the break-even stated; cancel with fewer than 1,024 rows left finishes and writes a full tail (5.2). |
| 14 | tower statement | fixed (6.3). |
| 15 | restore path | fixed. Verify all files before the first host-to-device copy; skip section D when t > 128; blake3 counted, with a verified-since-startup bit; fixed buffer pool (6.6). |
| 16 | snapping | fixed. Inside `plan_chunk`, which both callers use (5.5). |
| 17 | in-memory tier | added as 2.3. It is a separate tier, positioned relative to M3 and M4; not designed here. |

### Review round 2

| # | finding | resolution |
|---|---|---|
| N1 | the namespace over-separates | fixed. Namespace = semantics/ABI only (model, tower, Engram, store layout, C, `SWA_WINDOW`, CED, keys on/off, `KV_EPOCH`). Knob hash + build id in headers only. The epoch is bumped only for (a) fixes that corrected stored KV or (b) changes beyond the null, measured G7-style as a release-checklist step; everything else via `PURGE_BUILD` (4.4, 4.4.1). The classification test covers whole crates (`v4flash-kernels`, `v4flash-core`), which catches `f16.rs` and `dispatch.rs`. Risk 8 added. |
| N2 | a failed job evicts its restored tail | fixed. Only data-attributable errors evict (checksum, token ids, shape, ABI). Device and RPC errors keep the tail (the suspect state machine introduced here was replaced by log-only `kv.suspect` in N12) (9.4). Risk 10 added. |
| N3 | M0 measures what does not exist / wrong direction | fixed. The null is last-row KL + KL over a 32-step teacher-forced continuation, over ≥ 20 prompts and cut placements, classed by grid position and last-chunk length; per-store KV statistics are diagnostics only. Item 3 forces the exact arms (`=1`), checked against their per-row twins first, and is attribution only. Item 1 (matched partitions, bitwise) is the restore-path proof (7.2, G3/G7, 15.1). |
| N4 | the anchor misses short first messages | fixed. The anchor tail is full, through a side replay of a copy of the job's replay buffer: one 2.6 s replay per distinct system prefix. Condition A − pos0 ≥ 128. Measurement cited. ⌈A/C⌉C added to the snap set (5.2, 5.5). |
| N5 | shadow phase A | fixed. (a) Backfill is tagged, kept out of the chunk-tick stall metric, and budgeted per admission (`V41_KV_BACKFILL_MS`). (b) Reason codes on `kv.plan`; only `unexplained` is gated. (c) Phase B starts from phase A's store, an eviction storm used as a stress test (11.1). |
| N6 | deletions on the IO thread | fixed. Index removal on the scheduler thread, unlinks on the IO thread, the cap enforced on the index, the backlog reported. Stale namespaces are renamed to `trash/` at startup and unlinked in the background (2-10 s ESTIMATE per 37K files). Startup never waits on unlinks (4.4, 8.3, 9.5). |
| N7 | demotion mechanics | fixed. Pins block demotion and thinning; truncate before the header, with the startup scan checking section lengths; propagated touches are memory-only and do not increment hits, and are recomputed from the leaves at startup (4.5, 8.2, 8.3). |
| N8 | number nits | fixed. 1.2 uses one run (273 prompts, 32.78M tokens) and notes the second run's agreement; 2.3 now says ~250 MB / ~91K tokens at p50 573 ms; 6.6 M3 warms the page cache and verifies instead of staging 911 MB in a 90 MB ring; `kv.dedup_mismatch` cost stated (≤ ~9 chunks × ~5 ms per affected job) and sampled 1 in 8 after `on` (7.1). |

### Review round 3

| # | finding | resolution |
|---|---|---|
| N9 | demotion and thinning erase the anchor | fixed. Anchor tails carry a header flag `anchor` and are exempt from demotion and thinning; they leave only by score eviction, and the hits of every new conversation keep them young (4.5, 5.2, 8.2). New gate G8: (i) a job with and without the side replay ends bitwise equal (state + logits); (ii) the anchor tail is bitwise equal to the full tail of a cold prompt ending at A (7.3; M2 exit). |
| N10 | 9.4 evidence | fixed. There are no prefill-path failures in any log; the faults seen were step-level (one 09-28 decode-step failure fanned out by `abort_all`, outside the window; the "engine wedged" aborts are also 09-28). Conclusion unchanged. |
| N11 | epoch workflow, drift, scan scope | fixed. (a) The developer runs a reduced G7 (3 prompts × 1 cut, ~10-15 min) inside the fidelity gate they already run, against the previous builds' stored fixtures, so one harness run is enough. Unmeasured changes default to no bump and are marked in the release note; class-(a) fixes always bump. (b) `KEEP_BUILDS` = 3 purged older builds at startup. **Removed in round 4 (N16):** it counted deploys and its purge cascaded through shared prefixes. (c) The test scans knob-name literals (`V41_`, `DEEPSTRIX_`, `VIT_`) in the kernels, core, vision and server crates, which catches `VIT_GEMM` and the wrapper reads (4.4, 4.4.1). |
| N12 | suspect state machine | took the simplest option: never evict on device or RPC errors; `kv.suspect` is log-only, in memory, acted on by nothing (9.4, risk 10). |
| N13 | M0 classes and cost | fixed. (a) The switches apply per lane, so the classes are ≤ 16 / 17-32 / 33-128 / 129-256 / > 256 chunk rows (E3, 7.2). (b) Cost stated (~2.4M prefilled tokens for the null, ~2 h+); M0 split into M0a (~1 h: control, attribution, G1/G2/G4) and M0b (the null, longer windows, then G3/G7) (7.2, 13). |
| N14 | shadow bootstrap, aggregate, provenance | fixed. (a) `bootstrap` is defined from a writer frontier hashed independently of the chunk chain; a walk below a recorded frontier counts as `unexplained`. (b) The aggregate ≥ 96.8% gate is computed over non-bootstrap admissions and reported over the last 24 h, next to the per-admission check. (c) Backfilled chunks carry provenance `backfill-v6` and build id `v6` (4.5, 11.1). |
| N15 | which thread mutates files | fixed. Every mutation of an existing file (hits, mtime, truncate, header) runs on the IO thread in order; the scheduler keeps its view in memory and sends messages (4.5, 8.3). |

### Review round 4

| # | finding | resolution |
|---|---|---|
| N16 | `KEEP_BUILDS` counted deploys and cascaded through shared prefixes | **removed**, with all deletion-based retention by build. 10-01 alone shipped 5 production binaries (83200ec, c003112, aeda639, a4b2a0c, f76492f), none changing prefill numerics; by 10:33 the rule would have purged everything written before ~09:54, every shared system prefix and anchor included, and in shadow the backfill too. Drift is now bounded by `KV_NUMERICS_GEN`, a constant bumped in the same commit as any bit-changing prefill change and recorded in headers only. The release G7 runs against the **oldest generation present** in the store; beyond the null it is an explicit epoch-class decision. Routine retention is eviction only; a cascading purge happens only on explicit epoch or bug-fix decisions. Fixtures are keyed by generation, with a fallback when a generation skipped the check (4.4, 4.4.1, 4.5, 11.1, 13, 14). |
| N17 | the writer frontier was ambiguous | fixed. The frontier is the deepest tail whose whole chunk path had **completed** (written and renamed) when the tail completed. Before `unexplained`, the plan is tested for `dropped`, `evicted`, `thinned` and `demoted`. Lookup is one incremental blake3 pass over R, cloning at the sorted f values (~1 ms). The 10K LRU is memory-only and resets at restart; key-derivation changes across a deploy are G5's job (11.1). |
| N18 | most anchors are never reused | fixed by **removing work**: the anchor is written on **second sight** only, by the job whose walk matched another job's chunks through ⌊A/C⌋C but found no tail at A. Corpus: 13 distinct anchor prefixes among 16 first turns, 11 of them unique, so those pay nothing; one miss per recurring prefix. Snapping at A and ⌈A/C⌉C happens only in that job; the "reused many times" claim is withdrawn (0, 5.2, 5.5). |
| G8 nit | (ii) assumes a cold job | stated: (ii) is bitwise for pos0 = 0, the case G8 runs; a job restored at an unaligned pos0 matches only within the null (7.3). |

### Review round 5 (APPROVE)

| # | finding | resolution |
|---|---|---|
| R5-1 | env-only knob changes bypass `KV_NUMERICS_GEN` | fixed. The effective generation is the pair (`KV_NUMERICS_GEN`, knob hash): `kv.store gens=` and the oldest-present release check use pairs, and a new pair is logged at startup (`kv.gen_new`) like an unmeasured generation (4.4, 4.4.1). |
| R5-2 | frontier entries that selection would reject | fixed. Only frontier entries usable for this request count; entries 6.2 would reject (an encoder tail with L − f ≤ 128, or below T ≥ 64) are ignored, so a plan that legitimately stops below them is not `unexplained` (11.1). |
| R5-3 | wording | "per distinct system prefix" became "per recurring prefix" (8.2, 11.1). |

## Appendix B: M1 implementation (code review round 1)

M1 is on branch `worktree-kv-prefix-store`. Its deviations from revision 6, all accepted in code
review round 1, and the review's fixes that change what this document says:

| item | what M1 does | where |
|---|---|---|
| files | `keys.rs`, `scan.rs`, `knobs.rs` and `tests.rs` beside the four files of 9.7 | 9.7 |
| tail header | `origin` byte and `demoted` flag; 16 B header checksum skipping `hits`; `hits` clamped | 4.5 |
| section D | opaque in M1; M2 defines it: `n_raw_dec` decoder rows, ring present iff drafter id != 0 | 4.5 |
| pins | path pins; a plan pin is owned by its job | 6.5, 9.4 |
| arrival order | chunks may land before their parent; `broken_path` drops a tail over a hole | 9.4, 9.5 |
| demotion | newest = deepest T; anchors exempt and not counted in N = 2; thinning keeps the deepest | 8.2 |
| cap | evict above the cap down to cap − 1% | 8.3 |
| GC | `last_active` stamp per namespace; persisted byte total; `tmp/` renamed into the trash | 4.4, 9.5 |
| unlinks | a write cancels a queued unlink of its path; an unlink is skipped while the key has a write in flight | 8.3 |
| same-key writes | queued per key, oldest first (an encoder tail, then the full tail of the same key) | 9.4 |
| late completions | a job may end with writes in flight; they land unowned | 9.4 |
| transient errors | only data-attributable failures evict; IO errors log `kv.suspect` | 9.4, 9.5 |
| namespace | the file format version is a namespace input | 4.4 |
| root lock | `flock` on `kvstore-v1/.lock`; the store stays off if it is held | 4.4 |
| knob hash | set knobs only; the classification rule stated; run-time and box-2 knobs: M2 | 4.4 |

**Measured** (M1 bench, 10-01, this box): blake3 7.2 GB/s on one thread (911 MB in 126 ms);
unlink 16K/s at 2.76 MB (~2.3 s per 37K files); create + fsync 2.1-3.5 ms per file.

**M2 notes** (from code review round 2):

- **Events.** `ChunkStored` is reported only if the chunk is still indexed; a chunk that leaves
  the store for any reason (a failed job's chunk deleted at once, an eviction, a corrupt file)
  is reported as `ChunkDropped`, including the chunks `job_finished(failed)` deletes. A job that
  subscribed to a pending write re-enqueues on `ChunkDropped` (9.4).
- **Re-enqueue a dropped parent before queueing the job's next tail:** a tail over a hole is
  refused as `broken_path`, and that refusal releases the job's pins on its path.
- **Keep the restore plan's pin until the job's first own tail lands or the job ends,** not only
  until the restore completes: until then the plan's tail and path hold the job's prefix.
- **A job writes nothing after `job_finished`.** The store refuses such writes while it still
  remembers the job; once its writes drain it forgets the job, and a later write would pin for a
  job nobody releases. M2 either keeps that contract or refuses writes from finished job ids
  with a monotonic high-water mark of job ids.
- **Box-2 knob reporting** (4.4) stays M2 work.
