# DECODE_M8_PLAN.md — from 2 tok/s paged decode to 30 tok/s (2026-09-13, plan rev 0)

Scope: the single-stream decode path of the V4.1 server (`forward_token_paged`) on box 1 + box 2. Everything here is against the code as of this evening; line numbers are from the uncommitted tree. Other agents own `forward_prefill.rs`, `expert_pager.rs` internals, the remote-experts module and the vision crate — this plan names the interfaces it needs from them and does not touch their files' internals.

Inputs treated as facts: PLAN §6/§7a–§7f, decode designs A/B/C, M7_EXPERT_TIER, ARCH_SPEC §6, memory notes (`project_decode_dgpu_bound`, `project_m54_decode`, `project_decode_at_floor`, `feedback_two_box_maximize_busy`), the server log `logs/v41-server.log`, and the measured link (32 µs RTT, 1.1 GB/s, single segment ≤ 64 KB).

## 0. Where the ~500 ms/token goes today (measured + read off the code)

`logs/v41-server.log` `het.token.summary`: 496–1847 ms/token at pos 973–977 (dgpu/igpu busy = 0 because the paged path never harvests events). Decomposition from the code:

| item | per token | where |
|---|---|---|
| **expert misses, serial** | **≈ 50 misses × ~10.7 ms ≈ 535 ms** at the 79 % hit rate (240 picks × 21 %); the log's later runs show 50–56 % hit → ~110 misses → 1.2 s | `expert_pager.rs:418-519 ensure()`: single-threaded `read_expert_into` ×3 into a staging Vec (2.33 GB/s measured → 8 ms) + 3 synchronous `copy_from_host` (2.6 ms) per miss |
| per-layer host sync + readback | 40 × (`de.compute.synchronize()` + 24 B `copy_to_host`) ≈ 40 × 60 µs ≈ 2.5 ms | `forward_layer.rs:1668-1669` |
| per-layer remap rebuild + H2D | 40 × (`remap` reset 384 + `lru.position()` O(n) scans ×6 + `remap_dev.copy_from_host` 1.5 KB sync) ≈ 40 × 60 µs ≈ 2.5 ms | `expert_pager.rs:428-431, :228-233, :517` |
| standalone-graphs structure | `is_first_layer = is_last_layer = true` → a standalone `mhc_pre_attn` + a pure `ffn_combine` per layer instead of the fused `combined_ffn_pre_attn`; `qkv_chain` graph off under v41 (`:477`); host-scalar `pos` in rope/kv_append/rope_inv (`:533-541, :608-622, :1297-1305`) → ~10 extra un-graphed launches/layer ≈ 3–5 ms | `forward_layer.rs:322-323, :477` |
| Engram gather on the critical path | `rows_for` runs synchronously before `forward_token_paged`; 2 tables × 24 spawned threads × 4 KiB random reads ≈ 1 ms warm, 2–3 ms cold | `engine_worker.rs:58-83, :126-143`; `engram_table.rs:52-88` |
| device compute (chain + local MoE + head) | ≈ 22 + 23 + 1.4 ≈ 47 ms (single box, every pick on the iGPU) | §2 |

Two consequences the plan is ordered around: **the miss path is 90 % of the token today and the sync removal is worth ~8 ms**; and the pool split in `ExpertPager::new` (`:186-204`) currently gives decode's LRU only the slots above the prefill windows (log: "decode LRU gets slots 3072..3141" = **69 slots**), which is why the hit rate fell from the 79 % measured with one window to 50–56 %. Step 0 fixes the split before anything else is measured.

## 1. Per-layer host syncs on the paged path and how they go away

### 1.1 The three syncs per layer

1. `forward_layer.rs:1668` `de.compute.synchronize()` + `:1669 d_selected.copy_to_host` — the host must know the picks to page them.
2. `expert_pager.rs:517` `remap_dev.copy_from_host` (synchronous H2D on the iGPU) after `ensure()` — the kernel's `remap` argument is rebuilt on the host every layer.
3. Inside `ensure()`, every miss does three synchronous `copy_from_host` from a host staging buffer (`:497-511`) — on the iGPU compute stream's device, serialised against the MoE.

Plus the per-token `sample_next` sync (`engine.rs:1229`, unavoidable, ~40 µs) and the final `dgpu.compute.synchronize()` (`engine.rs:918`).

### 1.2 Decision: device-side slot table (a) for the hot path, device-signalled stall (b) for misses; per-layer readback dead

Justification from the hit rates: at 79 % (one box, 3141 slots) 21 % of picks miss, so misses are not rare and any "lag-one / predicted picks" scheme still has to stall correctly on the 30 % it cannot predict (hidden-state probe recall 70 % at k=1, id→id 0.15, PLAN §7d) — prediction is at best a low-priority prefetch hint. At ~70 % combined residency (box 1 + box 2), misses are ~1–3.5/token (§7d trace: 3.6 at 66 %, 2.5 compulsory at ≥ 75 % on a 1006-token trace; steady-state on a warm server lower), i.e. ≤ 1.5 % of picks: the hot path must never touch the host, and the miss path may be slow-but-exact. Both point the same way:

**(a) Device-resident slot table.** `slot_table_dev: DeviceBuffer<i32>[N_LAYER × N_EXPERT]` on the iGPU (61 KB), row `l` = exactly today's `remap` for layer `l` in the kernel's existing encoding (`mxfp4_pair_matvec.hip:183-232`: `>= 0` dGPU dense slot, `-(slot)-1` iGPU pool slot). Two new sentinels: `REMOTE = i32::MIN+1` (owned by box 2 → the iGPU zero-fills that pick exactly like `!ours`, one-line kernel change at `:218`) and `MISS = i32::MIN` (must never be observed by a kernel; if it is, the kernel stores a poison word the host checks per token → fail loudly). The `routed_moe` graph for layer `l` bakes the row pointer `slot_table_dev + l*N_EXPERT` as its `remap` argument, so the graph is captured once and never re-captured; `ensure()`'s per-layer remap rebuild and H2D disappear. Table writes happen only when residency changes (miss fill, boundary eviction), as 4-byte `copy_from_host_async` on a dedicated iGPU side stream, ordered before the consumer by the value-signal below.

**(b) Misses as a device-signalled stall.** After `router_topk` (`forward_layer.rs:1476-1486`) a tiny kernel `check_resident(layer)` reads `slot_table_dev[l][sel_i]` (run it on the iGPU side in the pre-issued lane so no peer read is needed) and (i) writes `sel_log[l][0..6]` (device, for the per-token recency readback), (ii) counts sentinels; if `n_miss == 0` it stores `miss_done[l] = token_seq` (pinned, system-scope fence) itself, else it stores `miss_req[l] = token_seq`. The iGPU MoE lane is pre-enqueued at token start exactly like `issue_igpu_moe` (`forward_layer.rs:206-262`, `DECODE_PREISSUE` infrastructure: `moe_signal`, `token_seq`, `wait_value32_gte`) with one extra `wait_value32_gte(miss_done[l], token_seq)` before the graph. A host **miss-service thread** busy-polls `miss_req[0..40]` (pinned memory, no HIP calls in the poll), reads `sel_log[l]` from the pinned mirror, fills the missing experts (§1.4), updates the table row, and writes `miss_done[l] = token_seq`. On a hit-only layer nothing on the host runs. `hipStreamWaitEvent` is not used anywhere on the pre-issued lane (the M54 trap: event waits snapshot at call time).

**One readback per token, off the path:** at the end of the head, `sel_log` (40 × 6 i32 = 960 B) is copied D2H asynchronously into a pinned buffer on `de.xfer` behind an event; the host consumes it at the *next* token boundary to update LRU recency. Recency one token late is harmless for an LRU.

### 1.3 What is exactly wrong if a miss is served late — and the rule that makes "late" safe

A late miss is only ever a stall (the value-wait holds the iGPU lane; the dGPU's `ffn_combine.wait` on `moe_arrived` holds the dGPU) — provided three orderings hold. Violating any of them produces silent garbage, which is the failure class M7 hit twice (window collision; remap allocated on the wrong device):

1. **Bytes before mapping.** The table entry for the new (layer, expert) is written *after* the expert's bytes are fully in the slot and visible to the iGPU (the fill goes through the side stream; the entry write is stream-ordered after it; then the fence; then `miss_done`). Otherwise the kernel reads a half-written slot.
2. **Unmap before overwrite.** The victim's entry is set to `MISS` *before* its slot is touched. Otherwise a layer that picks the evicted expert computes with a torn/foreign expert.
3. **No eviction while a kernel may read the slot.** The check kernel of layer `l` and the MoE of layer `l` are separate kernels; a host eviction between them would invalidate a "resident" verdict. Rule: **eviction only at token boundaries** (after `forward_head`, before the next token's layer 0, when no MoE lane is in flight), from a pre-allocated spare-slot budget; misses during the token fill *free* spare slots only. If the spare budget (say 64 slots ≈ 1.2 GB) is exhausted mid-token, the miss thread evicts anyway but only slots whose `pin[slot] != token_seq` — the check kernel pins its picks (`pin[slot] = token_seq`) and the host re-checks the pin after writing `MISS` (Dekker order). With DSpark, "token" = "verify step".

Tripwires: the `MISS`-observed poison word; a `V41_PAGER_DBG` mode that re-derives the table from `slot_key` each boundary and diffs it.

### 1.4 The fill itself

`ensure()` stages through host Vecs and copies (10.7 ms/miss). Replace with the phase-B path already measured in `COLD_EXPERT_CACHING.md` (pread straight into the iGPU slot via `as_host_slice_mut`, 64 threads: **4.82 ms, 3.90 GB/s**) — the pool is host-addressable on Strix. Box 2 serves its own misses from its plaintext NVMe (~3 ms). The miss thread owns a persistent 64-thread reader pool (no per-miss spawns). Expected: 10.7 → 4.8 ms per miss on box 1. NOTE (2026-09-13): the prefill pager agent's measurement found direct-into-device 17 % SLOWER than host-staging-then-copy for its batched reads (`DEEPSTRIX_FAST_EXPERT_LOAD` lesson); re-measure both for the single-miss case before committing.

### 1.5 Per-token host work that remains

`sample_next` readback (40 µs); `embed_lookup` + 80 KB residual H2D (async, pinned); two `write_value32` (pos, kv slot; `engine.rs:487-497`); the Engram hash + 48 preads (§2.3); the `sel_log` consume + boundary eviction; miss service (rare). Nothing else touches the host.

## 2. Graph structure for the 40-layer V4.1 step with the pager

### 2.1 Not one mega-graph

V4-Flash's 29.4 tok/s is per-stage graphs (`mhc_pre_attn`, `qkv_chain`, `output_proj_post_rope`, `mhc_pre_ffn`, `shared_expert`, `dgpu_hot_moe`, `routed_moe`, `combined_ffn_pre_attn`) with host-enqueued event/value waits between them; `project_decode_at_floor` measured host enqueue (4.4 ms) fully overlapped and the mega-graph / capture fork-join variants as regressions (`project_m54_decode`: ~10 µs per captured event at replay). The goal is **zero host syncs**, not one graph. So: make the paged path take `forward_layer` (combined graphs, `next_dlw`) instead of `forward_layer_standalone_graphs_paged` (`engine.rs:659-670`), with the iGPU MoE lane pre-issued.

### 2.2 What has to be parameterised (device scalars or graph variants)

| host scalar today | where | fix |
|---|---|---|
| `pos` in rope forward (q, kv, indexer q), rope inverse | `forward_layer.rs:533-541, :575-583, :1041-1049, :1297-1305` | `launch_forward_pdev` / inverse pdev exist (`rope.rs:94-119`) reading `dgpu_scratch.pos_dev` — already used by V4-Flash's `qkv_chain` |
| KV append slot `raw_off + n_raw` | `:608-622` | `kv_append.launch_slotdev` (`kv_cache_append.rs:92`) reading `kv_slot_dev`; needs the V4.1 window quant (`fp4kv.launch_fp8_window`, `:589`) fused into a v41 twin of `kv_post_fused` — the M7 item noted at `:473-476` |
| attention window pointer `kv_win = kv_cache.slice_view(raw_off*hd, n_raw*hd)` and `n_raw` | `:977-980, :1216-1234, :1252-1275` | kernels take `raw_off`/`n_raw` from device scalars (`launch_score_b1_htiled_wmma` already has a `raw_off` argument, passed 0); base pointer = `kv_cache` (stable). The wrap copy (`:634-649`, once per ~1024 tokens) stays host-driven outside the graphs |
| compressor boundary `(pos+1) % ratio == 0`, `row`, `comp_pos` | `:658, :676-677, :738` | ratio-2 layers (2, 8, 14): two graph variants keyed `(stage, layer, parity)` (`GraphCache` key gains a variant byte); ratio-1 layer 20 fires every token; comp row index `n_comp` → device counter + device-index `comp_kv_append` (the gap already listed for MTP verify in `project_decode_dgpu_bound`) |
| `n_index_comp`, dense-vs-sparse switch at 512 rows | `:1013-1020` | device counter for the score/top-k row count with grids pinned to `ATTN_MIXED_MAX_KEYS`; the dense→sparse switch is a one-time graph-variant flip at pos 512 |
| Engram rows | `stage_engram_rows` `:63-71` (sync H2D) | pinned host buffer + `copy_from_host_async` on `de.compute` before the layer-1/14 graph launch (the buffer pointer is baked) |
| slot table row, `hot_cap`, `token_seq` waits | `:1720-1736` | table row pointer per layer (constant); value-waits are enqueued by the host between graph launches, never captured |

### 2.3 Engram gather: where it overlaps

Rows depend only on token ids (ARCH_SPEC §1.7). The token is known at `sample_next`; layer 1 needs its rows ~0.9 ms later (layer 0), layer 14 ~13 ms later. Issue the 48 preads from a **persistent** 24-thread pool the moment the id is read back (today `gather_position` spawns 24 OS threads per call, `REVIEW` item); warm page cache ≈ 0.15 ms, cold NVMe at QD24 ≈ 0.3–0.5 ms wall — under layer 0. The host enqueues layer 1 only after the gather future resolves; since host enqueue runs ahead of the device, a 0.5 ms gather never surfaces on the device timeline. Under DSpark the K+1 candidates are known at step start → all their rows are gathered before layer 0 (design C) and the term vanishes.

### 2.4 Expected per-token time (V4-Flash graph numbers scaled to V4.1 bytes)

V4-Flash: 34 ms / 43 layers = 0.79 ms/layer = **0.50 ms dGPU chain** (launch-bound, ~2× its 117 MB byte time) + ~0.29 ms MoE phase. V4.1 (ARCH_SPEC §6): attention + hc + gate ≈ 134 MB/layer → chain **0.55 ms**; Engram wkv 2 × 167 MB Q8 → +0.56 ms/token; head 662 MB Q8 → 1.1–1.4 ms; local MoE 6 picks × 18.8 MB at the measured 214 GB/s = 0.53 ms + q8k + 20 KB push ≈ **0.58 ms**.

| configuration | per layer | ms/token | tok/s |
|---|---|---|---|
| single box, all picks local, no misses | 0.55 + 0.58 | 45.2 + 0.56 + 1.4 + 0.5 = **47.7** | 21 |
| two boxes (§4), no misses | 0.55 + 0.36 | 36.4 + 2.5 = **38.9** | 25.7 |
| + misses, m/token exposed at 4.4 ms (4.8 − the 0.36 branch already hidden) | | + 4.4 m | m=1: 23; m=3.5: 19 |

## 3. DSpark integration

### 3.1 What the drafter is (reference `model.py:1100-1160`, harness `dspark_accept.py`)

`forward_embed`: `main_x = main_norm(main_proj(cat(mean-copies of the attention INPUT of layers 37, 38, 39)))` (15360→5120, 79 MB) and `x = embed([sampled, noise×4])` expanded to 4 copies. Three `DSparkBlock`s (full `Block` widths, SWA-only attention whose window KV is `kv_norm(wkv(main_x))` of past positions, block-causal among the 5 drafts, 128 experts top-3 + shared). `forward_head`: `hc_pre` with the carried pre-mix, norm, tied head at full logits for 5 positions; the markov bias is sequential (`markov_head(output_ids[:, i])` → bias, sample, next), rank 256; confidence head on `cat(h, markov_emb)`. So drafts 2–5 are one parallel pass; only the markov chain (5 × 33 MB matvec) is serial. Prefill only seeds the window rings.

Not presented by `V41HfWeights` yet (`hf_v41.rs:23`): `mtp.{0,1,2}.*` — attention/hc/norms/gate/shared/experts (128 × 3 × 18.8 MB = 7.2 GB MXFP4), `main_proj`, `main_norm`, `markov_head.{embed,head}`, `confidence_head`.

### 3.2 Verify batch: a small-B decode path, not the prefill path

Two hard reasons the batched prefill path (`forward_layer_batch_v2`, `forward_prefill.rs:959`) cannot be the verify path: it pages **whole layers** through `ensure_layer_dense` (`:3016`, 7.2 GB per layer per call), and it was measured to recover nothing at small B on V4-Flash (`project_decode_dgpu_bound`: "verify forward MUST be built from DECODE primitives"). So `forward_verify_batch(B ≤ 6)` = the decode chain with a row dimension:

- projections: `matvec_batched`/q8 GEMM twins at B ≤ 6 (weights read once per batch; the prefill kernels' tiled-B rule from `PREFILL_BALANCE` applies: never `grid.z = batch`);
- attention: `launch_score_batched_htiled_wmma_f16s` (`attention.rs:656`) + batched softmax/wsum over the window + selected compressed rows, per-row `n_raw_offset` (rows attend causally within the batch — same as prefill's `n_raw_offset_per`); `bench_verify_attention` showed C_attn(3) ≈ 0.9;
- **MoE union kernels (the one new kernel that matters):** today's `*_batch_hetsplit` (`mxfp4_pair_matvec.hip:189`, `mxfp4.rs:75`) has `blockIdx.y = pick slot` and no token dimension, so B launches would re-read shared experts B times. New `gate_up_union` / `down_union`: a tiny device kernel builds the per-layer union of the B×6 picks (≤ 36 entries: distinct expert → member list), fixed grid = 36 max slots with empty slots exiting early; each WG processes one (expert, row tile) and loops over its ≤ B member tokens, so **expert bytes = distinct experts**, not B×6. Same `remap`/slot-table semantics; `ew[t]` per member. Graph-capturable (fixed grid);
- head at B (662 MB once, ~1.2 ms), then `spec_accept`.

Expert bytes per verify step: distinct experts ≈ 15/layer at B=3, ≈ 26 at B=6 (V4-Flash measured ~16 for 3 positions; V4.1's 384 experts are flatter).

### 3.3 Acceptance and rollback (all exact)

`spec_accept` kernel (new, next to `sampler.rs:65-187`): inputs `p = softmax(main_logits[B, V]/T)`, `q_i = softmax(draft_logits_i + markov_bias_i)` kept on device from the drafter's head (5 × 129280 f32 = 2.6 MB), `K+1` uniforms from the host `SamplerRng` (seeded → runs stay reproducible). For `i in 0..K`: accept `d_i` if `u_i < min(1, p_i(d_i)/q_i(d_i))`, else sample from `norm(max(0, p_i − q_i))` and stop; if all accepted, sample the bonus from `p_K`. Output `n_acc` + the emitted token → one 8-byte readback per step. `T = 0` reduces to greedy match (the metric `dspark_accept.py` reports).

Rollback = counters, because every store is monotonic:

| state | rollback |
|---|---|
| window KV `n_raw`/`raw_off` (`state.rs:237-257`; append at `raw_off + n_raw`) | restore pre-step values, then advance by `n_acc + 1`; rows past that are dead. Do the wrap copy (`forward_layer.rs:634-649`) *before* a step if `raw_off + n_raw + B ≥ KV_CACHE_ROWS` (`SWA_WINDOW + B_MAX`) |
| compressed store `n_comp` (ratio-1 layer 20; ratio-2 layers 2, 8, 14) and index-key cache `n_comp` | restore to pre-step + accepted boundaries; rows past are dead |
| ratio-2 parked partial group `state_kv/state_score` (`compressor_state_write` row = `pos_mod`, `:700-712`) | the verify path writes per-position `kv_cur/sc_cur` into a `[B, 512]` scratch (as prefill does) and commits only the accepted prefix's last partial row into the 2-row state after acceptance (one 4 KB D2D). Never write the state rows during the pass |
| candidate pool (layer 20 → 24/28/32/36) | recomputed per position inside the pass; nothing persists |
| Engram `compressed` sequence | `rows_for` already `truncate(pos)`s (`engine_worker.rs:67`); rollback-safe |
| DSpark window rings (3 layers × 128) | same monotonic append + `raw_off` as the backbone; entries `kv_norm(wkv(main_x_t))` for all B candidates are produced during layers 37–39 of the verify pass (design C) and committed for the accepted prefix. Prefill's last chunk must emit `main_x` for its last 128 rows to seed the rings (new output of `forward_layer_pre_moe_v2` at layers 37–39); snapshots must carry the rings or re-seed |
| mHC carry `hc_pre_carry` | per-step transient (`[B, 4]`), reset at layer 0 |

### 3.4 Expected tok/s at draft-1 acceptance 0.6–0.7

Conditional acceptance beyond position 1 falls steeply (teacher-forced 0.44 → 0.26 → 0.14, i.e. ×0.6, ×0.55): at p1 = 0.65 take p2 ≈ 0.40, p3 ≈ 0.25, p4 ≈ 0.15 → E[tokens/step] = 1.65 (K=1), 1.91 (K=2), 1.98 (K=3), 2.0 (K=5); at p1 = 0.70: 1.70 / 2.02 / 2.11 / 2.13. Drafter cost (attention/hc/head on the dGPU, its 7.2 GB of experts pinned in the box-1 pool): main_proj 0.13 + 3 × (0.65 chain@b=5 + ~12 distinct picks 1.06 + hops 0.08) + head@b=5 1.2 + markov 0.35 + misc ≈ **6–7 ms/step**, serial with the verify (its input is the sampled token). Two boxes, no misses, chain at b: 0.55 + 0.03(b−1):

| schedule | step ms | tokens/step (p1 0.65 / 0.70) | ms/token | tok/s |
|---|---|---|---|---|
| batched K=2 (B=3): 40 × (0.61 + 0.72) + head 1.5 + drafter 6.5 + boundary 0.5 | 61.7 | 1.91 / 2.02 | 32.3 / 30.5 | **31 / 33** |
| wave 2+2 (K=3, two B=2 sub-batches trailing by one layer): dGPU 40 × 2 × 0.58 + 8.5 (experts + RTT hidden) | 54.9 | 1.98 / 2.11 | 27.7 / 26.0 | **36 / 38** |
| + misses: m distinct misses per step exposed at 4.4 ms | +4.4 m | | | m=2: 30–33 (wave); m=7: 21–23 |

### 3.5 How the verify batch lights the tracks (7b rule 4)

Single stream: dGPU idle during the expert phase (~40 % of a layer), both iGPUs idle during the chain. Batched verify amortises the launch-bound chain over B rows (the chain is ~flat in b) but does not overlap steps. The **layer wavefront** (design B) does: sub-batch A = [bonus, d1], B = [d2, d3]; B trails A by one layer; A's experts (box-1 iGPU ∥ link → box-2 ∥ dGPU shared/hot) run while the dGPU does attention(B, l), then attention(A, l+1) while B's experts run. Legal because B depends on A only through the window/compressed KV that A's *attention* wrote at the same layer (write-before-read is by stream order on the dGPU); same logits as batched verify. In code it is the two-lane structure `forward_prompt_batch_v2_pipelined` already uses: a second `DgpuScratch/IgpuScratch` lane set and `sync_events_t1` (`engine.rs:331-335`), one KV edge per layer (`attn(B,l)` waits on `kv_appended(A,l)`). Every RTT sits under the other lane's attention; the link always has one message each way; box 2 always has a request queued.

## 4. The remote branch: join point and client API

**Where the partial is added.** In `forward_layer_impl_inner`, the combine graph (`forward_layer.rs:1866-1878` last layer, `:1886-1893` combined) does `vec_add(ffn_moe_recv, ffn_shared)` [+ `vec_add(ffn_moe_recv, ffn_moe_dgpu)` when `hot_experts` is Some], then `hc_post` → `residual_next`. The box-2 partial lands in a new dGPU buffer `dgpu_scratch.ffn_moe_remote[N_EMBD]` (20 KB, pointer-stable so the graph bakes it) and the combine gets a third `vec_add`. The wait: next to `de.compute.wait_event(&sev.moe_arrived)` (`:1820-1823`) add `de.compute.wait_value32_gte(remote_done[layer], token_seq)`. The receiver thread writes the reply into `ffn_moe_remote` with `copy_from_host_async` on a dedicated dGPU stream `de.remote_rx` and then `write_value32(remote_done[layer], seq)` on that stream (write ordered after the copy). Exactness: fp32 partials, summed in fp32 in the same order every token (shared + dGPU + iGPU + remote).

**Where the request leaves.** Right after `sev.selected_ready.record(&de.compute)` (`:1517`), on `de.xfer` (already FIFO after the `ffn_input_norm` push): D2H of the Q8_K-quantised activation (`moe_xq`, 5.4 KB — quantise once on the dGPU with `de.q8k.launch` as the hot path does at `:1588`; bit-identical to what the iGPU consumes) + `sel_ew_pack` (48 B) into a pinned ring `remote_tx[layer]`, then `write_value32(tx_ready[layer], seq)`. The sender thread busy-polls `tx_ready` and writes one message per layer (TCP_NODELAY, busy-poll recipe from SECOND_BOX). Box 2 owns the directory `owner[layer][expert]` and computes only its picks, replying with a 20 KB f32 partial or a 4-byte "none of mine" so the wait always resolves. For verify batches: B × 5.4 KB out, B × 20 KB back (B=6 → 120 KB, split into ≤ 64 KB segments per the measured single-segment cliff).

**Client API the engine needs (what the remote-experts agent exposes as "send at router time, join before combine"):**

```rust
pub trait RemoteExperts: Send + Sync {
    fn begin_step(&self, seq: u32, rows: u32);                       // once per token / verify step
    fn tx_ready_word(&self, layer: u32) -> *mut u32;                 // engine's de.xfer writes seq here after the D2H
    fn tx_buf(&self, layer: u32) -> &PinnedBuffer<u8>;               // [rows × (5.4 KB q8k + 48 B sel/ew)]
    fn partial_buf(&self, layer: u32) -> &DeviceBuffer<f32>;         // dGPU-resident [rows × N_EMBD], written before the word
    fn partial_ready_word(&self, layer: u32) -> *mut u32;            // engine's de.compute waits GTE seq
    fn owner_table(&self) -> &[u8];                                  // [N_LAYER × N_EXPERT]: 0 local, 1 box-2 (feeds REMOTE sentinels in the slot table)
    fn residency_events(&self) -> Receiver<(u32 /*layer*/, u32 /*expert*/, bool /*resident*/)>;  // box-2 LRU changes → directory
}
```

Anything the engine's streams touch is a pinned word or a device buffer; the transport thread never calls into HIP on the compute streams. Balance rule (7b.1): the hottest experts live on box 1 (its branch carries no RTT), the tail on box 2, so box 2's pick share is *below* its byte share; the directory is what the slot table's `REMOTE` sentinels are built from.

## 5. Ordered steps, gates, and the ms/token budget

Gates used throughout: **P** = parity harness `V41_PAGER=1 V41_LAYER_MAJOR=1 V41_LAYER=39` in decode mode (per-token residual + HEAD gate, argmax 11111); **S** = server "capital of France → Paris", 17×23 → 391, the 612-token prompt; **A/B** = back-to-back tok/s on the same prompt at the same pool size, plus per-track busy fraction from the perfetto export (`analyze_pftrace_gaps.py`; not in this tree — port from the V4-Flash working copy, or read `TokenTiming` busy/idle) which is the §7b acceptance number.

| # | step | files | gate | ms/token after (single stream unless noted) |
|---|---|---|---|---|
| 0 | Pool split: decode LRU gets the pool (`V41_PAGER_WINDOWS=1` or a decode-vs-prefill budget in `ExpertPager::new`), pool 70–80 GB, log misses/token | `expert_pager.rs:186-204`, `engine_worker.rs:858` | hit ≥ 79 % on the S prompts; P unchanged | ~550 (miss-bound, 50 × 10.7) |
| 1 | Miss fill = direct pread into the slot, 64-thread persistent pool, no staging copy | `expert_pager.rs:486-511` (pager agent) | P (bytes identical); A/B | ~300 (50 × 4.8 + 55) → 3.4 tok/s |
| 2 | Device slot table + check/pin kernel + miss-service thread + value-wait iGPU lane + boundary eviction + `sel_log` readback; paged path → `forward_layer` with combined graphs | `forward_layer.rs:1658-1742, :206-262, :1820`, `engine.rs:510-535, :659-670`, new `het/slot_table.rs` | P; S; host syncs/token = 1 (the sampler); A/B | −8 ms of sync/remap/host gaps; still miss-bound until residency |
| 3 | Graph-friendly V4.1 chain: pdev rope, fused v41 window quant + slotdev append, device `raw_off/n_raw/n_comp`, parity graph variants, Engram async H2D + persistent gather pool | `forward_layer.rs:473-541, :608-622, :977-980, :1297`, `rope.rs`, `kv_cache_append.rs`, `engram_table.rs` | P; A/B on a **100 %-resident synthetic** (`V41_PAGER_SLOTS` ≥ the prompt's union) to isolate the chain from misses | 47.7 no-miss single box → 21 tok/s |
| 4 | Two-box branch: `ffn_moe_remote`, tx/rx words, `REMOTE` sentinels from the directory, disjoint LRUs, box-2 miss service | `forward_layer.rs:1517, :1820, :1866-1893`, `scratch.rs`, remote-experts client | P with `V41_REMOTE_ALL=1` (every pick remote) vs resident; S; busy fraction dGPU/iGPU1/iGPU2/link; **measure steady-state misses/token on real traffic** | 38.9 + 4.4 m → 26 (m=0), 23 (m=1), 19 (m=3.5) |
| 5 | DSpark: present `mtp.*`; drafter forward; `forward_verify_batch` (B ≤ 6 decode primitives, MoE union kernels); `spec_accept`; rollback; ring seeding from prefill | `hf_v41.rs`, new `het/forward_verify.rs`, `het/dspark.rs`, `kernels/moe_union.hip`, `sampler.rs`, `engine_worker.rs:2156-2361` | drafter ids + confidence == `dspark_accept.py` at T=0 on the gen2 transcript; B=6 teacher-forced verify vs 6 sequential decodes: argmax equal, residual Δ within the 5e-2 vector-scale drift budget; live acceptance ≈ 0.6–0.7; S; A/B | batched K=2: 32.3 + 2.3 m per token → **31** (m=0) |
| 6 | Wavefront: two sub-batch lanes trailing by one layer, KV edge per layer | `forward_verify.rs` (lane structure from `forward_prefill.rs:297`), `engine.rs:331-335` | logits identical to batched verify (same kernels per row); busy fractions ≥ 0.85 dGPU / ≥ 0.7 iGPUs / ≥ 0.8 link | 27.7 + 2.2 m → **36** (m=0), **31** (m=2) |
| 7 | If m is too high: box-2 plaintext tier (3 ms), predictor prefetch at low priority into page cache (§7d: 2.5–4 ms/token free), adaptive K from the confidence head, DSpark experts IQ2/IQ3 on the dGPU (drafter 6.5 → ~3.5 ms) | | A/B | +2–4 tok/s each |

**Is 30 reachable?** Yes on the arithmetic of steps 2–6 *if the steady-state exposed miss count per generated token is ≤ ~2* (wave 2+2 at p1 ≥ 0.65 gives 36 no-miss; each exposed miss per token costs 4.4 ms ≈ 4 tok/s at that point). The **risk step is 4** — the miss term. What we know: the trace says 3.6 misses/token at 66 % on a *cold* 1006-token trace and 2.5 compulsory at ≥ 75 %; the server's own short sessions hit 79 % at only 20 % residency, so real traffic is far more concentrated than flat routing and the warm steady state should sit well below the trace. Step 4's gate is therefore a measurement, and step 7 is the fallback ladder. Second risk: the 0.55 ms launch-bound chain is taken as-is (fusion, design A, is outside this plan; it is worth +4–6 tok/s later). Third: the MoE union kernel and the verify chain at b ≤ 6 (new kernels; the batched attention exists).

## 6. Budget table (ms per generated token)

| item | today | S1 | S2+S3 (1 box) | S4 (2 box) | S5 K=2 | S6 wave 2+2 |
|---|---|---|---|---|---|---|
| dGPU chain (40 × 0.55; ×b-scaled under verify) | ~25 | ~25 | 22 | 22 | 24.4/1.91 = 12.8 | 46.4/1.98 = 23.4 |
| expert phase (local ∥ remote ∥ shared) | 23 | 23 | 23 | 14.5 | 28.8/1.91 = 15.1 | hidden under the other lane (≈ 0) |
| head + sampler + embed + Engram | 3 | 3 | 2.5 | 2.5 | 1.9/1.91 = 1.0 | 1.0 |
| drafter | – | – | – | – | 6.5/1.91 = 3.4 | 6.5/1.98 = 3.3 |
| host syncs / remap / gaps | ~8 | ~8 | 0.3 | 0.3 | 0.2 | 0.2 |
| misses (m exposed/token × 4.4; today 50 × 10.7) | 535 | 240 | 4.4 m | 4.4 m | 2.3 m | 2.2 m |
| **total, m = 0** | ~590 | ~300 | **47.7** | **38.9** | **32.3** | **27.7** |
| **tok/s (m = 0 / 1 / 3.5)** | 1.7 | 3.4 | 21 / 19 / 16 | 26 / 23 / 19 | 31 / 29 / 25 | **36 / 33 / 28** |

## Critical files for implementation
- `crates/v4flash-kernels/src/het/forward_layer.rs` (paged MoE block `:1658-1742`, router/xfer `:1436-1560`, combine `:1820-1893`, `issue_igpu_moe` `:206-262`, host-scalar launches `:533-622, :977-980, :1297`)
- `crates/v4flash-kernels/src/het/engine.rs` (`forward_token_impl` `:447-1014`, pager branch `:659-670`, pre-issue `:510-535`, `sample_next` `:1184-1233`, sync structs `:283-387`)
- `crates/v4flash-kernels/src/het/expert_pager.rs` (`ensure` `:418-519`, pool/window sizing `:140-204`, LRU `:228-233`)
- `crates/v4flash-kernels/kernels/mxfp4_pair_matvec.hip` (hetsplit remap semantics `:183-232`; base for the slot-table sentinels and the MoE union kernel)
- `crates/deepstrix-server/src/engine_worker.rs` (decode loop `:2156-2361`, `forward_one!` `:124-156`, Engram `rows_for` `:58-83`, pager construction `:858-865`)

## Review rev 0 (2026-09-13)

Reviewer: senior systems architect (independent of the planning agent). Method: every number and code claim in §0–§2 checked against the uncommitted tree (line numbers below are from it), `logs/*.log`, PLAN §6/§7a–§7f, the three decode designs, M7_EXPERT_TIER, ARCH_SPEC §1/§6, COLD_EXPERT_CACHING, `model.py`, `dspark_accept.py`, and the memory notes the task lists. No GPU jobs were run; no source was edited.

### R1. Verified / refuted claims (§0–§2)

| # | claim | verdict | evidence |
|---|---|---|---|
| 1 | miss cost ≈ 10.7 ms: "single-threaded `read_expert_into` ×3 into a staging Vec (2.33 GB/s → 8 ms) + 3 sync `copy_from_host` (2.6 ms)" | **total supported, decomposition wrong** | 10.70 ms is COLD_EXPERT_CACHING's "naive pread into host + copy" row. The three staged reads + three sync copies are real (`expert_pager.rs:490-492`, `:497-511`). But each `read_expert_into` → `read_expert_raw` (`hf_v41.rs:584-621`) allocates two fresh Vecs (6.3 MB weight + scale), does two cached preads, then a scalar nibble repack loop over `out × nb × 8` (2304 × 160 × 8 ≈ 2.9 M iterations per role). Per miss that is 6 allocations + 6 preads + ~9 M scalar iterations of CPU repack before the 2.6 ms copy. The "8 ms read" bucket hides several ms of CPU that no reader-thread count removes. |
| 2 | "50 misses × 10.7 ≈ 535 ms at the 79 % hit rate; later runs 50–56 % → ~110 misses → 1.2 s" | **refuted as stated; the conclusion is stronger than claimed** | No log under `logs/` contains a 79 % figure. The `hit_rate` counter mixes prefill and decode: `ensure_layer_dense` adds 384 requests per layer per chunk (`expert_pager.rs:267`, misses `:369`) into the same counters `ensure` uses (`:438`, `:445`). The logs' request totals are prefill multiples: meas-A/B `61680 = 4 × 15360 + 240` (four layer sweeps + ONE decode token), meas-D `123360 = 2 × 61680`. With one window (meas-A/B, decode LRU = slots 384..3141 = 2757) the counter reads **0.498**; with eight windows (meas-D, decode LRU = 69 slots) it reads **0.542–0.564** — the opposite direction from the plan's story, because the counter is measuring prefill's dense-window residency. Decode's real miss count is only visible in token time: `v41-server.log` (pool 45 GB = 2570 slots → 6 windows, 5 dense → decode LRU = slots 1920..2570 = 650) shows **0.50–2.85 s/token** = 46–266 misses at 10.7 ms. Today's token is ~4× more miss-bound than §0 says (the "miss path is 90 %" is really 95–98 %). |
| 3 | "decode LRU gets slots 3072..3141 = 69 slots" (log) | **supported, wrong log cited, and worse than stated** | The line is `logs/meas-D.log:23` (run with `V41_PAGER_WINDOWS=8`), not `v41-server.log`. Code: `expert_pager.rs:190-195` defaults `dense_windows = total_windows − 1`, and `:453-456` sets `lru_lo = dense_windows × 384`, so by default decode's LRU gets **`384 + (n_slots mod 384)` slots — between 384 and 767 (7–14 GB) — regardless of `V41_PAGER_POOL_GB`**. Pool size buys decode nothing today. `V41_PAGER_WINDOWS=1` (`:192-193`) gives decode `n_slots − 384` but drops prefill's effective residency to 0 (M7: 14.75 → 7.79 tok/s on the 1825-token prompt). |
| 4 | per-layer sync + readback ≈ 40 × 60 µs; remap rebuild + H2D ≈ 40 × 60 µs | **code supported, timings unmeasured** | `forward_layer.rs:1668` `de.compute.synchronize()`, `:1669` `d_selected.copy_to_host`; `expert_pager.rs:429-431` remap reset, `:228-233` `lru.position()` O(n) per touch (3000-entry VecDeque × 6 touches × 40 layers ≈ 720 K element scans/token), `:517` sync `remap_dev.copy_from_host`. ~2.5 ms each is plausible; nobody has measured it because the paged path harvests no device events (every `het.token.summary` in every log has `dgpu_busy_us=0 igpu_busy_us=0`). |
| 5 | standalone-graphs structure; `qkv_chain` off under v41; host-scalar `pos` | **supported** | `forward_layer.rs:322-323` (`standalone_graphs` forces both flags; Engram layers 1/14 and the layers before them force them too, so even the combined path keeps 4 standalone transitions/token), `:477` (`&& !cfg!(feature = "v41")`), rope `:533-541`, kv_append `:608-622`, rope⁻¹ `:1297-1305`, compressor `:658, :676-677, :700-712`. |
| 6 | Engram gather on the critical path; 24 spawned threads per table | **supported** | `engine_worker.rs:58-83` (`rows_for` runs synchronously in `forward_one!` `:126-143` before `forward_token_paged`); `engram_table.rs:72-88` spawns `ENGRAM_COLS = 24` scoped threads per `gather_position`, two tables → 48 thread spawns per token; `stage_engram_rows` `forward_layer.rs:63-71` is a sync H2D. |
| 7 | chain 0.55 ms/layer, "flat in context" (§2.2 "dense-vs-sparse switch at 512 rows") | **derived, not measured; the flat chain does not exist in the tree** | 0.55 = PLAN §7a's 0.50 × 1.1, never read off a V4.1 trace (see #4). More important: `forward_layer.rs:1017` `use_sparse = ratio == 4 && n_index_comp > INDEXER_TOP_K` (also `:775, :829, :844`) — V4.1's ratios are 1 and 2, so the sparse indexer/top-512 path **never fires**; decode attends densely over `n_comp_full` rows (`:1200-1203`). ENGINE_PORT.md M5 status confirms: "the hierarchical candidate pool and the index keys are still not exercised". By the M58 kernel timings (score 58 µs + smwsum 260 µs at 16 K rows), 38 compressed-attention layers cost ≈ 4–5 ms/token at the server's `--ctx 8192` and ≈ 10–18 ms/token at 32 K, growing linearly. §2.2's "one-time graph-variant flip at pos 512" describes V4-Flash code that V4.1 bypasses. |
| 8 | miss fill 4.82 ms / 3.90 GB/s "pread straight into the iGPU slot, 64 threads" | **number real, not applicable to the bytes the kernel needs; step 1 cannot be built as written** | The bench (`crates/v4flash-kernels/tests/bench_expert_miss_cost.rs:266-366`) reads 18.8 MB at a random raw shard offset (`read_exact_at(slice, off)` `:306`, `:359`) into a fresh device buffer: raw HF bytes, no HF→ggml repack, no GPU-side verification, never a slot the GPU had cached. The kernel reads ggml 17-B blocks (byte i = elems i \| i+16, scale interleaved per block) while the checkpoint stores `[out, in/2]` (byte i = elems 2i \| 2i+1) plus a separate scale plane — the transform at `hf_v41.rs:605-617` is not optional. Two in-tree end-to-end attempts at direct-to-device also LOST (memory `fast_model_load`: 95.9 s vs 82 s; the prefill pager agent: −17 % at 4 threads). Real options: (i) teach the MXFP4 kernels the HF layout (a lane's 16 elements are 8 *contiguous* HF bytes — simpler than ggml's split; `mxfp4_unpack16` `mxfp4_pair_matvec.hip:315-332`, `dot_super_quarter_mxfp4` in `mxfp4_matvec.hip`) with the scale plane as a second region per slot, so a raw pread lands directly; or (ii) pread raw into a staging slot + a GPU repack kernel (18.8 MB at ~200 GB/s ≈ 0.1 ms). Either way the honest step-1 target is 10.7 → 5–7 ms (NVMe floor 18.8 MB / 4.31 GB/s = 4.36 ms), not 4.8. |
| 9 | slot-table encoding = today's remap; `REMOTE = i32::MIN+1` is "one-line kernel change at :218"; `MISS = i32::MIN` never observed | **encoding supported; the sentinels as specified produce OOB reads** | Encoding: `mxfp4_pair_matvec.hip:196, :212-226`; identical in the down kernel `mxfp4_matvec.hip:143-190`. `REMOTE` is negative → `:217` `dense >= 0` false → `dgpu_takes` false → `:218` `ours = true` in mode 0 → `:226` `e = -dense - 1 = i32::MAX - 1` → 18.8 MB × 2³¹ offset → garbage or a GPU page fault; same at `mxfp4_matvec.hip:183`. The change must be `ours = !dgpu_takes && dense > REMOTE` in **both** kernels (gate/up and down; 8 more hetsplit siblings if the expert quant ever changes), and MISS needs the same guard plus the poison store. Also `:224-226`: an over-cap hot expert (`dense >= 0`) indexes the iGPU buffer by RAW id — valid only for a 384-slot resident buffer; with the pool (`n_slots ≠ 384`) and §4's dGPU hot set, cap overflow reads a random pool slot. Pin `dgpu_hot_cap = N_EXPERT_USED` whenever a dGPU hot set coexists with the pool (the M63 rule), or extend the encoding. |
| 10 | table = 61 KB; per-layer row pointer baked into the graph | **supported** | 40 × 384 × 4 = 61 440 B. `GraphCache` key is `(&'static str, u32)` (`graph_cache.rs:29`); the paged graph already captures `&pg.remap_dev` (`forward_layer.rs:1738`); a `slice_view` of the table is pointer-stable. |
| 11 | `sample_next` sync; final sync; `token_seq`/`moe_signal`; `DECODE_PREISSUE` | **supported, one omission** | `engine.rs:1229`, `:918`, `:337-344`, `:519-524`, `:509-517`. Omission: pre-issue is guarded OFF when `hot_experts.is_some()` (`:515-517`, "not composed yet") and pre-issue itself measured **neutral** on V4-Flash (memory `m54`). §4's dGPU shared+hot branch (7b rule 1) on a pre-issued lane is therefore new composition work not in the step table. |
| 12 | no `hipStreamWaitEvent` on the pre-issued lane | **supported** | `issue_igpu_moe` `:206-262` uses `wait_value32_gte`; the `ie.xfer.wait_event(&sev.moe_done)` at `:249` is record-then-wait in host order (safe). The dGPU's `wait_event(&sev.moe_arrived)` `:1820-1823` is called after the pre-issued record call, so it binds to this token's record (the M54 trap is the reverse order). |
| 13 | §2.1 V4-Flash facts | **supported** | memory `decode_at_floor`: host 4.43 ms concurrent, A2 graph capture −1.3 %; `m54`: pre-issue neutral, "FUSE don't graph". |
| 14 | §2.2 pdev rope; `launch_slotdev`; batched attention takes device arrays | **supported, one gap** | `rope.rs:94-119`; `kv_cache_append.rs:92-108`; `attention.rs:656-668` (`n_raw_per / n_raw_offset_per / n_comp_per` device arrays). The B=1 kernels take scalar `raw_off/n_raw/n_comp` (`attention.rs:351, :395`) → pdev twins are new kernel work, unsized in the table. |
| 15 | §2.4 arithmetic | **internally consistent; RTT input wrong** | 47.7 / 38.9 ms sums re-derived. But 32 µs is the 4 KB ping-pong; the per-layer pair is 5.8 KB out (20 × 292 B Q8_K, not 5.4) + 20 KB back → measured 20 KB p50 67 µs (p99 90), 32 KB 97 µs (SECOND_BOX §6, memory `second_box`) → +35–65 µs/layer → **+1.4–2.6 ms/token** on every two-box row. |
| 16 | §3.1 drafter description | **supported, one correction** | `model.py:1032-1075` (DSparkAttention: window = `kv_norm(wkv(main_x))`, drafts attend window + each other), `:1128-1135` (`forward_embed`), `:1137-1156` (`forward_head`: serial markov chain, confidence on `cat(x, markov_embed)`), `:1275-1283` (`forward_spec`). Correction for §3.3: the drafter samples from `(logits_i + bias_i)/T` (`:1150-1152`, `sample()` `:1286-1293`), so `q_i = softmax((draft_logits_i + markov_bias_i)/T)` — the plan omits `/T`. |
| 17 | §3.2 cites and "the one new kernel" | **cites off; the kernel is not new** | `forward_prefill.rs:959` is inside the chunk loop, `:3016` is `mhc_pre_ffn` (the `ensure_layer_dense` call is `:3298`), the two-lane structure is `:342/:376` (not `:297`). The MoE "union" kernel already exists in pieces: the M61 dGPU hot path runs a **static work-items list with early-exit so grid.y never waits on a host readback** (`forward_prefill.rs:3362-3372`, `batch_scratch.rs:554-576`) over the member-looping kwide kernels (`mxfp4_pair_matvec.hip:334-492`) and `moe_group_builder_hetsplit` (`moe_group_builder.hip:73`). The verify MoE is that machinery with grid.y sized to B × 6 and a graph around it — say so; it removes most of step 5's kernel risk. |
| 18 | §3.3 rollback table | **supported; one omission** | `state.rs:237-257` (`raw_off` monotonic append), `KV_CACHE_ROWS = SWA_WINDOW + B_MAX = 1152` (`state.rs:20`), 2-row compressor state with `row = pos_mod` (`forward_layer.rs:676-677`). Omission: the server samples with `top_p = 0.95` by default (`v41-server.log` line 12) — `p` in `spec_accept` must be the sampler's truncated/renormalised distribution (temperature, top_p, min_p), or rejection sampling is not distribution-preserving. |
| 19 | §3.4 acceptance p1 = 0.65–0.70 | **refuted by the project's own measurement** | PLAN §7f/§7f.1 (2026-09-13, `dspark_accept.py`, model's own continuation): p1 = **0.44 greedy / 0.376 rejection-sampling**, prefix E[tokens/step] K=1/2/3 = **1.44 / 1.67 / 1.78**. The plan's values are 1.5–1.6× that with only "prose may accept higher (pending)". Re-priced at measured: batched K=2 61.7 / 1.67 = 36.9 ms → **27 tok/s**; wave 2+2 54.9 / 1.78 = 30.8 ms → **32 tok/s** (m=0), 63.7 / 1.78 → **28** at m=2; under rejection sampling at T=1 (the deployment metric, E ≈ 1.67) wave 2+2 → **30**. §7f already concluded "batched K=2 ≈ no gain, wavefront 28–30". |

### R2. Correctness holes (concrete interleavings)

- **H1 — shared per-layer words under two lanes (step 6, silent garbage).** `miss_done[l]`, `miss_req[l]`, `moe_signal[l]` (`engine.rs:341`, `[N_LAYER]`), `remote_done[l]`, `tx_ready[l]` are all indexed by layer only, and both lanes of a verify step share `token_seq`. Interleaving: lane A's `check(l)` finds no miss → stores `miss_done[l] = seq`; lane B's `check(l)` later finds a miss → stores `miss_req[l] = seq`; lane B's `wait_value32_gte(miss_done[l], seq)` is **already satisfied** by A's store → B's MoE runs on a `MISS` entry (OOB per R1.9, or a stale slot if the host has since refilled it). Fix: per-`(lane, layer)` words, or counting semantics (`miss_done[l]` waits for `2·seq + lane`), and the same for the remote words. Prerequisite for step 6, but the words should be shaped this way in step 2.
- **H2 — boundary eviction's `MISS` write is asynchronous (step 2, silent garbage).** §1.3 puts eviction at the token boundary via `copy_from_host_async` on a side stream with no signal ordering it before the next token's first check kernel (only fills are ordered by `miss_done`). Interleaving: boundary frees slot S (old occupant Y at layer 5; the `MISS` DMA is queued, not visible) → token t+1 layer 0 misses X → miss thread takes free slot S, fills X, writes `table[0][X] = S`, `miss_done[0]` → layer 5's check reads the stale `table[5][Y] = S` (resident) → MoE reads slot S = X's bytes. Fix: boundary table writes synchronous (≤ 10 copies/token, ~10 µs each) or event-ordered ahead of layer 0's lane.
- **H3 — single `ffn_moe_remote[N_EMBD]` (step 4, silent garbage).** The receiver writes layer l+1's partial on `de.remote_rx` while `combine(l)` on `de.compute` may not yet have consumed layer l's; the two streams are unordered. Fix: `[N_LAYER × N_EMBD]` (800 KB; × B under verify) so each layer's combine reads its own row.
- **H4 — "none of mine" reply (step 4, silent garbage).** A 4-byte reply leaves the previous partial in the buffer and the third `vec_add` adds it again. Fix: `hipMemsetAsync` on `remote_rx` before the word, or a per-layer flag the combine reads.
- **H5 — value-waits nothing satisfies (steps 2 and 4, hang → 30-min watchdog abort, `DEEPSTRIX_HANG_DEADLINE_MS=1800000`, log lines 5–6).** (a) miss thread I/O error or panic → `miss_done[l]` never written → iGPU stream stuck behind the wait; (b) box 2 down or a message lost → `remote_done[l]` never reaches `seq` → dGPU compute stuck. A pre-issued lane cannot be cancelled once enqueued. Fix before step 2 ships: every waiter word has a producer-side timeout that writes a poison marker + the word so the stream drains, and the host fails the request on the poison; document it as part of the protocol, not an afterthought.
- **H6 — CPU writes into a slot the GPU has cached (step 1, silent garbage).** A direct pread into `hipMalloc` memory bypasses the iGPU's L2; the next kernel sees fresh bytes only if its launch acquire invalidates L2 for coarse-grained memory — no HIP guarantee, and the phase-B bench never verified bytes on the GPU nor reused a slot. Fix: step 1's P gate = refill the SAME slot ≥ 2× after a kernel has read it and verify via the parity harness, not just first-fill identity.
- **H7 — sentinel OOB** (R1.9): both MXFP4 hetsplit kernels, both sentinels.
- **H8 — §1.3 rule 3's pin/Dekker is dead code in step 2 and under-specified in step 6.** Single lane: while the miss thread serves layer l the iGPU stream is blocked at `wait_value32(miss_done[l])`, so no check kernel of layer > l can run — no concurrent reader exists and pins buy nothing. Two lanes: the "Dekker" pair is a DMA table write vs a kernel read; the host must event-sync the side stream before reading the pin, which the text omits. Simplify: mid-token victims = any slot not in this token's `sel_log` so far (later layers re-check and re-fill), and defer the two-lane rule to step 6 with H1's per-lane words.
- **H9 — one iGPU compute stream (`engine.rs:194-195`).** `hipStreamWaitValue32` blocks the whole stream, so a lane-A miss stall at layer l blocks lane B's MoE(l) even though B's router is done. Design B's "wavefront covers sub-batch-B misses (~65 %)" needs a second iGPU compute stream per lane; the prefill pipeline shares streams and only splits events (`sync_events_t1`), so it is not the template the plan says it is.
- **H10 — the wavefront edge is not "one KV edge".** B's layer-l attention reads A's window rows *and* A's `comp_kv` rows, index-key rows, and — for a ratio-2 group straddling the lanes — A's `kv_cur/sc_cur` in the `[B, 512]` scratch. The edge must sit after A's compressor stage and cover that scratch, or a straddling group is pooled from stale rows.
- **H11 — host-written word consumed by a GPU value-wait is unmeasured on this stack.** `PinnedBuffer::new` uses flags 0 (`buffer.rs:401-403`, coherent/fine-grained) so the CP poll should see the store, but every existing signal (`moe_signal`) is written by a *device* stream. A 30-line test (host store → `wait_value32_gte` release latency, both devices) gates the whole §1.2 mechanism.
- **H12 — `sel_log` D2H "on `de.xfer`".** `sel_log` is written by an iGPU kernel; either it lives in pinned memory (then the miss thread's "pinned mirror" is the buffer and no copy exists) or the copy is on `ie.xfer`. Pick one.

### R3. Structural critique

- **What the plan optimises within.** The 40-serial-layer chain of host-enqueued per-stage graphs plus a per-layer expert phase. Its levers inside that structure: delete host syncs (−8 ms of a 500 ms token), amortise the launch-bound chain over b rows (wavefront). It explicitly defers the structural rewrite of the binding dGPU chain (design A, ~2× over byte floor, +4–6 tok/s) and owns the miss term only through residency. That is a legitimate M8 scope — provided the arithmetic does not borrow from unmeasured inputs, which it does (R1.19).
- **Is a small-B decode verify path the right structure?** Yes as the shared prerequisite (§7d step 3: every speculative design needs B=k on decode primitives; `project_decode_dgpu_bound`: the prefill path recovers nothing at small B). But the plan should carry the CED-boundary candidate pipeline (§7e/§7f: 30–31 at *measured* p, position-1 only, a B=2 encoder pass, rollback limited to the 20 encoder layers' stores, no markov chain) as the priced fallback — at measured acceptance it ties or beats the wavefront for less machinery.
- **"Same logits as batched verify".** True in exact arithmetic: B depends on A only through what A's pre-attention stages wrote at the same layer, per-row kernels, and both schedules append row i at `raw_off + n_raw + i` (`state.rs:237-257`). Bit-identity holds only if the per-row reduction orders match between one B=4 launch and two B=2 launches (score/smwsum tile over keys, not rows; the down kernel accumulates a row's 6 experts in slot order — plausible, unverified) and with the H10 edge in place. Gate it as argmax-equal + the 5e-2 vector-scale drift budget (the M54 oracle), not bit-exact.
- **Is the 30 tok/s arithmetic honest?** No double counting in §6 (S5/S6 sums re-derive). Hidden or mis-priced serial terms, in order of size: (1) acceptance — the plan's 1.5–1.6× on p1 is worth 15–20 % of tok/s (R1.19); (2) dense compressed attention (R1.7): +4–5 ms/token at 8 K, +10–18 at 32 K, absent from every row; (3) RTT at real message sizes: +1.4–2.6 ms/token single stream (R1.15); under the wavefront it is 80 round trips/step at 67–100 µs (B=2 replies are 40 KB, above the 32 KB point; p99 182 µs) that must each fit under a 0.58 ms window — budget +2–3 ms/step for tails; (4) DSpark's 7.2 GB of experts pinned in box-1's pool = −9 % box-1 residency → by §7d's curve ≈ +0.5–1 miss/token ≈ +2–4 ms; (5) 4.4 ms/miss is the NVMe floor (4.36 ms at 4.31 GB/s): two misses in one layer serialise to 8.7 ms, and more reader threads cannot lower it; (6) the wavefront's "expert phase ≈ 0" rests on iGPU/layer ≈ 2 × (10 picks × 0.088 + 0.05) ≈ 1.0 ms vs dGPU 1.16 — hidden with a 14 % margin; (7) box 2's per-layer launch + host RX/TX copies (~30–50 µs) beyond the "+0.02". Net: with measured p and items (2)–(4) the wave 2+2 single-stream number is ~27–30, not 36–38; 30 is reachable only if the chain is fused (design A) or acceptance on real traffic exceeds ~0.55.

### R4. Ordering and what to measure before step 2

Steps 0–1 first: agreed, they are 95 % of today's token. But their numbers and step 1's design need rework (R1.2/3/8), and four cheap measurements (no kernels, ≤ 1 day total) precede step 2 because they decide whether steps 2–6 can reach 30 at all:

- **M-a. Split the pager counters** (decode vs dense; `expert_pager.rs:267/369` vs `:438/445`) and log misses/token, ms/token, and a per-miss phase timer (pread / repack / copy, the `DEEPSTRIX_EXPERT_LOAD_PROFILE` pattern). Re-bases §0 and fixes step 1's design.
- **M-b. Harvest device events on the paged path** (`dgpu_busy_us = 0` in every summary today). Gives the real V4.1 chain ms/layer (R1.7) — the 0.55 assumption has never been read off a trace.
- **M-c. `DEEPSTRIX_EXPERT_TRACE`** (`engine.rs:735-770`) on real agentic traffic with the split fixed and the pool at 70–80 GB → steady-state misses/token on V4.1 routing. The plan makes this step 4's gate; it is available now and is the single input that decides the whole budget table.
- **M-d. Host-store → `hipStreamWaitValue32` wake latency and a refill-a-cached-slot byte check** (H6, H11): ~50 lines, gates the §1.2 mechanism.
- **M-e (for step 5).** The pending plain-prose acceptance run (§7f.1) → replaces the 0.65–0.70 in §3.4/§6 with a measured p, and a decision between wavefront and the CED-boundary pipeline.

Then: step 0 = a **phase-aware split** (dense windows are prefill's; at decode start they become LRU entries — they already carry `slot_of` keys — and prefill reclaims them next request), gated on prefill tok/s unchanged (14.75 on the 1825-token prompt) as well as decode misses/token; not `V41_PAGER_WINDOWS=1`. Step 1 = the format decision (HF-native kernel reads vs GPU repack) plus the parallel reader, target 10.7 → 5–7 ms. Insert **"V4.1 indexer top-512 + candidate pool on decode"** (the M5 leftover) before step 3's 0.55 figure is used at ≥ 8 K context. Step 2 after M-d with H2/H5/H7/H8 designed in; step 4 with H3/H4/H5 and the pre-issue × hot-set composition; step 6 with H1/H9/H10.

### R5. Verdict: NOT SIGNED OFF — changes required before implementation, ranked

1. **§3.4, §5 step 5–6, §6 — re-price every DSpark row at the measured acceptance** (p1 = 0.44 greedy / 0.376 RS; E = 1.44/1.67/1.78) and state the result: wave 2+2 ≈ 30–32 at m = 0, 28 at m = 2, batched K=2 ≈ 27. Add the CED-boundary pipeline as the priced alternative. Make the prose acceptance run a gate for building step 5.
2. **§1.4, §5 step 1 — replace "pread straight into the slot" with a format decision** (HF-native MXFP4 kernel layout or GPU repack after a raw pread), include the CPU repack in the per-miss decomposition, set the target to 5–7 ms, and gate on a refilled-cached-slot byte check (H6).
3. **§0, §5 step 0 — fix the residency facts and the split**: cite `meas-D.log:23`; state the default `384 + n_slots mod 384` decode cap from `expert_pager.rs:190-195/:453-456`; split the counters; make step 0 phase-aware with a prefill-unchanged gate.
4. **§1.2(a), `kernels/mxfp4_pair_matvec.hip:218-226` and `mxfp4_matvec.hip:171-183` — sentinel handling in both kernels** (`ours = !dgpu_takes && dense > REMOTE`, MISS poison), and pin `dgpu_hot_cap = N_EXPERT_USED` when a dGPU hot set coexists with the pool.
5. **§1.2(b), §1.3, §4 — the wait/signal protocol**: per-`(lane, layer)` words (H1); synchronous boundary table writes (H2); a producer-side timeout + poison for every waiter word (H5); drop the mid-token pin/Dekker text for step 2 (H8).
6. **§4 — per-layer (× B) `ffn_moe_remote` and an explicit zero on "none of mine"** (H3, H4); RTT at 20 KB/40 KB message sizes in §2.4/§3.4/§6; list the pre-issue × dGPU-hot composition (`engine.rs:515-517`) as step-4 work.
7. **§2.2 / §5 step 3 — add the V4.1 sparse indexer + candidate pool for decode** (`forward_layer.rs:1017` gates on `ratio == 4`; ENGINE_PORT M5 leftover) as a prerequisite of the flat 0.55 chain, and price dense attention until then (+4–5 ms @ 8 K, +10–18 ms @ 32 K).
8. **§3.5 / §5 step 6 — a second iGPU compute stream per lane** (H9) and an edge that covers the compressor scratch and index keys (H10); gate on argmax + 5e-2 vector-scale, not "logits identical".
9. **§3.2 — say the verify MoE reuses the M61 static-work-items + kwide + group-builder machinery** (`forward_prefill.rs:3362-3372`, `batch_scratch.rs:554-576`), fix the three cites (`:3298`, `:342/:376`), and add `/T` to `q_i` and top_p/min_p to `spec_accept` (§3.3).
10. **§5 — insert measurements M-a…M-d before step 2** and move the misses/token trace (step 4's gate) to step 0.

## Measured (2026-09-13)

Everything below replaces the estimates in §0–§2 and §5 steps 0–1. Method: the V4.1
server (`--ctx 8192`, CED on, `V41_PAGER_POOL_GB=45` = 2570 slots × 18.8 MB = 48.3 GB,
6 windows of 384), five back-to-back runs of the same three requests, new
instrumentation described in "What was added" below. Raw logs: `logs/m8-R1..R5.log`.
Requests: **S** = 17-token prompt / 48 generated; **L** = 2702-token prompt / 96
generated; **S'** = S again, warm LRU. Runs R4/R5 ran while two other agents held the
box at load-average 20–40, so their *wall* numbers are contaminated; their *counting*
numbers (hit rate, misses/token) are not.

### M-a. The counters were conflated; decode's real hit rate, first measurement ever

`requests`/`misses` were incremented by BOTH `ensure_layer_dense` (prefill, 384 per
layer per chunk) and `ensure` (decode, ≤ 6 per layer per token). A 2702-token prefill
issues ~30 000 requests against decode's 240/token, so every hit rate this project has
quoted — including `v41-vision.log`'s `hit_rate=0.5446` — was prefill's dense-window
residency. The counters are now split four ways (`prefill_requests/misses`,
`decode_requests/misses`) plus per-phase nanosecond timers, logged per request
(delta) and cumulatively.

**Decode's true numbers** (`pager_misses` from `het.token.summary`, steady state):

| decode LRU | run | S | L | S' | decode hit (req S/L/S') |
|---|---|---|---|---|---|
| 650 slots, 12.2 GB (**old default**) | R1 | 99.0 | 104.2 | 107.2 | 0.574 / 0.559 / 0.538 |
| 1418 slots, 26.7 GB (frac 0.50) | R2 | 74.6 | 89.3 | 81.0 | 0.665 / 0.617 / 0.639 |
| 2186 slots, 41.1 GB (frac 0.85) | R4 | 63.0 | 75.7 | 69.4 | 0.704 / 0.670 / 0.681 |
| 2186 slots, 313-token generation | R5 | – | 64.3 | – | 0.727 (0.754 by token 256) |

(cells = misses per generated token). **The plan's §0 numbers — "79 % hit, ≈ 50
misses/token" — are wrong by 2×.** Decode's hit rate on the old default was **0.54–0.57
with 89–110 misses per generated token**, and it is the miss term that is 93 % of the
token, not 90 %.

### M-b. The real per-token breakdown (paged decode path)

`dgpu_busy_us=0` in every log was not a paged-path bug: `EventPool` is disabled unless
`attach_perfetto` is called, which the server never does. `DEEPSTRIX_TOKEN_PROFILE=1`
now enables both pools and emits the per-stage rollup at INFO. R1, request L, mean of
92 steady-state tokens at pos 2705–2796:

| item | ms/token | share | where |
|---|---|---|---|
| `ExpertPager::ensure` (miss service) | **914.6** | **92.9 %** | host read 796.1 + H2D 112.3 + LRU/remap ~6 |
| per-layer `synchronize()` + `d_selected` readback | **50.2** | 5.1 % | `forward_layer.rs:1667-1670` — **1.25 ms/layer**, not the 60 µs §0 assumed |
| everything else (host enqueue, iGPU MoE, head, Engram) | 20.1 | 2.0 % | |
| **total** | **984.9** | | 1.02 tok/s |

`host_us = 978 ms`, `sync_us = 2 ms`: the token is **99.8 % host-bound**. The dGPU's own
stage sum is 109 ms, of which the `.wait` scopes (`peer_push_ffn_input_norm.wait` 44.8,
`ffn_combine.wait` 21.3, `peer_push_selected.wait` 1.8) are stalls; real dGPU work is
**~35 ms/token**.

**dGPU chain, per layer.** Real work 35 ms − head 1.29 − Engram 0.59 = 33.1 ms / 40 =
**0.83 ms/layer** of stage time, and **1.25 ms/layer** of *exposed* time (what
`sel_sync` actually waits for). §2.4's **0.55 ms/layer is optimistic by 1.5–2.3×**;
every row of §6 that starts from `40 × 0.55 = 22 ms` should start from 33–50 ms.
Largest chain stages (ms/token, pos 2750): output_proj 6.73, q_chain 4.67,
shared_expert 3.89, kv_chain 3.39, mhc_pre_ffn 2.40, mhc_pre_attn 2.35, router 2.35,
attn_smwsum 1.99, peer_push_selected 1.97, head 1.29, attn_score 0.97.

### M-c. Dense compressed attention (R1.7 confirmed) — but the gate is not the fix

Ablation `V41_ATTN_COMP_CAP=512` makes the dense kernels do exactly the row count a
working top-512 indexer would leave them (wrong rows — timing only). R2 vs R3, same
pool split, same prompts:

| pos | attn_score + attn_smwsum, dense | capped at 512 | delta |
|---|---|---|---|
| 20–120 (n_comp < 512, cap inert = **control**) | 1.698 ms | 1.678 ms | −0.02 ms (no bias) |
| 2705–2796 | 2.966 ms | 2.009 ms | **−0.957 ms** |

Slope: **0.605 µs per average compressed row per token** over the 38 compressed layers.
Extrapolated cost of having no indexer: **+3.5 ms/token at 8 K**, **+14.8 ms/token at
32 K** — R1.7's estimate confirmed (it said 4–5 @ 8 K, 10–18 @ 32 K).

**But widening `forward_layer.rs:1017`'s `ratio == 4` does nothing**, for three
independent reasons, so R5.7's "make the gate correct" is not a gate fix:

1. `het/weights.rs:379` loads `indexer` and `indexer_compressor` weights **only at
   ratio 4** — under `v41` `dlw.indexer` is always `None` and the sparse branch would
   panic on the first layer that took it.
2. `het/state.rs:329` builds `indexer_compressor` state **only at ratio 4**, so
   `n_index_comp` is always 0 and `n_index_comp > INDEXER_TOP_K` is false regardless
   of the ratio test.
3. V4.1's indexer is a *different mechanism*. Checkpoint (`model.safetensors.index.json`):
   `attn.indexer.wq_b` exists on layers **[2, 8, 14, 20, 24, 28, 32, 36]** and
   `attn.indexer.wk` only on **[2, 8, 14, 20]** — matching ARCH_SPEC §1.4/§1.5: eight
   index-source layers whose top-512 is *shared* with their reuse layers, keys owned by
   the four kv-source layers, plus the layer-20 hierarchical candidate pool
   (2048 blocks × 8) that layers 24/28/32/36 mask against. None of that exists in the
   tree; ENGINE_PORT M5 already says so.

So this is the **M5 leftover in full** (index-key cache, 8 shared top-512 sites,
candidate pool), priced at 3.5 ms/token @ 8 K and 14.8 @ 32 K. No numerics change was
shipped and the parity harness was therefore not run; the ablation env is measurement-
only and defaults off.

### M-d. Miss-phase split, and the real floor (§1.4 / R1.8)

Per decode miss, single-threaded, uncontended (R1; 18.8 MB across 3 roles, 6 pread
ranges):

| phase | ms/miss | rate | removable? |
|---|---|---|---|
| host `Vec` allocation (2 per role) | 0.16 | | yes — reusable buffers |
| pread of the HF shard (`read_range_into_cached`) | **5.9–6.2** | **3.10 GB/s** | **no — this is the floor** |
| HF → ggml MXFP4 repack (scalar nibble loop) | 1.40–1.50 | | yes — GPU kernel (~0.1 ms) or HF-native kernel layout |
| 3 × synchronous `copy_from_host` | 1.08 | 17.4 GB/s | yes — pread into the host-addressable slot |
| **total** | **8.6–8.9** | | |

`V41_PAGER_MISS_THREADS=3` (one thread per role, opt-in, default 1) gave **8.89 → 6.01
ms wall on request S and 9.17 → 9.12 ms on request L** — i.e. a partial gain that
vanishes as soon as anything else touches the disk. Under the other agents' load the
whole miss path went to **17 ms**. Two secondary findings: the read does **not** scale
with threads at role granularity (matching `feedback_no_expert_streaming` /
`fast_model_load`'s four lost attempts), and with `GLIBC_TUNABLES=arena_max=2` the
3-thread path's allocation cost rises 0.16 → 0.96 ms/miss on malloc-arena contention.

**Verdict on the plan's 4.82 ms.** `bench_expert_miss_cost.rs` measured a raw 18.8 MB
read only. Adding the repack, allocation and H2D this path actually needs costs +2.6 ms
measured, so the same bench end-to-end is **~7.5 ms**, not 4.82. The honest floor is:
**6.0–6.5 ms today** (pread at the 3.1 GB/s the device delivers, everything else
removed), and **4.4–5.0 ms only if the read itself reaches the 4.31 GB/s** the
COLD_EXPERT_CACHING bench got with 64 threads on one contiguous range. R1.8's "5–7 ms"
is right; §5 step 1's "→ 4.8 ms" is not.

### M-e. The pool split (§5 step 0), fixed and measured

`ExpertPager::new` defaulted `dense_windows = total_windows − 1`, giving decode's LRU
`384 + n_slots mod 384` slots whatever `V41_PAGER_POOL_GB` was (650 at 45 GB). It is now
a deliberate budget: `V41_PAGER_DECODE_FRAC` (default **0.75**) reserves that fraction
of the pool for decode's LRU, capped by a CED-aware prefill ceiling (`CED_DECODER_START
+ 2` windows — with CED on, prefill only sweeps the 20 encoder layers plus one rotating
window for the 128-token decoder replay, so windows beyond that buy prefill *nothing*).
`V41_PAGER_WINDOWS=<n>` still forces the count.

Prefill gate (A/B, uncontended): 2702-token prompt, `ced_replay total_s` **115.8 s at 5
dense windows → 126.0 s at 3** (+9 %; 68 → 74 layer-pagings of 7.2 GB), **→ 186.9 s at
1** (80 pagings; the rest is external load). Decode gain over the same step: 104.2 →
89.3 → 75.7 misses/token. One window is worth (chunks − 1) × 7.2 GB to prefill and
~13.6 misses/token ≈ 122 ms/token to decode: **break-even at ~82 generated tokens**.
Pinning only pays if the *whole* encoder fits (21 windows = 151 GB), which no pool here
affords — so at these sizes the pin line is nearly worthless and decode takes the pool.

**Back-to-back tok/s (R1 old split → R2 frac 0.50):** S 1.05 → **1.20** (+14 %),
S' 1.03 → **1.18** (+15 %), L 1.02 → 1.00 (−2 %). The L row did not move because
ms/miss rose 8.75 → 10.35 in the same interval: with 289 GB of experts against 26 GB of
page cache, more distinct experts resident means *less* page-cache reuse on refill. **The
two effects partly cancel; the split buys miss count, not automatically wall time.**

### M-f. The residency curve — the number the whole plan hinges on

`DEEPSTRIX_EXPERT_TRACE` over a real 313-token generation on the 2702-token prompt
(`scratchpad/v41_expert_trace.bin`, 40 layers × 6 picks, no dedup ever fires). Replaying
the pager's exact policy offline reproduces the live server (1418 slots: sim 88.4 vs
live 89.3; 2186: sim 66.6 vs live 69.4), so the simulator can be trusted for sizes the
box cannot hold:

| slots | GB | % of 15 360 | LRU miss/token (warm) | LRU hit | static top-N miss/token | static hit |
|---|---|---|---|---|---|---|
| 650 | 12.2 | 4.2 % | 134.2 | 0.441 | 116.3 | 0.515 |
| 1418 | 26.7 | 9.2 % | 88.4 | 0.632 | 71.7 | 0.701 |
| 2186 | 41.1 | 14.2 % | 66.6 | 0.723 | 44.6 | 0.814 |
| 3072 | 57.8 | 20.0 % | 47.0 | 0.804 | 25.2 | 0.895 |
| 4096 | 77.0 | 26.7 % | 33.3 | 0.861 | 11.8 | 0.951 |
| 6144 | 115.5 | 40.0 % | 14.7 | 0.939 | 0 (trace exhausted) | 1.000 |

Concentration over the generation: top 20 % of (layer, expert) pairs carry **85 %** of
picks, top 26.7 % carry 91.4 %, top 36.5 % carry 97.1 %. Only 7 395 of 15 360 pairs
(48 %) are touched at all in 256 tokens, and the discovery rate decays 111 → 7.4 new
pairs/token between token 17 and token 256, so the ≥ 6144 rows above are compulsory
misses of this trace, not a steady state.

Three consequences:

1. **≤ 2 exposed misses/token needs ≈ 99.2 % of picks resident**, which this
   distribution reaches at roughly **45–50 % residency (≈ 7 000 slots ≈ 132 GB)** and
   only with near-oracle placement. Box 1 alone (45–70 GB) cannot get there; box 1 +
   box 2 (224 GB raw, realistically ~180 GB for experts) can — **but only just, and only
   if placement beats LRU.**
2. **Static frequency-ordered placement beats LRU at every size** — 44.6 vs 66.6
   misses/token at 2186 slots, 11.8 vs 33.3 at 4096. (The static figure is an oracle fit
   on the same trace, so it is an upper bound; but `project_placement_analysis_2026-09`
   found V4-Flash routing stable across prompts and the hot-expert-file machinery already
   exists.) A frequency-pinned core + LRU for the tail is the cheapest remaining lever
   and needs no new kernels.
3. **Requantising the routed experts is the structural answer.** At MXFP4 (4.25 bpw) the
   set is 289 GB; at ~3.4 bpw it is 231 GB, at ~2.2 bpw 150 GB — which fits two boxes
   *entirely* and takes the miss term to zero, unlocking every "m = 0" row in §6.

### M-g. Recommendation on §5 step 2

**Do not build the device-resident slot table next.** Its entire prize is the
50.2 ms of per-layer sync + the ~2 ms remap + the ~18 ms of host gaps = **~70 ms of a
985 ms token (7 %)**. Executed perfectly it moves 1.02 → 1.09 tok/s, while carrying the
whole correctness surface the review enumerated (H1–H12: four distinct silent-garbage
interleavings and a hang class). It is a prerequisite for steps 3–6, but it cannot be
*validated* — there is no A/B that shows it working — until the miss term is small
enough for a 70 ms change to be visible.

Ordered by measured ms/token recovered:

| # | move | recovers | risk |
|---|---|---|---|
| 1 | Residency: frequency-pinned core + LRU tail at the largest pool the box holds | 89.3 → ~45 misses/token ≈ **−400 ms** | low, no kernels |
| 2 | Miss cost: reusable buffers, GPU (or HF-native) repack, pread into the slot | 8.7 → ~6.2 ms/miss ≈ **−220 ms** at 89 misses | low–medium (H6 byte check required) |
| 3 | Prefill: page the *union* of a chunk's picks instead of all 384 per layer | a 17-token prompt currently pages 40 × 7.2 GB = 288 GB and takes 60 s; the union at B = 17 is a few hundred experts | medium |
| 4 | Requantise routed experts to ~2–3 bpw (two-box full residency) | the miss term → 0 | high, quality gate |
| 5 | V4.1 indexer + candidate pool (M5) | **3.5 ms/token @ 8 K, 14.8 @ 32 K** | medium, numerics-gated |
| 6 | Device slot table (§5 step 2) | **70 ms** | high (H1–H12) |

Step 2 becomes the top item only once (1)+(2)+(4) have pushed misses/token below ~6 —
at which point the token is ~40–70 ms and the sync term is 30–50 % of it. Until then,
every row of §6 that assumes `m` ∈ {0, 1, 3.5} is describing a machine that does not
exist: the measured `m` is **64–104**.

### What was added to the tree (all off by default except the pool split)

| change | file | switch |
|---|---|---|
| four-way pager counters + per-phase ns timers + `PagerCounters` delta type | `het/expert_pager.rs` | always on (counting only) |
| per-request and per-heartbeat split logging, miss-phase-per-miss line | `deepstrix-server/src/engine_worker.rs` | always on; `DEEPSTRIX_HEARTBEAT_TOKENS` |
| CED-aware prefill/decode pool budget | `het/expert_pager.rs` | `V41_PAGER_DECODE_FRAC` (default 0.75), `V41_PAGER_WINDOWS` |
| EventPool enable + per-stage INFO rollup + host phase fields on `het.token.summary` | `het/trace.rs`, `het/engine.rs`, `het/forward_layer.rs` | `DEEPSTRIX_TOKEN_PROFILE=1` |
| pread/repack/alloc split inside the HF expert read | `v4flash-core/src/hf_v41.rs` | always on (atomics) |
| dense-attention row cap (timing ablation, output invalid) | `het/forward_layer.rs` | `V41_ATTN_COMP_CAP` (default 0 = off) |
| one reader thread per role on a decode miss | `het/expert_pager.rs` | `V41_PAGER_MISS_THREADS` (default 1 = old behaviour) |
