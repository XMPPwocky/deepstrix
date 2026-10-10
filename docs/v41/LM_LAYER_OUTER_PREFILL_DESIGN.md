# Layer-outer prefill, frozen expert split, bounded prefetch: design (rev 3, 2026-10-10, after review round 2)

## Review round 2 (verdict APPROVE WITH CHANGES) -> dispositions (rev 3 text below already includes them)
| # | sev | finding | disposition |
|---|---|---|---|
| R2-1 | MAJOR | hub free slots are ~441-921, not ~1,300: unified LRU = 5,425 - 384 (idle dense window, `V41_PAGER_WINDOWS=0` -> 1 window, `ep:2064-2077`) = 5,041; decode-hot owned `top` 103x40 = 4,120 or INTERLEAVE (live) 115x40 = 4,600; filler 46 or 22 per layer; box-2 share 235-247/layer = 4,700-4,940 = 103-108% of ~4,570 usable: DOES NOT FIT as specified | ACCEPTED: filler = min(knob, (n_slots - lru_lo - owned_now)/20) from LIVE counts at job start; reclaim the idle dense window (allow `dense_windows = 0` under the unified pool, ~10 lines + startup assert; AUDIT B12) => 9,995 slots vs 7,680 encoder + hub decoder-hot head 2,060-2,300 = 9,740-9,980 (fits with 15-255 slack under `top`, ~0 under interleave); per-job log share1/share2/usable2; honest steady re-reads = max(0, share2 - usable2) per window (0-~400, not "~0") (sec 4) |
| R2-2 | MINOR | consumer list wrong both ways: look-ahead builders, cache-prior, `wants_for_box2` are arena-only; MISSING `readahead_layer` (`fp:10296-10303`) | ACCEPTED: real consumers = pick loop `fp:9616`, `lm_prefetch_words` `fp:870`, `readahead_layer` `fp:10302`; `pre_moe_route` has no job handle (5 call sites) => process-global `PREFILL_SPLIT: RwLock<Option<Arc<PrefillSplit>>>` set by a guard around the quantum call (single scheduler thread; decode/verify between quanta see None) |
| R2-3 | MINOR | 5a fix smaller: pass `cur_layer` instead of `d.layer` as `for_layer` at `re:5672` (and `re:5677`): repairs `own_sweep_rank` (tier 3 long) AND `sweep_rank` (tier 2/4); demand claims unaffected (landings optional, dropped without victim) | ACCEPTED (sec 5a); to pin down in the loopback test: (i) speculative words on a PREFILL-flagged frame land prefill-class (`me_tiered` needs `prefill_landing = d.stage || d.prefill`, `re:5649`; `prefetch_words_prefill` `re:5163-5165`); (ii) drop the hub `pf_sent` dedup for P3 (box 2 DROPS words with no free staging set, `re:5116-5118`): re-derive each layer's words from the mirror |
| R2-4 | MINOR | missing ordering invariant for reusable pinned staging (`engram_pin`, `remote_ffn_moe_pin` + `remote_upload_done`, `rb_*`); `indexer_saved_store` per lane-unit confirmed sufficient | ACCEPTED: I4b (sec 2); sel/nsparse copy-OUT right after the source task's PRE, copy-IN skipped when flag = -1 (sec 3.2) |
| R2-5 | MAJOR | producer thread needs an ownership refactor: `EngramCtx` borrowed mutably by decode; `token_embd_bytes: Vec<u8>`; image rows; absolute-range contract; lifecycle; RAM; P2b-2 cannot hold the embedding table in VRAM | ACCEPTED (sec 6) |
| R2-6 | MINOR | P1 estimate 5.5-6.0 s unless R2-1 fixed; P2a alone moves input prep in front of the first quantum; sec 9 still says W=8192 | ACCEPTED: land P2a+P2b together or attribute; estimates and sec 9 fixed |
| R2-7 | NIT | 16 requests/layer => 192 words/layer | FIXED |
| R2-8 | NIT | `lm_mid_group` also gates the embed phase (`ms:421`) | ACCEPTED: define "window open, between quanta"; the embed phase is safe there (lanes drained) |

Status: DESIGN FOR REVIEW (round 3, confirm). No production code written for it. Derived from reading the code and from the measurements in
`IGPU_MOE_F16LUT_PLAN.md` (POST-MOVE BREAKDOWN, ROOT CAUSE, streaming arithmetic). Numbers marked (inferred) are NOT measured.
Owner direction (10-10): rebuild around layer-outer; for long prefills changing the expert->box allocation is acceptable.

## Review round 1 (Fable reviewer, verdict NEEDS REWORK) -> dispositions
| # | sev | finding | disposition in rev 2 |
|---|---|---|---|
| 1 | BLOCKER | background prefetch LANDINGS rank victims with `for_layer = word's layer` (`re:5672`, `own_sweep_rank` `re:4081`), so a landing for layer L+k evicts layers L+1..L+k-1 first; Belady-correct only for demand at the current layer | ACCEPTED. P3 is no longer hub-only: box-2 landing-rank fix (rank with the pass's CURRENT layer; `(cur, word_layer]` = "ahead") + loopback test; the existing `V41_LM_PREFETCH` has the same flaw (likely why it measured "not worth enabling") |
| 2 | MAJOR | hash split makes box 1's prefill half a random half => LRU evicts box 1's whole decode hot set; post-job decode misses are SYNCHRONOUS dm-crypt reads on the critical path; no re-warm path | ACCEPTED. Split = box 1 takes its decode-hot encoder head UNION a hash filler sized to its free slots, FROZEN per job (sec 4); measured displacement budget and a post-job decode-p90 gate |
| 3 | MAJOR | box 1 has NO async residency in the prefill pipeline: `ensure` is synchronous; `b1_page_misses` short-circuits `b1_prefetch`; `drain_prefetched` once per tick | ACCEPTED. P1 cold fill on box 1 is a synchronous, bounded stall (filler only, ~24 GB ~6 s) and is stated as such; in-pipeline admission is an explicit P3 sub-item with its own gate |
| 4 | MAJOR | `bd.indexer_saved_store` (host i32 on lane scratch, retracted to -1 when the source does not fire) missed: bit-identity would break silently | ACCEPTED. Added to the inventory as a per-lane-unit field (sec 3.2) |
| 5 | MAJOR | capacity: floors/reserve/nursery | CHECKED: box-2 startup line "pin budget 4232 = 4760 - 0 - reserve 480 - floors 0 - nursery 48"; global pool floor 0.00. Usable for a long prefill ~ 4,760 - ~136 pinned - nursery 48 ~ 4,570. Reserve is a guarantee, not lost space. Sizing in sec 4 uses 4,570 |
| 6 | MAJOR | pair merge never fires: partner arrives >= 10 ms later (> 400 us wait) | ACCEPTED. P4 needs DEFERRED submit (route first, submit the pair together) or a long wait; gate is golden/KLD, not bitexact (WMMA row-composition independence unverified) |
| 7 | MAJOR | arithmetic: 2K rows/s at 16K windows needs 8.8 GB/s/box (> 4 GB/s); streaming cannot reach the 2K target | ACCEPTED, big windows CUT. W stays 4096 (8192 only if the existing halve-on-OOM store allows). Arena-tail lend removed (R5 gone). The resident regime (P1) is the only path; P3's value is bounded refill |
| 8 | MAJOR | `lm_prefetch_words`, the look-ahead word builder, `wants_for_box2` select with `partition_box2` | ACCEPTED. One frozen `PrefillSplit` object consulted by every prefill-path consumer (sec 4) |
| 9 | MAJOR | `EngramAhead` runs one block per scheduler tick; `chunk_inputs` builds HC on the scheduler thread | ACCEPTED. Sec 6: continuous producer thread + device-side HC build (P2b) |
| 10 | MINOR | replay-row capture unspecified | ACCEPTED, sec 3.5 |
| 11 | MINOR | U odd / empty lane B | ACCEPTED; simulator covers it; parity flips per layer, harmless once all per-unit state is relocated |
| 12 | MINOR | per-task re-uploads (`pos_per_b`, vis, tokens slice, SYNC Engram `copy_from_host` ~12.6 MB at layers 1,14, `dump_layer_pos`, `events.reset()`/perfetto flush) | ACCEPTED, sec 3.2 table + 3.6 |
| 13 | MINOR | `layer <= CED_DECODER_START` includes layer 20 | FIXED: encoder split uses `layer < CED_DECODER_START` |
| 14 | MINOR | f16 arm P0 estimate wrong at 8 members/expert | ACCEPTED; P0 sequenced last, estimate removed |
| 15 | MINOR | lending not exception safe | ACCEPTED: `restore_compressor_lending()` at quantum entry/exit + guard |
| 16 | MINOR | bitexact oracle: split OFF for the P2 arm; it has no box-2 client | ACCEPTED, sec 9 |
| 17 | NIT | quantum time 2.1 s today (~1.0 s at the P1 target) | FIXED |

## 0. Why (measured)
* **Wall.** Steady 4096-row window 8.02 s (511 rows/s; 10-10 16:44 cold 192K job). The hub is not the wall (reply->next submit
  10.4 ms; wire 7 ms/round trip); box 2 is: two 512-row lanes served serially at a 19.4 ms kernel floor (40 ms period floor) plus
  ~30 ms/unit-layer parked on page-ins (mean 7 missing experts/request; run 19.6/26.1/47.5/79.6 ms at 0-2/3-6/7-12/13+ missing).
  Hub-only gaps (Engram/G0 prep 1.25 s + group boundaries 0.63 s) are 23% of the window while box 2 idles.
* **Paging is a partition problem.** `partition_box2` = box 1 owns each layer's hottest ~103-115 ids by DECODE pick mass, box 2 the
  rest (~270/layer); prefill touches ~every expert => box 2's set ~5.2-5.8K vs ~4.57K usable => ~1,146 re-reads/window.
* **Streaming cannot reach the target; residency can.** Streaming a box's half (72 GB/pass) needs window time >= 72 GB / disk rate:
  at 2K rows/s a 16K window needs 8.8 GB/s per box (> 4 GB/s measured, 4.2 box 1). Only a RESIDENT set removes steady-state reads.

## 1. Goals / non-goals (each phase behind a default-OFF knob, one-flip rollback; OFF = today's code path, I7)
P1 **Frozen prefill split** (hub-only). P2 **Layer-outer window driver** + window-input producer. P3 **Bounded refill prefetch**
(box-2 landing-rank fix; plan-based words; optional box-1 in-pipeline admission). P4 **Pair merge** with deferred submit. P0 **f16
MoE arm** into dispatch (last: it ties at <= ~16 members/expert, i.e. 512-row lanes of a ~200-expert share).
Non-goals: decode, DSpark, KV arena, CED/replay semantics, weight format, big windows, arena-tail lending.

## 2. Invariants (the review verified I1,I2,I3,I5,I6,I7; I4 was false as written and is restated with the full inventory)
I1. For a FIXED expert->box partition, layer-outer output (KV bytes, logits) is bit-identical to chunked prefill (oracle:
    `tests/lm_prefill_bitexact.rs`, run with the prefill split OFF for the P2 arm).
I2. For every layer, lane-units run in increasing position order (A_0 B_0 A_1 ...), exactly as chunked.
I3. pre(L,u) needs only post(L-1,u), layer L's KV/ring/compressor after pre(L,u-1); never post(L,u-1). (Verified r1.)
I4. Everything in lane/shared scratch is produced and consumed within one task's pre..post EXCEPT the items in sec 3.2 (relocated).
I4b. Reusable PINNED host staging per lane (`engram_pin` per-table slots if async Engram staging is adopted, `remote_ffn_moe_pin` +
    `remote_upload_done`, `rb_*` router-readback staging) is safe across units only because the lane's router-readback wait of task u
    completes every earlier async copy on `de.compute` before task u+2 rewrites the slot: keep the per-task readback wait and do not
    batch or skip it. (`prior_pin` is decode-only; `candidate_block_score/threshold` are written at layer 20 and consumed only by the
    replay call, which recomputes layer 20; `remote_ffn_moe_valid/_layer`, `remote_upload_pending`, `mhc_ffn_split_pending` are pre->post.)
I5. Lane scratch is shared with decode: the driver drains, stores, and restores compressor lending before ANY return.
I6. Checkpoints/snapshots only between windows. I7. Knobs unset => existing path byte-for-byte.

## 3. The layer-outer driver (P2)
### 3.1 Tasks and schedule
Window of W rows (`lm_plan_window`/`plan_chunk` cuts, lane cut = `plan_chunk`'s, assert kept). Lane-units in position order
u = 0..U-1 (A_0 B_0 A_1 ...; **U is NOT always 2n**: an empty lane B (`b_b == 0`, tiny or span-cut chunk) drops its unit, so U can be
odd and `lane(t mod 2)` flips parity between layers; harmless because all per-unit state is relocated to per-unit homes).
Tasks T[t] = (L,u), t = (L-a)*U + u, lane scratch = `lane(t mod 2)`, sync events `sync_events.layers[L]` (even) /
`sync_events_t1.layers[L]` (odd), as today. Schedule (the existing steady state generalised; at U=2 it IS the existing loop):

    warm:  pre(T[0]); pre(T[1])
    step:  for t in 2..N: post(T[t-2]); store_out(T[t-2]); pre(T[t])      (N = U*(b-a); store_in(T[t]) is the first thing pre(T[t]) does)
    drain: post(T[N-2]); post(T[N-1]); store_out both

Legality for U >= 2 (verified r1): pre(L,u) needs only post(L-1,u) = task t-U <= t-2. `sync_events` reuse is safe (post(L,u) enqueues
its wait before pre(L,u+2) re-records).

### 3.2 Relocated / per-task state inventory (incl. round-1 additions)
| state | today | layer-outer |
|---|---|---|
| residual [rows x HC_DIM], `hc_pre_carry` [rows x HC_MIX_DIM] | lane scratch, swapped residual<->residual_next | `lm_store` (82 KB/row, exists): store_in = async D2D into `bd.residual`/`bd.hc_pre_carry` (layers > a); store_out after post+swap. ~0.14 ms per 512-row task |
| layer-0 seed | host `hcs` -> `bd.residual`; carry reset via `hc_pre_onehot_rows` | unchanged for layer-0 tasks |
| compressor lending (KV-source -> reuse layers) | take before post_A(L)/pre_A(L+1), return after pre_B(L+1) | bracket per layer's pre-span: take before pre(L+1, first), return after pre(L+1, last); post never touches it; **`restore_compressor_lending()` at quantum entry and on every error path (guard)** |
| `bd.indexer_sel_saved` [rows x 512 i32], `indexer_nsparse_saved` | lane scratch, valid inside a group | per-window-row store (2 KB/row + nsparse): copy-OUT enqueued right after the source task's PRE (the selection is produced in pre, `fp:7439-7455`), copy-IN at reuse tasks, SKIPPED when the unit's `saved_store` flag is -1 |
| **`bd.indexer_saved_store` (host i32 flag, retracted to -1 when the source does not fire; read by `s2_reuse`)** | lane scratch; written `fp:7511-7515`, retracted `fp:7566-7568`, read `fp:6874-6880` | **per-lane-unit flag** `saved_store[u]` on the job (a `Vec<i32>`, U entries), written by the source task of unit u, read by the reuse tasks of unit u. Without it unit 0 would read unit 14's flag at layer 3 (fresh prompt, units with pos < ~2048 do not fire) |
| `pos_per_b` (device), `vis_a/vis_b`, tokens slice, `pos0` | once per range | per task: tiny async uploads (<= 4 KB) before store_in |
| Engram rows staging (`stage_engram_lane`, SYNC `copy_from_host` ~12.6 MB at layers 1 and 14) | per (lane, layer) | per task; make the layer-1/14 copy async from pinned (as decode slice 1e did) or it blocks the pipeline 2x per unit |
| `events.reset()` + perfetto flush (`emit_prefill_perfetto`) | per unit | **per task** (a whole quantum of stage events overflows the pool) |
| KV ring / raw SWA / compressor caches / index keys | `state.layers[L]`, positional | unchanged (I2) |
| `split`, `after_attn_hc`, `ffn_shared`, `ffn_input_norm`, `remote_*`, tickets | lane scratch pre->post | unchanged (I4) |
| pager / `b2_mirror` windows / prefetch queue / `expire_incoming` / `remote_set_phase_busy_poll` | per range call | per quantum (global, time-ordered) |
| MTP capture / `mtp_lane_cut` | decode / non-CED only | n/a (verified r1) |
(Reviewer: find what is STILL missing.)

### 3.3 Quantum and scheduler interface
`prefill_job_lm_pass(job, a, b)` runs layers [a,b) over ALL lane-units of the open window and returns, after drain+store,
the rows completed (W when b is the window's last layer, else 0). Default quantum `V41_LM_QUANTUM_LAYERS` = 4: ~2.1 s today
(4 x 8 x 66.5 ms), ~1.0 s at the P1 target; drain+store ~50 ms (~2-5%). Layer 20 (`KvSourceOnly`, `PreMoePhase::Skipped`) is a
one-layer quantum: pre only, no post, residual untouched (r1 verified). `lm_mid_group` / `V41_MS_FINISH_GROUP` (which also gates
the embed phase, `ms:421`) get explicit semantics: a quantum is atomic, and "window open, BETWEEN quanta" is a legal point for decode
and the embed phase (lanes drained, residuals in the store, lending restored); sticky open-window job (`ms:1287-1294`) kept; checkpoints only when no window is open (I6).

### 3.4 Replaces / keeps
Replaces `prefill_job_lm_unit`, `lm_groups` cuts and `pf_sent` for the new path; the group-major driver stays when
`V41_LM_LAYER_OUTER` is off (I7).

### 3.5 Replay capture
Today the last 128 rows ENTERING layer 20 are copied from the lanes (`fp:1380-1395`). Layer-outer: after post(19, *) the store
holds every row's layer-20 input, so the replay rows are read from `lm_store` rows [W-128, W) (residual + carry, host copy) at the
layer-20 quantum; token ids from the job. (Same bytes as today's lane read.)

### 3.6 Per-task hygiene
Per-task `events.reset()`; stage-event pool bounded per task; evtrace unit id per quantum (the pf2 join keys by seq: unaffected);
`dump_layer_pos` per task.

## 4. Frozen prefill split (P1)
Today: `home2 = partition_box2(layer, e)` in the pick loop (`fp:9616`) and in two OTHER prefill-path consumers: `lm_prefetch_words`
(`fp:870`) and `readahead_layer` (`fp:10296-10303`, `V41_PREFILL_READAHEAD`, default off, would warm the wrong box's page cache).
(NOT consumers, all arena/decode-only: the look-ahead builders `fp:9750/9789`, cache-prior mask `fp:8469`, substitution predicates
`fp:9224-9484`, `wants_for_box2` `fp:9837`.) **One object** `PrefillSplit` (per job, built at job start, long jobs only:
tokens > `V41_LONG_PREFILL_TOKENS`, `V41_PREFILL_SPLIT=1`), a bitmap `box2[layer][e]` for encoder layers (`layer < CED_DECODER_START`;
layer 20 excluded); decoder layers (replay) keep `partition_box2`. `pre_moe_route` has no job handle (5 call sites), so the object is a
process-global `PREFILL_SPLIT: RwLock<Option<Arc<PrefillSplit>>>` set by a guard around the quantum call (single scheduler thread;
decode/verify steps between quanta see `None`). `verify_routing_exactly_once` is unaffected (it checks `owns_eff` vs `remap`, both
produced by the same loop).
Construction (deterministic, frozen for the job even if `hot_set` refreshes mid-job), **sized from LIVE counts at job start**:
* box 1 = its decode-hot encoder head at job start (`hot_set::box1_owns`) UNION a hash filler of
  `filler = min(V41_PREFILL_B1_FILLER, (n_slots - lru_lo - owned_now) / 20)` ids/layer. Hub numbers (review r2): pool 5,425 (hub env
  `V41_PAGER_POOL_GB=95`) minus the idle dense window 384 (`V41_PAGER_WINDOWS=0` still reserves one, `ep:2064-2077`) = 5,041 LRU
  slots; decode-hot owned = `top` 103x40 = 4,120 or INTERLEAVE (live) 115x40 = 4,600; free = 921 / 441 => filler 46 / 22 per layer
  (NOT 64) so the displaced decode-hot set is ~0 only while filler <= free/20;
* box 2 = the rest: 235 / 247 per layer = 4,700 / 4,940 vs ~4,570 usable (103% / 108%) => **does not fit as is**. Fixes: (i) reclaim
  the idle dense window under the unified pool (allow `dense_windows = 0`, ~10 lines + a startup assert; AUDIT B12 calls it 7.2 GB wasted):
  hub 5,425 + box 2 4,570 = 9,995 slots vs 7,680 encoder experts + the hub's decoder-hot head 2,060-2,300 = 9,740-9,980 => fits with
  15-255 slack under `top`, ~0 under interleave; (ii) accept the residual: **expected steady re-reads = max(0, share2 - usable2) per
  window** (0 to ~400, vs 1,146 today), logged per job as share1/share2/usable2; (iii) grow the pool if hub RAM allows (~6 GB free: no).
  After the reclaim the live free slots are 5,425 - 4,120 = 1,305 (`top`) / 5,425 - 4,600 = 825 (interleave): filler 65 / 41 per layer,
  box-1 share 168 / 156, **share2 = 216 / 228 per layer = 4,320 / 4,560 <= 4,570** (the numbers the job-start log line prints). Un-reclaimed
  box-1 share is (103+46)/384 = 39% (`top`) or (115+22)/384 = 36% (interleave); reclaimed 44% / 41%: **36-44%**.
  Cold hot set (`box1_owns == None`): share1 = `hot_set::per_layer()` (or `B1_HOT_IL_PER_LAYER` under interleave), filler computed with
  `owned_now = per_layer x 40`, ALL chosen by hash (NOT `owned_now = 0`, which would give `n_slots/20` = 271/layer).
  The `Arc<PrefillSplit>` is cloned ONCE per `pre_moe_route` call, not locked per pick in the 384-wide loop.
  Note (review r3): prefill-class landing of frame-carried words is keyed on the request's ROW COUNT (`b > PIN_DECODE_MAX_ROWS` = 16,
  `re:8571-8596`), not the hub's phase flag; irrelevant for 512-row lanes, state it in the loopback test.
**Variant P1H "hub-heavy" (owner 10-10: why keep the decoder-hot head resident during a LONG prefill?).** It need not stay: decode does
not run during a prefill burst, replay touches only 128 rows, and in the unified LRU pool those ids are the LRU-oldest, so a filler
larger than the free slots simply EVICTS them (no special mechanism). Dropping the decoder-layer hot head (2,060-2,300 ids) frees
~2,100 slots + the dense window 384: filler up to ~150/layer ~ the balanced split of the MoE-leg model (sec 8: both legs ~16 ms,
-36% vs ~25 ms). The PRICE is the re-warm at the next decode burst: ~2,100 experts ~ 40 GB ~ 10 s of hub disk (~6 s with a Gen4-class
SSD), hottest-first (top-25/layer = ~50% of decode picks), during which decode steps take synchronous misses (6.8 ms each,
`ep:3446-3548`); burst cadence 120 s prefill / 30 s decode => maybe 15-25% of each burst's throughput while a long job is in flight
(inferred). Needs a hottest-first BULK async re-warm at decode entry (the hub only prewarms 2 interleave newcomers/step today,
`take_prewarm`, `ms:1667-1682`; box 2 already does release/restore for pins). Middle option P1M: keep the top ~50/layer decoder-hot
(~1,000 ids, ~70% of decode mass), drop the rest. Knob `V41_PREFILL_B1_FILLER` (conservative default = live free slots) and
`V41_PREFILL_KEEP_DECODER_HOT` (all / top-N / none). Gate: decode p90 over the first 60 s of each burst after a long job, and
prefill rows/s, both vs the conservative setting.

Cold fill (stated honestly): box 1 pages ONLY its filler synchronously (`ensure`; 22-46/layer x 20 = 440-920 experts ~8-17 GB, 2-4 s at
4.2 GB/s); box 2 fills its set (<= ~79 GB, 20-26 s) as today's windows 1-2 do. After each decode burst some prefill-class pages are
lost on both boxes and re-read: measure (gate).
Post-job: box 1's decode head is inside the prefill share so it is not evicted by construction; the filler overflow (if any) evicts
decoder-hot ids; gate = decode p90 over the first 60 s after a long job vs the same traffic before.
Numerics: which box sums which expert changes => f32 association only; golden/KLD gate for the partition change.

## 5. Bounded refill prefetch (P3) -- small k, no big windows
Purpose: refill the resident set after decode bursts and the cold start; not steady-state streaming (sec 0).
5a. **Box-2 landing-rank fix (BLOCKER r1; smaller than first described, r2).** Background landings call
    `pick_victim_any(region, global, band, cur_layer, want, &pinned, d.layer, ..)` (`re:5672`, also `re:5677`); `for_layer = d.layer`
    feeds `own_sweep_rank` (tier 3, long jobs, `re:4081-4083`) AND `sweep_rank` (tier 2/4 with `encoder_victims_first`, `re:4085`,
    `re:2861-2875`). Change: pass `cur_layer` as `for_layer`; then every `l >= cur` in the region is "ahead", i.e. the rule
    `(cur_layer, d.layer]` for free, and both rankings are repaired. Demand claims keep `for_layer = want_layer` (`re:3806`, `re:4227`)
    and cannot be starved (a landing is optional and dropped when no victim, `re:5674-5682`; the rank orders, never excludes). Other
    landing paths checked: nursery (LIKELY/decode only), restore (requeued in a prefill phase), partial completions (no victim),
    staging band off in prod. Needs a box-2 redeploy (two-box restart order) and a loopback test for the landing rank. To pin down in
    that test: (i) speculative words on a PREFILL-flagged frame land prefill-class via `prefetch_words_prefill` (`re:5163-5165`;
    `me_tiered` needs `prefill_landing = d.stage || d.prefill`, `re:5649`) and not decode-class (else they rank in no tier); (ii)
    drop the hub's `pf_sent` dedup for P3 (it counts a TAKEN word as sent, but box 2 DROPS words with no free staging set,
    `re:5116-5118`): re-derive each layer's words from the mirror. Also repairs `V41_LM_PREFETCH`.
5b. **Plan-based words** from `PrefillSplit` for layers L+1..L+k (k = 2-3), carried on request frames (queue cap 4,096, per-request
    take cap, unsent dropped at unit end). A 4096-row window = 8 sub-chunks x 2 lanes = 16 requests per layer, so ~12 words/request x
    16 = 192 words/layer (r1 verified box 2 can start ~80 reads/request); raise the take cap for the refill case. Words filtered by the
    box-2 residency mirror as today.
5c. **Box-1 in-pipeline admission (optional, own gate).** Victim search that skips the other lane's in-flight picks (as the code
    itself proposes at `fp:10169-10178`) + a per-task drain; until then P1's box-1 fill is the synchronous bounded stall of sec 4,
    and `b1_page_misses` keeps short-circuiting `b1_prefetch`.
Lookahead arithmetic (inferred): k ~ 2-3 layers; resident need ~(k+1) x ~205 experts x 18.8 MB ~ 11-16 GB per box.

## 6. Window-input producer (P2b)
The 23% hub-only gaps are input prep (embeddings->HC, Engram gather: disk-latency-bound) between units. Inputs depend only on token ids.
A **continuous producer thread per job** (not scheduler-tick driven: `EngramAhead` does one block per tick, ~5 ticks/window with
quanta) builds window n+1's inputs during window n's compute into a ring (host RAM ~0.6 GB/job at 4K: HC 336 MB + ~200 MB Engram +
kept rows, against ~6 GB available; HC is built synchronously today on the scheduler thread, `ms:2703-2742`).
Ownership refactor required (review r2): (i) `gather_prefill_engram` takes `&EngramCtx` (`ms:2749`) living in `WorkerState.engram`
(`engine_worker.rs:847`), which decode borrows MUTABLY (`rows_for` mutates `compressed`, `engine_worker.rs:65-83`); today's scoped
thread works only because it ends inside the tick. Split into `Arc<EngramShared{hasher, tables, st}>` (`st` is its own checkpoint
handle for exactly this) + the mutable `compressed`; `token_embd_bytes: Vec<u8>` (`engine_worker.rs:849`) -> `Arc<[u8]>`; image rows
(`EncodedImages::row_at`, `engine_worker.rs:5167`) shared or copied. (ii) Contract: produce by ABSOLUTE token range
`[chunk_start, chunk_start + lm_rows)`; the window takes a PREFIX of it (`lm_open_window` can halve at open, `fp:1189-1193`, and spans
shorten it, `fp:934-936`); restored suffixes are already absolute (`ms:2712`); tokens are complete at `PrefillJob::new`. (iii)
Lifecycle: cancel drops the job (`ms:1296-1334`); the thread owns only Arcs + a channel; one producer, bound to the sticky job.
Slice P2b-1: producer thread + whole-window `set_window_inputs` (replaces `next_chunk_range`'s group-0 sub-chunk contract,
`fp:1122-1132`, and per-sub-chunk `set_chunk_inputs`). Slice P2b-2: upload N_EMBD rows (20 KB) and replicate x4 on the device at the
layer-0 seed (the embedding table is host bytes and the dGPU has ~0.5 GB free, so it cannot move to VRAM); this also removes today's
1,024 SYNCHRONOUS per-row `copy_from_host` at the layer-0 seed (`fp:2674-2686`), a per-task cost never isolated in the gap analysis.
First window of a job pays the prep once (~1.5 s). **P2a and P2b land together or are attributed separately**: P2a alone moves the
whole window's input prep in front of the first quantum.

## 7. Pair merge (P4) -- deferred submit, golden gate
Pairs `pre(2j); pre(2j+1); post(2j-2); post(2j-1)` need the two requests to reach box 2 within `V41_B2_MERGE_WAIT_US` (400 us); today
the partner follows its own dGPU chain + router readback drain (>= 10 ms later) so no merge fires. Options: (a) split `pre_moe_route`
so the first request's `submit_dispatch` is DEFERRED until the second is routed and submitted back to back (new plumbing for xq/
sel/ew/ticket/PinNote/evtrace per lane); (b) raise the wait to ~20 ms (box 2 idles per pair). +2 lane scratch sets (~290 MiB) for
depth 4. Gate = golden/KLD (row-composition independence of the WMMA kernels is UNVERIFIED), not bitexact. Do after P1-P3 if box 2's
19 ms floor is still the pole.

## 8. Phases, gates, estimates (inferred; steady window 8.02 s = 511 rows/s today)
| phase | change | gate | rollback | estimate (steady window / rows/s) |
|---|---|---|---|---|
| P1 | frozen split, all consumers via `PrefillSplit` (long jobs) | `verify_routing_exactly_once`; golden/KLD vs old partition; one long-job A/B with the pf2 evtrace analysis (`n_missing`, window wall); hub `ensure` stall in window 1; decode p90 of the FOLLOWING burst | knob | ~5.0-5.6 s / 730-820 |
| P2a | layer-outer driver, W=4096 (+ fixes 4,10,12,15) | extended `lm_prefill_bitexact` (split OFF) with a layer-outer arm; live-window golden with box 2 attached (per-task `remote_ticket` ordering) | knob | -0.7 s (boundaries+fill/drain) |
| P2b | window-input producer | no output change; rows/s | knob | -1.0..-1.2 s |
| P3 | landing-rank fix + refill words (+5c optional) | loopback landing-rank test; bitexact unaffected; post-decode-burst re-read counts | knobs; box-2 redeploy | robustness (refill) |
| P4 | deferred-submit pair merge | golden/KLD | knob | box-2 floor -25%: -10..-15% |
| P0 | f16 MoE arm | oracle + golden/KLD | knob | pays only with 7b/bigger batches |
P1 estimate is 5.0-5.6 s only after the dense-window reclaim of sec 4; without it box 2's share overflows by 130-400 experts (~150-400
re-reads/window) and the window is nearer 5.5-6.0 s. Target after P1+P2a+P2b: ~3.6-4.1 s / 1.0-1.1K rows/s (inferred).

**MoE-leg model (owner, 10-10): leg time = max(hub iGPU MoE, box-2 queue + run + wire) per lane-layer.** Measured (16:44 job, 6,008
prefill requests of 512 rows): box 2 receives n_sel p50 1,344 of 3,072 picks = **44% of picks** (hub already ~56%), over n_distinct p50
147 experts (~9 members each; ~270 owned). Both boxes are weight-bandwidth-bound at these member counts, so leg time ~ distinct
experts touched x ~0.09-0.125 ms: hub ~115 hot experts ~ 10 ms (matches the 10.4 ms reply->submit), box 2 147 ~ 18.4 ms floor + 7 wire
(= ~25 ms; lane 1 also queues ~8.6 ms behind lane 0). Moving f (box-2-owned, ~54%-touched) experts/layer to the hub: hub ~ 9.7 +
0.049 f ms, box 2 leg ~ (147 - 0.54 f) x 0.125 + 7 ms. Residency-capped f = 46: box 2 leg 22.3 ms (-12%, -17% on the serial pair);
balanced f ~ 135/layer (hub 250, box 2 135): both ~16.3 ms (-36%). So the hub's RESIDENT-EXPERT CAPACITY is the lever (f is capped
at ~22-46 by free slots, ~70 with the dense window; ~135 would need ~2,700 more hub-resident encoder experts = ~51 GB, i.e. dropping the
decoder-hot head during long prefills, which costs ~43 GB of re-warm per decode burst: not viable at a 120 s/30 s burst cadence). The
cost-per-expert coefficients are from two points (inferred). CORRECTION to earlier text in these docs: "box 2 does ~70% of the work"
was a count of EXPERTS; by picks it is 44%.

**Is the hub's compute used? (owner question 10-10)** Yes, but at ~22% busy: 1.75 s of the 8.02 s window (hub work ~21 ms per
unit-layer vs a ~66 ms period; blocked on box 2 53%, gaps 24%). Its dGPU runs attention/dense for every row; its iGPU runs only the
experts it owns (~30% today: the decode-hot head), and what limits that share is hub RESIDENCY (5,041 LRU slots, 441-921 free beyond the
decode-hot head), not compute. P1 raises the hub's expert share to 36-44%; a 50/50 EXPERT split (the two iGPUs are identical) would need +500-1,100
more hub-resident encoder experts, i.e. hub streaming from its disk (the 9100 Pro helps) or temporarily dropping the decoder-hot head
and re-warming. At ~1K rows/s the hub is fine (~50% busy, inferred); the 2K rows/s target needs <= ~12 ms of hub work per unit-layer (inferred: half the 25.6 ms period) vs ~21 ms now,
and the split between dGPU chain and iGPU MoE inside those 21 ms is UNKNOWN (device timing is off in the trace: enable
`V41_EVTRACE_DEV` for a short prefill-only window). So 2K additionally needs the box-2 kernel floor AND the hub's per-unit-layer work to
roughly halve; it is not reachable by P1-P4 alone.

## 9. Test plan
* CPU: task-schedule simulator: random U in 2..16 INCLUDING odd U and empty lane B, quantum sizes, layers: every (L,u) once, I2 order,
  pre(t) strictly after post(t-2), residual flow, per-unit index-selection/`saved_store` flow (reuse reads its own unit's source write),
  compressor bracket balanced, drain leaves nothing in flight, window open->quanta->close, replay rows = store rows [W-128, W).
* GPU (server down): `lm_prefill_bitexact` + `layer_outer` arm (fresh / multi-window / restored / lazy+foreign; split OFF); W stays 4096 (an 8192 case only if the existing halve-on-OOM store allows it).
  The test builds its own engine/pager and has no box-2 client, so add a loopback remote or a live-window golden with box 2 attached
  (reference_gpu_gate_windows: box 2 must be attached).
* Live A/B: pf2 evtrace join, steady-window rows/s, decode p90 during and after a prefill; box-2 `n_missing`; box-1 `ensure` stalls.
* P3: loopback test for the landing rank; pin/restore tests unchanged (decode-keyed, r1 verified).

## 10. Risks / open questions (round 2: attack these)
R1 drain cost and hidden per-task host syncs (`drain_before_ensure` `fp:10184-10189`; sync Engram copy).
R2 completeness of the 3.2 inventory beyond `indexer_saved_store` (every field of `BatchDgpuScratch`/`BatchIgpuScratch` that survives a task).
R3 P1 capacity margin 90-94% and the box-2 landing/tier behaviour when the set barely fits; filler sizing if the hub pool is not 5,425.
R4 image spans / lane cuts under layer-outer (units reproduce `plan_chunk`).
R5 hub `ensure` synchronous stalls (window 1 and after decode bursts).
R6 scheduler: FINISH_GROUP removal, other prefill jobs, `lm_store` lifetime, producer thread lifecycle and RAM.
R7 numerics: partition change (golden/KLD), box-2 f16 replies still ungated.
R8 whether the dGPU chain / hub MoE becomes the pole after P1 (unknown).
R9 post-job decode warmth: is "head inside the prefill share" enough, or does decode-class LRU on the hub still evict it during the job
   (decode steps interleaved every 120 s/30 s)?
R10 replay/decoder layers unaffected by the encoder split (verify).
