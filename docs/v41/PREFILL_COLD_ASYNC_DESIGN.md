# Prefill cold reads on box 1, off the hub thread (design, stage 2, 2026-10-12)

Status: rev 3 (review r1 REVISE -> rev 2; r2 APPROVE WITH CHANGES -> rev 3: 2 MAJOR + 9 minor/nit folded in, section 10). Owner 10-12: "ttft fix" -> "let's write up
stage 2" -> "reviewer". Hub only; box 2 untouched.

## 0. Why (measured 10-12, timing metadata only)

- A turn's TTFT is dominated by its FIRST prefill window: 82% of the hub's wait on box 2 over 34,560 prefill requests
  (10:12-20:57 UTC) was in each job's first window; short turns (<= 1,024 rows) spend 3.9 of 4.6 s there blocked on box 2
  reading cold experts (decode evicts ~87% of the next turn's prefill set between turns; owner: no partitioned pool).
- Box 1 can read as fast as box 2: the E100 (box 1's model drive since 20:52 UTC, no dm-crypt) does 4.42 GB/s with 8 x
  6 MiB O_DIRECT readers; box 2's measured prefill read rate is ~4.4 GB/s over two drives. Box 1's drive idles in first
  windows.
- Stages 0/1 (`V41_PREFILL_COLD_B1`, `V41_PREFILL_PIN_INFLIGHT`) read box 1's share INSIDE `ensure`: the hub's box-2 wait
  fell 82% -> 6-13% but first windows got SLOWER (4-unit p50 6.82 -> 9.87 s with pins). `ensure` reads synchronously on the
  hub thread; meanwhile the hub issues nothing and the dGPU and box 2 starve. Production since ~21:45 UTC: free-only
  (`FRAC 0`) + pins on.

## 1. Idea: issue early at route, join at the NEXT unit's launch (review r1 simplification)

Op order (lm_schedule.rs:54-71): Pre(t) = chain, route, prep, launch; then Post(t-1); then Pre(t+1); then Post(t).
A cold pick given to box 1 becomes LATE: excluded from MoE(t), its read queued to background readers at route(t). The
join happens at the START of `pre_moe_launch(t+1)` -- before MoE(t+1) is queued -- so the late pass runs on `ie.compute`
right behind MoE(t) with nothing queued in between; `si.d_xq_q8k` / `d_x16` still hold unit t's activations (no copy).
MoE(t)'s `moe_done` record + peer push move to the join: the late result is `vec_add`ed into `bi.ffn_moe(t)` on the
iGPU, then ONE push, `moe_arrived` as today. Post is unchanged. The reads get route(t) .. launch(t+1): prep(t), launch(t),
Post(t-1) (incl. its box-2 wait), chain + route + prep of t+1. A join that still waits only delays MoE(t+1), whose box-2
leg is already submitted (route precedes launch). The join runs BEFORE launch(t+1)'s
`ie.compute.wait_event(&sev.selected_pushed)` (~11468) so late(t) does not also wait for unit t+1's chain pushes (r2 #7).
The LAST task of a pass has no t+1: its join runs in the exec closure's `Op::Post` arm (forward_prefill.rs ~1898-1903,
which holds `lanes`, `si`, the pager and the weights) BEFORE `forward_layer_post_moe_v2` (r2 #5). Late is enabled only on
the sparse pool path (`sparse_resid_layer`, absolute slots, group bound = pool; asserted, r2 #8).
State across the gap: a per-lane PENDING-LATE record next to `bi` (layer, rows, late ids + slots, ticket, deferred-push
flag), since `PreMoeCarry` dies with `forward_layer_pre_moe_v2` (r2 #6).

## 2. Pieces

### 2.1 Route (`pre_moe_route`, `cold_route` block)
- In-flight registry (r1 #4): `(layer, e) -> ticket` for late reads not yet landed. A cold pick already in flight is
  taken LATE by this unit too, with no new read (counted `cold_late_shared`); `cold_b1_take` takes those first, then new
  ones up to FRAC/MAX and the per-layer budget (2.4).
- Late ids go into the carry's `late: Vec<u32>`: in neither `ids` nor `extra_remote`. Box-2 request mask and
  `remote_sel_override` stay `extra_remote` only (r1 #5).
- Main-pass EXCLUSION (r2 #1): `ensure` resets the remap to the self-map, so prep's exclusion closure becomes
  `owns_eff[e] || late[e]` (`mark_remote_after_ensure`); the box-2 mask stays `extra_remote`.
- Three-way exactly-once (r1 #5), run in PREP where `verify_routing_exactly_once` runs (~11075; the main remap does not
  exist at route time, r2 #1): every distinct pick in exactly one of {main remap < 0, `extra_remote`, `late`}; at the join,
  every late id's private-remap entry is negative and resolves to its landed slot. Fails closed.
- A SHARED late id (registry hit) that is no longer resident at this unit's join (evicted by `ensure(t+2)` in between)
  takes the read-failure path (synchronous read + land), not a step error (r2 #4).
- Reads: `pg.read_async(layer, &new_late) -> ticket` before the box-2 submit (both disks start together).

### 2.2 Late pass (at the join)
0. Unit t+1's slots are PINNED at the end of its prep (`pin_inflight` on `c.ids` right after `ensure(t+1)`), so the
   join's claims cannot evict a slot MoE(t+1) is about to read; launch(t+1)'s own pin then only extends it (r2 #2). Same
   for the read-failure claim.
1. `pg.join(ticket)`: wait for the reads (normally done); claim slots with a fresh `call_tick` (2.4); queue all repacks
   on the repack stream with NO per-expert `st.synchronize()` (r1 #6); record one `late_landed` event; `ie.compute`
   waits on it. Staging reuse is gated by that event (query).
2. PRIVATE late remap (r1 #1, #2): a per-lane device table: late id -> `-(slot+1)`, every other id -> 0 (not mine), cap
   `N_EXPERT_USED` (as `split_cap`, forward_prefill.rs:9501), uploaded through its own pinned stage on `ie.compute`
   (a done-event gates the stage's rewrite, as expert_pager.rs:2359-2373; r2 #9).
   The pager's shared `remap` / `remap_dev` / `cur_layer` / `remap_dirty` are never touched.
3. Late MoE on `ie.compute`: the same group builder / gate-up / q8k / down / reduce as MoE(t), over the lane's REAL
   `bi.d_selected` / `bi.d_ew` with the private remap (the builder AND the reduce both skip non-late picks: no -1 masking,
   which the reduce would still sum, r1 #1), into a per-lane `bi.late_moe` [rows x N_EMBD]. The reduce OVERWRITES
   (`out = acc`): rows without late picks come out 0, no zeroing. Debug assert: `d_selected` holds no negative pick (the
   reduce would take it in both passes, r2 #11).
4. `vec_add(bi.ffn_moe, bi.late_moe)` on `ie.compute`; then MoE(t)'s deferred `moe_done` record, the push, `moe_arrived`.
   Add order: (main + late) + shared + remote.
5. `pg.pin_inflight(layer, late_slots, ie.compute)` right after the late launch, in the same host step as the claims.
No late picks for unit t -> today's path exactly (moe_done + push stay in launch(t)).

### 2.3 Background reads (`ExpertPager::read_async`)
- Reader pool `V41_PREFILL_COLD_READERS` (static, default 8: the E100 optimum, 16 is slower), O_DIRECT into TIGHT pinned
  stages (nibbles + scales of the coalesced layout, ~19 MB/expert; an expert stored in the per-role layout is simply not
  taken late, r1 #7). `V41_PREFILL_COLD_STAGES` (static, default 16 = ~300 MB pinned; 0 disables); no free stage -> the
  pick stays box 2's (never waits). Peak `MemAvailable` logged once per pass beside the Engram cache (2 GB).
- Read failure (r1 #8): a synchronous read into the same stage at the join, then the normal landing (never `ensure()`).
- Pass end / error (r1 #9, r2 #10): drop QUEUED reads, WAIT for reads in progress (a pread cannot be aborted), then
  `unpin_all_wait` and clear the registry; no reader writes a stage and no claim lands after the pass.

### 2.4 Landing and budget
- Late claims skip pinned slots and hot-set-OWNED slots (`hot_set::box1_owns == Some(true)`); nothing eligible -> plain
  LRU (r1 #10). Decode misses after prefill are attributed: late claims that evicted an owned slot are counted
  (`cold_late_owned_evict`), separate from the main `ensure`'s.
- Budget `V41_PREFILL_COLD_LATE_MAX_LAYER` (default 20 ~ spare slots per layer) per layer per window.

## 3. Gates (r1 Q7)
- EXACTNESS (debug knob `V41_PREFILL_COLD_FORCE_LATE`): a fixed late set made of BOX-1-RESIDENT picks (no reads), late pass
  vs the same picks in the main pass, per-layer `ffn_moe_recv` compared on the dev path: per-pick partials identical
  (same kernels, same row count -> same `_rows` arm), only f32 reassociation in the sum -> tight relative bound; plus the
  three-way exactly-once check and a group-audit count on the late builder.
- End gate: golden KLD on fixed prompts (late on vs off), within the split null.
- Live: `pin_smoke.py` (synthetic, arithmetic answers) off vs on before any A/B.

## 4. Knobs (live unless noted)
| knob | default | meaning |
|---|---|---|
| `V41_PREFILL_COLD_ASYNC` | off | late reads instead of `ensure` for the FRAC/MAX share (needs `V41_PREFILL_COLD_B1`, `V41_PREFILL_PIN_INFLIGHT`) |
| `V41_PREFILL_COLD_B1_FRAC` / `_MAX` | 0.5 / 8 when async | share / cap of NEW late reads per request (8 so 16 stages serve both lanes, r2 #3) |
| `V41_PREFILL_COLD_LATE_MAX_LAYER` | 20 | late reads per layer per window |
| `V41_PREFILL_COLD_FORCE_LATE` | off | gate only: resident picks through the late pass |
| `V41_PREFILL_COLD_READERS` (static) | 8 | reader threads |
| `V41_PREFILL_COLD_STAGES` (static) | 16 | tight pinned stages (~19 MB each) |

## 5. Expected effect (inferred)
First-window request ~50 cold experts: box 2 reads them at ~4.4 GB/s, mostly exposed. Half read late on box 1 in parallel
cuts box 2's reads ~2x; the hub thread adds a join that normally finds the reads done and a late pass whose cost does NOT
scale with late picks alone (post-quantize over all rows x slots, reduce over all rows, devcount grid; r1 #11): measured
per late lane-layer before trusting the estimate. First-window blocked 3.6 -> ~1.8-2.2 s; short turns 4.3 -> ~2.7-3.2 s.

## 6. Measurement
`first_windows.py` (per job first-window wall + hub box-2 wait by units) vs the 10-12 baseline (2-unit 4.27 s, 4-unit
6.82 s), alternating on/off blocks of real traffic; counters `cold_late`, `cold_late_shared`, `cold_late_join_wait_ms`,
`cold_late_fallback`, `cold_late_owned_evict`, late-pass ms (evtrace) on the "stream admitted" line.

## 7. Risks
| risk | mitigation |
|---|---|
| late output wrong (stale partials, wrong remap) | private remap + real selection (2.2), exactness gate (3) |
| a late expert evicted under its kernel | pinned in the same host step as its claims |
| both lanes read the same expert | in-flight registry (2.1) |
| join waits | counted; FRAC/MAX/budget live; a wait delays only MoE(t+1) |
| pinned host RAM | tight stages, 16 by default, static, MemAvailable logged |
| `V41_LM_PREFETCH` words for an expert box 1 reads late (r1 #12) | knob off in prod; route skips late ids when building words |

## 8. Slices
1. `read_async` / `join` / cancel + tight stages + in-flight registry; host tests (tickets, failure path, no-stage refusal,
   pass-end cancel).
2. Route late class + three-way exactly-once + carry plumbing (late empty -> today's path exactly; host test).
3. Private late remap + late pass + deferred push at the next launch; last-task Post join.
4. Exactness gate (FORCE_LATE, dev path / scratch hub) + golden KLD.
5. Deploy default off; `pin_smoke`; flip live; on/off blocks.

## 9. Review round 1 (2026-10-12, REVISE): where each finding went
1 masked -1 summed by the reduce -> 2.2 step 3 (real selection + private remap). 2 shared remap corrupted -> 2.2 step 2.
3 late pass behind MoE(t+1) -> 1 (join at launch(t+1)). 4 duplicate reads across lanes -> 2.1 registry. 5 vacuous verify
-> 2.1 three-way check. 6 per-expert repack sync -> 2.2 step 1. 7 staging size / RAM -> 2.3 tight stages, 16. 8 fallback
must not call `ensure` -> 2.3. 9 tickets at pass end -> 2.3. 10 owned-filter fallback + attribution -> 2.4. 11 late-pass
cost unmeasured -> 5. 12 join placement / LM_PREFETCH words -> 1, 7. Q7 exactness gate -> 3.

## 10. Review round 2 (2026-10-12, APPROVE WITH CHANGES): where each finding went
1 main pass must exclude late ids + check in prep -> 2.1. 2 join claims vs unit t+1's unpinned slots -> 2.2 step 0 (pin
at the end of prep). 3 stages vs MAX -> MAX 8 (4). 4 shared id evicted before its join -> 2.1 read-failure path. 5 last
task's join in `Op::Post` -> 1. 6 per-lane pending-late record -> 1. 7 join before `selected_pushed` -> 1. 8 sparse-only
-> 1. 9 remap stage done-event -> 2.2 step 2. 10 cancel = drop queued + wait running -> 2.3. 11 negative-pick assert ->
2.2 step 3. Verified safe by r2: the deferred push/moe_done (no wait on unit t between launch(t) and launch(t+1); `sev`
reuse ordered for U >= 3), builder/devcount/reduce over the private remap, pass/window boundaries.

## 11. Range passes outside windows (`V41_PREFILL_COLD_ASYNC_RANGE`, 2026-10-14)

Why: the 10-14 TTFT analysis (1,251 jobs) found short prefills paging-bound: for < 2K fresh tokens, 2.71 s (p50) of the
5.64 s TTFT is box-1 SYNCHRONOUS reads, ~620 of the ~760 misses per job the `V41_PREFILL_COLD_B1` share. Those reads
happen in the passes that are NOT layer-outer windows -- a short job's chunked encoder chunks (`prefill_job_chunk`,
layers 0-20 `KvSourceOnly`) and EVERY job's CED replay (`prefill_job_finish`, layers 20-39) -- which run through
`pipelined_range_impl`, where stage 2 never applied.

What: `pipelined_range_impl` takes `late_ok` (true only from those two call sites; every public wrapper passes false).
With the knob on (plus `V41_PREFILL_PIN_INFLIGHT`, running late readers, no prefill hot split in the range) it runs
`pipelined_range_late` instead of its own loop: the SAME op order (warm-up `pre_A pre_B`, then `post_A(L) pre_A(L+1)
post_B(L) pre_B(L+1)`, cool-down) -- which is `lm_schedule::ops` at U = 2 -- with each pre split into its four phases so
a unit with late ids parks its carry, joined after the next unit's prep and before its launch (section 1), or at the
start of its own post when no unit follows first (lane B empty; the last lane-layer). Setup / teardown are the window
pass's: one iGPU sync, the pin pass, `LATE_PASS` + the per-layer budget (per range pass), cancel + unpin on every exit.
Route, the late join, the private remap and the late MoE pass are unchanged code.

Gate: `tests/late_exactness.rs::late_pass_matches_main_pass_range` -- FORCE_LATE resident picks in the encoder chunks
only (verdict at encoder layer 0) and in the replay only (verdict at replay layer 20, with the encoder bit-identical to
the reference's: its KV and its layer-0 MoE), each with a bit-identical null and a dropped-late negative control.

## 12. The replay's late reads, issued from the encoder's tail (`V41_PREFILL_EARLY_REPLAY`, 2026-10-14)

Owner 10-14: cut TTFT by hiding the CED replay's expert reads under the encoder; "an extension of prefetch: on the last
prefill batch, or the last encoder layer of a replay, we know we'll need decoder layers next". Hard rule: what is
computed does not change -- only WHEN bytes are read; knob on / off bit-identical.

### 12.1 Measured (10-14, short jobs n=64; hub log + evtrace, clock-corrected 1.5-1.7 s)
- Replay 1.24 s p50, late JOIN WAIT 0.85 s p50 (0.8-1.0). ~320 new late reads per replay (16/layer = MAX 8 x 2 lanes),
  ~6 GB. Box-1 drives during the replay: 4.45 GB/s, **E100 98% busy**, YMTC 41%: the replay is box-1-DISK bound.
- Box 2 pages ~0.75 s (p50 sum of `page_us` over the replay's 40 requests), today hidden under box 1's joins: the
  replay's floor with box 1's reads fully hidden is ~0.8-0.9 s, not ~0.45 s (box 2 becomes co-critical).
- Encoder (range pass): 3.1 s p50, 1.77 GB/s, E100 48% / YMTC 24% busy: the slack is in the encoder.
- What box 1 reads in the replay is NOT its home set: box-1-home = decode's owned hot set (113/layer), kept resident
  (`V41_B1_PROTECT_OWNED` + rewarm; owned-not-resident after a prefill p50 1-2, p90 26-65 per JOB). The 0.85 s are the
  LATE picks: box-2-home experts box 2 does not hold (mirror), `cold_b1_take` of them read by box 1.

### 12.2 What can be read early, exactly
- A replay layer's picks are known only at its route (layer L's router reads layer L-1's output): "read all 20 decoder
  layers once layer 19 is done" is impossible, and issuing earlier inside the replay cannot help a saturated E100.
  Bytes must move OUT of the replay, into the encoder's tail, so the replay's late set must be PREDICTED.
- Predictor: decayed counts of the replay's own picks per (layer, expert) (x4/5 per replay), ranked among the experts
  COLD right now exactly as the replay's route defines it (box-2 home, mirror says box 2 does not hold it, not on box 1).
  Offline (10-12 pick trace, 143 replays, two-box residency proxy) recall of the late set at R per layer: R=8 0.26,
  R=16 0.43, R=24 0.56, R=32 0.66 (random 0.08 / 0.15 / 0.23 / 0.31). Precision at R=16 ~0.4.
- Plus the prefill read-ahead's own set (`readahead_layer`'s predicate: box-1-home, not resident) for decoder layers,
  landed VISIBLE (the main pass reads box-1-home picks whatever their residency) -- small, see 12.1.

### 12.3 Mechanism (extension of the late readers + read-ahead)
- Reads: the late readers (`LateReader`), a second, LOWER-priority lane in their queue (`replay_early::ReadQueue`):
  a late (demand) job always pops first; at most `V41_PREFILL_EARLY_READERS` (2) early reads run during the encoder,
  `V41_PREFILL_EARLY_READERS_REPLAY` (1) during the replay. Own pinned stages (`V41_PREFILL_EARLY_STAGES`, static,
  default 0 = feature absent): a late read's stage acceptance -- hence the late set -- never sees them.
- TAIL START: the book opens when the encoder's remaining time, from this job's measured per-layer rate, drops to
  `V41_PREFILL_EARLY_LEAD_MS` (1000, ~6 encoder layers of a short job), at the latest at its last MoE layer; in the job's
  last chunk / last window only. LAYER-AHEAD: only layers below (first replay layer not yet routed) +
  `V41_PREFILL_EARLY_DEPTH` (6) are issued; the window slides as the replay routes (`early_settle_below`).
- SLOT-AWARE cap: early reads in flight + landed-unused <= min(`V41_PREFILL_EARLY_MAX` 160, box 1's free non-owned
  slots at the open - `V41_PREFILL_EARLY_RESERVE` 400 = one replay's late budget). Landed early experts sit in PINNED
  HIDDEN slots (key = `hidden_key`, `is_resident` false): the encoder's later chunks cannot evict them, routing cannot
  see them. Landing only in pin passes, never evicting a slot used since the pass started or an owned one, no fallback.
- USE: the replay's `late_read(layer, ids)` -- ids chosen exactly as with the knob off -- finds an early read of a pick
  (landed / arrived / still reading) and ADOPTS it: registered in flight like a late read (the other lane shares it, as
  off), no new read, no stage. The join publishes a landed hidden slot (the point where the off path lands the same
  expert; one repack-stream event covers its earlier repack) or lands an adopted result like its own.
- END: layers both lanes routed release their unused hidden slots (wasted); the replay's end closes the book; a job
  that never reaches its replay expires after 60 s (`ensure`), another job's tail closes it.

### 12.4 Exactness
The late set, the late pass, its kernels, the landed bytes (same read, same repack) are the off path's. Classification
inputs are untouched: hidden slots are invisible to `is_resident`; adoption happens only where the off path reads.
Residual: hidden slots are claimed from LRU victims older than the pass, which can shift a LATER residency-dependent
`cold_free` choice as any pool churn does (box 2's mirror is timing-dependent anyway; production classification is not
run-to-run deterministic). Gate (`tests/late_exactness.rs::early_reads_match_off_replay`): deterministic late set
(`COLD_B1=off`, `FORCE_LATE=N` in the replay, plan = that set via `V41_PREFILL_EARLY_GATE`), every arm from an empty
pool: null bit-identical; early-on arms bit-identical to off (replay MoE captures, logits, seed rows, KV of all 40
layers) with early reads USED; negative control (late pass dropped) must differ. Soak:
`early_replay_soak_prod_selection` (production selection, pairs of runs, reads used, nothing left pinned / held).

### 12.5 Expected effect
At R=16 / layer with a 1 s lead at 2 readers (~1-1.6 GB before the replay), ~40% of layers 20-25's late bytes are
already read: ~0.15-0.25 s of the 0.85 s join wait hidden (~0.3-0.4 s at most: box 2's ~0.8 s paging floor). Early
reads during the replay mostly move bytes the E100 reads anyway (hits) and add the misses' bytes: A/B
`V41_PREFILL_EARLY_READERS_REPLAY` 0 vs 1. Waste ~55-60% of early reads (~1 GB disk, <= 160 slots held, released per
layer). Judge on per-job `replay_join_wait_ms` DOWN without `enc_join_wait_ms` UP (`prefill_job_replay` line, with
`early_*` counters; both arms log them once early stages are configured).
