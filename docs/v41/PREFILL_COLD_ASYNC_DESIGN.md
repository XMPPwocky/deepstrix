# Prefill cold reads on box 1, off the hub thread (design, stage 2, 2026-10-12)

Status: rev 2 (review round 1: REVISE -> all 12 findings folded in, section 9). Owner 10-12: "ttft fix" -> "let's write up
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
leg is already submitted (route precedes launch). The last task of a pass joins in its own Post (no t+1).

## 2. Pieces

### 2.1 Route (`pre_moe_route`, `cold_route` block)
- In-flight registry (r1 #4): `(layer, e) -> ticket` for late reads not yet landed. A cold pick already in flight is
  taken LATE by this unit too, with no new read (counted `cold_late_shared`); `cold_b1_take` takes those first, then new
  ones up to FRAC/MAX and the per-layer budget (2.4).
- Late ids go into the carry's `late: Vec<u32>`: in neither `ids` nor `extra_remote`. Box-2 request mask and
  `remote_sel_override` stay `extra_remote` only (r1 #5).
- Three-way exactly-once (r1 #5): every distinct pick is in exactly one of {main remap < 0, `extra_remote`, `late`}; at the
  join, every late id's private-remap entry is negative and resolves to its landed slot. Fails closed (step error).
- Reads: `pg.read_async(layer, &new_late) -> ticket` before the box-2 submit (both disks start together).

### 2.2 Late pass (at the join)
1. `pg.join(ticket)`: wait for the reads (normally done); claim slots with a fresh `call_tick` (2.4); queue all repacks
   on the repack stream with NO per-expert `st.synchronize()` (r1 #6); record one `late_landed` event; `ie.compute`
   waits on it. Staging reuse is gated by that event (query).
2. PRIVATE late remap (r1 #1, #2): a per-lane device table: late id -> `-(slot+1)`, every other id -> 0 (not mine), cap
   `N_EXPERT_USED` (as `split_cap`, forward_prefill.rs:9501), uploaded through its own pinned stage on `ie.compute`.
   The pager's shared `remap` / `remap_dev` / `cur_layer` / `remap_dirty` are never touched.
3. Late MoE on `ie.compute`: the same group builder / gate-up / q8k / down / reduce as MoE(t), over the lane's REAL
   `bi.d_selected` / `bi.d_ew` with the private remap (the builder AND the reduce both skip non-late picks: no -1 masking,
   which the reduce would still sum, r1 #1), into a per-lane `bi.late_moe` [rows x N_EMBD].
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
- Pass end / error (r1 #9): join or cancel every ticket, then `unpin_all_wait`; no reader writes a stage and no claim lands
  after the pass.

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
| `V41_PREFILL_COLD_B1_FRAC` / `_MAX` | 0.5 / 16 when async | share / cap of NEW late reads per request |
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
