# Code review: 9f713ac -- two-phase expert landing + gate/up-only hints (on 5232557)

## Findings

1. **MAJOR -- the price in 3.4 assumes a role read costs a third of an expert read; the data says ~90%.**
   The three roles are read CONCURRENTLY, one thread per role, and each role is its own ~2 ms stream: measured
   per-role walls (spec_reads.log, isolated reads) r0 2.09 / r1 2.08 / r2 2.21 ms against a whole-expert wall of
   2.26 ms; certain reads r0 1.96 ms. So `ROLE_DOWN` alone (6.3 MB on one thread) takes ~2.1 ms, not ~0.75, and
   `ROLES_GATEUP` ~2.1 ms, not ~1.5. Under `land_two_phase=1` the Down phase therefore hides at most the gate/up
   kernel's wall (~0.1-0.7 ms at decode sizes) of a ~2.1 ms read: per unhinted miss ~2.2 -> ~1.6-2.0 ms exposed,
   not 1.75 -> and per hinted partial miss ~1.5-2.0 ms, not 0.25. And `likely_gateup_only` saves bytes
   (-33%) but ~0 wall per hint. The lever only pays if a single role is read with the whole expert's
   parallelism: split the down role into 3 chunks over the three reader threads / both drives (`read_miss_into`
   already stripes roles under `route=split`; stripe the chunks of one role the same way). Re-price 3.4 on the
   measured per-role walls before the bundle; without the chunked role read, keep both knobs 0.

2. **MAJOR -- the Down phase re-runs the whole `ensure` prologue: `note_serve` twice per pass breaks the
   nursery's unused rule, and the stats double-count.** `ensure_layer_inner` (6292) calls `pool.note_serve(layer)`
   unconditionally; two-phase calls it for GateUp and again for Down, so `serves[layer]` advances by 2 per pass and
   an entry is "unused" (`nursery_victim` tier 1) after ONE pass at `nursery_lanes=2` -- the other lane's request
   has not come yet, so its entry is first in line for recycling (round-3 finding 2 again, now under the knob).
   Also doubled per pass under two-phase: `pg.requests` (6294), `ev_hits`, the `touch_hit` recency tick (harmless),
   `admit_prefetched` (lands twice as often -- harmless), and the `B2_ENSURE` record (6519) is emitted TWICE per
   pass with the same seq, which doubles `n_want`/`n_hits` in b2tail's joins. Fix: in the Down phase skip
   `note_serve`, `pg.requests`, the hits accounting and the `B2_ENSURE` emit (or emit one record with both phase
   waits); the only work Down should do is `admit_prefetched` (optional), the partial loop and the down reads.

3. **MAJOR -- `likely_gateup_only=1` without `land_two_phase=1` turns a hidden read into an exposed one.**
   `is_resident_pool` counts a PARTIAL entry (5866-5874), so the early-page hook issues NO certain read at arrival
   for a hinted partial expert; under `Full` ensure the pass then reads its down role synchronously at the top of
   `ensure` (~2.1 ms per finding 1, ~0.7 with chunking) -- where today the hook would have read the whole expert
   in the background under the previous request. The knob doc says "else it is on the critical path"; make it a
   rule: `likely_gateup_only` is forced off (or the hook issues a certain `ROLE_DOWN` read for partial entries
   of queued frames) unless `land_two_phase` is on. The latter is the better fix: it hides the down read even
   when the partial entry was promoted earlier than its pass.

4. **MINOR -- `nursery_protect` and partial entries.** `protect_nursery` (5950) is keyed by `in_nursery`, which
   includes partial entries, so an arrived frame's partial entry is protected from recycle until served -- good.
   But a recycled partial entry wastes 2/3 of a read (counted in `partial_evicted`, not in `nc.recycled`'s
   cost): add the read fraction to the design's churn price.

5. **MINOR -- the no-miss hot path under `land_two_phase=1` is not ~zero.** With no misses the second `ensure`
   still does: `pinned` Vec build, `admit_prefetched` (a `try_recv` + the `want_pre` Vec), the dirty check, the
   want dedup Vec, the hits loop (6 `touch_hit`s + `is_partial` lookups), `two_phase_stats` x2 per request
   (walks every layer's pager for the evtrace deltas, 40 layers) and the `B2_ENSURE` emit. ~5-15 us per pass on
   the daemon's serve thread; with the knob 0 the cost is one `knobs::land_two_phase()` atomic and nothing else
   (verified: the Full branch is the old call sequence; `batched_pass` is now `gateup` + `down` with the kernel
   order unchanged, see below). Trim: early-return the Down phase when `down_pending` would be empty (a single
   `is_partial` scan over `want` before any of the prologue), and compute `two_phase_stats` only under the knobs.

6. **MINOR -- `gateup_wait_us` / `down_wait_us` measure the read loop, not the wait.** `phase_ns` (6609-6665)
   spans the chunk loop including repack and commit; under two-phase the down "wait" is the whole read wall
   even though it overlaps the gate/up kernel -- the exposed part is `down wall - gate/up kernel wall`. Record
   the kernel-side overlap too (an event after `batched_pass_gateup`'s q8k, queried after the Down phase: done
   -> fully exposed, not done -> hidden), or the A/B cannot tell hidden from exposed.

7. **NIT -- the `paged` predicate on partial slots** (6114-6120): `p.partial[(-row[e]-1).max(0)]` is evaluated
   even when `row[e] == 0` (then indexes slot 0 -- harmless because the `||` short-circuits only on the left).
   Write it as a `match`. `ev_partial_fields`' field order (`partial_lands, promotions, slots, gateup, down`)
   matches `B2_REQ` but the named fallback at 8923-8925 lists `partial_slots` last -- fine for `set_named`, but
   keep the two lists in one order.

## Verified
- **Split pass equivalence (1).** Kernel order on the one compute stream is unchanged by the refactor (5232557
  `batched_pass`: group_count zero, builder, n_work_items zero, work items, [decode_down: mid zero], gate/up,
  q8k; then [decode_down diag], partials zero (`first && !fast`), down) -- `batched_pass_gateup` ends at q8k and
  `batched_pass_down` starts at the diag; `first` is consumed only by the down half (`let _ = first` in the
  gate/up half is correct). The down half reads `d_midq_cat`, `group_count`, `expert_members`, `work_items` and
  `n_work_items` (devcount), none of which the host touches between the halves; `io` (sel/ew/xq) is not needed
  by the down half. The Down-phase `ensure` writes only the DOWN region of partial slots (separate per-role
  buffers; `repack_in_place` permutes on the repack stream and `st.synchronize()`s before returning, so the down
  kernel queued after sees landed bytes) and can set `dirty` for other layers only; a `remap_dev` upload for this
  layer during the Down phase (a nursery landing in this layer at `admit_prefetched`) changes entries for experts
  the running gate/up kernel never reads (wanted ids are never victims) -- the same argument the existing
  pass-A/ensure overlap rests on. A merged partner's picks are in `sel`, so `want` covers both. Pass A's
  `resident_mask` excludes partial slots (6163-6170), so a partial expert goes to pass B whose GateUp phase has its
  gate/up and whose Down phase completes it. Bit-identity vs the single pass is asserted by loopback run 3 --
  compiled only, like runs 1-2.
- **Partial state (2).** `partial` cleared by `land`, `land_nursery`, `unclaim`, `evict` (counted) and
  `complete`; masked from `residency_words`, `soft_words` (`landed_whole`), `PinBook::report` (never pinnable,
  even granted), `resident_mask`; `is_resident_pool` and the early-page hook see it (finding 3); PAGED for the
  reply (the pass reads its down) -- consistent with the hub's `paged_total` (reply-experts that needed a read)
  and b2tail's definition, with a smaller read behind the bit. `touch_hit` promotion keeps the state and counts
  it; `me_account`/refill unchanged (the refill victim search excludes `want`, so a wanted partial slot is never
  the victim); eviction mid-pass is impossible for wanted slots (victim searches exclude `want`/`pinned`), and a
  partial entry wanted by two queued requests is completed by whichever pass runs first (the interleaved one on
  exec2 runs its own two-phase `ensure`). The sim's odd seeds and `partial_slots_are_masked_until_completed`
  cover the state machine; the pin/soft/held invariants hold with `partial` in the subset check.
- **LIKELY gate/up-only read (3).** A 2-role read holds one staging set (all three buffers) for ~its wall (no
  shorter per finding 1); the reserved pair is unchanged; chunking/pause apply per role thread, so fewer roles =
  fewer chunks; `pick_set` unchanged. The coalesced span falls back to per-role reads for a partial mask
  (6682-6692); the join zips handles with the roles read (6709-6711) -- correct.
- **Knobs (5).** Both are `Knob::flag(...).alias(...)` in the daemon knob file: live via SIGUSR2. `two_phase` is
  read once per pass and used consistently for GateUp -> gateup -> Down -> down; a flip between passes leaves
  partial slots that the next `Full` `ensure` completes (`down_pending` under Full) -- safe. Both default 0 =
  today's path verbatim.

## Tests
Rules: `phase_plan` as a pure function, the partial state machine, the sim's odd seeds (never served partial,
never pinned/held, promotions <= lands, completions = the sim's down reads). Missing before the GPU gate: a
host test that two `ensure` calls (GateUp then Down) leave the pool and the counters exactly as one `Full`
call (finding 2 would fail it: `serves`, `pg.requests`); a test that the early-page hook issues a down read for
a partial entry (finding 3); the loopback run itself (all three runs).

VERDICT: APPROVE WITH CHANGES -- both knobs stay 0 in the bundle.
Minimum before the slice-D bundle (the knobs ship dark): finding 2 (Down phase skips `note_serve`, request/hit
accounting and the duplicate `B2_ENSURE`), finding 3 (`likely_gateup_only` requires `land_two_phase`, or the hook
issues a `ROLE_DOWN` certain read), re-price 3.4 on the measured per-role walls (finding 1) and decide whether
the chunked single-role read is built before either knob is ever turned on. Findings 5-6 before the A/B that
turns them on. The earlier minimum lists (loopback run, late = `moe_arrived`, SOFT clear at the switch) stand.
