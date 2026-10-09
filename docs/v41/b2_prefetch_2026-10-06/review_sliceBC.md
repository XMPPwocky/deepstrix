# Code review: prefetch slices B + C and the slice-A amendment (ec3cf7f..6220e10)

Read: the full diff by area (daemon 5646edf/e652936/b35279a, hub 129c55c), the new loopback test, and the
touched sites in context. Line numbers are post-change unless marked (diff).

## Daemon: nursery vs ShardPool invariants -- holds
- `slot_of`/`remap_hosts`: a nursery entry is resident like any slot (`land_nursery` 3727-3738 maps it; `evict`
  3783-3808 unmaps it); `is_resident_pool` sees it, so the early-page hook issues no certain read and the hit at
  `ensure` promotes it. `held` vs `nursery_held` kept apart everywhere (carve 3657-3676, land, evict, relabel
  3816-3829) -- round-3 finding 3 fixed; the floor test (`nursery_entries_do_not_count_toward_the_floor`) proves it.
- Relabel INSIDE `touch_hit` (3816-3829): flag off, counts moved, no remap write, no `dirty`, the decode stamp is
  `touch_hit`'s own. Refill at the call site (ensure 5839-5851): `pick_victim_any(Main, want, extra=pinned incl.
  parked, ignore_hub_pins=false, floors via for_layer)`, `me_account` BEFORE `evict` (3764-3765, the `claim_miss`
  order), shrink on no victim, never a revoke. `pick_victim` skips nursery slots (3706-3712), so no claim, landing
  or restore ever takes one. Pin budget minus `nursery_target` (5584-5296 diff), `report` masks nursery slots
  (3157-3177), `residency_words` masks them (`landed_main`), `on_evict` false by construction.
- Served while being recycled: not possible. A nursery slot is read only by a pass that WANTS the expert, and
  wanting it promotes it (relabel) before the pass runs; recycling happens only at `ensure` top (`admit_prefetched`,
  the compute thread) and `nursery_victim` excludes the pass's `want` and `extra` (= `pinned` + `parked_pins`, so the
  park loop's `admit_landed` -> `admit_prefetched(layer, &[])` is covered). Readers write staging only.
- Served bytes identical: the landing is `repack_in_place` / `copy_from_host` into the slot exactly as a main
  landing (5082-5087 diff) and the kernel reads the slot through the same remap. Asserted by the loopback test --
  which has only been COMPILED (finding 1).

## Findings

1. **MAJOR -- the daemon-side I1 gate has not run.** `remote_experts_nursery_loopback` (361 lines) is the only
   test that serves bytes from a nursery slot, exercises `likely_words_in` at arrival + dequeue, `append_nursery`
   on partner replies, and `lands = hits + recycled + delta(occupied)` across the real serve loop. It is `#[ignore]`
   and has never been executed. Required before (b), on box 1 with the live hub down (one GPU test process).

2. **MINOR -- a nursery entry an arrived frame relies on can be recycled before that frame is served.** The
   early-page hook (7940-7960 diff) skips the certain read for a pick that `is_resident_pool` -- including a
   nursery entry -- but `nursery_victim` protects only the pass being served (`want`, `extra`). Between R2's
   arrival and R2's `ensure`, R1's `ensure` may land a LIKELY read that recycles R2's entry (when the nursery is
   full and the entry is the oldest fresh one, or "unused" at `nursery_lanes` 2 after two passes). R2 then
   demand-reads at `ensure` with its early-page lead lost -- correctness intact, a late reply manufactured. Fix:
   in `early_page`, add the nursery-resident picks of the queued frame to a protection set keyed by seq (the
   `EarlyPaged` ring already tracks arrivals), and have `nursery_victim` skip them until that seq is served.

3. **MINOR -- the margin is a weight margin, the rank is a biased-score rank.** `classify_w` (lookahead.rs
   diff 752-759) computes `(w[0]-w[1])/sum(row)` from `sd.look_ew`, the normalized top-k WEIGHTS of the selected
   experts; the column order (rank) comes from `router_topk`'s selection on score + bias. Inversions (rank-1 by
   bias, lower weight) clamp to margin 0 and land in bucket m0 only. Fine as an empirical proxy since the four
   buckets are measured against `hits_prot`, but say so in the doc comment and in the amendment note; if the dry
   run shows m0 >> m1 with flat precision, the right quantity is the biased selection score, which would need a
   `look_score` segment (and there is no pack room under k2: 10 of `RB_PACK_MAX_SEG` 10 with alts + sub3 + xq).

4. **MINOR -- `Bars::allow` counts "allowed", not "sent".** `bars_allow(hints.len())` (9330 diff) adds `n` to
   `sent` before the budget plan may take fewer (`t.hints <= allowed`); the surplus is re-queued and counted again
   at the next submit, so the step bar can trip on words that never went. Count after the plan (`allow` on
   `t.hints`, or `commit` the actual count). Also: once tripped, every later submit of the step re-takes the whole
   queue and re-queues it (O(<=128) per submit, bounded, dropped at `begin_step`) -- acceptable, but a tripped step
   should skip `take` altogether.

5. **MINOR -- `pump_restore`'s throttle now sees the reserved pair.** `free.len() * 2 <= stages.len()` (5010)
   divides general free sets by ALL sets (general + 2 LIKELY), so the restore pump stops one set earlier than
   before when the nursery is on. Use `n_general`.

6. **MINOR -- amendment and slice B share a commit (129c55c).** The hub-only relaunch for the second dry run
   carries slice B's wire code (inert without `k1`/`k2`: `Mode::wire` needs `RESP_FLAG_NURSERY`, which the live
   daemon cannot send). Acceptable, but the relaunch note must say the binary is B-capable and that `k1` on the
   live box 2 = dry + a probe flag on every decode request (safe: `decode_request` 874-905 uses flags only for
   lengths, no unknown-bit check; the old daemon echoes bit 11).

7. **NIT -- `restore_pace` and `lh2_cand` doc comments** still assume one lane / per-step distinctness (round-A
   findings 5 and 9); `Cfg` is two atomics with a documented harmless tear; `HintQueue::take` still rebuilds a
   `VecDeque` per submit.

## Verified against the code
- LIKELY scheduler: `push` order Certain > Likely > Spec (1673-1683 diff); `pop_mode`: Likely pops when
  `running_spec + running_likely < max_spec` regardless of `running_certain`, Spec additionally needs
  `running_certain == 0` (1753-1775); the certain reserve reader is never taken by LIKELY; `promote` and
  `reclassify_certain(from_likely)` move counters correctly; `finished_cls` releases the right counter; `PfFinish`
  carries `likely`. Pause rules: Spec pauses for running LIKELY, LIKELY pauses for certain/urgent only, under
  `likely_pause_for_certain` (chunked) else unchunked (4871-4890 diff). Starvation of certain reads: impossible
  by the queue (certain always first) and the reader cap; drive sharing is the knobbed trade (round 3, finding 1).
- Reserved set pair: last two of `pf_stages`, `pick_set` (369-379 diff) takes the pair first then a general set
  under the speculative rule, never lets a speculative word touch the pair, certain takes any general set;
  `release_set` returns to the right list; every former `pf.free.push(set)` is routed through it (the only
  remaining `free.push` is inside `release_set` 2015). Drops fold into `nc.drops` (4949-4955 diff).
- Wire compatibility. Old daemon + new hub: bit 11 echoed, no payload change, no `RESP_FLAG_NURSERY` -> hub stays
  dry (`nursery_reply_seen(false)`, logged once); marked words never sent before detection; if one ever reached an
  old daemon, `(w >> 16) >= N_LAYER` skips it at 4324 (proto test asserts). Old hub + new daemon: no bit 11, no
  block appended (7958 diff gates on the request flag). Bit 31 is free in every word encoding on the PREFETCH block
  (layers 16-21, experts 0-9; restore and LM words are plain). `RESP_FLAG_NURSERY` 1<<13 disjoint from PIN/RESID/miss
  mask (test). Block framing: after resid + pin blocks, length-checked in `decode_response_meta` (1038-1041 diff),
  `NURSERY_NONE` padding, own layer first then rotation (`nursery_words`, tested).
- Hub detection/loss: `nursery_reply_seen` on every LIKELY-flagged reply; loss -> support 2 -> `wire()` false ->
  taken words are counted (`dry_words`) and dropped, not re-queued (they would be stale next step anyway);
  `on_connect` resets to unknown. Queued words cannot grow without bound: `HINT_QUEUE_MAX` 1024, per-step drop at
  `begin_step`, re-queue only of words taken this step.
- Pack growth: one `look_ew` segment (`n_sel` u32 = 6 x rows words, 24 B/row) only under the knob (8621-8631
  diff); the pack is one kernel, so `before` placement cost is unchanged to first order; `rb_pack` sized 6 x
  n_used. Segment count under k2 + alts + sub3 + xq = exactly `RB_PACK_MAX_SEG` 10.
- I1/I2 off and dry: knob off -> `look_next` None (unless legacy), no `look_ew` segment, `hints` empty,
  `allowed` 0, budget 0 -> the legacy drain path verbatim. Dry -> launches + pack + filter, no words, no
  `note_incoming` (only under `nursery_prior && rank >= 2` and a non-empty `sent`). Mirror NURSERY bits feed
  only `queue_hint_words`' dedup and counters; `resident()` ignores them (b2_mirror test asserts `Some(false)`).
- Dry-hit matching per lane (`LaneState::pending` on `bd.lh2`) unchanged in structure; `take_pending` now keeps
  the match for `dry_hits_demanded`'s target layer -- correct.

## Tests
Rules, not implementation: queue order/cap/pause (`likely_queue_order_cap_and_pause_rules`), set pair, carve/
victims/recycle order, relabel + refill (pins, floors, `me_account`), floor exclusion, pinnable/resident masks,
the lands invariant over 6 random seeds, proto roundtrip incl. the old-daemon case, block rotation, mirror bits
and capability transitions, bars, margin buckets + protected set, re-queue; the 6-seed pin sim extended with
LIKELY words and the NURSERY mirror. Missing before a GPU gate: (a) `likely_words_in` dedup by seq + ring
eviction; (b) a submit-path test with `wire` on: bars trip -> re-queue -> next submit picks the same words,
budget > 0 with hints first; (c) finding 2's scenario (entry recycled between a frame's arrival and its ensure);
(d) the loopback run itself.

VERDICT: APPROVE WITH CHANGES.
Minimum before (a) the hub-only relaunch (second dry run): nothing in the code -- `dry` is unchanged in kind
from slice A plus the `look_ew` segment and the new counters; note finding 6 in the relaunch message; keep
`V41_B2_SPEC_BUDGET=0`.
Minimum before (b) the box-2 bundle: finding 1 (run the loopback on box 1, hub down; it must pass bit-identical,
0 surprises, 0 pinned evictions, the lands invariant); finding 2 (protect nursery entries an arrived frame relies
on); finding 4 (bars count the words that went); finding 5 (`n_general` in the restore throttle); then the hub
gates of design section 7 (G5a-h `k1` vs `off` bit for bit with a remote attached, the determinism recipe).
