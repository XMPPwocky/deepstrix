# Review round 3: B2_PREDICTED_MISS_PREFETCH_DESIGN.md rev 3

## Round-2 fixes against the code
All hold: knob gates `look_next`/`look_next2` (8319/8323/8875/8879) and the 9515-9539 block is replaced; per-step
budget replaces the per-request caps (drop/discard at 4342-4350, 4190 cited correctly); ns units (evtrace.rs:287),
handling 0.1; paged-only ceiling 0.71; clear at multistream.rs:1119; RESP capability flag 1<<13 (free beside 416/446);
absolute bars; `note_admits` correctly retired (hints never pin). Pricing table matches price2.py with the two fixes.

## Findings

1. **MAJOR -- LIKELY "neither pauses for certain" slows the reads it is meant to help.** Certain (src=1) reads are
   the early-page reads of the queued request's own picks = the paged replies themselves (4,187 and 3,905 in the two
   dumps, ~7/step, 2.3 ms each -> a certain read is running ~19% of the time; a 2.3 ms LIKELY read overlaps one with
   P ~ 0.38). The busy-wall data (3 concurrent -> ~3x wall) says the drives share bandwidth ~linearly under
   `route=split`, so an unpaused LIKELY read stretches an overlapped certain read by ~1 ms: ~3 hints x 0.38 x ~0.7
   ms ~ 0.8 ms/step of ADDED lateness on already-late replies, against a 3.6 ms gain. Pausing instead costs the hint
   its lead in those 38% (~1 ms/step of gain). Same magnitude either way; the data cannot pick. Change: LIKELY reads
   stay CHUNKED (isolated chunked walls = certain walls, 2.26 ms, so chunking is free) and pause for certain reads
   under a daemon knob `likely_pause_for_certain` (default on: protect the known-late reply), never for speculative;
   speculative pauses for LIKELY (`background_should_wait` 1590-1596 gets `|| running_likely > 0`). A/B the knob.

2. **MAJOR -- LIKELY must count against the speculative reader cap.** `pop_mode` (1629-1655) hands out a
   speculative job only while `running_spec < max_spec` (3 of 4) AND `running_certain == 0`; the 4th reader is the
   certain reserve. A LIKELY class that "runs on any reader" without counting toward `max_spec` can occupy all four
   and make a certain job wait -- "never pre-empts a certain one" in the queue order does not cover the readers.
   Specify: `running_likely + running_spec < max_spec`; LIKELY pops regardless of `running_certain` (it does not
   yield), subject to finding 1's pause; `promote()` (1559-1581) must handle Likely -> Certain like Spec -> Certain
   (the park/early-page certain path, 7222/7258, relies on it); `reclassify_certain` likewise.

3. **MINOR -- per-slot state the relabel must touch / must not.** Verified against `ShardPool` 2240-2271:
   - `last_use`: nursery landing stamps 0-class (fine: main searches skip nursery slots anyway); promotion is the
     decode-hit stamp `tick + PREFILL_AGE` -- exactly `touch_hit`'s else-branch (3327), so the relabel belongs
     INSIDE `touch_hit` (3307): `if nursery[slot] { clear; refill }` before the stamp. Not at the top of `ensure`
     as 3.2 says: `admit_prefetched` runs there (5053-5056) and is where LANDINGS go; promotion is the hit path
     (5136).
   - `held[layer]`: 3.2 counts nursery entries in `held`. `pick_victim` 3232 protects a foreign layer only while
     `held > floor`, so one nursery entry per layer lets a main slot of a layer AT its floor be evicted. Keep a
     separate count or exclude nursery entries from `held`.
   - `me_account` must run on the REFILL VICTIM before `evict`, with the serving request's `prefill_mode`
     (`claim_miss` order 3378-3379), not on the promoted slot; `me.phase_touch` is per slot and carries over
     harmlessly. Restore list / `decode_delta`: untouched by promotion (decode landing), as 3.2 says.
   - `pins`: `PinBook::report`'s `state()` (2864-2869) needs `pinnable = slot < stage && !nursery[slot]`; `grant`
     skipped for LIKELY words; `on_evict` on a nursery recycle returns false by construction (gate-test it).
     `residency_words` (4841-4858) masks nursery slots; `dirty` is set by the landing (`land` 3407) and not by
     the relabel (the remap already points at the slot) -- correct.
   - `is_resident_pool` (7212) sees a nursery entry, so the early-page hook does not issue a certain read for it:
     the hit at `ensure` is what makes the hint pay. Correct, state it.

4. **MINOR -- recycle "never a wanted slot" is per request.** The other lane's layer-L request arrives one period
   later; the draft's "unused = served by >= lanes requests since landing" rule handles it -- keep that rule in 3.2
   (the doc text has only LRU + not-wanted).

5. **MINOR -- restore floor.** Hints <= 10/step (bar) + admissions 4-11/step leave >= 39 of 60 for restores;
   leftover restore words survive a phase flip (`pin_enter_prefill` 1086-1088 pushes them back), so starvation is
   impossible and the pool always re-warms; worst case is the daemon's own start capacity (~40/step under busy
   walls), not the hub budget. A floor of 16/step is free insurance against an admissions spike: adopt it.

6. **NIT -- "dequeue is 1.35 us at the median" is from the reservoir (all decode requests).** For two-lane cells
   late.log's b2q is 16-18 us median, 0.6-1.0 ms p90 on late requests; the arrival-time hook is worth that tail
   (~0.1-0.2 ms/step), as 3.2 says. Fine.

## Open questions (rev 3)
1. 32 (k1 burst bound 16; room for cap bursts and k2); 48 only if the dry run shows dump-rate volume. Carve from
   the pool on the +270 restart (net +238); 32 from today's 4480 (-0.7%) is also fine.
2. The request's own `pin_grant` suffices: it is already per REQUEST including a merged partner (dab7e25,
   `ExpertShard::pin_grant` 4924), so the partner's pick pins too. No promotion grant.
3. Two new pinned buffers (+113 MB). The 16 sets are sized by `V41_B2_MISS_PAR` and `miss_par` caps demand-read
   concurrency at `min(knob, stages.len())` (5177); taking two changes demand concurrency and the 2x reserve rule.
4. Yes, floor 16 (finding 5).

## Slice A
Safe to build now, provided it ships with: the knob gating + the 9515-9539 replacement in the same change (else
`dry` either launches nothing or leaks the legacy words); `dry` never calls `push_prefetch_words`; `look_next2` off
unless `k2`; `lookahead_hints_ok` left as is in A; host timers + the `ms.step` per-cell watch; the hub budget
(2.4) may ship in A as well -- it is independent of hints and a win on its own.

VERDICT: APPROVE WITH CHANGES -- findings 1-2 (LIKELY pause knob, reader cap, promote/reclassify) into 3.2 before
slice C is specced; finding 3 (relabel in `touch_hit`, `held` excludes nursery, `me_account` on the victim,
`pinnable` mask) into 3.2's bullets; floor 16. Slice A proceeds now; B-E sequencing stands.
