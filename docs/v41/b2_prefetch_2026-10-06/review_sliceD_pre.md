# Code review: 78663a2 (refined-objective counters), abce02e (dry run 4 counters + 2 fixes), 4715b2a (SOFT map)

## Findings

1. **MAJOR -- "Post found the reply not ready at least once" is not "reply after the MoE"; it is "reply slower
   than one host loop turn".** `remote_post_spun` (forward_prefill.rs:4660-4663) is set on the FIRST
   `Ph::Post(l)` check that finds `r.ready(seq)` false. The ready-first loop reaches Post(i, l) one iteration
   after Route(i, l): after the other lane's phase (a chain enqueue, ~0.1-0.3 ms of host time at most) -- i.e.
   ~0.1-0.3 ms after the submit, against a box-2 RTT of 0.47 ms median (min ~0.3). So the flag is true for most
   replies, late or not; b2tail's real definition (reply consumed after the lane's MoE end) marks 12-14%. The
   `*_late_*` counters and `prot_paged_late_total` / `paged_late_total` therefore carry almost no information,
   and recall "against the reads we block on" is really recall against all paged reads. Check: in dry run 4,
   `lh2_paged_late_total / lh2_paged_total` should be ~0.13; if it is ~0.9+, this is confirmed. Fix (hub-only,
   cheap): at the Post that finds the reply READY, query the lane-layer's `moe_arrived` event
   (`self.sync_events_lane(i).layers[l].moe_arrived.query()`, recorded on `ie.xfer` at 11427): already complete
   -> the MoE finished before the reply was consumed -> late; else the reply beat the MoE. That is the b2tail
   definition to within the polling granularity. Lane/step leakage otherwise OK: the flag is per lane scratch,
   set only from that lane's own `remote_ticket`, cleared at that lane's submit (9987); the lockstep/pipelined
   drivers never set it (always "not late" there -- say so).

2. **MAJOR -- the classify fix changes the meaning of the slice-A counters; dry runs 1-3 are not comparable with
   run 4.** Before abce02e a box-1 expert first seen at a low rank and again at a better rank was un-marked
   (255 -> rank) and surfaced as box 2's; a box-1 expert is never in box 2's maps, so `lookup` says non-resident
   and it became a CANDIDATE, a NON-RESIDENT, a hinted word, and -- if the row picked it -- a DRY HIT (own picks
   are not ownership-filtered). Rows picking the same hot box-1 expert at different ranks is the common case.
   Affected in runs 1-3: `lh2_cand_rN`, `nonres_rN`, `nonres_m*`, `dry_words`, `dropped_cap`, the hint volume
   (the "~43 words/step" of dry run 1 and the "17-65 words/step" bar note), `dry_hits_rN`, `dry_hits_prot_rN`,
   `hits_prot_m*`, and every precision ratio built from them (the 0.93-at-margin-0.1 of dry run 2 included).
   Unaffected: the `b2_*`/`sub_*` step counters, `paged_hinted`, `nursery_*`. Mark runs 1-3 as superseded in the
   design doc's results section and re-derive the margin choice from run 4+. The MAX_RANK 3 -> 6 change does not
   alter the per-R <= 3 counters (`count`/`dry_hits` stop at `HINT_MAX_RANK`); only the new any-rank counters
   see ranks 4-6.

3. **MAJOR -- SOFT rows go stale across a prefill phase, exactly where it matters under `SOFT_PRIOR=1`.** A
   layer's SOFT row is replaced at its next decode reply (`update_soft` 4715b2a); during a prefill phase there
   are no decode replies, the band is released and prefill evicts freely, so at the first decode step after the
   switch every SOFT row describes the pre-prefill pool. With `V41_B2_SOFT_PRIOR=1` the prior then treats
   evicted experts as resident for one lane-layer per layer (no swap, no admission, `n_pred_miss` low) -- in the
   0-20 s window that already carries 2x paged replies. Clear SOFT (all layers) beside the hint-queue clear at
   multistream.rs:1119 (`Phase::Prefill`) and in `expire_incoming` (the per-chunk path), as INCOMING is. Counters
   only (both knobs 0): harmless, but `lh2_paged_mirror_soft` will spike after every switch and must be read
   per phase bin.

4. **MINOR -- the before-submit denominators now run on EVERY decode submit by default.** 4715b2a gates
   `protected_nonres` / `nonres_bits` / `pick_bits` on `lookahead::active()`, and `active()` is true whenever
   `V41_B2_SOFT_MAP` is on (default). Three passes over `sent_sel` with a `lookup` per distinct expert
   (~1-3 us) on the Route path before `submit_dispatch`, plus a `hub_lh2` record every step, with the prefetch
   knob off. Acceptable, but it is a default-on cost on the turnaround (0.03 ms slack): price it in
   `lh.remote_submit` once, or gate on `mode.on() || soft_prior()` (the soft counters are only needed when the
   consumer is on or being evaluated). Note the fix in abce02e is correct: the predicate must run before
   `note_submitted` marks the picks PENDING, and it uses `nonres_for_miss` = the `n_pred_miss` predicate (differs
   from the prior's `resident()` only by `pending_on()`, live 0 -- round 1, finding 5).

5. **MINOR -- per-reply denominators double-count across lanes, by design; say so.** `PinReply.paged` is per
   request: the hook's arrival bits are OR-ed in (8f7899f; `note_early_paged` 5897/7993/8143), so a queued
   request's early-page certain reads DO show as paged (the denominator is not under-counted on that account),
   but an expert read once for lane A and waited on by lane B is paged in both replies, as in b2tail's
   `b2_paged_replies`. Consistent with the lateness curves; document that `paged_total` is reply-experts, not
   drive reads. A merged partner is scored against its own bits (8353) -- fine.

6. **MINOR -- `RESP_FLAG_SOFT` reserves bit 12 of the shared flags word.** Request flags use 0-11 (LIKELY is
   11); a future request flag 12 would be echoed and read as a SOFT block by a new hub. Add a const assert /
   comment that request flags stop at 11, or move response-only flags above 12 together (13 NURSERY, 14 PIN, 15
   RESID already are).

7. **NIT -- `soft_words` is O(384) per decode reply with a `pins.is_pinned` call each** (5660-5679); fine (~1 us),
   but it runs on the daemon's serve thread per reply; a bitset over `pins.state` would make it ~12 word ops.

## Verified
- Wire compatibility both ways. The second flags word was ALWAYS written as `0u32` by every earlier hub
  (8996153 `encode_request`: `[layer, b, flags, n_used, xq_bpt, 0u32]`) and never read by any earlier daemon
  (`decode_request` had no `u(5)`), so an old daemon ignores the ask and an old hub never asks; `REQ_FIXED` 32
  and `REQ_T1_OFF` unchanged. The block sits after RESID, PIN and NURSERY in that order on both sides
  (`response_soft` offset, `decode_response_meta` length), length-checked; the proto test covers new/new,
  old-daemon (echo without bit 12 -> no block) and the flagged-but-absent frame.
- Soft never leaks into `held` or the pin ledger: `update_soft` writes only `SOFT`; `BITS` are written by
  `update` / `update_pinned` / `pin_begin_step` / `band_release` only; `pin_note_submit` reads `BITS`; the
  ledger's release/restore logic reads counts and `held`. The surprise check (`check_surprises` on
  `ticket.held`) is unaffected. The admission planner (b2_mirror.rs:1579 `resident() == Some(false)`) and the
  prior vector (forward_prefill.rs:8393-8402, `resident() == Some(true)`) both switch with `SOFT_PRIOR`: under
  it a soft expert is also a BOOST TARGET, so the prior can swap a true miss INTO a soft expert (the intended
  trade: a hit if it stays, a paged read of a mirror-soft expert if evicted -- `lh2_paged_mirror_soft`).
- Exactness with both consumer knobs 0: `resident()`, `nonres_for_hint`, `nonres_for_miss` reduce to the old
  predicates; `sub_soft_unswapped` is a count; the only wire change is +48 B per decode reply.
- `remote_post_spun` cannot leak across lanes (per-lane field, lane's own ticket) or steps (cleared at submit;
  a Post of step s always follows that lane's submit of step s).
- Tests test rules (reply scoring per R / bucket / late, denominator from the sent sel, any-rank scoring, totals
  vs held/nonres bits, classify's box-1 mark, SOFT row replacement + knob gating, proto roundtrips, daemon soft
  set). Missing: the late proxy (finding 1) has no host test and needs the `moe_arrived` form to be testable
  at all; SOFT staleness across a phase switch (finding 3); a `pick_bits` test with a repeated expert whose
  predicate is false (the old `nonres_bits` re-evaluated it; the new one does not -- same result, worth pinning).

## What the `SOFT_PRIOR=1` A/B must pre-register
Per-turn knob flip, >= 2 h warm, cells with >= 200 steps, phase-binned (0-20 s after a switch separately):
- Primary: `sub.picks_swapped` per step DOWN (the dry-run claim is that ~24 of ~43 swaps are against soft
  experts: pre-register >= -30%), `sub.predicted_miss` down, `sub.soft_unswapped` ~ the difference.
- Guards: `b2_paged_replies` not up by more than `lh2_paged_mirror_soft` + 0.3/step in any cell;
  `b2_surprises` 0; `pin.released_unused` unchanged; `ms.step` p50 <= 1.00x every cell, no cell > 1.01x;
  `box2.compute_us` +<= 5%. Abort: `lh2_paged_mirror_soft` > 1/step outside the post-switch bin, or paged
  replies up > 10% in any bin.
- Fidelity: swaps can only fall or move to a different held target of the same gap rule; no golden gate needed,
  but the determinism recipe (`V41_T2_CATCHALL=2`, T=0) once with the knob on/off.

VERDICT: APPROVE WITH CHANGES.
Minimum before the slice-D box-2 bundle: finding 1 (late = `moe_arrived` already complete when the reply is
consumed; hub-only, can ship with the next relaunch) and finding 3 (clear SOFT at the phase switch and in
`expire_incoming`) -- both are needed before any `SOFT_PRIOR=1` turn and before the run-4 lateness numbers are
quoted; finding 2 is a doc change (runs 1-3 superseded). The B+C minimum list (loopback run, nursery protect,
bars count sent, `n_general`) still stands.
