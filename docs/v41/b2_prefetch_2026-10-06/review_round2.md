# Review round 2: B2_PREDICTED_MISS_PREFETCH_DESIGN.md rev 2 (+ owner's nursery direction)

## Round-1 fixes against the code
Hold: lookup() predicate (b2_mirror.rs:374 / 9725 form); no gate at rank <= 2; `note_admits` per sent word
(1426 -> 745; `released_unused` 892 then sees hints); hints-become-pins stated; `before` kept, host timers,
captured-stage plan; G5 assert conditional on a remote; R=1 default with per-R dry counts.
Do not hold yet (see 1, 2, 9 below): the knob does not actually drive the look-ahead; the clear-at-switch hook is
mis-cited; the 0.26 ms "dequeue handling" was my unit error.

## Findings

1. **MAJOR -- the knob does not turn the look-ahead on, and the old word path is still live.** Doc 2 says
   `V41_LOOKAHEAD_PREFETCH` stays untouched and the new filter ships words. In the code `look_next` /
   `look_next2` exist only under `lookahead_prefetch()` (forward_prefill.rs:8319, 8323, 8875, 8879) and
   `look_next2` also under `lookahead_depth() >= 2` (default 2). With the env var unset the launches never run
   and `dry` counts nothing; with it set, the OLD block 9515-9539 fires too (top-`lookahead_topk()` = 5 ranks of
   every look-ahead pick, box-2-owned, no residency test -> `push_prefetch_words`) once `lookahead_hints_ok =
   rb.packed`: the 09-21 40-words/step path rides along with the new one. Change: gate `look_next` on
   `lookahead_prefetch() || miss_prefetch() != off`, `look_next2` on `k2` only (not depth), and REPLACE the
   9515-9539 block with the filter (old behaviour only under the legacy env var with the new knob off). Host
   test: knob `k1` + legacy var unset -> `PREFETCH_WORDS` receives nothing from 9515.

2. **MAJOR -- pacing at 4/request does not do what 3.3 says; the right rate is ~1/request and it is a net
   WIN, not a cost.** Daemon speculative START capacity = 3 readers / (2.26-6.7 ms) ~ 0.45-1.3 reads/ms ~
   40-110 starts per 84 ms step; the hub sends restores at `pin_restore_per_request` 16 (b2_mirror.rs:505) ->
   up to 1,280/step after a switch. Words beyond the 16 staging sets are dropped at application (4342-4350) and
   `prefetch_words_full` DISCARDS the returned `dropped_words` (4190, `let _ =`): a dropped hub restore or
   admission is lost for good (6-8/step average, ~3,500 per 480-step phase period vs ~7,000 started). So (i)
   pacing restores to 4/request (320/step) still exceeds capacity 3-8x and still exhausts the sets; (ii)
   pacing to ~1/request (~80/step, about capacity) refills ~1,000 words in ~1 s with ~0 loss, versus today's
   burst that lands a fraction and loses the rest. The restore lever is therefore helped, not hurt. Admissions
   are ~4-9/step total (pred.log), so a per-request cap of 4 never binds; drop it. Price: refill time ~equal,
   lost restores -> ~0, free sets for hints. Make the pace a per-step budget (restores + admissions + hints <=
   ~60/step; hints first), not a per-request cap. (Answers open question 1: pacing is acceptable and should
   be on regardless; it does not by itself justify pulling (a) forward.)

3. **MAJOR (my error, corrected) -- the 1.35 ms `t_frame -> t_dequeue` is an artefact of my round-1 log,
   and so is the 0.26 ms "dequeue -> hints_end".** `evtrace::now()` is CLOCK_MONOTONIC_RAW **ns**
   (evtrace.rs:287-289). `spec_reads.py` block 1 auto-scaled: request stamps had a raw median < 1e4 so they
   were printed unscaled with a "us" label. Correct values: decode `frame -> dequeue` med **1.35 us** (p90
   23-27 ms = requests behind a park/long pass), `dequeue -> hints_end` med **0.26 us**, p90 0.65 us. late.log's
   b2q 16-18 us for two-lane cells stands. Consequences: `HANDLING_DEQ` 0.36 -> ~0.1 ms (link only), lead +0.26
   ms in every cell (prices up ~10%); the difference between (a)-at-dequeue and (a)-at-arrival collapses to the
   b2q tail (late requests' p90 0.7-1.0 ms), so "applied at frame arrival" is worth ~0.1-0.2 ms/step, not
   +0.5. Open question 4: artefact. Fix the two numbers in 3.1/3.2/4 and `price2.py`.

4. **MINOR -- normalisation mixes two ceilings.** `price2.py` scales the paged-only removal curves by 0.81 so
   their 11.1 ms ceiling meets the sim's +9.0. The sim's "box-2 leg fully hidden" hides ALL late replies
   (paged + no-page); the lever reaches paged ones only (paged share of exposed lateness r3 0.89, r4 0.84, r6
   0.88, M3 0.93, r7 0.95, M8 0.985). Correct paged ceiling ~ 9.0 x 0.87 ~ 7.9 -> factor ~0.71. Per cell the
   sim/measured ratio is 0.71-1.13 (r3 1.13, M8 0.71), so one factor is crude. Net with finding 3 (+10%):
   (b) R=1 ~2.1-2.8, (a) ~3.6-4.5 ms/step. The ranking and the (a)-(b) gap (+1.4-1.8) are unchanged; state
   +-15%. No double counting otherwise: the curve is paged-only and the regime shares sum to 0.98.

5. **MAJOR -- (a) should be the nursery, and it changes the (b)-first call.** Evaluation of the owner's
   mechanism against the pool code:
   - *Promote on use needs no copy.* Residency is `slot_of: HashMap<(layer,e), slot>` (2244) and the kernel's
     per-layer remap maps expert -> slot; weights never move between slots. Make the nursery a SET of slot ids
     (not a range like `Band::Stage` 2503/3283, whose confinement cost 3x on prefill): on a hit at `ensure(L)`
     (the request's `want` ∩ nursery), relabel the slot into the main pool (`land`-style accounting 3400,
     `me_account`, pin eligibility via the request's own `pin_grant` 4924 -- exactly what a demand read gets)
     and hand the nursery the main pool's victim slot (`pick_victim_any` 3264, never a hub pin) -- the same
     eviction the demand read would have caused. Zero bytes copied, ~us of bookkeeping, at the top of
     `ensure_layer_inner` where nothing reads the pool.
   - *Size.* Live entries ~ hints/step x (read ~1.5 lane-layers + lead 1 layer) / 80 lane-layers: < 1 at R=1,
     ~1-2 at R=2; burst bound 2 lanes x cap 8 x 2 layers = 32. 32-48 slots (0.6-0.9 GB) out of 4480 (-0.7% of
     the main band); never evicts from the main band; a wrong hint = one nursery slot recycled LRU by a later
     hint, no pin, no ledger entry, no `released_unused` needed. Recycle only at `ensure` time (slots are read
     by in-flight passes otherwise) and only slots not wanted by the request being served.
   - *Reader / sets.* A reserved staging set pair + queue position ahead of speculative jobs (behind certain)
     and no yield/chunking (the owner's "nursery" is what removes the TinyLFU/pin question; priority removes
     the busy wall 6.7 -> 2.26 ms). Nursery reads bypass `pin_apply_words` grants (skip the grant entirely --
     open question 3; the pick's own `pin_grant` pins on use) and `admit_passes`.
   - *Maps and the prior.* Do NOT report nursery slots in the pin-mode map (`apply_map` would count them as
     hub pins). Then the hub mirror says "missing" for a hinted expert: fine at rank 1 (protected), but at R=2
     the prior may swap the hinted rank-2 pick away -> the nursery entry is never used -> waste. If R=2 is ever
     live, mark hinted words INCOMING on the hub (b2_mirror.rs:300) so the prior sees them resident; with a
     nursery that is safe (own set, no drop).
   - *Restore list.* Independent: restores land in the main pool; `prefetch_words_core` skips a key already in
     `slot_of` (4327), so a nursery-resident expert is not re-read; on promotion the restore stamp logic does not
     apply (decode landing).
   - *Detection.* "Falls back to (b) when the daemon does not echo the bit" is wrong: the daemon echoes ALL
     request flags (proto comment 424-425: an older daemon "echoes it but never sets RESP_FLAG_PIN"). Use a
     RESP flag (next free 1 << 13 beside RESP_FLAG_PIN 446 / RESP_FLAG_RESID 416).
   - *Verdict on sequencing.* The nursery removes findings 2/3/7 of round 1 at the root (no gate, no drops, no
     pin pressure) and lifts hint walls to the isolated 2.26 ms. Hub-only (b) live keeps all three and buys
     ~1.0-1.6 ms/step at the modal r3-r4 cells, below the per-cell p50 resolution (~1.5%, the graph-keys A/B)
     and visible only on the paged-late count. Recommendation: slice A (dry, hub-only) as written -- it is the
     measurement (a) needs and decides R; then (a)+nursery bundled with the already-queued box-2 restart (pool
     +270, mode-evict), with the hub-side filter/queue/pacing from slices B-C shipped in the same hub binary
     but hints live (`k1`) only once the daemon answers the RESP flag. (b) live stays as the fallback if the
     box-2 restart slips past ~2 weeks, gated on the paged-late metric alone. This is a change from round 1.

6. **MINOR -- clear-at-switch hook cited wrong.** Doc 2.2 "beside `expire_incoming` 294": that is called from
   forward_prefill.rs 1409/2253/2372/2480 (per-chunk paths), not at the phase switch. The decode -> prefill
   hook is `Phase::Prefill => pin_enter_prefill()` at multistream.rs:1119; clear `MISS_HINT_WORDS` there.

7. **MINOR -- (a)'s "LIKELY ahead of spec, behind certain" vs the park loop.** Park/early-page certain reads
   (`prefetch_words_cls(.., true, ..)` 7222/7258) run at 11% `blocked_on` (src=1 data); a LIKELY hint behind
   them is right. But `background_should_wait` (1590-1596) makes every speculative job pause while a certain
   one runs; LIKELY jobs must not pause for certain ones (they are 2.3 ms and the lead is 1.4-3) -- or the
   nursery gains nothing in the 11% of windows with a certain read running. Specify: LIKELY neither pauses nor
   is paused by speculative jobs; it does not pre-empt certain ones.

8. **MINOR -- open question 2.** An absolute bar: abort when `lh2_hints_sent/step` > 10 at <= 4 rows or > 2.5
   per lane-layer at any size (a bound must not read a runtime estimate); keep "2x the dry estimate" as a
   diagnostic. Volume at R=1 is ~3/step by the trace scaling, ~27/step by the dump's own routing -- the
   absolute bar is what protects the drives if the dump is closer to the truth.

9. **NIT -- doc 3.2 "P(any running) <= 0.23".** `pf_run_spec` is sampled at dequeue (E[N] over 0..3), so
   P(N >= 1) <= 0.23 is a correct bound, but it ignores spec reads STARTING during the hint's 2.3 ms; the
   per-read 0.60 is biased to clumps. Pricing both ends is right; pacing (finding 2) pushes toward 0.23.

10. **NIT -- `before` and the shared-expert note.** Under `before` the look-ahead sits on the turnaround
    (~32 us x k dGPU + host launches); under `after` it delays the shared expert. The dry run measures the
    first; fine as planned.

## Open questions (rev 2)
1 Pace at ~1 restore/request and a per-step speculative budget; it is a win for the restore lever itself (lost
words -> 0); do not pull (a) into C for that reason -- pull it in because of finding 5. 2 Absolute bar
(finding 8). 3 Skip the grant entirely; the pick's own `pin_grant` on use is the promotion. 4 Artefact of my
round-1 log (finding 3); the daemon dequeues within ~1.4 us at the median.

VERDICT: APPROVE WITH CHANGES -- slice A may be built as written once finding 1 (knob gates the look-ahead,
9515-9539 replaced) and finding 6 are in; sections 3-4 re-run with handling ~0.1 ms and the paged-only
normalisation (finding 3, 4); pacing becomes a per-step budget (finding 2); section 3.3's (a) becomes the
nursery (finding 5) with a RESP capability flag, and the plan sequences A -> (a)+nursery on the queued box-2
restart, with (b) live as the fallback only.
