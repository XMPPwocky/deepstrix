# Predicted-miss look-ahead prefetch on the decode step

Status: DESIGN rev 4, 2026-10-06: APPROVED WITH CHANGES folded (3 review rounds). **Slices A, B and C
BUILT** on `worktree-b2-prefetch` (base main 8996153 = production b5def35): A live since 10-06 (dry),
B (hub wire) and C (daemon nursery + LIKELY class) built and host-tested 10-06, NOT deployed -- slice D
(section 11) needs the box-2 restart. Owner direction 10-06 "speculative reads into a small nursery of
slots, keep only if used in the next layer" is the target mechanism (section 3.2; changelog section 10).
Slice A live finding (10-06 11:00 UTC, 1 h): rank-1 ANY-rank recall 0.92-0.95, host 0.2 ms/step, but
~44 hint words/step at R=1 against ~2 paged replies and 1-3 `sub.blocked`: ~90% of predicted rank-1
non-resident picks land at actual ranks 2-6 where the prior swaps them away. Hence the **slice A
amendment** (section 11.1: protected-set counters + a gate-margin filter). Scratch
`~/.claude/jobs/749c61d3/tmp/prefetch_design/` (`trace_lru.py`, `recall_protected.py`, `recall.log`,
`price2.py`, `price2.log`, `nursery_draft.md`); reviews and `spec_reads.*` in `b2_prefetch_2026-10-06/`.

## 0. What and why

The box-2 reply tail (`DECODE_IDEAS_SWEEP_2026-10-06.md` section 2, `b2tail/`): ~9 replies per step arrive
after the lane's iGPU MoE; 85-99% of that lateness is replies carrying exactly ONE paged expert (a certain
demand read, 18.8 MB, 2.0-2.5 ms p50). The hub predicted 96-98% of them before the submit
(`hub_req.n_pred_miss`, forward_prefill.rs:9716-9731): the cache prior's residue, PROTECTED
(`V41_SUB_PROTECT` **live 1**; b2_mirror.rs:207-214) or out of lambda's gap. Paged replies never late
would be worth ~7.9 ms/step weighted (the sim's +9.0 for ALL late replies times the paged share 0.84-0.985).

The lever: at layer l's Route, run layer l+1's router on layer l's router input (`V41_LOOKAHEAD_PREFETCH`,
b9a2122, exists), keep the box-2-owned, mirror-non-resident, top-ranked picks and ship them as
`REQ_FLAG_PREFETCH` words in THIS layer's request. Box 2 reads them one lane-layer early into a small
NURSERY of slots, outside TinyLFU admission and the pin ledger; an entry enters the main pool only when the
next layer's real request picks it; unused entries are recycled by later hints.

## 1. Step 0: does the one-layer-early router find THESE picks? (CPU only)

`ROUTER_LOOKAHEAD_AND_PREFETCH.md` (09-14) closed "predict and prefetch" on recall-ON-MISSES 0.08-0.47:
the cold tail. Today's exposed misses are rank-1 and out-of-gap picks. `recall_protected.py`: the 09-14
predictor on its 1,006-token agentic dump (layer (l+k)'s gate on the mean-over-copies residual after
layer l; true picks `topk_ids`, rank order); ownership = per-layer top-103 by decode pick count from
`picks-sub-20261005-2337.trace`; weight `p_miss(layer, e)` from a box-2 LRU replay of the same trace
(`trace_lru.py`, 4000 slots = the live pin budget); "cold" = `p_miss >= 0.5`.

| k | pred top-N | recall true rank-1 (all / dec / enc) | rank-1, cold only | recall rank 1-2 | hints/row (sum p_miss) | precision: any true pick / true rank-1 |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.710 / 0.764 / 0.640 | 0.713 | 0.425 | 0.084 | 0.94 / 0.71 |
| 1 | 2 | 0.861 / 0.909 / 0.799 | 0.867 | 0.728 | 0.176 | 0.87 / 0.41 |
| 1 | 3 | 0.912 / 0.950 / 0.863 | 0.918 | 0.836 | 0.279 | 0.80 / 0.27 |
| 1 | 6 | 0.955 / 0.980 / 0.923 | 0.959 | 0.919 | 0.668 | 0.56 / 0.12 |
| 2 | 2 | 0.772 / 0.832 / 0.691 | 0.810 | 0.639 | 0.189 | 0.76 / 0.35 |
| 0 (shortcut bound) | 6 | 0.970 / 0.985 / 0.951 | 0.977 | 0.938 | 0.648 | 0.58 / 0.12 |

- **The 09-14 verdict does not transfer**: rank-1 recall is the same on the cold set as overall.
- One layer early costs 0.015 against the shortcut's own k=0 bound: a FLOOR for the production predictor
  (exact router input). k=2 costs 0.05 more and halves precision.
- Trace replay: box-2 picks miss at 9.7% (rank 1) .. 21.5% (rank 6); 0.383 misses/request = 31/step at 80
  requests (production `sub_predicted_miss` 31 at r4: calibrated); 0.027 rank-1 misses/request = ~2.2/step
  (production `sub_blocked` 3, paged replies 3.4).
- Volume (cross-dataset, caveat): by the trace's miss rate ~3 hinted reads/step at R=1, ~6-7 at R=2; by the
  dump's own routing ~27/step at R=2. The dry run's per-R counts and the absolute abort bar (section 5)
  settle it.

## 2. Hub mechanism

Code map (forward_prefill.rs): knobs 1576-1591 / 1823-1831 / 1874-1878; `look_next*` 8318-8325 (and
8875-8879), gated TODAY on `lookahead_prefetch()` and `lookahead_depth() >= 2`; launches 8523-8534 into
the shared `sd.look_sel*` (batch_scratch.rs:721-726) after the captured router stage (`cap.end()` 8312),
before the readback pack 8579-8600; unpack 8972-8985; the OLD word block 9515-9539 (top-5 ranks, box-2
owned, no residency test) -> `push_prefetch_words` (remote_experts.rs:3688) -> `submit_inner` 8510-8518.

1. **Knob gating (round-2 finding 1).** `look_next` exists iff `lookahead_prefetch() || miss_prefetch() !=
   off`; `look_next2` iff `k2` (not `lookahead_depth`). The 9515-9539 block is REPLACED by the filter; the
   old behaviour survives only under the legacy env var with the new knob `off`. Host test: `k1` with the
   legacy var unset pushes nothing from 9515 into `PREFETCH_WORDS`.
2. **Filter** (pure `het::lookahead::hint_words`): predicted rank `<= R` (`V41_B2_MISS_PREFETCH_RANK`,
   default 1; dry counts 1/2/3 at once), `expert_pager::partition_box2(L, e)` (507-515), and
   `b2_mirror::lookup(L, e)` with `!held && !pending && !incoming` (the 9725 predicate; not `resident()`
   357, which drops `pending` under live `V41_SUB_PENDING=0`). No admit gate at rank `<= 2` (a protected or
   gap-blocked pick is read anyway; `admit_passes` refuses 22 of 26 candidates/step at r4). Dedup within
   the lane-layer, against the queue, and against the mirror's NURSERY bits (3.2). Cap
   `V41_B2_MISS_PREFETCH_CAP` 8 words/request.
3. **Own queue, step-tagged, first on the wire.** `MISS_HINT_WORDS` beside `PREFETCH_WORDS` (3685), drained
   by `submit_inner` before admissions and restores; each word carries the mirror STEP (`begin_step` 284);
   a word from an earlier step or whose layer `<=` the carrying request's is dropped stale; the queue is
   cleared at the decode->prefill switch, i.e. beside `Phase::Prefill => pin_enter_prefill()`
   (multistream.rs:1119), not `expire_incoming` (a per-chunk path). Any lane's next decode submit may
   carry a word.
4. **Per-step speculative budget (finding 2).** Box 2 can START ~40-110 speculative reads per 84 ms step
   (3 readers / 2.26-6.7 ms); the hub sends up to 16 restores per request (`pin_restore_per_request`,
   b2_mirror.rs:505) = 1,280/step after a switch, and words beyond the 16 staging sets are dropped at
   application (4342-4350) and discarded (`let _ =` 4190): 6-8 lost per step on average, ~3,500 per
   480-step phase period. Replace the per-request caps with one budget: restores + admissions + hints
   `<= V41_B2_SPEC_BUDGET` (default 60) per step, hints first (bar <= 10), then admissions (4-11/step),
   then restores (~1/request) with a FLOOR of 16/step for restores (insurance against an admissions
   spike; leftover restore words survive a phase flip via `pin_enter_prefill` 1086-1088, so starvation is
   impossible; the real cap is the daemon's start capacity ~40/step under busy walls): a 1,000-word
   restore refills in ~1 s with ~0 loss instead of a burst that lands a fraction. A win for the restore
   lever on its own; on regardless of the hint knob. SHIPS IN SLICE A (code review 10-06, finding 1): knob
   default 0 = today's per-request caps, so the slice-A restart changes nothing with the hint knob off; flipped
   to 60 live as its own per-turn A/B (restore refill time after a phase switch, `pf_d_dropped`, `b2_pinned`,
   paged late replies in the 0-20 s phase bin).
5. **Shared scratch is safe under the pack.** The drivers' `lookahead_hints_ok = false` (4189-4195,
   4507-4510) guards the COPY path (`rb_stream` 9017-9036); the pack reads `sd.look_sel*` on `de.compute`
   in stream order (8573-8575). Change: `lookahead_hints_ok = rb.packed && knob on`.
6. **Placement and host cost.** ~4-6 host API calls (~40-100 us) per lane-layer on the single ready-first
   thread (4601-4668; turnaround slack 0.03 ms at r4) plus ~32 us x k of dGPU before `selected_ready`.
   Slice A measures `lh.pre_moe`, `lh.sel_sync`, `lh.remote_submit`, `ms.step` per cell, dry on/off,
   with the launches where they are (`before`). If the deltas show it: (i) capture the look-ahead as its
   own stage keyed by (stage, rows) under the graph-keys infrastructure (3eccfdf9); (ii)
   `V41_B2_MISS_PREFETCH_PLACE=after`: launches after `selected_ready`, a 1-segment pack into a per-lane
   `rb_look`, a per-lane `look_ready` event, a non-blocking `hipEventQuery` at submit (not done -> no hints,
   `lh2_look_late`); `after` delays the shared expert by ~35-70 us, so `before` is its control.

Why this is not 09-21 again (1823-1831): ~3-7 words/step not 40; a mirror residency filter; rank cut 1-2
not 5; the words land in a nursery, not the pool; `route=split`, the priority queue and certain early
paging (7196-7224) postdate 09-21.

## 3. Box 2

3.1 **The path a plain PREFETCH word takes today, measured** (`spec_reads.log`, 9,750 src=3 reads =
cache-prior admissions + pin restores; stamps are CLOCK_MONOTONIC_RAW ns, so round 1's "1.35 ms dequeue"
was a unit error: frame->dequeue is 1.35 **us** median over all decode requests, 16-18 us on two-lane
cells (`late.log` b2q), 0.6-1.0 ms p90 on late requests; dequeue->hints_end 0.26 us). `prefetch_words_core` (4321-4355) pushes the word to the BACK of
`PfQueue` (1541-1552); a reader yields while demand/certain reads run (4245-4262, 1590-1598), reads in
1 MB chunks that pause for certain ones. Wall med **5.0-5.2 ms** (isolated 2.26 = a certain read; busy
6.6-6.8, 60-67% of these reads, clumped admissions/restores; time-weighted `P(any running) <= 0.23` from
`pf_run_spec`, ignoring reads that start during the hint). hint->pop p90 29-31 ms behind queued bursts;
drops 6-8/step (3.2 of round 2). `admit_prefetched` (4439-4528) promotes a pending word to certain at the
demand `ensure` and waits for it; landed words are granted pin eligibility (`pin_apply_words` 4899-4919)
and pinned at the next report. This path is the INTERIM (b) in section 4; its three defects (gate, drops,
pins) are what the nursery removes at the root.

3.2 **The nursery (target; owner direction, round-2 finding 5).** Pool facts (remote_experts.rs):
residency is `slot_of: HashMap<(layer, e), slot>` (2244) plus the per-layer `remap_hosts` (2253,
`-(slot)-1 | 0`); weights never move between slots; `pick_victim` 3205-3257 (never `want`/`extra`/pinned,
floors), `evict` 3291-3303 (THE choke point), `land` 3400-3408, `touch_hit` 3307-3330, landing of
background reads 4528-4617; `PinBook::report` 2858-2893 already treats a staging slot as "landed but never
pinned"; `residency_words` 4841-4858 = `remap_hosts != 0`.
- **A SET of slot ids, not a range.** Per-slot `nursery: Vec<bool>`, N slots. Main-band searches (claims,
  landings, restores) skip nursery slots; a `Band::Stage`-style range (2503/3283) is NOT used: confinement
  cost prefill 3x (09-27).
- **Landing** (in `admit_prefetched`, 5053-5056, like every background read). The hint lands in the
  nursery slot recycled by: UNUSED first (an entry whose target layer has been served by `>= lanes`
  requests since it landed -- the other lane's layer-L request arrives one period later), then LRU; never
  a slot the request being served wants; only at `ensure` time (slots are read by in-flight passes
  otherwise). `evict` the occupant (unpinned by construction, `on_evict` false; `nursery_recycled` if it
  was an unused hint), repack in place, map it with a nursery stamp (`last_use` 0-class; main searches
  skip nursery slots anyway); `land` (3407) sets the layer `dirty`. Nursery entries are NOT counted in
  `held[layer]` (a separate `nursery_held`): `pick_victim` 3232 protects a foreign layer only while
  `held > floor`, so a counted nursery entry would let a main slot of a layer AT its floor be evicted.
  No `pin_apply_words` grant for LIKELY words, no `admit_passes`, no ledger entry. `is_resident_pool`
  (7212) does see the entry, so the early-page hook issues no certain read for it: the hit at `ensure` is
  what makes the hint pay.
- **Promote on use, no copy -- inside `touch_hit` (3307), the hit loop (5136).** `if nursery[slot] {
  clear the flag; nursery_held -= 1; held[layer] += 1; nursery_hits += 1; refill }` before the decode-hit
  stamp `tick + PREFILL_AGE` (3327), which is exactly what a demand hit gets; no remap write (the remap
  already points at the slot), no `dirty`. Pin eligibility comes from the request's own `pin_grant`
  (4924; per request including a merged partner, dab7e25), so no promotion grant. Refill: the main
  pool's victim via `pick_victim_any` (3264, never a hub pin, floors honoured, the serving request's
  `want` excluded), `me_account` on that VICTIM with the serving request's `prefill_mode` (the
  `claim_miss` order 3378-3379), then `evict`, then flag it nursery -- the eviction the demand read would
  have caused. Zero bytes copied, ~us of bookkeeping. No victim -> the nursery shrinks by one
  (`nursery_shrunk`) and refills at the next free. Restore list / `decode_delta`: untouched (a decode
  landing).
- **Size.** Live entries ~ hints/step x (read ~1.5 lane-layers + lead 1) / 80: < 1 at R=1, 1-2 at R=2;
  k1 burst bound 2 lanes x cap 8 = 16. Knob `nursery` (`V41_B2_NURSERY`, default 0 = off), **32 by
  default** (room for cap bursts and k2; 0.6 GB), 48 only if the dry run shows dump-rate volume; carved
  from the pool on the +270 restart (net +238; 32 from today's 4480 is -0.7% and also fine).
- **LIKELY reader class.** `PfJob.class = Certain > Likely > Spec`; `push` puts a Likely job behind
  certain, ahead of speculative. Readers: LIKELY counts against the speculative cap, `running_likely +
  running_spec < max_spec` (3 of 4; the 4th reader stays the certain reserve), and pops regardless of
  `running_certain` (no yield), so it can never occupy all four readers and make a certain job wait.
  Reads stay CHUNKED (an isolated chunked wall = a certain wall, 2.26 ms, so chunking is free) and PAUSE
  for certain reads under daemon knob `likely_pause_for_certain` (default on: a certain read is running
  ~19% of the time and is itself a late reply; an unpaused LIKELY read would stretch it ~1 ms under
  `route=split`'s linear bandwidth share -- ~0.8 ms/step of added lateness vs ~1 ms/step of hint lead
  lost when pausing; the data cannot pick, so the knob is A/B'd), never for speculative; speculative pauses
  for LIKELY (`background_should_wait` 1590-1596 gets `|| running_likely > 0`). `promote()` (1559-1581)
  and `reclassify_certain` handle Likely -> Certain like Spec -> Certain (the park/early-page certain path
  7222/7258 relies on it). A reserved staging-set pair = two NEW pinned buffers (+113 MB; the 16 sets are
  sized by `V41_B2_MISS_PAR`, and `miss_par` caps demand concurrency at `min(knob, stages.len())` 5177, so
  taking two of them would change demand concurrency and the 2x reserve rule). That is what takes the
  hint wall 6.7 -> 2.26 ms. Request bit 2048 (`REQ_FLAG_LIKELY`, next free after `LONG_JOB`) marks a
  decode frame's PREFETCH words as Likely; the early-page hook (7196-7224) may apply them at frame
  arrival, worth the b2q tail only (~0.1-0.2 ms/step). `b2_read.src` 4.
- **Capability = a RESP flag.** The daemon echoes ALL request flags (proto 424-425), so an echoed bit
  proves nothing: `RESP_FLAG_NURSERY = 1 << 13` (beside PIN 1<<14, RESID 1<<15), set by a daemon that
  understood `LIKELY`; the hub sends hints live only after seeing it, else (b).
- **Maps and the mirror.** `residency_words` (4841-4858) masks nursery slots and `PinBook::report`'s
  `state()` (2864-2869) becomes `pinnable = slot < stage && !nursery[slot]` (else `apply_map` would count
  them as hub pins); with the flag, 12 NURSERY words are appended after the pin block. Hub
  `b2_mirror` keeps a NURSERY bitset per layer for dedup (never re-hint a nursery entry) and counters
  (`lh2_nursery_covered`), NOT for `held`: the prior stays hint-blind, so I2 holds exactly even with the
  prior on and `n_pred_miss` keeps its meaning. At R=1 "missing" is right (the rank-1 pick is protected).
  If R=2 is ever live, mark hinted words INCOMING (b2_mirror.rs:300) so the prior does not swap the hinted
  rank-2 pick away and waste the entry; safe with a nursery (own set, no drops).
- **Restore list / pins.** Independent: restores land in the main pool over decode victims
  (`RESTORE_INFLIGHT` 2); `prefetch_words_core` skips keys already in `slot_of` (4327), so a nursery
  resident is not re-read; promotion is a decode landing (no restore stamp). Nothing a hint does touches
  the TinyLFU watermark, `released_unused`, or a pin until the pick itself pins it.

3.3 **Interim (b), hub-only.** Hints as plain PREFETCH words down 3.1's path, with the per-step budget
(2.4) keeping the readers idle for them and the sets free. Keeps the gate/drop/pin defects in weakened
form; live only as the fallback (section 8).

## 4. Pricing (`price2.py`, `price2.log`; quote +-15%)

Removal curves = `price.log` "paged lateness removed if every paged reply were X us earlier" per cell,
weighted by regime share (`whatif.py`), normalised x0.71 so the paged-only ceiling is 7.9 ms (the sim's
+9.0 x paged share 0.87; per-cell sim/measured ratios 0.71-1.13). Lead = lane period (2 x the lane's MoE
median: r4 1.66 ms, r6 2.1, r7 2.5, M8 3.2; single lane ~1.4) minus 0.1 ms link. Saved per caught hint:
idle min(lead, 2.26); busy lead x 2.26/6.7. Times rank recall (R=1 0.71 / R=2 0.86) x residency 0.97.

| variant | R=1 ms/step weighted | R=2 | per cell R=1, raw (r3 / r4 / r6 / M3 / r7 / M8) |
|---|---|---|---|
| (b) hub-only, busy share 0.60 (per-read) | 2.2 (2.2%) | 2.6 | 1.7 / 2.0 / 3.3 / 2.0 / 5.0 / 6.3 |
| (b) hub-only, busy share 0.23 (time bound) | 2.9 (2.9%) | 3.4 | 2.4 / 2.7 / 4.6 / 2.8 / 6.4 / 7.4 |
| **(a) nursery + LIKELY** | **3.6 (3.7%)** | **4.2 (4.3%)** | 3.1 / 3.5 / 5.6 / 3.7 / 7.7 / 8.5 |
| (a) + k2 | 4.3 (4.6%) | 5.0 (5.4%) | 5.0 / 4.7 / 5.9 / 6.0 / 7.4 / 8.3 |
| ceiling (paged never late) | 7.9 | | |

Reading: the nursery is worth 3.6-4.2 ms/step (R=1-2), ~half the paged ceiling; hub-only 2.2-3.4, and at
the modal r3-r4 cells only 1.7-2.7 raw -- below the per-cell p50 resolution of a graph-keys-style A/B
(~1.5%), visible only on paged late replies per step. k2 adds ~0.7-0.8 at 2x the hinted reads (and 2x the
nursery). (a)-at-arrival vs at-dequeue is the b2q tail only.

## 5. Policy safety

- Hub knob `V41_B2_MISS_PREFETCH = off | dry | k1 | k2` (default `off`; live knob file); `_RANK` 1-3
  (default 1), `_CAP` 8, `_PLACE=before|after`, `V41_B2_SPEC_BUDGET` (default 0 = today's caps; 60 is the A/B value). Box 2: `nursery` slots
  (0 = off; file key, restart), LIKELY class on iff nursery > 0.
- Nothing adaptive. If R is ever adaptive: epsilon-uniform over {1,2,3}, forget by time, bounded sample.
- Scope: `RowLayout::Arena` decode rows, `remote_split_on`, learned routers (8319, 1848-1860).
- **Absolute abort bars (a bound must not read a runtime estimate):** `lh2_hints_sent/step` > 10 at <= 4
  rows, or > 2.5 per lane-layer at any size -> `off`; "2x the dry estimate" stays a diagnostic.
- Rollback: knob `off`; box 2 `nursery=0` + SIGUSR2 (reads already in flight land as speculative).

## 6. Observability (behavior-free counters, slice A)

`hub_step` (evtrace_kinds.rs:106-142, beside `sub_*` at multistream.rs:2170-2185): per R in {1,2,3}
`lh2_cand_rN`, `lh2_nonres_rN`, `lh2_dry_hits_rN` (non-resident AND in the router's OWN picks `sel_orig`
at layer L next lane-layer: the live Step 0); `lh2_hints_sent`, `lh2_demanded`, `lh2_demanded_prot`,
`lh2_paged_hinted` (tail recall), `lh2_nursery_covered`, `lh2_look_late`, `lh2_dropped_cap`, `lh2_stale`,
`lh2_budget_deferred` (restores/admissions held by the budget). `hub_req`: `n_hint_words`. Host timers
`lh.pre_moe` / `lh.sel_sync` / `lh.remote_submit` dry on/off. Box 2 `b2_req`: `nursery_lands`,
`nursery_hits`, `nursery_recycled` (= wrong hints, churn), `nursery_drops` (no set), `nursery_occupied`,
`nursery_shrunk`; invariant per step `lands = hits + recycled + delta(occupied)`. `b2_read` src=4 rows
joined to hint words by (layer, expert, time): queueing, wall, `wanted`. Precision = hits/lands (Step 0:
0.71 at R=1); churn/step; main evictions/step unchanged vs nursery off minus promotions. Lateness per class
reuses `b2tail/late.py`.

## 7. Invariants and gates

I1 (numerics): `d_selected`, `d_ew`, `xq`, every kernel input and partial byte-identical knob on/off;
the look-ahead writes only `sd.router_logits` after its last reader (8516-8522), `sd.look_*`, the pack.
I2 (routing): with `V41_SUB` unset the pick trace is identical; with the prior on, the prior never sees a
nursery entry (3.2), so picks are identical too; promotion changes only when an expert enters the main
pool, never what is computed. I3 (safety): a hint never blocks the host, never names an unpaged layer
(4324), never pins, never evicts a hub-pinned or floor-protected slot, and a nursery slot is never in
RESID or the pin map.

Gates:
- Hub host tests: `hint_words` (predicate, dedup incl. NURSERY bits, cap, step tag, stale, clear at
  multistream.rs:1119, knob-off = empty, legacy-var-unset pushes nothing from 9515); queue order and the
  per-step budget; RESP-flag detection and fallback.
- Daemon unit tests: nursery recycle order (unused-first then LRU, never a wanted slot, only at `ensure`),
  relabel inside `touch_hit` + refill victim honouring pins/floors with `me_account` on the victim,
  `held` excludes nursery entries (a layer at its floor keeps its main slots), shrink/refill, maps and
  `pinnable` mask nursery slots, `on_evict` false on a recycle, LIKELY queue/reader rules (behind certain,
  ahead of spec, `running_likely + running_spec < max_spec`, pause knob both ways, promote/reclassify
  Likely -> Certain), reserved set pair; the 6-seed randomized protocol sim extended with LIKELY words and
  the NURSERY map; `lands = hits + recycled + delta(occupied)`.
- iGPU loopback (beside `remote_experts_pin_loopback`): hints on/off partials bit-identical, promoted
  entries served from their slot, 0 pinned evictions, 0 surprises.
- `tests/multistream_step.rs` G5a-h, `V41_SUB` unset, `k1` vs `off`: logits/final state bit for bit, pick
  trace diff, every b in 1..8; the `hints_sent > 0` assert conditional on a remote (box-1-only windows).
- Determinism recipe: `V41_T2_CATCHALL=2`, T=0, same prompt x3, knob on/off, identical sha.
- A/B (pre-registered, per-turn knob flip, >= 2 h warm, box 2 on the nursery build): PASS = paged late
  replies/step -25% at r4 AND `ms.step` p50 <= 1.00x in every cell with >= 200 steps; diagnostics: src=4
  wall and hint->pop, `nursery_hits/lands`, `nursery_recycled`, `b2_pinned` flat, box-2 `compute_us`
  +<= 5%. Aborts: section 5 bars; `nursery_drops` > 10% of sent; any cell p50 > 1.01x.

## 8. Build plan (sequence per round-2 finding 5)

- **A. Hub-only dry run = the measurement (a) needs (CLEARED, round 3).** Ships together: knob gating
  (2.1) WITH the 9515-9539 replacement (else `dry` either launches nothing or leaks the legacy words);
  `dry` never calls `push_prefetch_words`; `look_next2` off unless `k2`; `lookahead_hints_ok` left as is
  in A; filter, queue, per-R counters, section 6 hub counters, host timers + the per-cell `ms.step` watch
  (dry on/off per turn); the per-step budget (2.4) ships in A at default 0, A/B'd live on its own. Decides R
  and the host-cost question.
- **B. Hub filter/queue/pacing/RESP detection** in the same hub binary: `k1` sends only once the daemon
  answers `RESP_FLAG_NURSERY`; the per-step speculative budget on regardless; `lookahead_hints_ok =
  rb.packed`.
- **C. Daemon nursery + LIKELY reader** (`nursery`, `REQ_FLAG_LIKELY`, `RESP_FLAG_NURSERY`, masked maps,
  NURSERY words) with the unit + loopback gates of section 7.
- **D. Two-box deploy** bundled with the already-queued box-2 restart (pool 4480 -> ~4750, mode-evict), in
  the two-box order; then the per-turn knob A/B.
- **E. k2** (and INCOMING for R=2) only if A's dry counts say the remainder is worth 2x the reads.
- **Fallback:** (b) live, hub-only, only if the box-2 restart slips past ~2 weeks, gated on the paged-late
  metric alone.

Rollback at every slice = knob `off` / `nursery=0`.

## 9. Risks and what would kill it

- Live `lh2_dry_hits_r1 / sub_blocked` well below 0.71 (proxy predictor, one conversation, in-sample
  ownership: box-2 share 12% in the dump vs ~27% live).
- Hint volume at the dump's rate (~27/step at R=2) rather than the trace's: the absolute bars stop it; the
  nursery then needs 48 slots and R stays 1.
- Host cost on the ready-first thread (40-100 us per lane-layer vs 0.03 ms slack) exceeds the gain at
  r3-r4: captured stage, then `after`, measured in A.
- The LIKELY class starves certain reads if mis-ordered, or a nursery refill victim search runs inside a
  pass: both are gate-tested invariants (never pre-empt certain; recycle only at `ensure`).
- The 990 Pro's concurrency scaling is unmeasured (`b2_read` r*_start/end answers it).
- Box-2 restart slips: hub-only fallback is worth 1.7-2.7 ms/step raw at r3-r4, invisible at p50.

## 10. Changelog

rev 4 (reviewer round 3, APPROVE WITH CHANGES; slice A cleared): 1 MAJOR LIKELY reads stay chunked and
pause for certain reads under `likely_pause_for_certain` (default on, A/B'd), never for speculative;
speculative pauses for LIKELY (3.2). 2 MAJOR LIKELY counts against the speculative reader cap, pops
regardless of `running_certain`, `promote()`/`reclassify_certain` handle Likely -> Certain (3.2). 3 MINOR
per-slot state: relabel inside `touch_hit`, landings in `admit_prefetched`, nursery excluded from `held`,
`me_account` on the refill victim, `pinnable` mask, grant skipped, `dirty` by the landing only,
`is_resident_pool` sees the entry (3.2). 4 MINOR recycle rule = unused-first (served by >= lanes requests)
then LRU, never a wanted slot (3.2). 5 MINOR restore floor 16/step (2.4). 6 NIT b2q medians by cell (3.1).
Open questions: nursery 32 (48 only at dump-rate volume; carve on the +270 restart), the request's own
`pin_grant` suffices, reserved sets = two new pinned buffers, floor 16. Slice A contents fixed (8).
rev 3 (reviewer round 2 + owner nursery): 1 MAJOR knob gating fixed, 9515-9539 replaced, host test (2.1).
2 MAJOR pacing = per-step speculative budget, hints first, ~1 restore/request; stated as a win for the
restore lever (2.4). 3 MAJOR stamps are ns: handling 0.36 -> 0.1 ms, lead +0.26 everywhere; arrival-vs-
dequeue collapses to the b2q tail (3.1, 3.2, 4, `price2.py`). 4 MINOR paged-only ceiling 7.9, factor 0.71,
+-15% (4). 5 MAJOR (a) = the nursery as the target: slot SET with relabel-on-use, refill victim, size 32-48,
LIKELY class with a reserved set pair, masked maps + NURSERY words, RESP capability flag, INCOMING only at
R=2, sequencing A -> (a)+nursery on the queued box-2 restart, (b) fallback only (3.2, 8). 6 MINOR clear at
multistream.rs:1119 (2.3). 7 MINOR LIKELY pause semantics (3.2). 8 MINOR absolute abort bars (5). 9 NIT
busy-share bound caveat stated (3.1). 10 NIT `before` measured first (2.6). Rev-2 open questions closed
per the reviewer (pace ~1/request; absolute bar; skip the grant; 1.35 ms was an artefact).
rev 2 (round 1): pricing on the measured speculative path; no admit gate at rank <= 2; drops measured;
host cost priced; `lookup()` predicate; `note_admits` (now moot: hints never pin); pins stated; per-R dry
counts; step tag; slice A a measurement; G5 assert conditional; `before` control.

## 11. Build notes: slices B and C (2026-10-06, `worktree-b2-prefetch`, commits 5646edf..)

Code: hub `het/lookahead.rs` (filter, queue, budget, bars, counters), `het/b2_mirror.rs` (NURSERY bits,
capability), `het/remote_experts.rs` (proto, `submit_inner` drain, `wait` parse; and the whole daemon:
`ShardPool` nursery, `PfQueue` LIKELY class, `admit_prefetched` landings, serve loop), `knobs.rs`,
`het/evtrace_kinds.rs`, `het/forward_prefill.rs` (pack `look_ew`, demanded/protected counting),
`het/batch_scratch.rs` (pack sizing); tests `tests/remote_experts_nursery_loopback.rs` (GPU; compiled,
not run: the hub was live on both GPUs).

11.1 **Slice A amendment (protected set + margin).** The pack carries L+1's look-ahead gate WEIGHTS
(`sd.look_ew`, 6 f32/row; the pack already carries `xq` at 5,840 B/row, so +24 B/row at r4; L+2's are
not packed: `RB_PACK_MAX_SEG` 10, slice E). Per rank-1 prediction `margin = (w1 - w2) / sum(row)`, the
largest over its rank-1 rows. Dry counters: `lh2_dry_hits_prot_rN` (own pick at ACTUAL rank `<=
V41_SUB_PROTECT`), `lh2_nonres_m{0..3}` / `lh2_hits_prot_m{0..3}` (margin `>= 0 / 0.1 / 0.2 / 0.3`); the
precision that matters is `dry_hits_prot_r1 / nonres_r1`, and `hits_prot_m{k} / nonres_m{k}` picks the
threshold. Knob `V41_B2_MISS_PREFETCH_MARGIN` (default 0 = off) gates rank-1 hints on the wire path; an
unknown margin (no weights packed) passes.

11.2 **Wire.** `REQ_FLAG_LIKELY` (2048) on every decode request under `k1`/`k2` (the probe); hint words
lead the PREFETCH block marked by `LIKELY_WORD_BIT` (bit 31: an older daemon skips them by its layer
check); `RESP_FLAG_NURSERY` (1<<13) + a 12-word NURSERY block (own layer first, others in rotation,
`NURSERY_NONE` padding) on every reply to a LIKELY-flagged request while `nursery > 0`. The hub mirror
replaces the reply layer's NURSERY row and dedups hints against it (`lh2_nursery_covered`), never `held`.
`Mode::wire = asks && nursery_supported`; detection and loss logged once per transition; a reconnect
re-probes. The drain honours the budget and the bars; the surplus is re-queued at the front.

11.3 **Daemon.** Nursery = per-slot set carved from the main band at `enable_paging` (spread `i * main /
n`; occupants become unused entries); `pick_victim` skips it; `touch_hit` relabels on a hit and `ensure`
refills via `pick_victim_any` + `me_account` (shrinks when every candidate is pinned; refills from free
slots later); `held` excludes entries; `PinBook::report` and `residency_words` mask them; pin budget minus
`nursery`. LIKELY class in `PfQueue`; reserved staging pair = the LAST two sets (+2 when `nursery > 0`);
LIKELY reads chunked and pausing for certain reads under `likely_pause_for_certain` (off = unchunked:
nothing to pause for); under `route=urgency` LIKELY behaves as speculative for routing (primary drive, no
pause). A hint the serving pass wants lands straight into main (a land + a hit). Counters in `b2_req`
(`pf_run_likely`, `pf_q_likely`, `n_likely_words`, `nursery_*`) and the 2,000-request stats line.

11.4 **Knobs.** Hub (live, `knobs.rs`): `V41_B2_MISS_PREFETCH` off|dry|k1|k2, `_RANK` 1, `_CAP` 8,
`_MARGIN` 0, `_MAX_WORDS_STEP` 0 (= `10 + 5 * max(0, rows - 4)`), `_MAX_PER_LL_X10` 25 (judged from 8
lane-layers), `V41_B2_NURSERY_PRIOR` 0, `V41_B2_SPEC_BUDGET` 0. Daemon (`~/expertd-knobs.txt` keys):
`nursery` 32 (STARTUP ONLY; 0 = off), `nursery_lanes` 2 (live), `likely_pause_for_certain` 1 (live, the
A/B knob; SIGUSR2).

11.5 **Slice D checklist.** (1) Build expertd ON box 2 (clock skew: `project_v41_box2_build_clock_skew`),
`nursery=32` in its knob file, `V41_B2_PREFETCH_SETS` as today (+2 reserved sets, +113 MB pinned). (2)
Server-down window with box 2 attached: `tests/remote_experts_nursery_loopback` and
`remote_experts_pin_loopback` on box 1's iGPU; `tests/multistream_step.rs` G5a-h `k1` vs `off` bit for
bit (needs a remote); the determinism recipe (section 7). (3) Two-box restart bundle in the two-box order
(box 1 down, box 2 down, box 2 up, box 1 up): pool 4480 -> ~4750 slots (+270, minus the 32 carved) and
mode-evict per `project_hardware_move_2026-10-03` / `project_v41_lm_mode_evict_deployed_2026-10-01`.
(4) Hub up with `V41_B2_MISS_PREFETCH=dry`: confirm `hub_lh2` emits, `b2 mirror: box 2 answers
REQ_FLAG_LIKELY` is NOT logged (dry does not probe), then flip `k1` per turn: expect the detection log
once, `lh2_hints_sent > 0`, `lh2_bar_trips` 0 at the chosen margin, box 2 `nursery lands/hits`. (5) A/B
per section 7 (paged late replies/step -25% at r4, `ms.step` p50 <= 1.00x); rollback knob `off` /
`nursery=0` + restart.

## Open questions for the reviewer (rev 4)

Rev-3 questions 1-4 are closed (nursery 32, own `pin_grant`, two new pinned buffers, floor 16). Open for
slice C: the `likely_pause_for_certain` A/B design (per-turn daemon knob flip via SIGUSR2 judged on paged
late replies and src=1 walls), and whether the nursery's unused-first rule should count merged partners as
one request or two.
