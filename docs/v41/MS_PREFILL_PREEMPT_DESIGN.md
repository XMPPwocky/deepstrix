# Multistream scheduler: preempt decode for FRESH prefills (design, rev 2 APPROVED (Fable r2), 2026-10-11)

Owner 10-11: "preempt definitely, interleaving sounds bad bc eviction". Interleaving decode steps INSIDE a prefill
window is out of scope -- and rev 2 closes the places where today's code already does it (3.1).

rev 2 = review r1 (REVISE): one window/group HOLD for every phase-ending path (MAJOR-1); the no-spare-state limitation
stated and measured, suspend-at-checkpoint as a follow-up (MAJOR-2); interrupts CAPPED in length, GAP default 2 s
(MAJOR-3); the stalled-streams flip routed through the same decision (MINOR-4); A/B with prefill_budget_last set
(MINOR-5); service accounted per unit incl. finish/replay (MINOR-6); SJF by remaining rows with aging (MINOR-7);
interrupt charged to neither phase via `phase_since += elapsed` (simplification); knob snapshot per tick.

## 1. Today (multistream.rs `Sched::tick`, ~1086-1184) and what it costs

Bursts with hysteresis: with both prefill work and live streams, the scheduler stays in a phase until its budget
elapses (`V41_MS_PREFILL_BURST_MS` 120 s, `V41_MS_DECODE_BURST_MS` 30 s; 3 s in production since 10-11 19:01 UTC) or
the phase runs out of work; a GROUP-MAJOR layer-major group holds the prefill phase (`finish_group` &&
`lm_mid_group`); queue order SJF with aging (60 s) and a starvation rule (600 s); two prefill jobs in flight
(`V41_MS_PREFILL_JOBS`), round-robin when no window is open (`checkpoint_ok` picks a job with an open window first).

Measured 10-11 (1,486 `ms.phase` switches, ~14 h, decode budget 30 s): every decode phase that ended had a prefill
waiting; 464 of 744 ran the full 30 s -> a ready prompt waited ~15 s mean for its prefill to START (more than its
whole first window, ~9 s). Prefill phases froze live streams p50 13.4 s / p90 45 s / max 170 s. The 3 s budget cuts
the wait to ~1.5 s mean but leaves live streams 3 s per ~123 s during a long prefill.

## 2. Goal

A new prompt starts prefilling at the next tick (one decode step, ~60-100 ms); interrupts are bounded so live
streams never freeze longer than one capped interrupt; long prefills keep a fair alternation with decode; no decode
step ever runs inside an open prefill window.

## 3. Mechanism

### 3.1 One HOLD for every phase-ending path (review r1 MAJOR-1; also fixes today's code)

`lm_mid_group` (`w.sub > 0 && w.group + 1 < groups`) is FALSE for an open LAYER-OUTER window (its passes advance
`lo_layer`, not `group`/`sub`), so today the budget switch (ms:1128) and the embed gate (ms:444) can end the prefill
phase BETWEEN passes of one window: decode steps inside a window, box 2 evicts the window's set, the next pass re-pages
it. One predicate, used by EVERY path that ends a prefill phase (budget end, interrupt end/cap, embed gate):

    hold = prefills.iter().any(|p| !p.job.checkpoint_ok()) || (finish_group() && prefills.iter().any(|p| p.job.lm_mid_group()))

(review r2 MINOR-A: the OPEN-WINDOW half is unconditional; `V41_MS_FINISH_GROUP` keeps gating only the group-major
hold it documents.)

(`checkpoint_ok` = no window open.) Bound: one window (3.6-4 s steady at 1024-row lanes, ~9 s cold) or the rest of
one group-major group. The starvation rule still wins (it only ever forces PREFILL).

### 3.2 Fresh vs continuing prefill jobs

`Prefill.service: Duration` accumulates the wall time of each of this job's `prefill_job_tick` calls (chunks AND
finish/replay/admission; not wall since start -- a job waiting through a decode phase does not age). A job is FRESH
while `service < V41_MS_PREEMPT_FRESH_MS` (live, default 20000). FRESH work = a fresh job in `prefills`. A
checkpoint-resumed prompt is a new `Prefill` and fresh again (correct: its remaining work is new). A job past the
threshold is CONTINUING and follows the burst alternation.

### 3.3 Preemption (decode -> prefill) and its cap

Decision state: `interrupt_since: Option<Instant>`, `last_interrupt_end: Instant`. In a Decode phase with live
streams, if `V41_MS_PREEMPT` (live flag; ships OFF) and FRESH work exists and
`now - last_interrupt_end >= V41_MS_PREEMPT_GAP_MS` (live, default 2000 = today's decode budget scale): switch to
Prefill as an INTERRUPT (`interrupt_since = now`), whatever the decode budget says.

An interrupt ends (back to Decode) when (a) no FRESH work remains, or (b) it has lasted `V41_MS_PREEMPT_MAX_MS` (live,
default 10000 ~ one cold window + replay) -- in both cases only at a point where `hold` is false (3.1). On ending,
`phase_since += interrupt elapsed` (the interrupted decode phase resumes with its remaining budget; the interrupt is
charged to neither burst -- as the embed phase already does at ms:451) and `last_interrupt_end = now`. Remaining fresh
work preempts again after the GAP (its jobs stay fresh until FRESH_MS of service). Decode share floor with continuous
fresh work: GAP / (GAP + MAX + hold) ~ 2 / (2 + 10 + 4) = ~12%; with FRESH_MS = 20 s one job takes at most ~2
interrupts before it becomes continuing.

A NORMAL (budgeted) prefill phase is unchanged except job selection (3.4); fresh jobs simply run first in it.

Every OTHER exit from an interrupt clears it WITHOUT shifting `phase_since` (review r2 MINOR-B): (i) starvation while an
interrupt runs -> reason `starved`, a NORMAL prefill phase from now (`phase_since = now`); (ii) all live streams
finish during an interrupt (`have_dec` false) -> a NORMAL prefill phase, nothing to resume; (iii) `stalled` (3.6).

### 3.4 Job selection inside a prefill phase (review r1 MINOR-7)

Unchanged: a job with an open window keeps the prefill until the window closes. Otherwise: an AGED fresh job
(queued >= `V41_MS_AGING_S`) first, then fresh jobs by fewest REMAINING rows (`total - done`), then continuing jobs
round-robin. Inside an interrupt only fresh jobs run (a continuing job's open window aside, which the hold finishes).

### 3.5 No spare scratch state (review r1 MAJOR-2): stated limitation

A queued prompt becomes fresh work only through admission, which needs a spare scratch state (`MS_PREFILL_JOBS` = 2).
With BOTH states held by continuing jobs (two long jobs at once; 68 of 519 jobs on 10-11 were multi-window), a new
prompt waits for one of them to finish -- today's worst case, unchanged by this design. Measured as `state_wait_ms`
(queued with no spare state). Follow-up (not in this change): SUSPEND a continuing job at its next checkpoint boundary
(`checkpoint_ok` + `checkpoint_tokens` + `snapshot::save`, the cancel path at ms:1324-1361) and re-queue it, or a third
state reserved for fresh jobs if its footprint fits the loan's slack.

### 3.6 The stalled-streams flip (review r1 MINOR-4)

`decode_step`'s "every stream stalled -> Prefill" (ms:1631-1636) sets the phase directly (no hint/soft clear, no
`ms.phase` line, no evtrace). Route it through the same decision with reason `stalled`; it is a NORMAL prefill phase
and clears any interrupt state.

### 3.7 Structure, knobs, logging

The decision becomes a pure function `next_phase(&Snapshot) -> (Phase, Reason, interrupt: bool)` over a per-tick
snapshot (phase, phase_since, interrupt_since, last_interrupt_end, have_pf, have_dec, fresh work, starved, hold,
stalled, knob values read ONCE per tick). Reasons: budget / no_work / preempt / interrupt_end / interrupt_cap /
starved / stalled / embed. `ms.phase` and the `hub_phase` evtrace record carry reason + interrupt;
`prefill_job_progress` logs `fresh` and `service_ms`; admission logs `start_wait_ms` (queued -> first unit) and
`state_wait_ms`.

Make `MS_PREFILL_BURST_MS`, `MS_DECODE_BURST_MS`, `MS_PREFILL_BURST_MIN_MS`, `MS_DECODE_BURST_MIN_MS`,
`MS_BURST_SCALE` `.live()` (one atomic load per read; snapshotted per tick) plus the new `MS_PREEMPT` (default off),
`MS_PREEMPT_FRESH_MS` (20000), `MS_PREEMPT_GAP_MS` (2000), `MS_PREEMPT_MAX_MS` (10000).

## 4. Interactions (review r1 answers)

* Job lifecycle: finish + replay + `try_admit` run inside one `prefill_job_tick`, so a finished fresh job leaves
  `prefills` in the same tick and "no fresh work" is seen next tick; parked jobs are retried every tick regardless
  of phase; a cancelled job is dropped in one tick. Unaffected.
* Loan: one `lend_loan` + one `end_loan` per interrupt (0-111 ms each, swaps only on overlap); bounded by the GAP.
  `lm_store` lives in the loan pool, so nothing is corrupted by a switch -- perf only.
* Box-2 pins: one release + one restore per interrupt (`pin_enter_prefill` / `pin_enter_decode`); measured.
* Box-2 mode-evict: every interrupt opens a box-2 prefill phase with a fresh decode-victim budget; production runs
  `prefill_budget_last=0` (last-unit evict-behind) which caps that for single-unit jobs -- the A/B runs with it set.
* Starvation rule precedence unchanged (checked first; only forces PREFILL).

## 5. Measurement

A/B `V41_MS_PREEMPT` 1 vs 0 live, 20-min slots, >= 6 per arm, decode budget 3 s in both, `prefill_budget_last=0`;
discard the first 2 min of each slot. ~37 prefill jobs/h -> ~12 per slot. STRATIFY by "continuing job in flight at
arrival" (yes/no). Primary: `start_wait_ms` and stream-freeze p50/p90. Guards: decode tok/s, first-window time,
stalled-stream events, loan/pin ms per switch, `state_wait_ms`. Then, with preempt on: decode budget 3 s vs 30 s (the
long-job fairness knob).

## 6. Tests

Host unit tests of `next_phase`: preempt only with fresh work + GAP; interrupt ends on no fresh work and on the cap,
never while `hold` (incl. an open LAYER-OUTER window); `phase_since` shifted by the interrupt; continuing jobs alternate
as today; starvation precedence; stalled -> normal prefill clearing the interrupt; embed gate respects the hold. Job
selection: window holder first, aged fresh, fresh by remaining rows, continuing RR.
