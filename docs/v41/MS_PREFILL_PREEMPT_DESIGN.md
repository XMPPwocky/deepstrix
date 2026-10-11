# Multistream scheduler: preempt decode for FRESH prefills (design, rev 1, 2026-10-11)

Owner 10-11: "preempt definitely, interleaving sounds bad bc eviction". Interleaving decode steps INSIDE a prefill
(between layer-outer passes) is out of scope: it would re-evict what the next pass needs.

## 1. Today (multistream.rs `Sched::tick`, ~1086-1184) and what it costs

Bursts with hysteresis: with both prefill work and live streams, the scheduler stays in a phase until its budget
elapses (`V41_MS_PREFILL_BURST_MS` 120 s, `V41_MS_DECODE_BURST_MS` 30 s; since 10-11 19:01 UTC 3 s in production) or
the phase runs out of work; a layer-major window / group in flight holds the prefill phase (`checkpoint_ok`,
`finish_group`); queue order SJF with aging (60 s) and a starvation rule (600 s). Two prefill jobs in flight
(`V41_MS_PREFILL_JOBS`), round-robin between them when no window is open.

Measured 10-11 (1,486 `ms.phase` switches, ~14 h, decode budget 30 s): every decode phase that ended had a prefill
waiting, 464 of 744 ran the full 30 s budget -> a ready prompt waited ~15 s mean for its prefill to START (more than
its whole first window, ~9 s). Prefill phases froze live streams p50 13.4 s / p90 45 s / max 170 s (16 of 742 hit the
120 s budget: prefill phases end when the work runs out). The 3 s budget cuts the wait to ~1.5 s mean but, for a long
prefill, leaves live streams 3 s per ~123 s.

## 2. Goal

A newly arrived (short) prompt starts prefilling at the next tick (one decode step, ~60-100 ms), while long prefills
keep a fair alternation with decode -- so the decode budget can go back up for long jobs without costing new prompts.

## 3. Mechanism

### 3.1 Fresh vs continuing prefill jobs

`Prefill.service: Duration` = wall time spent in this job's prefill units (accumulated around each unit in
`prefill_tick`). A job is FRESH while `service < V41_MS_PREEMPT_FRESH_MS` (live; default 20000 = covers a cold
single window + tail + replay, ~9-15 s) and it has not finished. FRESH work = any fresh job (started or still to
start its first unit). A job that crosses the threshold becomes CONTINUING and from then on follows today's burst
alternation.

### 3.2 Preemption (decode -> prefill)

In `tick`, with both sides having work: if the phase is Decode, `V41_MS_PREEMPT` is on (live; ship OFF for the A/B),
FRESH work exists, and the decode phase has lasted at least `V41_MS_PREEMPT_GAP_MS` since the last preemption-
triggered prefill phase ended (live; default 1000; the anti-starvation floor: under a continuous stream of fresh
prompts, live streams still get >= 1 s of decode between them), switch to Prefill now -- regardless of the decode
budget. The preempted decode phase's budget is NOT reset: its remaining time is saved and resumed after the
interrupt (3.4).

### 3.3 Job selection inside a prefill phase

Unchanged invariant: a job with an open window keeps the prefill until the window closes (`checkpoint_ok`). Otherwise
(today round-robin): FRESH jobs first, shortest remaining tokens first among them; then continuing jobs round-robin.
So a fresh prompt never alternates units with a 150K job; it waits at most for that job's open window to close.

### 3.4 Interrupt semantics (prefill -> back)

A prefill phase entered by preemption is an INTERRUPT: it runs while FRESH work exists (plus the open-window hold);
when no fresh work remains it ends and the scheduler RESUMES the interrupted decode phase with its saved remaining
budget (so a continuing long job does not get an extra prefill burst out of someone else's interrupt). If the phase
was a normal (budgeted) prefill phase, fresh jobs simply run first in it (3.3) and the budget rules are today's.
Interrupt time counts toward neither burst budget.

### 3.5 Live knobs (no restarts for tuning)

Make `MS_PREFILL_BURST_MS`, `MS_DECODE_BURST_MS`, `MS_PREFILL_BURST_MIN_MS`, `MS_DECODE_BURST_MIN_MS`, `MS_BURST_SCALE`
`.live()` (they are read per tick already), plus the new `MS_PREEMPT` (flag, default off), `MS_PREEMPT_FRESH_MS`,
`MS_PREEMPT_GAP_MS`. With preemption on, the decode budget can return to e.g. 30 s for long-job fairness live.

### 3.6 Logging / trace

`ms.phase` gains `reason` (budget / no_work / preempt / interrupt_end / starved / stalled) and `interrupt` (bool); the
`hub_phase` evtrace record gets a reason code. `prefill_job_progress` logs `fresh`.

## 4. Interactions to check (review)

* Loan: `lend_loan` before prefill units, `end_loan` at `decode_step` / `try_admit` -- more frequent switches = more
  lend/end cycles (each ~0-111 ms, swaps only on overlap). Cost per interrupt.
* Box-2 pin band: `pin_enter_prefill` / `pin_enter_decode` at every switch (release / restore of pinned decode
  experts). More switches -> more pin churn; measure `pin_released` / `pin_restore` in `ms.phase`.
* Box-2 mode-evict phases: an interrupt's prefill requests start a box-2 prefill phase (fresh budgets); the resumed
  decode ends it after `ME_DECODE_STREAK` requests. The last-unit evict-behind keeps decode's layers for single-unit
  jobs, which fresh jobs mostly are.
* Admission / KV room: a fresh job that waits for KV ROOM (`room_wait`) is not in `prefills` -> not fresh work;
  unchanged. Starvation rule unchanged and takes precedence.
* `decode_step`'s "every stream stalled -> prefill phase" path unchanged.
* Two fresh jobs + a continuing job's open window: the window hold first, then SJF among the fresh ones.

## 5. Measurement

A/B `V41_MS_PREEMPT` 1 vs 0 live, 20-min slots, decode budget 3 s in both arms (today's production) -- then a second
A/B with preempt on: decode budget 3 s vs 30 s. Score: prefill START wait (queue -> first unit; new log field),
time to first token, stream freeze durations (prefill phases with live streams), decode tok/s, first-window time
(switch-induced cold pages), switch count, loan/pin churn.

## 6. Tests

Host unit tests of the phase decision as a pure function (`next_phase(state, knobs) -> (Phase, reason, interrupt)`)
extracted from `tick`: preempt only with fresh work + gap; resume with saved budget; continuing jobs alternate as
today; starvation precedence; window hold; SJF among fresh jobs.
