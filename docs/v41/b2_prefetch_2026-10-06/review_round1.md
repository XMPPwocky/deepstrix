# Review round 1: B2_PREDICTED_MISS_PREFETCH_DESIGN.md rev 1

Grounded against worktree b2-prefetch (forward_prefill.rs, remote_experts.rs, b2_mirror.rs), the b2tail
evtrace dumps (new script `spec_reads.py`, log `spec_reads.log` in this dir), recall.log, price/pred/phasebin logs.

## Findings

1. **MAJOR -- pricing uses the wrong read time.** Doc 3/4: a hint "turns a 2.0-2.5 ms certain read into one that
   started ~1.3-2.7 ms earlier"; "serialisation is not the limit". Data (b2tail evt, 9,750 src=3 speculative reads =
   the exact daemon path a hint word takes, `prefetch_words_core` 4321-4355 -> chunked `read_miss_into` 4283-4289):
   wall **med 5.0-5.2 ms, p10 2.19, p90 10 ms**. Isolated (run_spec_at_start<=1, no pause, no demand): med 2.26 ms =
   a certain read; busy (2-3 spec readers running, **60-67% of spec reads**): med 6.6-6.8 ms. Today's admissions +
   restores already keep the 3 speculative readers busy two thirds of the time (pf_d_hinted 0.17-0.19/request =
   14-15 starts/step, restore bursts to 38-57 words/frame); hint->pop p90 29-31 ms (FIFO behind queued spec jobs,
   `PfQueue::push` 1547). A hint is promoted at the demand `ensure` (4487) and then waited for; competing spec reads
   pause within <=1 MB, so the remainder runs at full speed -- the gain per caught hint at r4 is ~0.5 ms when busy,
   ~1.35 ms when isolated, average ~0.8 ms, not the 1.35 ms lead assumed. Share of spec walls <= breakeven 3.65 ms
   (lead 1.35 + certain 2.3): 36-38%. Re-price with the measured busy/isolated mix: k=1 ~2-2.7 ms/step weighted,
   not 3.5-4.5. Fix options: (a) a daemon hint class that runs ahead of admissions/restores (slice D's LIKELY bit)
   -- then the "no box-2 change" premise goes; (b) hub-side: cap admission+restore words while hints are queued, so
   hints do not queue behind them on box 2 (hub-only, cheap, measurable in the dry run).

2. **MAJOR -- the admit gate would drop most R=2 hints.** Doc 2.1: non-protected hints pass `admit_passes` (1248).
   Live `V41_SUB_PROTECT=1` (KNOB_AUDIT hub=1), so under R=2 every rank-2 hint goes through the gate, and pred.log
   shows the gate refusing **22 of 26 candidates/step at r4** (`sub_admits_gated` 22, `sub_admits_queued` 4): the
   watermark is non-zero because pinned sits at 3614-3634 with releases in bursts. R=2's recall of 0.86 assumes all
   rank-2 hints are sent; gated, you get ~R=1 (0.71) plus wasted filter work. A hint for a protected or gap-blocked
   pick is a read box 2 makes anyway; the gate's purpose (do not read what will be released unused) does not apply.
   Change: no gate at predicted rank <= 2; gate only rank 3 if R=3 is ever tried. (Answers open question 2.)

3. **MAJOR -- `pf_d_dropped ~0` is false today.** Doc 7 A/B watch and 8.D premise. Data: pf_d_dropped 3,690 and
   5,309 over 50-57K decode requests = **6-8 drops/step**, bursty (max 25-29 per request; free sets mean 15.4 of 16,
   so drops come when restore pumps fill the sets after a phase switch -- exactly the 10-20 s window where paged
   replies are 2x, phasebin.log). Hints first on the wire does not help: the drop rule is per word at application
   (4342), and words of earlier frames already hold the sets. Change: measure `lh2_hints_dropped` from `b2_read`/
   `pf_d_dropped` deltas in slice A, and either pace restores (hub `pin_restore_per_request`) or accept that hints
   are lost in the window that matters most.

4. **MAJOR -- host launch cost not priced.** Doc 2.3 prices 2 x 32 us of device time. Under `after` (and `before`)
   the look-ahead adds ~4-6 host API calls per lane-layer (2 launches + pack + event record + `stage()` events) ~
   40-100 us of host time inside the chain enqueue (8523-8534 is host code in `forward_layer_pre_moe_v2`), on the
   single ready-first thread (4603-4668) that also polls the other lane's `selected_ready`. The sweep's turnaround
   slack is 0.03 ms. The `hipEventQuery(look_ready)` + filter at submit sits on the Route path proper. Change: price
   `lh.*` host deltas in the dry run (`before`/`after` both), and plan the look-ahead as a captured stage keyed by
   (stage, rows) like the router (graph-keys infra 3eccfdf9) rather than deferring to "lever #4".

5. **MINOR -- the residency predicate is not `n_pred_miss`.** Doc 2.1 cites `resident()` 353-358 as "the exact
   `n_pred_miss` predicate". `resident()` 357 uses `pending && pending_on()`; live `V41_SUB_PENDING=0` (KNOB_AUDIT
   hub=0), so PENDING picks (the other lane's just-sent certain reads) look non-resident and get hinted (daemon
   dedups at 4330, but they count as sent hints and cost frame words). 9725 uses `!held && !pending && !incoming`
   unconditionally. Change: use `lookup()` and the 9725 predicate.

6. **MINOR -- `b2_pin_released_unused` is blind to hints.** Doc 3 and the A/B watch. `released_unused` increments
   only when `admitted != 0` (892), set by `note_admitted` (745) via `note_admits`, called only for cache-prior
   admissions (forward_prefill.rs 9254, 9345; b2_mirror.rs 1415). Hint words never call it. Change: `note_admits`
   on every sent hint word (not `note_incoming`, which would change the prior's input -- rev 1 rightly defers that).

7. **MINOR -- wrong hints become pins, not band entries.** Doc 3 "one slot in the unpinned band". A landed hint is
   granted eligibility (`pin_apply_words` 4911-4917) and pinned at the layer's next report within budget (pinned
   3614-3634 of 4000; headroom line 3744). ~1 wasted hint/step plus rank-2 early admissions push the estimate to
   the release line in 1-2 minutes, after which `step_ranked` releases every step and the watermark stays
   non-zero -- i.e. hints move the ledger from "no release pressure" to "constant pressure", strengthening finding 2.
   Watch `b2_pinned`, `b2_pin_released` (median 0 today) in the A/B; consider not granting pins for hint words
   (needs a daemon change) or letting R=1 be the default until measured.

8. **MINOR -- Step 0 volume is cross-dataset.** Doc 1: "~6-7 hinted reads/step at r4". recall.log k=1 N=2:
   hints/row 0.176 = ~27/step on the dump's own routing (its rank-1 box-2 miss rate 0.084/row is 6x the trace
   replay's 0.027/request); 6-7 comes from scaling the trace's `sub_blocked` by the dump's precision. Fine as an
   estimate, but make `lh2_hints_sent/step` a pre-registered abort (>2x the estimate = stop) and report dry-run
   counts per R (1, 2, 3 simultaneously) so the A/B picks R. The LRU replay vs the live pin ledger (decayed-count
   LFU with TinyLFU) is acknowledged; the dry counter is the real Step 0 -- agreed.

9. **MINOR -- stale-word rule is by layer only.** Doc 2.2: drop a word whose target layer <= the carrying
   request's layer. A word queued late in step s and drained by step s+1's layer-0 request is 35 layers stale yet
   kept. Tag words with the mirror STEP (`begin_step` 284) and drop across steps; also clear `MISS_HINT_WORDS` at
   the decode->prefill switch (`expire_incoming` analogue), or the first prefill chunk's `submit_inner` carries them
   and `prefetch_words_prefill` (7149) lands them prefill-class. (Answers open question 7: other-lane carry is fine.)

10. **MINOR -- slice A is output-free, not behavior-free.** Doc 8.A. `dry` enables the look-ahead launches,
    `rb_look`, `look_ready` and the host query on every decode lane-layer: a timing change on the turnaround (finding
    4). State it as such and run the dry slice with an `ms.step` watch, not as a silent ride-along on the C restart.

11. **MINOR -- gates need box 2 attached.** Doc 7: G5a-h with `hints_sent` asserting the path ran. Hints exist
    only with `remote_split_on` (9515); on a box-1-only gate window the counter is 0 and the assert cannot pass
    (reference_gpu_gate_windows: box-1-only is the usual window). Say so, and make the counter assert conditional
    on a remote being configured.

12. **NIT -- `after` placement delays the shared expert.** Look-ahead on `de.compute` between `selected_ready`
    (8615) and `issue_shared_expert_prefill` (8640) pushes the shared expert's dGPU start by ~35-70 us; the combine
    waits on it later, so it is not on the turnaround, but it is not "idle time" either when lane B's chain is
    queued behind. The `before` control in the A/B covers it.

13. **NIT -- what is verified correct.** `REQ_FLAG_PREFETCH` wire (383, 737, 819) and both-side proto v4; daemon
    grant/dedup/drop/queue/promote/wait path (4899-4919, 4321-4355, 1542-1581, 4486-4508); words applied at
    dequeue not arrival (7147 vs the `early_page` hook 7200-7225, which reads `nreq.sel` only; measured
    dequeue->hints_end 0.26 ms med / 0.65 p90, matching the doc's 0.3 ms); `sd.look_sel*` pack-in-stream-order
    argument (8573-8591) and the `lookahead_hints_ok = rb.packed` change are sound for both drivers; the two
    look-ahead launches write only `sd.router_logits` after its last reader (8516-8522), `look_*`, and the pack
    buffer, so I1 holds; hints never block the host. `route=split` for spec reads confirmed (route code 0 on all
    9,750). Spec readers = 3 of 4 (`with_readers(n_par - reserve, n_par)` 4213).

## Open questions
1. Ship R=1 by default; dry-run counts per R decide. 2. No gate at rank <= 2. 3. Measure `before` first (slice A
is a measurement anyway); `after` only if `lh.route`/`lh.sel_d2h` deltas show the 64 us. 4. Correct: no INCOMING
for hints in rev 1. 5. Bar on paged late replies per step is right, add the hub `b2_read` src=3 wall and
`hint->pop` as diagnostics; keep `ms.step` p50 <= 1.00 every cell. 6. Own picks (`sel_orig`). 7. Fine, with the
step tag (finding 9).

VERDICT: NEEDS REWORK -- the mechanism and safety are right and slice A (counters + dry run, no gate, step-tagged
queue, host timers) may proceed as a measurement; sections 3-4 and 8 must be re-priced on the measured speculative
path (busy walls, queueing, drops) and decide up front whether the hub-only version can win at r3-r6 or the box-2
hint class is part of the plan.
