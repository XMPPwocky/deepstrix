# Code review: prefetch slice A (5f9c9f8 + b8fa391 on 88d28d1, worktree-b2-prefetch)

Read: full diff (+1,090/-31), lookahead.rs, the touched sites in forward_prefill.rs / remote_experts.rs /
multistream.rs / evtrace_kinds.rs / knobs.rs, scripts/evtrace.py.

## Invariants
- **I1 holds.** The only device work the knob adds is the two legacy look-ahead launches (8541-8557), unchanged,
  gated now by `look_gates` in both the chain (8336-8347) and the route (8899-8910) from the same `mp` snapshot;
  they write `sd.router_logits` after its last reader, `sd.look_*`, and the pack's `look` segments (8587-8591,
  present iff `look_next.is_some()` from the same `mp`). `pre_moe_route` only READS the pack (`look_ok`, 8994-9000).
  Nothing numeric is touched; `d_selected`/`d_ew`/`xq` untouched.
- **I2 holds.** The new block (9560-9600) never calls `push_prefetch_words`; the legacy block runs only under
  `legacy_words(mode, legacy_env, hints_ok)` = `Off && env && hints_ok` (lookahead.rs:167-169; site 9602). With the
  knob on and the env var set, the legacy block is off and the knob's words stay off the wire (`Mode::wire` false).
- **Knob read once per step.** `begin_step` (lookahead.rs:135-151) packs four `Knob` loads (plain atomics,
  knobs.rs:219-239) into `CFG`; the chain reads `cfg()` into `PreMoeCarry::mp` (8339), the route destructures the
  same copy (8878), `submit_inner` reads `cfg()` again (8570) -- same step, same value; all drivers call
  `lookahead::begin_step` right after `b2_mirror::begin_step` (3981, 4083, 4276, 4429), so `step` is the
  post-increment value. Correct.

## Findings

1. **MAJOR -- `V41_B2_SPEC_BUDGET` defaults to 60: the slice A restart changes production behaviour with the
   hint knob off.** `active()` (lookahead.rs:126-129) is true whenever the budget is non-zero, and
   `submit_inner` takes the budget branch for every decode request (8579-8612): restores drop from 16/request to
   ~1/request + floor, admissions get capped at the step budget, a `hub_lh2` record is emitted every decode
   step. The design says "on regardless of the hint knob", and I argued it is a win, but it has never run; a
   restart that is sold as "a measurement" must not also flip the restore pacing. Change: default 0 in code
   (knobs.rs:650) or `=0` in the deploy's env file, then flip to 60 live as its own per-turn A/B (restore refill
   time after a phase switch, `pf_d_dropped`, `b2_pinned`, paged late replies in the 0-20 s bin of phasebin.py).
   Keep the budget-0 path byte-identical as it is (verified: 8583-8591 is the old code verbatim, incl. the
   `pf.len() < 128` guard and `PREFETCH_TAKE_CAP` via `take_prefetch_words`).

2. **MINOR -- `is_box2` under the HELLO bitmap uses layer L's ownership for layer L+1/L+2** (9579-9580: `owns_remote`
   is built at 9401-9408 from `c.owns(layer, e)` for THIS layer). The comment claims "per layer; the legacy block
   used layer L's for L+1 too" -- the new code does the same. Production runs the T2 partition
   (`substitution_active` requires it; `partition_box2(nl, e)` is the live branch), so no live effect; fix the
   comment or resolve `c.owns(nl, e)` once per lane-layer (one lock, already taken at 9405).

3. **MINOR -- the filter sits on the Route critical path, before this lane's submit.** 9560-9600 runs between
   the `selected_ready` wait and `submit_dispatch` (9790s), by design (the words must ride THIS lane's request to
   get the one-layer lead). Per lane-layer: `classify` zeroes two 384-entry stack arrays, builds and sorts a
   `Vec`, allocates a `Vec<Pred>` (lookahead.rs:197-220); `Pending` allocates a filtered `Vec` (9591); `dry_hits`
   zeroes a 384-bool array; `queue_hint_words` locks `MISS_HINT_WORDS` and `hint_words` allocates; `lookup` per
   pred. Estimate 5-20 us against the 0.03 ms turnaround slack; `lh.look_filter` prices it -- good -- but the
   dry run should report it per lane-layer, not per step. Cheap trims if it shows: keep the three scratch arrays
   and the `Pending` vec on `bd` (reuse, no alloc), skip the sort (rank <= 3 only needs a 3-bucket scan).

4. **MINOR -- `HintQueue::take` reallocates a `VecDeque` per decode submit** (lookahead.rs:357, `with_capacity(len)`
   even when nothing is queued -- `VecDeque::with_capacity(0)` does not allocate, but any non-empty queue does).
   Use `retain`-style in-place compaction or a two-pass count; also `take(step, layer, 128)` hands out ALL fresh
   words and the budget branch then drops the surplus (`.take(t.hints)`, 8604) -- fine in slice A (wire off), but
   slice B must re-queue, not drop, or the `max` must come from the plan. Leave a TODO at 8597-8598 naming it.

5. **MINOR -- `lh2_cand_rN`/`lh2_nonres_rN` are distinct per LANE-layer, not per step.** `classify` dedups within
   one lane's rows; both lanes' predictions for the same layer are counted twice when they coincide. The
   `HUB_LH2` doc comment (evtrace_kinds.rs:153-156) says "distinct ... over the step's lane-layers". Fix the
   comment; the `hinted` dedup for the WORDS is correct across lanes and steps (lookahead.rs:331-349).

6. **MINOR -- dry-hit recall is per lane and so a floor.** `lh2_pending` is matched against the SAME lane's next
   picks (9566-9574); a hint that the OTHER lane's rows demand at L+1 counts as a miss. Correct for the
   two-stream ordered cut (lanes never cross) and robust to ready-first interleaving; state in the kind doc that
   `dry_hits/nonres` underestimates recall (the live 0.71 bar is then conservative, which is the safe direction).
   The silent drop of a stale `Pending` (p.layer != layer, p.step != step: 9569-9573) is right.

7. **MINOR -- `lh.pre_moe` changes meaning on the arena drivers.** 4510-4512 now scopes the whole ready-first chain
   enqueue; it was 0 there before (the verify path's `LH_PRE`). Any `ms.stage` reader comparing `lh.pre_moe`
   across deploys sees a jump unrelated to the knob. Acceptable (deviation 4), but say so in the deploy note and
   compare dry on/off within the new binary only.

8. **NIT -- `Mode::wire`** (lookahead.rs:79-82): `let _ = self; false` -- write `pub fn wire(self) -> bool { false }`
   with `#[allow(clippy::unused_self)]` or a `_mode` binding; and the `Stat` discriminants (533-537) must track
   `STATS` indices by hand -- the test pins two of them; pin all five.

9. **NIT -- `restore_pace() = budget / N_LAYER`** (lookahead.rs:452-454) assumes one lane; two lanes make 80 requests
   per step, so at 60 it is 1/request either way, but at budget 120 it is 3/request x 80 = 240 > 120 (the total
   cap then binds and the second lane's requests get nothing). Harmless; divide by `N_LAYER * lanes` or document.

## Checks that pass
- Stale/step-tag rule (lookahead.rs:354-369): earlier step OR target layer <= carrying layer; a lane-A word for
  L+1 rides lane A's own layer-L request (the filter precedes the submit in the same Route), or lane B's earlier
  layer -- both fresh. Clear at the switch sits at multistream.rs:1119 (`Phase::Prefill`), as specified.
- Budget ordering/floor (lookahead.rs:462-484): hints > admissions > restores; restores get `max(left,
  floor_left)` so the floor may overspend the budget by <= 16 (tested, intended); nothing is held forever:
  admissions and restores stay in their bounded queues (4096 / 8192) and the floor guarantees restore progress
  every step; `releases_queued() == 0` gate preserved (8599). Budget 0 and every prefill request take the old
  path verbatim.
- Knob gating (lookahead.rs:156-160 + tests 622-637): `look_next2` only under `k2`; legacy depth honoured only with
  the knob off.
- evtrace kind 15 is free (10-14 hub, 20-23 box 2); `scripts/evtrace.py:65` maps unknown ids to `kind{id}` with
  `f{i}` fields, and all readers in b2tail index `*_fields` by name, so the new `hub_req.n_hint_words` (appended
  last) and `hub_step.lh_look_*` break nothing. `emit_named` exists (evtrace.rs:322).
- Locks: `MISS_HINT_WORDS` (route + submit), `PREFETCH_WORDS` x2, `RESTORE_WORDS` x2, `BUDGET` x2 per submit --
  all on the one host thread, uncontended; ~8 mutex ops, negligible. Timers are `None` unless
  `layer_host_timing()` (11844), so zero cost by default.
- Tests test rules: gating/legacy (design 2.1 in both directions), filter dedup/cap/rank order, step-tag/stale/
  clear/bound, budget classes/floor/frame bound/reset, counters order. Missing: a `take` test where a lane-B
  request at an EARLIER layer carries lane A's word (the cross-lane case the design allows), and a `plan` test
  at budget 120 with 80 requests (finding 9). `cfg_packs_and_unpacks` covers the bit packing.

## Deviations
(1) `hub_lh2` kind: fine, field limit is real. (2) `lh2_dry_words`: good, it is the abort-bar quantity. (3) draining
under `dry`: good, it exercises the stale rule for free. (4) `lh.pre_moe`: fine with finding 7. (5) per-request
`deferred`: fine, documented at lookahead.rs:415-419. (6) `legacy_words` tested as a pure decision: acceptable;
the 9602 condition is a one-line call I read.

VERDICT: APPROVE WITH CHANGES -- finding 1 before the restart (budget default 0 or `=0` in the env file, flipped
live as its own A/B); findings 2-9 are follow-ups that do not block the dry run.
