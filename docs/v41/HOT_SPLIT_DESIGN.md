# Hot split: interleaved head (+ replicated top-K later) (design, 2026-10-08)

Status: DRAFT rev 3 (2026-10-08), branch `worktree-hot-split` (from `cb5b594`, the deployed
prefetch source). Owner 10-08: "let's go for hot split. no restarts w/o asking; otherwise, go nuts".
Rev 2 folded in review round 1 (16 findings), rev 3 round 2 (11 findings) and matches the slice-1
wiring; section 10 maps every finding.

Sources: `HOT_SPLIT_SIM.md` (policy (c)/(d), the replay simulator), `DECODE_IDEAS_SWEEP_2026-10-06.md`
lever #1, the 10-08 code map of every ownership consumer (section 3), the live hub log of 10-08.

## 0. Decisions

1. **Hub only.** Box 2 pins whatever the hub reports or words (`PinBook`; a plain prefetch word
   grants a pin, `remote_experts.rs:6144-6166`, and a pin lasts until the hub releases it). Box 2's
   head share is kept pinned by the hub's own words and ledger. No daemon change, no box-2 restart:
   the deploy is ONE hub restart (owner's go) with the policy `top`.
2. **One live knob picks the policy**: `V41_B1_HOT_POLICY = top` (default; today's code path,
   bit-identical) `| interleave`. Both directions of the switch are bounded migrations (section 2.4);
   rollback is the knob back to `top`, which RETURNS to today's placement layer by layer (2.4).
3. **Ownership states per (layer, expert):** `B1` (box 1 owns and computes), `B2H` (box-2 head: box 2
   computes, the hub keeps it PINNED), `B2` (box 2's cold tail, its LRU), and, in the second step,
   `REP` (resident on both; the route picks the leg). `partition_box2(l, e)` keeps its meaning "box 2
   is the home": true for `B2H` and `B2`, false for `B1` and `REP`.
4. **Sticky, swap-capped placement** instead of the sim's static alternation (section 2). Incumbents
   keep their side; box 1's mass share is pulled toward the target by at most `V41_B1_HOT_MOVES`
   box-1 reads per layer per refresh. A strict rank alternation flips both owners of every adjacent
   rank swap at every refresh; the sim never priced that churn (`HOT_SPLIT_SIM.md` 6, 8).
5. **A MOVING expert is served by whichever box holds it** until the destination holds it (the
   HOLDER FALLBACK, section 3.2). Demand reads from a refresh are bounded and counted, not zero: an
   expert neither box holds is still read (a tail id filled into a vacancy that box 2 holds
   unpinned, a dropped KEEP read after box 1 evicted the id).
6. **Ship policy (c) first: `REP = 0`.** The sim gives (c) at 0.55 -10.3% of (d)'s -12.6%. Without
   the replicated set there is no leg choice, no co-row dependence of a row's split, no REP pins and
   no residency-hint conflict, and the bit-exact gates stay meaningful. `REP` ships in the same build
   behind `V41_B1_HOT_REP` (default 0) and is its own A/B afterwards (section 8).
7. **Measure before flipping.** The sim's -12.6% was run at box 1 = 133 ids/layer, box 2 = 106
   slots with up to 90 pinned, and before the async upload; this design runs box 1 at 103-124 and
   box 2 at 119 slots. The deploy makes the pick trace a live knob; a fresh trace under `top`
   re-runs `scripts/split_sim` with this design's placement ported (section 8 step 2), and that
   number -- not the sweep's -- is the expectation and picks `N1`, `P2` and the target.

## 1. Problem and expected effect

The decode pole is box 1's iGPU MoE. Box 1 owns each layer's top-103 by decayed pick count (live
10-08: 0.74-0.77 of picks); box 2's identical iGPU serves the tail. The 10-06 sweep measured iGPU
MoE busy 72 of a 104 ms lone DSpark block and 151-157 GB/s at 4-6 rows.

The split trades iGPU time for box-2 time: in the sim `d_rep10_0.60` cut iGPU busy 62.9 -> 49.0
ms/step (-22%) while box 2's busy roughly doubled (20.5 -> 40.1), for -12.6% fwd held out (two-lane
4-6 rows -17..-21%, 1-3 rows -1..-7%); (c) 0.55 -10.3%. Those are pre-async and at the sim's sizes.
The 10-06 pipeline sim's "+11.3% for iGPU bytes -45%" what-if did not price box 2's growth and is
not this design's number. The gain shrinks with a slower box-2 leg (+400 us/request halves it) and
the model does not price prefill displacement of box 2's head (section 6), the cache prior, or
churn. The live A/B decides; the re-run sim (section 8) sets the expectation first.

## 2. Placement (`hot_set::refresh`, policy `interleave`; pure code in `het/hot_split.rs`)

Per layer, from the same decayed router-pick counts `c[e]` (counting and decay unchanged). Sizes:
`N1 = V41_B1_HOT_IL_PER_LAYER` (0 = `per_layer()`, today's 103 clamped to the decode LRU; includes
REP), `K = V41_B1_HOT_REP` (0 now; 10 in the second step), `P2 = V41_B1_HOT_B2HEAD` (default 60).
Box 1 side `cap1 = N1 - |REP|`, box 2 head `cap2 = P2`, region size `R = cap1 + cap2`. Ranks below
are among the non-replicated ids.

1. **Replicated set** (K > 0 only): incumbent REP ids stay while rank < `K + V41_B1_HOT_REP_HYST`
   (default 5); fill from the top ranks, at most `MOVES` REP newcomers per refresh (each a box-2
   read and pin; one box 1 does not hold also spends a box-1 move); if still short, the strongest
   departing incumbents stay (the set stays `K` wide while it migrates).
2. **Incumbents first.** Hottest first, each incumbent (`B1`, `B2H`, or an ex-REP) keeps its side
   while its rank < `R + V41_B1_HOT_IL_HYST` (default 40; NOT the top policy's 100, which would let
   incumbents squat to rank ~253) and its side has room; an ex-REP takes box 1 if there is room,
   else box 2's head; an incumbent with no room drops to `B2`.
3. **Vacancies.** Hottest first among ranks < `R` not yet placed: box 1 if its share of the
   non-replicated mass is below the target `s = V41_B1_HOT_TARGET` (default 0.60) or box 2's head is
   full, AND box 1 has room AND (box 1 already holds it or a box-1 move is left); else box 2's head
   if it has room. A vacancy left unfilled stays open until the next refresh (bounded reads).
4. **Balance.** While `|share1 - s| > V41_B1_HOT_TOL` (default 0.02) and moves remain: among the
   (`B1` a, `B2H` b) pairs whose swap IMPROVES `|share1 - s|`, prefer one that lands within the
   tolerance with the smallest moved mass `c[a] + c[b]`; else the one that improves most (ties:
   lower ids). Each swap spends one box-1 move (b becomes box 1's). O(|B1| x |B2H|) per swap, ~1 ms
   per refresh on the decode thread at the defaults.
5. **Outputs:** `STATE[l][e]` (`AtomicU8` per (layer, expert); `OWN`'s readers go through
   `box1_owns` / `partition_box2` / `keep`), per refresh: share1 (mean over layers), swaps, box-1 and
   box-2 newcomers, the KEEP size; the newcomer lists feed the pre-warm (section 5) and the KEEP
   words (section 4).

Moves: `V41_B1_HOT_MOVES` (default 3, live) bounds box-1 newcomers (vacancy fills + swaps) per layer
per refresh, and separately the REP newcomers.

### 2.4 Switching

- **`top -> interleave`:** the incumbents are today's top-103, all `B1`: share ~0.76, box 2's head
  empty. The first refresh fills box 2's head with ranks 103-162 (box 2's hottest today, mostly
  already resident and pinned there: no box-1 reads), then swaps toward the target, `MOVES` per
  layer per refresh (~3-8 refreshes at the measured curve; the test on a Zipf curve converges in a
  few). The holder fallback (3.2) keeps the swapped ids served from box 1 until box 2 holds them.
- **`interleave -> top` (RETURN MODE, per layer):** today's `refresh` would keep the interleaved
  set forever (every incumbent within rank `per_layer + hyst` = 203 stays, so nothing is ever
  replaced). On a switch to `top` EVERY layer enters return mode: a returning layer refreshes with
  incumbency limited to rank < `per_layer` (hysteresis off) and the change cap
  (`V41_B1_HOT_MAX_CHANGE`, made LIVE) on; its changes are MOVING (3.2) and its box-1 newcomers are
  pre-warmed. A layer leaves return mode on COVERAGE -- box 1 owns every rank below `per_layer -
  10` -- not on equality with the top set (live counts reshuffle the flat region at the cutoff by
  more than the cap every refresh, so equality never comes). Return mode ends when no layer is
  returning, or after 60 refreshes (forced, logged). Box 2's head ids return at <= MAX_CHANGE per
  layer per refresh; KEEP empties at once (STATE has no B2H under `top`), so the ledger releases the
  ex-head pins by its normal order. After it, `top` is exactly today's refresh again. Test:
  `hot_set::tests::interleave_switch_and_return` (counts jittered +-30% every refresh).
- `policy = top` outside return mode runs today's `refresh` unchanged, STATE = B1/B2 from `OWN`, KEEP
  empty, no MOVING id, no pre-warm, no holder fallback: every consumer sees exactly today's answers
  (G-top). Before warm-up `keep()` is false and `partition_box2` takes the hash path, as today.

## 3. Consumers (FP = `forward_prefill.rs`, BM = `b2_mirror.rs`, RE = `remote_experts.rs`)

| site | today | under `interleave` |
|---|---|---|
| route pick loop FP:9539-9556 | `partition_box2` -> `extra_remote`, else `ids` (b1_page_misses) | B1 -> `ids`; B2H/B2 -> `extra_remote`; each subject to the holder fallback (3.2); REP -> the leg choice (3.1) |
| prefill / verify rows (same loop, b > 8 or not Arena) | same | no holder fallback; REP -> box 1 (fixed), so `lm_prefetch_words` FP:836 and `prefill_readahead` FP:10210 see box 1 as home |
| cache-prior held mask FP:8405-8410 | box-1 owned: `pg.is_resident`; else `b2_mirror::resident` | held by the box that will SERVE it: `serve_on_box2` (3.2), the same function the route calls; differs from today only for MOVING ids (and REP) |
| substitution `predicted_miss` (mode 2, off) FP:9152, 9342 | | same rule as the held mask |
| displaced-pick admissions FP:9239-9308 | box-2 picks not held -> admission words | unchanged (an admission word grants a pin; B2H keeps its INCOMING mark and priority); REP never |
| `note_incoming_covered` FP:9229 | counter | counts box-2 home picks only |
| look-ahead `is_box2` FP:9658 (`V41_B2_MISS_PREFETCH`) | box-2 predictions -> LIKELY hints | unchanged (already skips held B2H); REP is not box 2 |
| `wants_for_box2` FP:9745 / ledger counts | masked to box 2's partition | REP and B2H count as box-2 wants (ranking; KEEP protects them anyway) |
| ledger stale `pin_begin_step` BM:1219-1222, `box2_lost` BM:1235-1237 | `!partition_box2` = stale/lost | `hot_set::box2_stale` = `!partition_box2 && !keep && !moving` (today's expression, hash regime included, less KEEP and less a MOVING id box 2 still serves); KEEP = STATE in {B2H, REP} (section 4) |
| residency hints `push_hint` on box-1 admissions (RE:5776-5787, sent RE:10017) | box 2 demotes its copy | never for a KEEP id |
| `touch_resident` FP:9310-9319 | prior-added box-1 substitutes | also every REP pick routed to box 2 |
| pick trace FP:9188-9207 | owner chars `1`/`2`; nothing offline reads them | `R` for REP; the leg chosen goes to evtrace `hub_req` (`n_rep_b1`, `n_rep_b2`) and the trace's new `L` line |
| hot-set counting FP:9501, BM:1586 | router picks | unchanged |
| single-stream `forward_layer` FL:2580-2602 | hash share | unchanged (not the multistream path) |

### 3.1 The leg choice (REP picks; second step)

After the non-replicated picks are classified (box-1 distinct `d1`, picks `n1`; box-2 `d2`, `n2`),
the REP distinct ids, hottest-first (by picks in this lane-layer, ties by id), go to the leg whose
finish time grows least -- the sim's `route_replicated`, plus an OPEN cost:

    ig(n, d)  = IG0 + IG_D*d + IG_N*n + IG_B*b           (ms: 0.100 / 0.0774 / 0.0046 / 0.0397)
    b2(n, d)  = d == 0 ? 0 : LINK(b) + (S0 + S_D*d + S_B*b)/1000 + (d2_before == 0 ? OPEN : 0)
                (us: 116 / 99.9 / 15.6; LINK 58..318 by b; OPEN 0.2 ms)
    t_box1 = max(ig(n1+k, d1+1), b2(n2, d2));  t_box2 = max(ig(n1, d1), b2(n2+k, d2+1))

`OPEN` (`V41_B1_HOT_LEG_OPEN_MS`, live) is the host cost of a submit + post + poll the sim's DES
charged after the fact but its greedy ignored; without it a REP pick opens a box-2 request whenever
box 1 has any other work, even at one row. With it, b = 1 with one other box-1 pick stays local,
but b = 1 with four box-1 picks and two REP picks still opens a request (0.49 vs 0.55 ms): `OPEN` is
validated in the sim replay before REP goes live (1L r1-r3 cells). All constants are live knobs
(`V41_B1_HOT_LEG_*`); `V41_B1_HOT_REP_LEG = cost | box1 | box2` pins the leg.

**Reproducibility, stated exactly:** the leg is a pure function of the lane-layer's picks, so the
same batch always splits the same way -- but a row's split now depends on its CO-ROWS (and, in a
speculative block, on rejected draft rows), which today's routing does not. Hence the gates in
section 7: bit-exact with the leg pinned, KL with `cost`.

Invariants (code map 4): every distinct pick lands in exactly one of `ids` / `extra_remote` per
lane-layer (`verify_routing_exactly_once`); `owns_eff == extra_remote`; the decision is made inside
the partition branch before `b1_page_misses`, before `sel_wants`, `owns_eff`,
`remote_sel_override`, `box2_idle` and the submit, and reads no post-`ensure` state. Two lanes may
choose differently for the same layer (`mark_remote_after_ensure` rebuilds the remap per lane;
the upload is stream-ordered on `ie.compute`).

### 3.2 MOVING ids and the holder fallback (decode rows; `V41_B1_HOT_HOLDER`, default on)

A refresh flips `STATE` before the destination box holds the moved expert, and the next step picks
it (the refresh runs at the head of `decode_rows`, MS:1637). An id whose HOME changed boxes at a
refresh (interleave, or a returning layer under `top`) is MOVING: `MOVED_AT[l][e]` = the refresh
counter, MOVING while fewer than 4 refreshes old and not confirmed. The destination confirms it:
the route clears it when the home box holds it (box 1 `pg.is_resident`, box 2 mirror `held`), and a
box-1 pre-warm admission clears it. Only MOVING ids are affected (applying the rule to every id
would route box-2 tail ids still in box 1's ~921 spare LRU slots to box 1 for good, pulling mass onto
the bottleneck above the target):

    serve_on_box2(l, e, home_box2, b1_has, b2_held, b2_incoming):
      not MOVING or knob off            -> home
      home box 1, !b1_has, b2_held      -> box 2   (pinned there: no read, no surprise)
      home box 2, !b2_held, !b2_incoming, b1_has -> box 1
      otherwise                         -> home

`held` is the mirror's pinned bit (the PENDING overlay is off in production, `V41_SUB_PENDING`=0);
INCOMING keeps a box-2-home id on box 2. The cache prior's held mask calls the same function, so
mask and route agree; a residency change between chain time and route costs at most one read. It
reads residency, so the split depends on history: OFF in the bit-exact gate arms. Counted:
`holder_b1` / `holder_b2` in the refresh log line. Test: `hot_set::tests::holder_decision`.

## 4. Box 2: the KEEP set

KEEP = `{e : STATE[l][e] in {B2H, REP}}` -- a predicate on hot_set's STATE, NOT ledger state (the
ledger is replaced on every reconnect, `on_connect` BM:1108-1126). Size at the defaults: 60/layer
(2,400) with REP = 0; 2,800 with REP = 10; box 2's pin budget is 4,232 (4,760 slots - 480 reserve -
48 nursery).

- **Never released:** `release_coldest_ex` (headroom and decay) skips KEEP candidates; stale = STATE
  `B1` only; `take_restore` accepts a KEEP id regardless of `last_want` and stale.
- **Pinned by words on the RESTORE queue** (`remote_experts::RESTORE_WORDS`, not
  `PREFETCH_WORDS`, whose 4,096 cap and LM-prefill guard it would share with the cache-prior
  admissions): decode requests only (a prefill entry moves the queue onto the ledger's restore
  list), after every queued release (`releases_queued() == 0`, so a grant never overtakes the
  release of an id that left KEEP and came back), paced per request like restores. `keep_fill`
  queues every KEEP id the mirror does not show held and that is not on the queue already, at the
  first decode step after a refresh that changed KEEP, and after a reconnect at the first decode
  step with pins on (`pin_apply_words` ignores grants before). At decode entry KEEP comes back
  through `take_restore_keep` only (KEEP first, regardless of `last_want` and stale). A grant that
  expires unpinned (nursery or partial slot, RE:3245-3250) is re-queued at the next fill.
  `keep_queued` (total) is in the refresh log line. Decode picks and admissions of B2H ids grant
  pins too. If `V41_B2_SPEC_BUDGET` is ever turned on, `budget_plan` sees these as restores.
- **Prefill band:** `V41_B1_HOT_KEEP_PREFILL` (live, default OFF). Off: today's band release order
  (KEEP last), and KEEP is restored first at decode entry (resident ids re-pin without a read). On:
  the band skips KEEP, so box 2's unpinned room for prefill falls from ~4,620 to ~2,360 slots (REP
  0) -- the 09-27 measurement (BM:648-660) put an unpinned prefill phase at ~2,700 slots and found
  560-880 up to 3x slower, so on is an A/B arm, not a default.
- **Budget clamp:** `P2 + K <= (budget - 1,024) / 40` per layer, budget from `est_pinned()`'s source
  (box 2's reported budget); before the first reply or after a reconnect, the static 4,232.

## 5. Box 1 side

- **Pre-warm:** box-1 newcomers (`b1_new`) at a refresh go to a new `prefetch_now` (the L1
  prefetcher, which production runs: `V41_B1_PREFETCH=1`) that bypasses the min-touch gate
  (`V41_B1_PREFETCH_MIN_TOUCH`, default 2), paced at `V41_B1_HOT_PREWARM_STEP` (default 2) queued per
  step against the 64-in-flight cap. Admission stays `drain_prefetched` at the step boundary (~1.3
  ms each, up to `V41_B1_PREFETCH_ADMIT` per step): at 2 per step a 120-newcomer refresh costs ~2.6
  ms/step for ~60 steps, which `hs_prewarm_ms` measures. Admissions of a KEEP id push no residency
  hint. Under `top` the pre-warm stays off (G-top bit-identical; today's arena path never reaches
  `prefetch_hint` because `b1_page_misses` continues first, FP:9553).
- **Size:** `N1` defaults to today's `per_layer()` (103; the decode LRU fits 124). Box 1 shares one
  global LRU with prefill (unified pool), so owning more means prefill evicts more owned experts;
  `V41_B1_HOT_IL_PER_LAYER` is live and the re-run sim picks it.
- **REP copies routed to box 2** are touched (`touch_resident`) so they do not age out.

## 6. Risks and what is measured for each

| risk | why | counter / bar |
|---|---|---|
| prefill displaces box 2's head | box 2 now holds ~35% of decode mass; today it holds the tail | `b2_paged_replies` per step in the 0-20 s bin after a prefill->decode switch vs `top` |
| prefill gets slower | KEEP pins (on arm) shrink box 2's prefill room; the head re-reads (off arm) cost box-2 disk | TTFT p50/p90 per prompt-size bin, prefill tok/s, box-2 prefill misses/window |
| churn | swaps move hot ids | per refresh: swaps, newcomers, `hs_fallback_*`, `b1_read` ms/step, `hs_prewarm_ms` |
| box 2 becomes the pole | its per-expert cost is ~20% higher | exposed `remote_wait`, late replies/step (`hub_req` reply after MoE end) per cell |
| cache prior | protect 2 now protects box-2 head picks too | `sub.picks_swapped`, surprises; protect/lambda A/B later |
| KEEP starves the tail | 2,400 of 4,232 pinned | box-2 cold misses/step, `pf_d_dropped`, nursery lands, `hs_keep_requeued` |

## 7. Gates (one hub window; box 2 untouched, stays warm)

- **G-top:** `V41_B1_HOT_POLICY=top`: `multistream_step` G5a-h + G6 against today's daemon, and the
  new hub's results equal the deployed hub's (6e3ede8e) bit for bit, as the slice-D window did.
- **G-il:** `interleave` with REP 0 and `V41_B1_HOT_HOLDER=0`: G5a-h + G6 pass bit-exact (the split
  is a function of STATE only). The harness forces a placement by running the refresh on a fixed
  count table (a test hook), so both arms of G5 see the same STATE.
- **G-rep** (before REP goes live, its own window or the same one): `REP_LEG=box1` and `REP_LEG=box2`
  each G5a-h bit-exact (holder off); `REP_LEG=cost` under `MS_ALLOW_INEXACT=1` with KL(dec||arena)
  mean <= 0.03, max <= 0.2 (the async-upload gate's envelope).
- **Unit tests (host):** the placement (sizes, incumbency, move caps, convergence, no flapping on
  jitter, dead ids, N1 shrinking), the return mode (interleave -> top reaches exact top-K within
  the predicted refreshes, never more than MAX_CHANGE newcomers per layer), `top` = today's
  `refresh` on the same counts, the holder fallback decision table, the leg choice, KEEP (never
  released by headroom/decay, restore accepts it, survives a ledger reset), `partition_box2` and
  `keep` on all states.

## 8. Rollout

1. Build + tests; ONE hub restart window with the owner's go (G-top, G-il; up with `top`): no
   change in behaviour.
2. **Re-run the sim** BEFORE the window, on production's own pick trace (`V41_PICK_TRACE` is set in
   the hub env: `picks-sub-20261007-0552.trace` since the current hub's start) with its evtrace,
   recalibrated post-async (fit_costs + calibrate; `params.json` reconstructed, the DES's drain
   coupling off): port `hot_split::interleave_layer` (and
   the holder fallback, the KEEP pins) into `scripts/split_sim/policies.py`, box 2 at 119 slots,
   `N1` 103 / 115 / 124, `P2` 50 / 60 / 75, target 0.55 / 0.60 / 0.65, REP 0 / 10 with `OPEN`.
   Quote that number; pick the defaults.
3. Flip `interleave` live; watch the transition for ~15 min (abort = `top`).
4. **Block A/B** `top` vs `interleave`, ABBA order. Block length and count from the block-to-block
   variance of existing `top`-only logs (block bootstrap per step cell); each block's discard window
   >= the measured transition time in that direction (return mode ~15-20 min). Judged per step
   cell (regime, rows, lanes) on step p50 and tok/s, plus the prefill metrics and risk counters
   of section 6. Bar: weighted step p50 down >= 5%, no cell up > 3%, TTFT p50 not up > 5%.
5. Then one at a time: `KEEP_PREFILL` on/off, `TARGET`, `IL_PER_LAYER`, REP 10 vs 0 (after G-rep),
   protect 1/2.

## 9. Knobs (all live; read at the refresh unless noted)

| knob | default | meaning |
|---|---|---|
| `V41_B1_HOT_POLICY` | `top` | `top` = today; `interleave` = this design |
| `V41_B1_HOT_TARGET` | 0.60 | box 1's share of the non-replicated mass |
| `V41_B1_HOT_TOL` | 0.02 | balance tolerance |
| `V41_B1_HOT_MOVES` | 3 | box-1 newcomers (fills + swaps) and, separately, REP newcomers per layer per refresh |
| `V41_B1_HOT_IL_HYST` | 40 | the interleave's region hysteresis (ranks) |
| `V41_B1_HOT_B2HEAD` | 60 | box 2's pinned head share per layer (clamped by the pin budget) |
| `V41_B1_HOT_IL_PER_LAYER` | 0 (= `per_layer()`) | box 1's owned + replicated ids per layer |
| `V41_B1_HOT_MAX_CHANGE` | env value (prod 3) | today's change cap, now live (return mode) |
| `V41_B1_HOT_HOLDER` | on | the holder fallback (decode rows, MOVING ids only) |
| `V41_B1_HOT_KEEP_PREFILL` | off | the prefill band skips KEEP |
| `V41_B1_HOT_PREWARM_STEP` | 2 | box-1 pre-warm reads queued per decode step (interleave only) |
| `V41_B1_HOT_REP` | 0 | replicated ids per layer |
| `V41_B1_HOT_REP_HYST` | 5 | rank hysteresis of the replicated set |
| `V41_B1_HOT_REP_LEG` | `cost` | `cost` / `box1` / `box2` (route; per decode step) |
| `V41_B1_HOT_LEG_*`, `..._LEG_OPEN_MS` | the 10-03 fit, 0.2 | the leg-choice constants (route; per decode step) |

## 10. Review round 1 (2026-10-08): where each finding went

1 return mode -> 2.4 + live MAX_CHANGE + test. 2 gates -> 3.1 "reproducibility" + 7 (G-il holder
off, G-rep pinned legs, cost under KL). 3 gain at the wrong operating point -> 0.7, 1, 8 step 2. 4
demand reads on swaps -> 3.2 holder fallback. 5 pre-warm -> 5 (`prefetch_now`, pacing, priced, no
KEEP hints). 6 prefill -> 4 (KEEP_PREFILL default off) + 6 + 8 bar. 7 KEEP predicate -> 4. 8 A/B
power -> 8 step 4. 9 region rule + separate hysteresis -> 2 steps 2-3, `IL_HYST` 40. 10 swap rule
-> 2 step 4 (improve-only, smallest mass within tolerance). 11 OPEN -> 3.1. 12 consumers -> 3 table
(residency hints, readahead, incoming, trace). 13 KEEP queue -> 4. 14 clamp fallback, prefill
room -> 4. 15 REP later -> 0.6 (and KEEP re-queue limited to refresh / decode entry / reconnect).
16 line refs fixed; swap cost noted in 2 step 4.

## 11. Review round 2 (2026-10-08): where each finding went

1 return exit unreachable -> 2.4 per-layer coverage exit (`per_layer - 10`), 60-refresh cap, noisy
test. 2 stale rule -> 3 table (`box2_stale` = today's expression less KEEP less MOVING). 3 holder
on every id -> 3.2 MOVING ids only (refresh-stamped, confirmed by the destination). 4 holder
predicate -> 3.2 (`held` only for box 2; not held and not incoming for box 1; the mask calls the
same function). 5 KEEP after prefill -> 4 (decode entry via `take_restore_keep` only; refresh and
reconnect fills on the restore queue; `KEEP_WORDS_STEP` dropped). 6 queue requirements -> 4 (decode
only, releases first, refill on the first step with pins on, dedup against the queue). 7 B2H
admissions unchanged -> 3 table. 8 return sub-mode -> 2.4 (MOVING + pre-warm for returning layers;
reads after the exit counted, not prevented). 9 overclaim -> 0.5. 10 sim prerequisites -> 8 step 2
(production's own trace; params reconstructed; recalibrated post-async). 11 nits -> `IlParams.hyst`
doc, 2.4 last bullet.
