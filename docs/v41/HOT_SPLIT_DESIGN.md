# Hot split: interleaved head + replicated top-K (design, 2026-10-08)

Status: DRAFT rev 1 (2026-10-08), branch `worktree-hot-split` (from `cb5b594`, the deployed
prefetch source). Owner 10-08: "let's go for hot split. no restarts w/o asking; otherwise, go nuts".

Sources: `HOT_SPLIT_SIM.md` (policy (c)/(d), the replay simulator), `DECODE_IDEAS_SWEEP_2026-10-06.md`
lever #1 (+10..15% traffic-weighted, the pipeline sim's +11.3%), the code map of 10-08 (every
ownership consumer, section 3), the live hub log of 10-08.

## 0. Decisions

1. **Hub only.** Box 2 already pins whatever the hub reports or words (`PinBook`), so its head
   share and the replicated set are kept pinned by the hub's own words and ledger. No daemon
   change, no box-2 restart: the deploy is ONE hub restart (owner's go), knob off.
2. **One live knob picks the policy**: `V41_B1_HOT_POLICY = top` (default; today's code path,
   bit-identical) `| interleave`. Every new behaviour sits behind `interleave`, so the restart
   changes nothing until the knob flips, and rollback is one knob-file line.
3. **Three-state ownership per (layer, expert):** `B1` (box 1 owns and computes), `B2H` (box-2
   head: box 2 computes and keeps it PINNED), `REP` (resident on both; the route picks the leg per
   lane-layer), and everything else `B2` (box 2's cold tail, its LRU). `partition_box2(l, e)` keeps
   its meaning "box 2 is the home": true for `B2H` and `B2`, false for `B1` and `REP`. The route
   loop is the only place a `REP` pick can go to box 2.
4. **Sticky, swap-capped placement** instead of the sim's static alternation. Incumbents keep their
   side; the box-1 mass share is pulled toward the target by at most `V41_B1_HOT_SWAPS` pair swaps
   per layer per refresh. A strict rank alternation flips both owners of every adjacent rank swap
   at every refresh; the sim never priced that churn (`HOT_SPLIT_SIM.md` 6, 8). The sim's
   held-out result already says the balance, not the exact ranks, carries the gain (a day-stale
   interleave keeps -11.6% of -12.6%).
5. **The leg choice for replicated picks is a pure function of the lane-layer's picks** (the sim's
   `route_replicated` greedy on fitted costs): deterministic given routing, no timing state, so
   `interleave` is as reproducible as today's routing.
6. **Pre-warm both sides at refresh.** Today a newly owned box-1 expert is a synchronous read on its
   first pick and a newly box-2 one a demand read; under the interleave a refresh moves HOT
   experts, so both sides are warmed asynchronously at the refresh (box 1: the L1 prefetcher; box 2:
   plain prefetch words, which also grant the pin).

## 1. Problem and expected effect

The decode pole is box 1's iGPU MoE: box 1 owns each layer's top-103 by decayed pick count (live
10-08: mass 0.74-0.77 of picks), box 2's identical iGPU serves the tail. The 10-06 sweep measured
iGPU MoE busy 72 of a 104 ms lone DSpark block and 151-157 GB/s at 4-6 rows. Halving box 1's
expert bytes per step is the only lever that moves the floor.

Expected: `HOT_SPLIT_SIM.md` policy `d_rep10_0.60` -12.6% fwd held out (two-lane 4-6 rows -17..-21%,
1-3 rows -1..-7%), measured BEFORE the async partial upload; with the async upload (live since
10-03) the sim says split + async = -21% vs neither, i.e. ~-12.5% on top of async. The 10-06
pipeline sim (calibrated on post-async, two-stream traffic) gives +11.3% weighted tok/s for iGPU
bytes -45%. The gain shrinks with a slower box-2 leg (+400 us/request halves it) and the model
does not price prefill displacement of box 2's head (section 6), the cache prior's interplay, or
refresh churn. Confidence medium; the live A/B decides.

## 2. Placement (`hot_set::refresh`, policy `interleave`)

Per layer, from the same decayed router-pick counts `c[e]` (unchanged counting and decay):

Sizes (per layer): `N1 = V41_B1_HOT_IL_PER_LAYER` (default 0 = `per_layer()`, i.e. today's 103,
clamped to the decode LRU; it includes the replicated set), `K = V41_B1_HOT_REP` (default 10),
`P2 = V41_B1_HOT_B2HEAD` (default 60: box 2's pinned head share). Keep set on box 2 per layer =
`P2 + K` = 70, 2,800 in all against box 2's pin budget of 4,232 (section 4).

1. **Replicated set.** Rank by `c`. Incumbent `REP` ids stay while their rank < `K + V41_B1_HOT_HYST`
   (the existing hysteresis, production 100 -- too wide for 10 ids: the rep band uses
   `V41_B1_HOT_REP_HYST`, default 5); fill to `K` from the top ranks not yet `REP`. Newcomers are
   capped by the swap cap below (they count as moves on both boxes).
2. **Region.** The non-replicated ranks `[0, N1 - K + P2)` plus incumbents (`B1`/`B2H`) whose rank
   is < that bound + `V41_B1_HOT_HYST` form the region; region size is fixed at `N1 - K + P2` (the
   weakest out-of-bound incumbents drop to `B2` first). The region is what the two boxes hold
   resident between them; the cold tail (rank >= ~163, ~4% of mass at today's curve) is box 2's
   LRU, as today.
3. **Sides.** Incumbents keep their side (`B1` stays `B1`, `B2H` stays `B2H`; a previous `REP` that
   left the rep set goes to `B1` if box 1 has a vacancy, else `B2H`). Region newcomers fill the
   vacancies hottest first, each to the side whose current mass share is further below its target
   (box 1's target `s = V41_B1_HOT_TARGET`, default 0.60, of the layer's NON-replicated mass, the
   sim's definition), subject to the vacancies (`N1 - K` on box 1, `P2` on box 2).
4. **Balance.** While `|share1 - s| > V41_B1_HOT_TOL` (default 0.02) and fewer than
   `V41_B1_HOT_SWAPS` (default 3) swaps were made this layer this refresh: swap the (`B1` id a,
   `B2H` id b) pair whose exchange `c[b] - c[a]` brings `share1` closest to `s` (ties: lower ids).
   Swaps keep both sizes fixed.
5. **Outputs.** `STATE[l][e]` in {B1, B2H, REP, B2} (one `AtomicU8` array replacing `OWN`; `OWN`'s
   readers go through `box1_owns`). The refresh returns, besides today's `(owned, mass, changed)`,
   per refresh: box-1 share achieved (mean over layers), swaps, box-1 newcomers, box-2 newcomers
   (B2H or REP), the keep-set size; and the two newcomer lists for the pre-warm (section 5).

The policy switch `top -> interleave` starts from today's top-103 (all `B1`): box 1's share is
~0.76, so the balance moves `SWAPS` hot ids per layer per refresh to `B2H` and the region fills box
2's vacancies -- a bounded migration (~15-25 swaps per layer, i.e. ~5-8 refreshes at `SWAPS` = 3;
`V41_B1_HOT_SWAPS` is live so a transition can run faster). `interleave -> top`: today's `refresh`
recomputes the top set from the counts under its own change cap (production 3 newcomers), so the
way back is bounded too.

`policy = top` runs today's `refresh` unchanged and sets `STATE` from its `OWN` (B1 / B2): every
consumer then sees exactly today's answers (gate G-top, section 7).

## 3. Consumers (the 10-08 code map; FP = `forward_prefill.rs`, BM = `b2_mirror.rs`)

| site | today | under `interleave` |
|---|---|---|
| route pick loop FP:9539-9556 | `partition_box2` -> `extra_remote`, else `ids` (b1_page_misses) | B1 -> `ids`; B2H/B2 -> `extra_remote`; REP -> collected, then the leg choice (section 3.1) |
| prefill / verify rows (same loop, b > 8 or not Arena) | same | REP -> box 1 always (fixed, so `lm_prefetch_words` FP:836 and readahead see box 1 as home) |
| cache-prior held mask FP:8405-8410 | box-1 owned: `pg.is_resident`; else `b2_mirror::resident` | REP: held if EITHER box holds it; B1/B2H/B2 unchanged |
| substitution `predicted_miss` (mode 2, off) FP:9150, 9339 | | REP never a predicted miss if either box holds it |
| displaced-pick admissions FP:9239-9308 | box-2 picks not held -> admission words | B2H handled by the keep set (section 4); REP never queued here |
| look-ahead `is_box2` FP:9658 (`V41_B2_MISS_PREFETCH`) | box-2 predictions -> LIKELY hints | REP is not box 2 (no hints); B2H hinted only if the mirror says not held (it should be pinned) |
| `wants_for_box2` FP:9745 / ledger counts | masked to box 2's partition | REP and B2H count as box-2 wants (ranking only; the keep tier protects them anyway) |
| ledger stale / `box2_lost` BM:1218-1237 | box-1-owned = stale, released first, never restored | REP and B2H are KEEP (section 4): never stale, never released by headroom, decay or band |
| `touch_resident` FP:9310-9319 | prior-added box-1 substitutes | also every REP pick routed to box 2 (its box-1 copy must not age out of the global LRU) |
| pick trace FP:9188-9207 | owner chars `1`/`2` | `R` for REP (the offline sim's parser takes it as box 1 + flag) |
| hot-set counting FP:9501, BM:1586 | router picks | unchanged |
| single-stream `forward_layer` FL:2567 | hash share | unchanged (not the multistream path) |

### 3.1 The leg choice (REP picks of one lane-layer)

After the non-replicated picks are classified (box-1 distinct `d1`, picks `n1`; box-2 distinct `d2`,
picks `n2`), take the REP distinct ids hottest-first (by picks in this lane-layer, ties by id) and
send each to the leg whose finish time grows least -- the sim's `route_replicated`:

    ig(n, d)  = IG0 + IG_D*d + IG_N*n + IG_B*b           (iGPU, ms; 0.100 / 0.0774 / 0.0046 / 0.0397)
    b2(n, d)  = d == 0 ? 0 : LINK(b) + (S0 + S_D*d + S_B*b)/1000   (116 / 99.9 / 15.6 us; LINK 58..318 us by b)
    t_box1 = max(ig(n1+k, d1+1), b2(n2, d2));  t_box2 = max(ig(n1, d1), b2(n2+k, d2+1))
    box 1 if t_box1 <= t_box2

Consequences the rule gets for free: a lane-layer whose other picks are all box 1's keeps its REP
picks on box 1 (`d2 = 0` makes box 2's leg a full round trip) -- no submit at all when every pick
is B1/REP, which today happens only by luck; a lane-layer already paying box 2's round trip moves
REP picks there while box 1's iGPU is the longer leg.

Fallbacks: the chosen side must hold the expert (box 1: `pg.is_resident`; box 2: the mirror's
held/pending); else the other side if it holds it; else box 1 (a sync read, counted). Constants are
knobs (`V41_B1_HOT_LEG_*`, live) so they can be refitted without a restart; `V41_B1_HOT_REP_LEG =
cost` (default) `| box1 | box2` pins the leg for gates and diagnosis. The other lane's load is not
visible in route (the code map: no cross-lane timing state); the sim's greedy ignored it too and
still calibrated within 2%.

Invariants kept (code map 4): every distinct pick lands in exactly one of `ids` / `extra_remote`
per lane-layer (`verify_routing_exactly_once`); `owns_eff == extra_remote`; the decision happens
before `sel_wants`, `owns_eff`, `remote_sel_override`, `box2_idle` and the submit, and reads no
post-`ensure` state. Two lanes may choose differently for the same layer (the remap upload is
stream-ordered per lane).

## 4. Box 2: the keep set (hub ledger, no daemon change)

KEEP = B2H ∪ REP (2,800 at the defaults).

- **Pin it:** at each refresh the hub queues a plain prefetch word (`push_prefetch_words`) for every
  KEEP id the mirror does not show held (the box-2 newcomers first). A resident one is pinned at its
  layer's next report with no read; a non-resident one is read speculatively and pinned when it
  lands. Words can be dropped on box 2 (no free staging set): the hub re-queues KEEP ids still not
  held at the next decode step, at most `V41_B1_HOT_KEEP_WORDS_STEP` (default 16) per step, under
  the speculative budget when `V41_B2_SPEC_BUDGET` is on.
- **Never release it:** the ledger gets a KEEP tier that `step_ranked`, `release_coldest_ex` and the
  decay never release. A KEEP id that leaves KEEP at a refresh becomes an ordinary entry (released
  by the usual order; a de-REP'd id that box 1 owns becomes stale as today).
- **Prefill band:** production runs the band at 4,096 (pins released down to ~136 at every prefill
  entry). `V41_B1_HOT_KEEP_PREFILL` (default on, live): the band release skips KEEP, so prefill gets
  `budget - |KEEP pinned|` unpinned slots at most (~1,430 instead of ~4,100). Off = today's release
  order with KEEP restored first at decode entry (`take_restore` accepts KEEP regardless of
  `last_want`). Which is better is a measurement (section 6): the default protects decode.
- **Budget:** box 2's budget is global, 4,232 (4,760 slots - 480 reserve - 48 nursery); KEEP at the
  defaults is 66% of it, leaving ~1,430 for the cold tail's ordinary pins and the nursery's
  promotions. `P2` is live; the refresh clamps `P2 + K` to `(budget - 1,024) / 40` per layer.

## 5. Box 1 side

- **Pre-warm:** box-1 newcomers (B1 or REP) at a refresh are handed to the L1 prefetcher
  (`prefetch_hint`, drained by the scheduler, MS:1652) so they are read asynchronously before
  their first pick. Today's code never feeds it on this path (the code map: its only arena caller
  is unreachable under production env), which is also why today's refresh churn shows up as
  `b1_read` in slow steps. The pre-warm applies under `top` too only with
  `V41_B1_HOT_PREWARM=1` (default 0 for `top` so G-top stays bit-identical; 1 under `interleave`).
- **Size:** `N1` defaults to today's `per_layer()` (103, the decode LRU fits 124). Box 1 shares one
  global LRU with prefill (unified pool), so owning more means prefill evicts more owned experts;
  `V41_B1_HOT_IL_PER_LAYER` is live and the A/B can try 103 vs ~115 after the policy itself.
- **REP copies routed to box 2** are touched (`touch_resident`) so they do not age out.

## 6. Risks and what is measured for each

| risk | why | counter / bar |
|---|---|---|
| prefill displaces box 2's head | box 2 now holds ~35% of decode mass; today it holds the tail | `b2_paged_replies` per step in the 0-20 s bin after a prefill->decode switch (hub_lh2 phase bins) vs `top` |
| refresh churn | swaps move hot ids | per refresh: swaps, newcomers b1/b2, `b1_read` ms/step, box-2 paged replies attributed to newcomers |
| box 2 becomes the pole | its per-expert cost is ~20% higher; the sim's optimum is box 1 at 55-60% | `remote_wait` exposed, late replies/step (`hub_req` reply after MoE end) per cell |
| cache prior retuning | protect 2 now protects box-2 head picks too | `sub.picks_swapped`, surprises; protect/lambda A/B after the policy A/B |
| keep set starves the tail | 2,800 of 4,232 pinned | box-2 cold misses/step, `pf_d_dropped`, nursery lands |
| leg-choice constants stale | fitted 10-03 | per-cell REP split (box 1 / box 2), and the iGPU / box-2 legs of lane-layers with REP picks |

## 7. Gates (one hub window; box 2 untouched, stays warm)

- **G-top:** the new hub with `V41_B1_HOT_POLICY=top` reproduces the deployed hub (6e3ede8e) bit for
  bit: `multistream_step` G5a-h + G6 against today's daemon, as the slice-D window did for `off`.
- **G-il:** `interleave` with `V41_B1_HOT_REP_LEG=cost`: G5a-h pass (exactly-once routing, the
  combine, lanes vs serial), G6 pass; KL(dec || arena) within the async-upload gate's envelope
  (mean <= 0.03, max <= 0.2): the router's picks are unchanged and only the box of a pick and the
  f32 grouping move (box 2's decode partials are f32, `remote_experts.rs:15`).
- **Unit tests (host, no GPU):** the placement (sizes, incumbency, swap cap, target convergence,
  `top` = today's `refresh` outputs on the same counts, the switch in both directions), the leg
  choice (the sim's cases), the KEEP tier (never released by headroom, decay or band; restore
  accepts it), `partition_box2` on all four states.

## 8. Rollout

1. Build + unit tests; one hub restart window with the owner's go (G-top, G-il, then up with
   `top`): no change in behaviour.
2. Flip `interleave` live; watch the transition (swaps, newcomers, b1_read, box-2 paging) for
   ~15 min; abort = knob back to `top`.
3. **Block A/B** `top` vs `interleave` (a placement cannot flip per turn: a switch moves ~2,000
   experts): alternating 60-min blocks, the first 15 min of each discarded, at least 3 blocks per
   arm, judged per step cell (regime, rows, lanes) on step p50 and tok/s, with the risk counters of
   section 6. Bar: weighted step p50 down >= 5%, no cell up > 3%.
4. Then the knobs one at a time: `KEEP_PREFILL` on/off, `TARGET` 0.55/0.65, `IL_PER_LAYER` 103/115,
   `REP` 10/0, protect 1/2.

## 9. Knobs (all live; read at the refresh unless noted)

| knob | default | meaning |
|---|---|---|
| `V41_B1_HOT_POLICY` | `top` | `top` = today; `interleave` = this design |
| `V41_B1_HOT_TARGET` | 0.60 | box 1's share of the non-replicated mass |
| `V41_B1_HOT_TOL` | 0.02 | balance tolerance |
| `V41_B1_HOT_SWAPS` | 3 | pair swaps (and REP newcomers) per layer per refresh |
| `V41_B1_HOT_REP` | 10 | replicated ids per layer |
| `V41_B1_HOT_REP_HYST` | 5 | rank hysteresis of the replicated set |
| `V41_B1_HOT_B2HEAD` | 60 | box 2's pinned head share per layer (clamped by the pin budget) |
| `V41_B1_HOT_IL_PER_LAYER` | 0 (= `per_layer()`) | box 1's owned + replicated ids per layer |
| `V41_B1_HOT_REP_LEG` | `cost` | `cost` / `box1` / `box2` (route; read per decode step) |
| `V41_B1_HOT_LEG_*` | the 10-03 fit | the leg-choice cost constants (route; per decode step) |
| `V41_B1_HOT_KEEP_PREFILL` | on | the prefill band skips KEEP |
| `V41_B1_HOT_KEEP_WORDS_STEP` | 16 | KEEP re-pin words per decode step |
| `V41_B1_HOT_PREWARM` | 0 under `top`, 1 under `interleave` | L1 prefetch of box-1 newcomers |
