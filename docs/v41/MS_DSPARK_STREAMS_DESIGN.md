# Two speculating streams on one row abstraction (StepRows)

Status: DESIGN rev 2, 2026-10-04 (review round 1: APPROVE WITH CHANGES; dispositions in section 8).
Branch `worktree-ms-dspark2` (base = production `worktree-lm-prefill-prod` 9762c6f, hub 9c6f8ea5).

## 0. What and why

Owner, 2026-10-04: "build two-simultaneous-dspark-streams, ensure the basic abstraction is
*unified* in a way that allows tree speculation (... some field on each token that says what
other tokens in this batch do i depend on ...). one unified abstraction."

A decode step is a batch of rows. Plain multistream, a lone stream's DSpark verify, two streams
both verifying, and (later) tree speculation differ ONLY in which earlier rows of the batch a row
reads at attention time:

| step | a row depends on (in-batch) |
|---|---|
| plain multistream | nothing (only its stream's committed KV) |
| DSpark chain (today, lone stream) | the previous row of its stream |
| two DSpark chains (this build) | the previous row of its stream |
| tree (later) | its parent node; ancestors = the parent chain |

Router, MoE, mHC, head, sampler are per-row and blind to it. What reads the dependency is
attention (raw window, compressed rows, indexer keys), what WRITES what attention reads (KV
append, compressor accumulator, index keys), the lane ordering (a later lane must not run a
layer before the rows it depends on wrote that layer's KV), and the host-side accept (which rows'
KV is kept). Today all four infer the dependency from "repeated slot ids are consecutive rows of one
stream". This build makes it an explicit per-row field and derives all four from it.

Why two streams (MEASURED 10-04 00:50-08:33 UTC, hub 9c6f8ea5, `ms.step` p50): a lone stream
runs DSpark at ~34.5 tok/s; as soon as a second conversation is live, steps go plain: 2 rows,
62.5 ms (n = 27K) = ~31 tok/s aggregate, 15.4 per stream, BELOW the lone rate. Rows of different
streams cost what verify rows cost (2 rows 62.5 vs 65.3 ms, 3 rows 76.7 vs 74.4; lone two-lane
verify 2..6 rows = 65.3 / 74.4 / 83.1 / 93.9 / 102.8 = ~9.4 ms/row). Priced at the overnight
acceptance (0.72-0.76): two streams speculating = ~37-41 tok/s aggregate (K 2-3 each, two serial
drafts 2 x 11.5 ms charged); ~50 at code-like acceptance (0.88). The alternative "time-slice lone
DSpark steps" stays at the lone ~34.5 aggregate and halves each stream's rate; batching both
streams' blocks beats it by ~10-20% at today's acceptance.

Owner: more than two speculating streams is not worth it. The code is written for N streams; the
knob is clamped to 2 (section 3).

## 1. The abstraction: `StepRows` (v4flash-kernels `het/step_rows.rs`)

```rust
/// One row of an arena step.
pub struct StepRow {
    /// The arena slot (stream) whose KV this row reads and appends to.
    pub slot: u32,
    /// The EARLIER row of the same slot whose output this row's input continues
    /// (its in-batch dependency); `None` = a ROOT: the stream's next token at its
    /// committed position, depending on committed KV only.
    pub parent: Option<u16>,
}

pub struct StepRows {
    rows: Vec<StepRow>,
    slots: Vec<u32>,  // rows[i].slot, for the callers that index by row
    depth: Vec<u16>,  // 0 for a root, depth(parent) + 1 otherwise
}
```

Construction validates (`StepRows::new`, an `Err`, never a panic): `parent < i` (rows are in a
topological order: a row's dependencies precede it), the parent has the same slot, every slot has
exactly ONE root, at most `ARENA_ROWS_PER_STREAM` rows per slot. A row's position is not stored:
it is the stream's committed position + `depth`. Siblings (two rows with one parent) share a
position -- legal in the type, refused by the kernels (below).

Derived queries (all O(rows), rows <= 16):
- `slots()`, `len()`, `slot(i)`, `depth(i)`, `parent(i)`, `ancestors(i)` (the parent chain),
  `depends_on(i, j)` (the queries a tree consumer needs; phase 1 uses them in tests only).
- `is_chain_layout()`: every non-root row's parent is row `i - 1`. Then each stream's rows are
  contiguous, row `j` of a stream's run sits at depth `j`, and its ancestors are exactly the rows
  before it in the run: what today's kernels compute (contiguous ranges). The ONE predicate the
  kernels gate on.
- `crosses(cut)`: some row at index `>= cut` has an ancestor `< cut` (checking parents suffices).
  A lane cut that crosses a dependency must be ORDERED. Replaces "a slot spans the cut".
- `streams()`: `(slot, root row, rows of the slot)` per slot, in root order.
- `root_of(slot)`, `rows_of(slot)`.

Builders: `StepRows::plain(&slots)` (all roots), `StepRows::chains(&[(slot, extra)])` (per stream a
root then `extra` chain rows: DSpark blocks), `StepRows::new(rows)` (general). The old reading of a
slot list (`chains_from_runs`: a run of one slot = a chain) exists ONLY under `#[cfg(test)]`, for the
kv_arena tests that compare tables over run lists; no production path or integration test infers a
dependency from slot repetition.

### 1.1 Where it is consumed (phase 1)

1. `KvArena::tables(rows: &StepRows)` instead of `&[u32]`: a row's offset inside its stream's
   block is `depth(i)`. Non-chain layouts: `Err` ("tree rows need ancestor-masked attention: not
   built").
2. `KvArena::compact_for_step(rows: &StepRows, ..)`: rows per slot from `rows_of` (was
   `step_rows(slots)`, a run count).
3. `check_lane_cuts(who, rows, cuts, ordered)`: refuses a cut iff `rows.crosses(cut)` and the
   driver does not order it (today: iff a slot spans the cut; identical for chain layouts, proven
   exhaustively by a host test).
4. Every driver makes ONE `tables(rows)` call and splits it by row range (`RowTables::rows(lo, hi)`,
   as ready-first already does); for cuts that cross nothing this equals per-lane calls (existing
   host test `lane_ranges_of_one_tables_call`).
5. The ready-first driver's per-layer `Ph::Chain` wait: on iff `rows.crosses(offs[i])`. Sufficient
   for any StepRows: any dependency edge crosses EVERY lane start between its two ends, so each lane
   in between is ordered too and the adjacent-pair waits chain.
6. NEW (review finding 1): on an ordered lane pair, lane `i` may also ROUTE layer `l` only after
   lane `i - 1` has. Today the host loop can route lane 1 first when both selection events fire
   between two polls; with the cache prior (`V41_SUB=3`) the earlier lane then sees the later lane's
   picks as box-2 PENDING (`b2_mirror` overlay), i.e. row `j`'s target would read the picks of rows
   whose inputs are drafts `>= j` (plan 2.5). Within a lane all rows route in one launch against a
   map fixed before it, so lane order is the whole channel. Cost: at most one host poll.
7. Server (`multistream::decode_rows`): builds the step's `StepRows` once and derives from it
   `row0` (roots), the token / hc / mode / Engram row arrays, the head and capture slicing and each
   stream's verify rows; accept keeps each stream's kept PATH (a chain: a prefix,
   `KvArena::accept(slot, keep)`, `keep <= rows_of(slot)` checked).

### 1.2 What trees still need (not built; the type does not change)

- Producer: node selection over each stream's candidate tree (siblings from the drafter's biased
  logits via the markov head per parent; conf is identical for siblings, rank them by calibrated q).
- Kernels: ancestor-masked raw-window / compressed-row / indexer reads (a per-row bitmask over the
  stream's block rows); the APPEND slots (`slot_per = .. + n_raw + j`, fire rows, index keys)
  keyed by block row, not depth (siblings share a depth); the compressor accumulator keyed by node,
  not position (today blocks are keyed by compressor group = position: siblings collide); accept
  that compacts a non-contiguous kept path. `is_chain_layout()` is the switch that flips.
- Verify: walk the tree (point-mass siblings test the residual after a sampled spine draft is
  rejected; exact for any shape).
- `V41_SUB_DEFER_ACCEPTED` keys deferred admissions by row POSITION and one global range; it stays
  lone-only (off in production: the 10-04 A/B found admissions -4%, t -0.3). A tree build re-keys
  it by row index.

## 2. Two streams in one step (server, `multistream.rs` + `ms_dspark.rs`)

Terms: S = live streams = the step's roots; S_d = the streams that drafted this step.

### 2.1 Who drafts

`V41_MS_DSPARK_STREAMS` (new live knob, integer 1..=2, default 1 = today; read ONCE per step):
while `1 < S <= V41_MS_DSPARK_STREAMS`, every live stream may draft; `S = 1` is today's lone path,
unchanged; above the knob every step is plain (as today at 2+). Per stream, unchanged from the lone
path: the stage-1 gate (`should_draft`), the KV cap (`can_step_rows(slot, 1 + cap)`),
`cap <= remaining - 1`, `pos > 0`, its own `draft_rng`. Drafts run one after the other (~11.5 ms
each; the drafter is single-sequence, `MtpState` swaps one ring in per slot). A failed draft costs
that stream's drafts only. S_d may be 1 (the other stream backing off, no residual yet, at its
token cap): still a multi-stream step (below).

### 2.2 The joint K policy (`ms_dspark::choose_ks*`, `MsDspark::ks_for`)

Used whenever S >= 2 and S_d >= 1. Input per drafting stream: `conf[0..5]`, `cap`; `base_rows` = S;
the step's lane rule; the multi-stream cost model `c(R)` (2.3) and the draft time `D` = S_d x the
per-draft EWMA. Output: K per drafting stream and the lanes.

Objective (plan section 6, unchanged): `(S + sum_s E_s(K_s)) / (c(S + sum K_s) + D)`,
`E_s(K) = sum_{k<=K} prod_{j<k} sigmoid(conf_s[j])`.

- Exploration (`explore`, at `explore_p`): drawn BEFORE any confidence is read, over the feasible
  `(R, lanes)` cells (staleness-weighted by the cell's time-aged weight), then the `R - S` drafts
  split uniformly among the K vectors with that sum (review NIT: drawing over K vectors over-weights
  cells many vectors map to). Exact (independent of drafts).
- All point-mass drafts: global search over the `prod (cap_s + 1)` grid (<= 36 at S = 2). Exact.
- Any sampled stream: the JOINT STOPPING RULE (`choose_ks_stopping`): from K = 0 everywhere,
  repeatedly price every extension of the frontier (each stream's depths beyond its current `K_s`
  forecast at `sigmoid(conf_s[K_s])`); if the best extension beats stopping, add ONE draft: the next
  draft of the stream with the highest immediate value `run_s * sigmoid(conf_s[K_s])` among the
  streams the best extension extends; repeat.
  - Plan 2.4 (inclusion): including draft `k` of stream `s` reads `conf_s[..=k]` (conf_k reads draft
    `k-1`, never `k`) and other streams' confidences, independent of `s`'s drafts (own drafter run,
    own `draft_rng`). No draft's inclusion reads its own value. Host tests: perturbing `conf_s[m]`
    leaves `min(K_s, m)` unchanged; for `m > K_s` it leaves every K unchanged; the joint chi-square
    over both streams' outputs (bias in either stream or coupling between them) with a negative
    control.
  - Plan 2.5 (the target must not depend on the draft it tests), stated honestly: `K_B` may read
    `conf_A[j+1]`, which reads `d_A[j]`; `K_B` changes the step's rows, so the lane cut and which
    rows share a lane. Rows' outputs depend on lane composition only through (a) the cache prior's
    cross-lane PENDING overlay (`V41_SUB=3`, production), (b) lane-wide regime switches
    (`need_mask` past 512 compressed rows, top-k chunking past 4096; the 8-rows-per-lane kernel
    regimes are excluded by the cap). With `V41_SUB` unset and both streams on the same side of
    those thresholds the targets are bit-identical whatever the cut (G5h). The lone path has the
    same class today (its K moves its own cut). Accepted and documented (KNOWN_BUGS): a
    substitution-level effect, not an LSB one, and far below the cache prior's own swap effect;
    1.1(6) closes the stronger same-stream route-order channel.
- Caps: rows per lane <= 8 (`SPEC_ROWS_PER_LANE`; plan 3.7 regimes, pin 16 / hot-set 8 per-lane
  bookkeeping); a row count no allowed lane count can hold costs infinity. `V41_MS_DSPARK_K`
  (fixed K) is clamped under the same cap (review finding 6): drafts are dropped from the deepest
  block first until the step fits.
- S = 1 (one root) never reaches these: `k_for` (unchanged). Host test: one block with
  `base_rows` = 1 gives exactly `choose_k` / `choose_k_stopping`, bit for bit.

### 2.3 Cost model

A THIRD `LaneTables` pair, `multi`: steps with S >= 2 roots and at least one verified draft row,
rows = the step's total rows (2..=12 at S <= 2), one-lane and two-lane cells, aged by those steps,
exploration as the other pairs. Not the lone tables (one KV stream, ordered cut at every split) nor
the plain tables (rows = streams, no drafts): the slopes differ (~9.4 ms per same-stream row vs ~14
for the 2 -> 3 independent rows, 10-04). Cells START from the 10-04 measured p50 ladders (review
finding 11; `DEFAULT_LADDER*` is 09-30 and reads 2 rows at 81-84 ms vs 62.5-65.3 now):
- two lanes, rows 1..=12: 62.5 (plain 2 streams), 65.3, 74.4, 83.1, 93.9, 102.8, then +9.4/row;
- one lane, rows 1..=8: 54.7, 71.1, 84.4, 97.9, 108.3, 117.1, then +9.7/row.

Each step feeds exactly ONE table, once (review finding 2): lone block -> `lanes` (today),
multi-stream step with drafts verified -> `multi`, plain multi-stream (incl. a multi step whose
drafts all came out K = 0) -> `PlainLanes` (today), three-lane -> none (today). Host test: a multi
step leaves the lone tables and `plain_ms` untouched.

### 2.4 Lanes and the cut

The lane rule is the spec rule (`spec_lane_rule`; two lanes only through the ready-first driver,
`V41_MS_STAGGER=2`, production), from the same per-step snapshot of `MS_STAGGER` the driver choice
uses (today it is read twice; a live flip between the reads sent an ordered cut to a refusing
driver). The cut stays the balanced contiguous `lane_rows(b, n)` shared by drivers, head and
captures. The server orders the streams' blocks by DESCENDING row count (stable; review finding
12): the balanced cut then falls between the two blocks whenever `|K_A - K_B| <= 1` (no ordering,
no Chain waits, no same-stream overlay channel). Otherwise `rows.crosses(cut)` turns the ordering on,
exactly as for a lone stream today. `lanes3` stays off for any speculating step.

### 2.5 Verify, emit, accept, rings, accounting

- Verify / emit: per stream (`verify_block` over the stream's rows from `StepRows`, emit token by
  token, stop at the first stop).
- Accept: `KvArena::accept(slot, keep)` per stream, `keep <= rows_of(slot)` checked.
- Rings (review finding 4): a stream's kept rows are ring-written when `ring_all()` (default) OR
  `S <= V41_MS_DSPARK_STREAMS` (was `S == 1`): under `RING=solo` two drafting streams would draft
  from stale rings (the 10-01 gap bug).
- Accounting API (review finding 2): per drafted block, calibration (K > 0), stats and the stage-1
  gate sample; per STEP, one cost sample into the one table it ran (2.3). Lone steps keep
  `record` / `record_k0` / `note_plain_step` as today.
- Stage-1 gain for a multi step, per drafted stream: `g_s = emitted_s x plain(S) / (step_ms + D)`,
  `plain(S)` = `PlainLanes`' best cost for S rows under the plain rule. It compares stream s's
  tokens per ms with the both-plain step; a K = 0 that came from competition for rows still wasted
  the draft, so backing off is right (the size of the gain is approximate: noted, not refined).
  An exploration draw of all K = 0 feeds no gain (as `record_k0`).
- Back-off is reset (gain to its optimistic start) for every live stream when S changes (review
  finding 5): a back-off earned beside another stream is not carried into lone mode.
- DEFER stays lone-only (1.2).

### 2.6 Fairness (review finding 10)

The objective is aggregate tokens per ms (plan section 6; the per-stream latency bound was left to
the owner). A stream backing off beside a K = 5 block rides a ~100-120 ms step for one token
(~8-10 tok/s, vs 15.4 at 2 plain rows). This build:
- logs per step each stream's emitted tokens (`ms.step` gains `spec_streams`) and the `ms dspark:
  blocks` line gains a per-stream rate summary of multi steps (`multi_tok_s` p10 / p50 over
  streams-steps);
- makes the A/B pass rule include the slower stream: per-stream p10 tok/s with the knob at 2 no
  worse than at 1, beside the aggregate gain;
- does NOT build a latency bound (owner's call; a knob `V41_MS_DSPARK_FAIR` bounding each stream's
  expected per-token time at `(1 + f) x plain(S)` fits the stopping rule -- it reads only confs up
  to each frontier -- and is the follow-up if the A/B shows the slower stream losing).

## 3. Knobs

- `V41_MS_DSPARK_STREAMS` (live, 1..=2, default 1). 1 = today, bit for bit. Rollback of the POLICY
  = 1. Rolling back the StepRows refactor and the route-order fix needs a binary revert (they change
  no numerics: G5a-g gate them bit-exact).

## 4. Kernel facts (code map 2026-10-04, base 9762c6f) and what changes

- `KvArena::tables(&self, slots: &[u32])` (kv_arena.rs:1092): the only per-row input is the slot.
  A run of one slot = consecutive positions from its committed `pos` (`StreamKv::step` per repeat,
  `j += 1`); runs may come in any slot order; refused: > 8 rows of a slot, a slot reappearing after
  another's rows, a dead slot, a raw window at its region end, a full store at a fire. So two
  streams' blocks in one call ALREADY work. CHANGE: `tables(&self, rows: &StepRows)`, chain layouts
  only, the per-row fields unchanged (`depth(i)` = the run index `j`).
- Accumulator blocks are keyed by compressor group = position (`slot_block(slot) + pos/ratio -
  pos0/ratio`): a tree blocker (1.2), irrelevant to chains.
- `accept(slot, keep, stream)` (:1247) checks only `1 <= keep <= 8`. CHANGE: the server checks
  `keep <= rows_of(slot)`.
- `check_lane_cuts` (forward_prefill.rs:252), ready-first `ordered_dep` (:4147), `may_enter`
  (:4239): CHANGE per 1.1(3, 5, 6).
- Drivers (`forward_step_arena` :3665, `_pipelined` :3760, `_lanes` :3951, `_ready_first` :4096)
  take `slots: &[u32]`; `_pipelined` / `_lanes` made one `tables` call per lane. CHANGE: `rows:
  &StepRows`; one call split by row range everywhere.
- HAZARD (existing, fixed here): `MS_STAGGER` read twice per step (2.4).
- HAZARD: with head-cands on, a step over `HEAD_BATCH_MAX` = 16 rows errors instead of falling back
  (`head_cands`, forward_prefill.rs:2981). Speculating steps are capped at 12 rows (S <= 2).
- Engram rows are keyed by row index (safe); their hash input is each stream's own token chain.
- Captures and head targets concatenate lanes in row order and are read by global row.
- Caps: `RowTablesDev` rows_cap = `n_slots x 8`; `MTP_CAP_ROWS` 128; scratch rows 512 per lane.

## 5. Gates

Host (cargo test, no GPU):
- `StepRows`: builders, every refusal, `depth`, `ancestors`, `crosses` == the old slot rule on every
  chain layout of <= 9 rows (exhaustive), a tree is not a chain layout and is refused by `tables`.
- kv_arena: the existing tables / lane-range / accept tests over `StepRows`.
- `choose_ks*`: one block + one root == `choose_k` / `choose_k_stopping` bit for bit; the global
  search is the grid optimum; the row cap (infinite cost) is respected; the frontier properties of
  2.2; `ks_for` with fixed K respects the cap; a multi step leaves the lone tables untouched.
- Exactness (spec_sample): two toy streams in one step under the joint stopping rule (sampled,
  q = p, point-mass), chi-square of the JOINT outputs against two independent plain samplers; the
  negative control (stream 0's first draft verified only when its value is `a`) is caught.
- knobs: the live-knob table.

GPU window (hub down, box 2 attached, `V41_REMOTE_SPLIT=1 V41_T2_CATCHALL=2` as the 10-03 gate):
- G5a-g unchanged (the refactor is bit-exact).
- G5h TWO BLOCKS IN ONE STEP (new arms in `multistream_step`): the G5f block schedule with two
  streams' blocks in the SAME step: one lane; two lanes ready-first with the cut (i) between the
  blocks (aligned), (ii) through the first block, (iii) through the second; junk tails (wrong tokens
  past the kept prefix) vs true tails; the forced-overtake hold and the unordered negative control
  (must differ) as G5g; prompts on the same side of `need_mask` for the bit-exact arms (every kept
  row == alone == G5f's one-block step), a straddling pair judged by the G5b KL bars.
- Drafter state: drafting stream B right after stream A (one `MtpState`, rings swapped) gives
  bit-identical drafts / q / conf to drafting B alone (`mtp_ring_async`-style).
- Server end to end: temperature 0, `V41_SUB=0`, knob 2, two concurrent requests: each output equals
  the same request's output run alone (greedy).

Production: A/B per turn, `V41_MS_DSPARK_STREAMS` 1 vs 2: aggregate tok/s of 2-stream steps,
per-stream p10 tok/s (2.6), lone-stream steps unchanged.

## 6. Sequencing and rollback

1. Refactor, no behaviour change: `StepRows`, tables / drivers / cut checks / compaction over it, the
   server building rows from it, the `MS_STAGGER` snapshot, the ordered route order (1.1(6)).
   Host tests + G5a-g.
2. Policy and server behind the knob (default 1): `choose_ks*`, `ks_for`, `multi` tables,
   accounting, block order, rings, back-off reset, logs. Host tests + G5h + the drafter-state test +
   the end-to-end greedy test.
3. Deploy with the knob at 1 (hub-only restart), then the per-turn A/B.

## 7. Out of scope (phase 2+)

- Batched drafting of both streams in one drafter pass (two rings, B = 10).
- Skipping the second draft when the first block's confidence already fills the row budget
  (exact: B's inclusion would read only conf_A).
- A dependency-aware cut beyond the size ordering.
- Trees (1.2). The fairness bound (2.6).

## 8. Review round 1 dispositions (reviewer: APPROVE WITH CHANGES)

1. 2.5 exactness: ACCEPTED -- documented (2.2), ordered lanes route in order (1.1(6)), KNOWN_BUGS
   entries for the lone route race (fixed) and the lane-composition coupling (accepted).
2. Accounting into lone tables: ACCEPTED (2.3, 2.5 API, host test).
3. Two live, one drafting: ACCEPTED (2 terms S / S_d; `multi` whenever S >= 2 and S_d >= 1).
4. Rings under solo: ACCEPTED (2.5).
5. Gain baseline: PARTLY -- `plain(S)` specified, back-off reset on S change; the marginal form not
   adopted (the both-plain baseline errs toward backing off a stream whose drafts lose rows, which is
   the economically right direction; noted).
6. Fixed K bypasses caps: ACCEPTED (2.2).
7. `chains_from_runs`: ACCEPTED -- test-only; the server derives every row array from StepRows.
8. Per-lane tables: ACCEPTED -- one call split by range everywhere (`StepRows::range` dropped).
9. NITs: append slots added to 1.2; the contiguity argument dropped (1.1(5)); `ancestors` /
   `depends_on` kept as the tree consumers' queries (tested).
10. Fairness: PARTLY -- logging + the A/B pass rule now; the bound is the owner's call (2.6).
11. Cold tables: ACCEPTED (2.3, measured 10-04 ladders, 12 rows); exploration over (R, lanes).
12. Cut alignment: ACCEPTED (2.4, descending block order).
13. DEFER for multi steps: NOT ADOPTED (off in production after the 10-04 A/B); the second-draft skip
   listed in 7.
14. Test gaps: ACCEPTED (section 5: frontier-min property, joint chi-square, G5h arms, drafter state,
   end-to-end greedy).
15. Sequencing: ACCEPTED (section 6).
