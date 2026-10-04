# Two speculating streams on one row abstraction (StepRows)

Status: DESIGN rev 1, 2026-10-04, for review. Branch `worktree-ms-dspark2` (base = production
`worktree-lm-prefill-prod` 9762c6f, hub 9c6f8ea5).

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
DSpark steps" (no new code beyond scheduling) stays at the lone ~34.5 aggregate and halves each
stream's rate; batching both streams' blocks beats it by ~10-20% at today's acceptance.

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
- `slots()`, `len()`, `depth(i)`, `parent(i)`, `ancestors(i)` (the parent chain).
- `depends_on(i, j)`: `j` is an ancestor of `i`.
- `is_chain_layout()`: every non-root row's parent is row `i - 1`. Then each stream's rows are
  contiguous, row `j` of a stream's run sits at depth `j`, and its ancestors are exactly the rows
  before it in the run: what today's kernels compute (contiguous ranges). The ONE predicate the
  kernels gate on.
- `crosses(cut)`: some row at index `>= cut` has an ancestor `< cut`. A lane cut that crosses a
  dependency must be ORDERED (the later lane enters each layer after the earlier one has written
  it). Replaces "a slot spans the cut".
- `streams()`: `(slot, root row, rows of the slot)` in row order.

Builders: `StepRows::plain(&slots)` (all roots), `StepRows::chains(&[(slot, extra)])` (per
stream: a root then `extra` chain rows -- today's block layout), `StepRows::chains_from_runs(&slots)`
(a slot list with repeats read as today: each run of one slot = a root then a chain; what the
tests and every existing caller hand over), `StepRows::new(rows)` (general).

### 1.1 Where it is consumed (phase 1)

1. `KvArena::tables(rows: &StepRows)` instead of `&[u32]`: a row's offset inside its stream's
   block is `depth(i)` (today: the count of earlier rows with the same slot -- identical for a
   chain layout). Non-chain layouts: `Err("tree rows need ancestor-masked attention: not built")`.
   KERNEL FACTS: section 4 (filled from the code map).
2. `check_lane_cuts(who, rows, cuts, ordered)`: refuses a cut iff `rows.crosses(cut)` and the
   driver does not order it (today: iff a slot spans the cut; identical for chain layouts, and
   correct for independent streams and trees).
3. The ready-first driver's per-layer `Ph::Chain` wait: on iff `rows.crosses(cut)` (today: iff
   the lanes share a slot at the cut).
4. The forward drivers (`forward_step_arena`, `_pipelined`, `_lanes`, `_ready_first`) take
   `rows: &StepRows` where they take `slots: &[u32]` today; inside, `rows.slots()` wherever a slot
   list is indexed. Tests build `StepRows::chains` / `plain` instead of `vec![slot; r]`.
5. Server (`multistream::decode_rows`): builds the step's `StepRows` from its streams and their
   drafts; Engram hashes each row's own path (`depth`); head targets, captures and verify read rows
   by index; accept keeps each stream's kept PATH (a chain: a prefix, `KvArena::accept(slot,
   keep)` as today).

### 1.2 What trees still need (not built; listed so the type does not have to change)

- Producer: node selection over each stream's candidate tree (siblings from the drafter's biased
  logits via the markov head per parent; conf is identical for siblings, rank them by calibrated q).
- Kernels: ancestor-masked raw-window / compressed-row / indexer reads (a per-row bitmask over the
  stream's block rows); the compressor accumulator keyed by node, not position (today blocks are
  keyed by compressor group = position: siblings collide, `blocks_per_slot`); accept that compacts
  a non-contiguous kept path. `is_chain_layout()` is the switch that would flip.
- Verify: walk the tree (point-mass siblings test the residual after a sampled spine draft is
  rejected; exact for any shape).
- `V41_SUB_DEFER_ACCEPTED` (`b2_mirror::defer_*`) keys deferred admissions by row POSITION:
  siblings, and rows of different streams, share positions. It stays lone-stream only (as today);
  a tree build re-keys it by row index.

## 2. Two streams in one step (server, `multistream.rs` + `ms_dspark.rs`)

### 2.1 Who drafts

`V41_MS_DSPARK_STREAMS` (new live knob, integer 1..=2, default 1 = today): every live stream may
draft while `live <= V41_MS_DSPARK_STREAMS`; above it every step is plain (as today at 2+).
Per stream, unchanged from the lone path: the stage-1 gate (`should_draft`), the KV cap
(`can_step_rows(slot, 1 + cap)`), `cap <= remaining - 1`, `pos > 0`. Drafts run one after the
other (~11.5 ms each; the drafter is single-sequence, `MtpState` swaps one ring in per slot). A
failed draft costs that stream's drafts only.

### 2.2 The joint K policy (`ms_dspark::choose_ks`)

Input per drafting stream `s`: `conf_s[0..5]`, `cap_s`, `sampled_s`; the step's lane rule; one
cost model `c(R, lanes)` and the total draft time `D`. Output `K_s` per stream and the lanes.

Objective (plan section 6, unchanged): expected tokens per ms,
`(S + sum_s E_s(K_s)) / (c(S + sum K_s, lanes) + D)`, `E_s(K) = sum_{k<=K} prod_{j<k} sigmoid(conf_s[j])`.

- Exploration (`explore`, at `explore_p`, staleness-weighted by the cost cell each choice feeds):
  drawn BEFORE any conf is read, over `(K_1..K_S, lanes)` candidates. Exact (independent of drafts).
- All point-mass streams: global search over the `prod (cap_s + 1)` grid (<= 36 at S = 2). Exact
  (point-mass tests are exact for any K rule, plan 2.4).
- Any sampled stream: a JOINT STOPPING RULE, the S-stream generalisation of `choose_k_stopping`:
  from `K = 0` everywhere, repeatedly price every extension of the frontier (each stream's depths
  beyond its current `K_s` forecast at `sigmoid(conf_s[K_s])`, as the one-stream rule does); if the
  best extension beats stopping, add ONE draft: the next draft of the stream with the highest
  immediate value `run_s * sigmoid(conf_s[K_s])` among the streams the best extension extends;
  repeat. Exactness: including draft `k` of stream `s` reads `conf_s[..=k]` (conf_k reads draft
  `k-1`, never draft `k`) and other streams' confs, which are independent of `s`'s drafts (each
  stream's drafts come from its own drafter run and its own `draft_rng`). No draft's inclusion reads
  its own value. Point-mass streams in a mixed step follow the same rule (safe).
- `S = 1` returns exactly what `k_for` returns today (same exploration candidates, same rule): the
  lone path is unchanged (unit test).
- Caps: rows per lane <= 8 (plan 3.7: kernel regimes switch above 8 rows per lane; production
  decode never runs there): one lane `R <= 8`, two lanes `R <= 16`. `K_s <= cap_s`.

### 2.3 Cost model

A THIRD `LaneTables` pair, `multi`: steps where two or more streams speculate, rows = the step's
total rows (2..=12 at S = 2), one-lane and two-lane cells, aged by those steps, exploration as the
other pairs. Not the lone tables (a lone block is one KV stream with an ordered cut at every split)
and not the plain tables (rows = streams, no drafts), though MEASURED 10-04 their per-row costs
agree within ~3 ms at 2-3 rows. Cells start from the two-lane / one-lane production ladders
(`DEFAULT_LADDER*`, extrapolated linearly past 8 rows) with `START_MARGIN` as `LaneTables::new`.
The draft term `D` = the per-draft EWMA x drafting streams (serial drafts).

Each step feeds exactly one table: lone block -> `lanes` (today), multi-stream spec -> `multi`,
plain multi-stream -> `PlainLanes` (today), three-lane -> none (today).

### 2.4 Lanes and the cut

The lane rule is the spec rule (`spec_lane_rule`: two lanes only through the ready-first driver,
`V41_MS_STAGGER=2`, production). The cut stays the balanced contiguous `lane_split(b, n)`
(computed once, shared by drivers, head and captures: DSPARK_SINGLE_STREAM_PERF section 1). With
two streams of `1 + K_A` and `1 + K_B` rows the balanced cut crosses a stream unless
`1 + K_A == ceil(b / 2)`; then `rows.crosses(cut)` turns the ordering on, exactly as it is for a
lone stream today. Chain waits are mostly 0 at 6 rows today (10-03 log); the step logs them.
LATER (not phase 1): a `StepRows`-aware cut (stream-aligned when the imbalance is small, no waits).
`lanes3` stays off for any speculating step (today).

### 2.5 Verify, emit, accept, rings, accounting

- Verify / emit: unchanged per stream (`verify_block` over the stream's rows, emit token by token,
  stop at the first stop). `spec_out` becomes per stream.
- Accept: `KvArena::accept(slot, keep)` per stream (today's loop).
- Rings: unchanged (`ring_all` default writes every live stream's kept rows each step).
- `MsDspark::record` per speculating stream: calibration and the block stats per block; the step's
  cost sample once per step into the table it ran (2.3).
- Stage-1 gain, one formula for any S: `g_s = emitted_s x plain(S) / (step_ms + D)`, `plain(S)` =
  the plain step cost of S streams (S = 1: `lanes.one.cost(1)`, as today; S >= 2: `PlainLanes`).
- K = 0 after a draft: `record_k0` per stream (issue #3 finding 2 semantics).
- The `ms dspark: blocks` line gains `multi_blocks` (blocks verified beside another stream's block)
  and the `multi` cells; `ms.step` logs `spec` as a per-stream list.
- DEFER stays lone-only (1.2).

## 3. Knobs

- `V41_MS_DSPARK_STREAMS` (live, 1..=2, default 1). 1 = today, bit for bit (the step builds the
  same `StepRows::chains` the slot list described). Rollback = 1.

## 4. Kernel facts (code map 2026-10-04, base 9762c6f) and what changes

- `KvArena::tables(&self, slots: &[u32])` (kv_arena.rs:1092): the only per-row input is the slot.
  A run of one slot = consecutive positions from its committed `pos` (`StreamKv::step` per repeat,
  `j += 1`); runs may come in any slot order; refused: > 8 rows of a slot, a slot reappearing after
  another's rows, a dead slot, a raw window at its region end, a full store at a fire. So two
  streams' blocks in one call ALREADY work: what the kernels need for two chains exists. CHANGE:
  `tables(&self, rows: &StepRows)`; it requires `rows.is_chain_layout()` and walks the same runs
  (the per-row fields are unchanged; `depth(i)` = the run index `j`).
- Accumulator blocks are keyed by compressor group = position (`slot_block(slot) + pos/ratio -
  pos0/ratio`): the tree blocker in 1.2, irrelevant to chains.
- `accept(slot, keep, stream)` (:1247) checks only `1 <= keep <= 8`, not `keep <=` the rows the
  stream ran. CHANGE: the server checks `keep <= rows of the slot in this step's StepRows` before
  calling it (a cheap guard on the new multi-block bookkeeping).
- `check_lane_cuts(who, slots, cuts, ordered)` (forward_prefill.rs:252): Ok when `ordered`, else
  refuses `slots[c-1] == slots[c]`. CHANGE: takes `&StepRows`; refuses `rows.crosses(c)` unless
  ordered.
- Ready-first ordering (:4147): `ordered_dep[i] = slots[offs[i]-1] == slots[offs[i]]`; lane `i`
  enters layer `l`'s chain only after lane `i-1` has (`may_enter`, :4239). CHANGE:
  `ordered_dep[i] = rows.crosses(offs[i])`. Sufficient for any StepRows: the waits chain (lane `i`
  behind `i-1` behind `i-2` ...), and a dependency from lane `i` into lane `i-k` crosses every cut
  in between (a slot's rows are contiguous), so each of those lanes is ordered too.
- Drivers (`forward_step_arena` :3665, `_pipelined` :3760, `_lanes` :3951, `_ready_first` :4096)
  take `slots: &[u32]` (with repeats), `input_hcs`, `tokens`, no positions; one `tables` call
  (ready-first: one call sliced per lane by `RowTables::rows(lo, hi)`; pipelined / lanes: one call
  per lane, which is why they must refuse cuts inside a stream). CHANGE: `rows: &StepRows` in place
  of `slots`; `compact_for_step(rows.slots(), ..)` unchanged in meaning.
- Lane cut = `lane_rows(b, n)` balanced contiguous (:248) everywhere (drivers, head sources,
  captures); stays.
- HAZARD (existing): `decode_rows` reads `MS_STAGGER` twice per step (`spec_lane_rule` and
  `stagger_mode`): a live flip between the reads sends a two-lane speculating step to `_lanes` /
  `_pipelined`, which refuse the cut, and the step errors. CHANGE: one snapshot per step feeds both.
- HAZARD: with head-cands on, a step over `HEAD_BATCH_MAX` = 16 rows errors (`head_cands`,
  forward_prefill.rs:2981, `head_out` sized 16 x stride) instead of falling back. Two blocks of 6 =
  12 rows; the policy caps a speculating step at 8 rows per lane, 16 in all.
- Per-lane bookkeeping limits: pin/want counting stops above `PIN_DECODE_MAX_ROWS` = 16 rows per
  lane request; box-1 hot-set picks are counted only while a lane has `<= max(small_b_catchall_max,
  8)` rows. The 8-rows-per-lane cap keeps speculating steps inside both.
- Keyed by position: `b2_mirror` deferral (`ROW_POS`, `defer_flush(lo, hi)`, one global range):
  lone-only, unchanged (1.2). Engram rows are keyed by row index (safe); their hash input is the
  stream's own token chain (`ext = compressed ++ drafts`): per stream, unchanged.
- Captures (`pre_moe_chain`, lane-local, last `n` rows) and head targets concatenate lanes in row
  order and are read by global row (`caps[row0[i] + j]`): already per stream.
- Caps: `RowTablesDev` rows_cap = `n_slots x 8`; `MTP_CAP_ROWS` 128; scratch rows 512 per lane.

## 5. Gates

Host:
- `StepRows`: builders, invariants (each refusal), `depth`, `crosses`, `is_chain_layout`.
- `KvArena::tables(StepRows::chains(..)) == tables(slots)` before the change, for one stream, two
  streams with blocks, mixed plain + block (the existing `multi_row_tables_match_one_row_steps`
  extended).
- `check_lane_cuts`: the old and new predicates agree on every chain layout of <= 12 rows and
  every cut.
- `choose_ks`: `S = 1` equals `k_for` / `choose_k_stopping` / `choose_k` on random confs and cost
  tables; the 8-rows-per-lane cap.
- Exactness (G-RS1 style, `spec_sample` tests): two toy streams verified in one block step under the
  joint stopping rule with a synthetic conf that depends on the drafts through the markov-prev
  channel; chi-square of each stream's emitted sequence against the target; the negative control
  (a policy that reads conf past the frontier) is caught.

GPU (window, hub down, box 2 attached, `V41_REMOTE_SPLIT=1 V41_T2_CATCHALL=2` as the 10-03 gate):
- G5h `TWO BLOCKS IN ONE STEP`: the G5f block schedule with two streams' blocks in the SAME step
  (one lane, and two lanes through the ready-first driver with the balanced cut crossing a stream);
  every kept row == alone == G5f's one-block step bit-exactly (rejected tails included), Chain waits
  recorded when the cut crosses; the existing G5a-g unchanged.

Production: A/B per turn, `V41_MS_DSPARK_STREAMS` 1 vs 2, judged on 2-stream aggregate and
per-stream tok/s (`ms.step` live = 2) and lone-stream steps unchanged.

## 6. Out of scope (phase 2+)

- Batched drafting of both streams in one drafter pass (two rings, B = 10).
- A dependency-aware cut (stream-aligned, no Chain waits).
- Trees (1.2).
