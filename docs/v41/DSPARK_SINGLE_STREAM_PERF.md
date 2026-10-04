# DSpark single-stream throughput: two-lane verify, early Engram, async ring writes

Status (corrected 2026-10-04): **items 1, 2 and 4 IMPLEMENTED (18ecbf0, 2026-10-01); item 3's
pre-draft Engram spawn NOT built** (only the `V41_MS_ENGRAM_THREADS` fan-out knob and
`engram_gather_ms`, "to measure before the pre-draft spawn"; the decode gather is still spawned
after the draft). Section 2 was replaced the same day: the code keeps `LaneTables` (a one-lane
and a two-lane `StepCost`) chosen by `LaneRule {Off, Threshold(min_rows), Learned}`
(`V41_MS_LANES_LEARNED`, 268b48b), and a `StepCost` is per-row-count cells by default
(`V41_MS_DSPARK_COST_SHAPE=cells`), not a line; `two_from` does not exist; the split helper is
`lane_rows(b, n)`. The two-lane verify engages only with `V41_MS_STAGGER=2` (code default 0;
production 2 per the hub env 2026-10-04) and `V41_MS_SPEC_LANES` on (default on).
Original: DESIGN rev 3, APPROVED (review round 3, 2026-10-01; dispositions at the end). Branch `worktree-ms-tail` (production = this branch's
f9ebe2c: merged head + nucleus candidates). Owner approved items 1, 3, 4 of the ranked list.

## 0. Where a lone stream's block goes (MEASURED 2026-10-01, production, 4-6 row verifies)

Block = draft 12.0 ms + verify ~126 ms (p50 at 4.9 rows) + ~3.5 ms between `fwd` end and step end
(accept, ring writes, record, emit) -> ~3.2 tokens.

Verify step, per layer (x40), from `hub_step` evtrace (1,286 single-lane steps, rows >= 4):

| part | per step | per layer |
|---|---|---|
| `lh_sel_sync` (dGPU attention + router until the selection is ready) | 25.4 ms | 0.64 ms |
| iGPU busy (routed MoE, box-1 share) | 60.0 ms | 1.50 ms |
| box-2 round trip (`remote_rtt`; link 0.43 + server 1.04 per layer) | 63.3 ms | 1.58 ms |
| `lh_remote_wait` (host blocked on box 2 after launching the iGPU MoE) | 58.0 ms | 1.45 ms |
| dGPU busy | 39.7 ms | 0.99 ms |
| `lh_engram_join` (exposed Engram gather) | 1.73 ms | -- |

(`lh_pager_block` 27.8 contains `lh_sel_sync`; host hop latencies are ~2 us.) One lane runs the
per-layer chain serially: dGPU attention -> [iGPU MoE || box 2] -> combine -> next layer. dGPU is
busy 32% of the wall, the iGPU 48%: the devices take turns.

Two lanes overlap them (lane B's attention runs while lane A's MoE runs). Same-day evidence from
the 2026-09-30 production log (27 h, 543K steps; V41_MS_PIPELINE_MIN_ROWS=4, so 1-3 rows run one
lane and 4+ rows two): step mean by rows 62.4 / 83.9 / 103.4 | 109.1 / 127.4 / 141.4. The 2->3 row
step costs +19.5 ms on one lane; 3->4, where the second lane switches on, costs +5.7 ms. Today's
single-lane verifies pay the one-lane slope at every size (3->4: 100.5 -> 119.4).

## 1. Two-lane verify: an ORDERED cut through the lone stream's rows

Today `pipelined = !spec && ...` and `check_lane_cuts` refuses a cut through one stream: row `j`
of a stream reads, at every layer, the KV that rows `< j` of the same stream wrote at that layer,
and lanes are not in lockstep.

VERIFIED (review round 1): every write a later row of the same stream reads at layer `l` is
enqueued inside `pre_moe_chain`, on `de.compute`: the raw SWA append (forward_prefill.rs:4527),
accumulator `state_write` (4775), the pool on a fire (4968), the comp row rms/rope/fp4/append
(4989-5117), the E2M1 `index_k` append (5060); reuse layers only read (5153). Route / prep /
launch / post take no `HetLayerState`; compaction, accept and grow run outside the step;
`with_kv_source` / `restore_compressor_lending` are host-only Option moves; mHC and the candidate
pool are lane-local. The reverse direction is clean (lane B writes only later positions; a firing
row in lane A pools a group ending at its own position).

Design (ready-first driver, `V41_MS_STAGGER=2` = production):

1. A spec step with `b >= V41_MS_PIPELINE_MIN_ROWS` runs two lanes when spec lanes are on. One
   `two_lane` flag drives the driver choice, the caps sizes and the head sources (today they key
   on `stagger2 || pipelined` with `pipelined = !spec`); `lanes3` is forced off for spec steps.
   Cut: the driver's contiguous split (lane A = the earlier rows, the odd row in A), computed ONCE
   (`lane_split(b, n)` in forward_prefill.rs, used by the drivers, the head sources and the capture
   sizes; today it is written out in three places -- review round 2 M1). No odd-row knob: a split
   that differs between the driver and the head would feed the head a stale row silently.
2. ORDERING: lane `i` may run `chain!(i, l)` only after lane `i-1` has, when they share a slot at
   the cut. A new phase `Ph::Chain(l)`: a lane leaving `Post(l-1)` waits there until the
   predecessor's entered-layer count is `> l`, and moves to `Route(l)` only after its chain is
   enqueued (`selected_ready` still holds the previous step's record and would query ready).
   Lane 0 enters layer 0 first already (3405-3407) and never waits on lane 1; a Done lane has
   entered 40 layers, so no deadlock. Counters: Chain waits and wait time per step -> hub_step
   (`rf_chain_waits`, `rf_chain_wait_us`) and the `ms.step` line.
3. TABLES: `arena.tables(&slots)` once for the step, split per lane by row range: per-row
   vectors sliced, each store's fire lists partitioned by owning lane and `fire_rows` re-based.
   (RowTables/StoreTables hold nothing else; `upload` builds no aggregate; equal to per-lane
   `tables` for lanes that split no stream.) The driver always uses the split form.
4. `check_lane_cuts(.., ordered)`: the ready-first driver passes true; the others keep refusing
   (they also run A before B per layer, but stay out of scope).
5. Unchanged: compaction, head sources in row order, captures (lane A then B), Engram staging by
   row offset, `accept` per stream after the step.
6. Knob: `V41_MS_SPEC_LANES` (default on) + run-time file `V41_MS_SPEC_LANES_FILE`.

Known cost: `stage_engram_rows_batch` is a blocking `copy_from_host` (2847) that drains the dGPU in
each lane's chain at layers 1 and 14; with two lanes a spec step pays it twice. Measure; a pinned
async staging is the follow-up if it shows.

Exactness scope:
- Bitwise vs single-lane AND vs `alone` (one-row steps, as G5f) with `V41_SUB=0` (the default).
  With `V41_SUB=3` (production) routing reads a host snapshot of residency when each lane's chain
  is enqueued, so two-lane rows differ from single-lane by design, as multistream lanes do today.
- Excluded from bitwise, covered with a KL bar instead: lane-wide host decisions -- `need_mask`
  (any row with `n_comp > 512`, 5469-5475: a lane-A row goes dense where the single lane went
  sparse at the TOP_K crossing; ratio 1 near pos 512, ratio 2 near 1024) and the top-k chunk
  count for `n_idx_max > 4096`.
- Batch-size arms crossed by 2-3 vs 4-6 rows (all documented bit-identical; G5f already shows
  rows of one stream are computed independently): `z16_comp_ratio1_for` (b in {3,4}, f16.rs:109,
  the layer-20 compressor), `Z16_PROJ_MAX_B = 4` for the indexer head weights (5832),
  `gemv_bpack_symbol(batch)` per-b twins (q8_0.rs ~599); `dn2 <= 8`, `dec_fuse <= 16`,
  `attn_dec <= 16` take the same arm.

Gate (G5g, `tests/multistream_step.rs`, one hub-down window, `V41_SUB=0`):
- a lone stream's verify blocks for b = 4, 5, 6 (2/2, 3/2, 3/3), at
  positions covering: both ratio-2 parities at the cut (B's first row firing on a group opened by
  A's last row), ratio 1 (fires every row), the raw window full (pos >= 128, `raw_off` sliding
  inside the block), compaction inside the step;
- two-lane vs single-lane vs `alone`: logits bitwise per row; after `accept(keep)` the arena state
  (raw windows, comp rows, index keys, accumulator blocks, counters) bitwise;
- a deterministic overtaking hook (`#[doc(hidden)] pub` atomics in v4flash-kernels, visible to the
  integration test: lane A's `Post(l)` is held until lane B has posted `l`): with the ordering,
  Chain waits > 0 and still bitwise; with the ordering disabled, the comparison MUST fail. Before
  each two-lane step in both arms the stream's DEAD state is poisoned with NaN (raw slots from the
  window end to the region end, comp rows and keys at or above `n_comp`, non-carried accumulator
  blocks), so neither arm can pass on stale identical data;
- box 2 attached if the window allows (each lane's remote request composition);
- the TOP_K crossing (n_comp = 511..513 at ratio 1) with a KL bar.

Expected: up to -10..-13 ms per 4-6 row verify (from unordered multistream lanes; the ordering
binds when box 2 answers lane B first, so the Chain-wait counters say what it costs).

## 2. K policy priced per lane regime

`StepCost` is ONE line `a + b * rows` with a strong slope prior; two lanes from MIN_ROWS make a
step change in cost (2->3 rows +19.5 ms, 3->4 +5.7) a line cannot fit, so K would be mispriced,
and an A/B through the run-time file would feed both arms into one ~500-sample fit.

Design: two fits, one per lane regime (`lanes = 1`, `lanes = 2`), each observed only from steps
that ran in it (`record` takes the lanes the verify actually ran). The regime is ONE snapshot per
step, taken before the draft: `two_from = Some(MIN_ROWS)` iff `V41_MS_STAGGER=2` (ready-first)
and `ms_pipeline()` and spec lanes on (env + run-time file); a lone stream is implied (only it
drafts). The K decision prices each candidate block `1 + k` with the two-lane fit when `two_from`
is `Some(m)` and `1 + k >= m`, else the one-lane fit (so with `two_from = None` only the one-lane
fit is used), and the same snapshot picks the driver. The draft time is one shared estimate.
(A single fit `a + b*rows + c*[two_lane]` would keep every lone step moving one level during pool
warming; two lines are kept for now because both regimes get samples every few blocks -- K spans
1..5.)
Priors: one-lane = `V41_MS_DSPARK_COST` (today's measured line); two-lane = the 2026-09-30
production two-lane means (rows 4..8: 109, 127, 141, 159, 177; `V41_MS_DSPARK_COST2`). The
`ms dspark: blocks` line logs both lines and the lane mix. Judging: item 1's mechanics by hub_step
verify step time per (rows, lanes) (the ~8% tok/s noise floor is the size of the effect); this
section's K policy by `tok_per_s` of `ms dspark: blocks` per arm, with the regime mix logged.

## 3. Engram gather: start earlier, fan out wider

Today the gather for all rows is spawned when the forward starts (= draft end) and joined at
layer 1: 1.73 ms p50 exposed at 4-6 rows. `gather_engram_rows` runs one thread per table, each
spreading the ids over <= 32 threads (engram_table.rs:153-170); cold reads cost ~0.8-1.7 ms each,
so a 5-row gather (120 ids per table) is ~4 read rounds.

Design (no plumbing through the drafter):
- row `next`'s gather is spawned BEFORE the draft (its hash depends only on the sequence);
- the draft rows' gather is spawned right after the draft with a wider fan-out than 32 threads
  (`V41_MS_ENGRAM_THREADS`; the spawns are serial at ~15 us each, so 120 threads start the last
  read ~1.8 ms late -- measure 32 / 64 / 128 first);
- `decode_rows` opens ONE `std::thread::scope` around the draft and the forward (the WorkerState
  destructure moves above the draft, adding `mtp` and `token_embd_*`; `engram` is a shared
  borrow); an assembler thread joins the per-row handles into today's `[table][b * ENGRAM_IN]`
  layout, so `LazyEngramRows` is unchanged. A plain or failed-draft step uses the pre-spawned
  `next` gather as its row. Hashing as today: draft row `j+1` hashes `s.compressed ++
  compress(d0..dj)`; DEAD rows stay zero.
- Bit-identical (same hashes -> same bytes; the cache returns the same values).
- Only the lone-stream draft path pre-spawns; 2+-stream steps and lone steps that never attempt a
  draft keep today's single batched gather. The profile / evtrace resets and `t0` stay after the
  draft (drafter stages stay out of `ms.stage` / `hub_step`; `record` keeps excluding the draft).

Expected: most of the 1.73 ms; the callback form (per-draft spawns from the exit loop) only if this
leaves exposure.

## 4. Ring writes: async at keep time (revised)

Round 1 showed the deferral design moves the sync instead of removing it: `igpu.compute` is a
BLOCKING stream, `copy_from_host` is a synchronous hipMemcpy that drains it (stream.rs:28-31,
measured in mtp.rs:614-619), and the draft starts with `inject_main_hidden`'s blocking copy. Also
`main_proj` is [5120, 15360] Q8_0 = 83.5 MB (weights.rs:222); with three `attn_kv` ~92 MB, ~0.45
ms at ~200 GB/s.

Design: `keep_rows` enqueues `ring_write_rows` WITHOUT the trailing `synchronize`
(`dspark_ring_write_rows_async`), and the rewind logic stays exactly as today. The write then runs
on the iGPU while the host does the ~3.5 ms tail (accept, emit, record, the next step's prologue);
the next draft's blocking copy finds it done. `rows_in` is filled by a blocking copy before the
kernels, so the caller's slice can go; the ring buffers persist in `SlotDraft`. On `ring_all`
2+-stream steps only the last stream's write is hidden (each `keep_rows` starts with a blocking
`rows_in` copy that waits for the previous one). Fault attribution (review round 2 N6): an event
is recorded on `igpu.compute` after each async write, per slot, and queried at the top of
`decode_rows` (before `drain_prefetched`'s blocking copies) and before that slot's next write; an
`Err` there resets exactly that slot. A sticky device fault poisons the context and aborts every
stream as today. Ring contents, order and rewind unchanged -> drafts
bit-identical.

Expected: ~0.5 ms per block (the write's device time + launches, now hidden), more on ring_all
steps. A `keep_ms` timer in `ms dspark: blocks` measures it before and after.

Gate: host/unit level the change is a removed sync; GPU: the keep/draft sequence test (hub-down
window, drafter weights): async vs sync writes -> ring buffers, draft ids and q bitwise equal, over
plain back-off steps, keep = 1, lone <-> 2+ transitions in both ring modes, slot reuse (reset +
seed), an injected draft error.

## 5. Rollout

1 + 2 behind the run-time file; 3, 4 env, default on. One hub-down window for the GPU gates (G5g +
the ring test; multistream_step loads the model once), then deploy and A/B 1 by step time per
(rows, lanes).

## Appendix: review round 1 dispositions

1 (G5g control could not fail) -> deterministic overtaking hook + mandatory failing control. 2 (deferral
moves the sync; 83.5 MB) -> section 4 rewritten: async write at keep time. 3 (deferral state cases) ->
moot with the rewrite; reset on Err kept. 4 (K policy) -> section 2. 5 (exactness scope) -> V41_SUB=0
gate, need_mask / top-k chunk exclusions with a KL bar. 6 (gate contents, ratios 2/1) -> gate list.
7 (batch arms) -> listed. 8 (Engram: fan-out first) -> section 3 simplified. 9 (counters, odd row,
one flag, lanes3, engram staging cost) -> section 1 items 1-2 + known cost. 10 (ring gate cases) ->
section 4 gate.

## Appendix: review round 2 dispositions

M1 (split computed in three places; odd-row knob) -> one `lane_split`, knob dropped. M2 (regime predicate,
decide/run snapshot) -> section 2: one snapshot per step incl. STAGGER=2 and pipeline on, used for K and the
driver. N3 (spawn cost) -> measure fan-out first. N4 (scope notes) -> section 3. N5 (hook visibility,
stale-data pass) -> doc(hidden) pub atomics + NaN poisoning + box 2 attached. N6 (fault attribution) ->
per-slot event. N7 (ring_all overstated) -> stated. N8 (judging section 2: tok_per_s per arm, regime mix
logged) -> rollout.
