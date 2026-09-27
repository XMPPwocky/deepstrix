# DSpark on the multistream arena — build plan

rev 0, 2026-09-27. Base: production branch `worktree-b2-pin-deploy` @ 04c00f3.
Status: PLAN, not reviewed. Supersedes the verify-path economics in
`DSPARK_VERIFY_ECONOMICS.md` / `DSPARK_WHAT_IS_LEFT.md` (they priced a prefill-shaped
verify that no longer needs to exist).

## 0. Ground rules

1. **Rejection sampling, always.** A draft `d ~ q` is accepted with probability
   `min(1, p(d)/q(d))`; on reject the emitted token is drawn from
   `norm(max(0, p - q))`; after K accepts a bonus token is drawn from `p_K`.
   Never accept on `argmax(p) == d` for a sampled request. (For a temperature-0
   request `p` is a point mass and the rule reduces to an exact match — that is the
   rule's own limit, not greedy accept.)
2. **A verify row IS a decode row.** No separate verify forward. A stream's draft
   rows ride the same `forward_step_arena*` step as every other stream's rows,
   with the same kernels, tables and routing, so `p` is the production decode
   distribution up to the LSB-level batch-composition differences decode already
   has (section 3.7).
3. **The drafter's precision is free.** Under rule 1 a worse `q` only lowers
   acceptance; it can never change what is emitted. Anything that makes the
   drafter cheaper (quantization, fusion, a smaller ring) is judged on
   acceptance x cost only, never on fidelity.
4. **Integrate, don't fork.** The legacy serial DSpark driver in
   `engine_worker.rs` (~1,500 lines) is retired at the end, not extended
   (owner decision 2026-09-24: "integrate cleanly, reuse the main path").

## 1. What it is worth (priced on today's arena step)

speedup = E x t(1 row) / (t(1+K rows) + t_draft), E = tokens emitted per block.

Arena step cost by rows (mean ms, prod log 2026-09-26 20:57 - 09-27 00:37,
kernel-sweep build): 1:76 2:105 3:121 4:138 5:160 6:175 — about 17 ms per added
row, the same slope in zero-paging (p10) steps, so it is not paging.

**Where the per-row cost goes** (ms.stage medians, same window):

| rows | wall | dGPU busy | box-1 iGPU MoE | box-2 compute |
|---|---|---|---|---|
| 1 | 71 | 33 | 22 | 13 |
| 3 | 116 | 39 | 54 | 25 |
| 6 | 154 | 77 | 107 | 45 |

The dense dGPU chain is flat inside a lane (it doubles at 4+ rows only because the
step splits into two lanes). The routed experts grow per row: routing is flat, so
**each added row brings ~4.35 new experts per layer** (~174 experts, ~3.3 GB of
MXFP4). MEASURED on `picks-sub-20260926-2054.trace` original picks: distinct
experts/layer for k consecutive tokens of ONE stream 6.00 / 11.00 / 15.57 / 19.84 /
23.89 / 27.77 (k=1..6) vs real cross-stream lane requests 11.74 (k=2) / 14.51 (k=3).
**A verify block's rows cost the same as rows of different streams.**

Speedup by acceptance profile (best K; temp-0 profiles unless noted; production
default is temperature 1.0, which lowers E — 09-17 one prompt lost 23% of DSpark
tok/s going 0 -> 1):

| acceptance profile | E at K=5 | 1 stream, t_draft 17 ms | 1 stream, drafter hidden | 3 streams |
|---|---|---|---|---|
| in-engine 09-18 warm | 2.35 | 1.15x | 1.32x | 1.20x |
| in-engine 09-17 best prompt | 2.91 | 1.26x | 1.43x | 1.22x |
| reference drafter, seeded (09-13 CPU oracle) | 4.38 | 1.73x | 1.90x | 1.42x |
| 0.90 per position | 4.69 | 1.85x | 2.03x | 1.51x |
| 0.95 per position | 5.30 | 2.10x | 2.30x | 1.69x |

2x on one stream needs E ~4.3 (drafter hidden) to ~5.0 (17 ms drafter) at K=5.
All-accepted ceiling is 2.4-2.6x. So there are exactly three levers, and the plan
pulls all three: **acceptance** (seeding, drafter temperature, K policy),
**drafter latency** (batch, hide, shrink), **per-row cost** (section 7.3).

UNMEASURED: E at the production recipe (T=1.0, top_p 0.95) on real agent traffic.
Milestone M0/M4 measure it; every number above moves with it.

## 2. Acceptance: the exact rule and what it must consume

### 2.1 Target distribution `p`
`p_j` is exactly the distribution `multistream::sample_row` draws from for row j:
temperature, then top-p over the tempered weights, then min-p. Checked: the OpenAI
request path accepts temperature (default 1.0), top_p (default 0.95) and seed;
`min_p_rel` is hard-coded 0.0; no penalties, logit_bias or top_k exist. Three
implementations of this distribution exist today (device kernel, legacy f64
`row_sample`, arena f32 `sample_row`); the arena one is the reference. Refactor `sample_row` into
`target_dist(row, mode) -> SparseDist {ids, probs}` (the survivors, normalized)
plus `draw(&SparseDist, u)`, and make BOTH plain sampling and rejection sampling
use `target_dist`. Gate: the refactored plain sampler emits bit-identical tokens
for the same RNG stream (G-RS2). One definition of `p`, or the output drifts
silently.

### 2.2 Draft distribution `q`
`q_i` is exactly the distribution `d_i` was drawn from, conditioned on whatever
the drafter conditioned on (`d_{i-1}` through the markov head). Any `q` is valid.
Two cheap, exact choices:

* **Point-mass drafts** (drafter argmax): accept with prob `p(d)`; on reject
  sample `p` with `d` removed. This is what the legacy "sample y ~ p, keep d if
  y == d" rule computes — it is already rejection sampling. It needs no `q`
  transfer at all, so it is the M5 starting point.
* **Tempered, truncated drafts**: `q_i` = drafter logits (after the markov bias
  from `d_{i-1}`) at temperature `tau_d`, truncated to its top-M (M ~ 64) and
  renormalized ON DEVICE, `d_i` drawn from that with a host-supplied uniform. Because the truncated `q` is literally what we
  sample from, it is exact, and only M (id, prob) pairs per position cross to the
  host. `tau_d` is a free, lossless knob; tune it for acceptance at the production
  temperature (point-mass is `tau_d -> 0`, the right choice for temp-0 requests).

### 2.3 The block procedure (host, f64 over survivors)

    rows = [next, d_0 .. d_{K-1}] at positions pos .. pos+K; row j yields p_j
    for j in 0..K:
        u ~ U(0,1)                                   # stream RNG
        if u < min(1, p_j(d_j) / q_j(d_j)):  emit d_j; if stop: break; continue
        y ~ norm(max(0, p_j - q_j));  emit y;  keep = j+1;  break
    else: y ~ p_K; emit y; keep = K+1
    next = last emitted; roll back rows keep..K

Stop conditions (turn end, EOS, max_tokens, cancel) are checked after EVERY
emitted token; a stop mid-block discards the rest and rolls back before any
snapshot is saved. The scheduler forces no tokens today (checked: no think budget
injection in `multistream.rs`), so there is no forced-token case.

Seeds: outputs for a fixed seed differ from non-speculative runs (different RNG
consumption); the DISTRIBUTION is identical. Document it on the API.

## 3. Arena: several rows of one stream in one step

Facts (V4.1, `config.rs` ~128-146): layers 0-1 dense, 2-19 ratio 2, 20-39 ratio 1;
KV-source stores at L2/8/14 (ratio 2) and L20 (ratio 1, read by L21-39 through
`with_kv_source`); index sources L2/8/14/20/24/28/32/36. At ratio 1 every position
fires, so on every decoder layer row p+1 must attend the compressed row that row p
produced earlier in the same step.

What already works per row: inside a layer the raw KV append (Stage 4,
`forward_prefill.rs` ~4447) is issued before attention (Stage 5, ~5369) on the
same stream, and the compressor's comp-row and index-key appends (~4993) precede
the indexer (~5820). Every table the kernels consume is already per row. So with
correct tables, raw KV, comp rows, index keys, the indexer and the reuse layers
are exact for same-stream rows with NO kernel change. The one structural problem
is the compressor accumulator.

3.1 **Multi-row tables** (`KvArena::tables`, `kv_arena.rs` ~650; drop the
    duplicate-slot refusal). A stream's rows are contiguous; for row j at
    `pos + j` with `f_j` = fires among its rows before j:
    `n_raw_per = n_raw + j`, `slot_per = region + raw_off + n_raw + j` (and the
    `_dec` twins), `n_comp_per = n_comp + f_j`, `fire_dst_row = base + n_comp +
    f_j`, region-full check over all of the stream's fires. The driver's window
    `n = min(nrp + 1, W)`, `offset = slp + 1 - n` then equals a one-row step after
    j advances, including the slide at `n_raw = W`. The positional asserts at
    ~4725 / ~5086 (`after == (pos+1)/ratio`) become free checks. CPU unit test on
    the tables.

3.2 **Compressor accumulator: stash, gather, pool, commit.** Today every row of a
    stream would state-write into, and pool out of, the stream's ONE block
    (`state_base_per = slot*ratio*width`); at ratio 1 even K=1 races. Replace:
    (a) state-write into a per-ROW stash `[rows_cap x ratio x width]`
    (`state_base_per[row] = row*ratio*width`, same launch count);
    (b) build per-fire snapshots with the existing `compressor_snapshot_gather`
    (`compressor_state_snapshot.hip` ~49-85: slot r of boundary k takes the stash
    row when `p >= pos0`, else the carried block), as a `_rows` variant or one
    launch per speculating stream on sliced views;
    (c) pool in snapshot mode, exactly as the Contiguous prefill path (~4911);
    (d) after acceptance, per ratio-2 store, if the next position P is odd copy the
    stash row of P-1 into block slot 0 (P-1 is always in the step: row 0 is always
    kept). Ratio 1 needs no commit.
    The block is never written during the step, so a rejected tail costs nothing
    to undo and a failed step can be retried. About one extra launch per KV-source
    layer. (The multistream plan's "K sequential launches with per-row snapshots"
    is not needed.)

3.3 **Advance moves out of the drivers** (`advance` calls at ~2852, 3045, 3180,
    3387): the caller runs `advance(slot)` `keep` times (already exact per
    position incl. the `n_comp`/`n_index_comp` lockstep), then the commit.
    `needs_compaction` checks `raw_off + n_raw + rows`, not +1; `RowTablesDev`
    `rows_cap` becomes `n_slots * (1 + K_max)` (`multistream.rs` ~124); no
    `compact_*` between a step and its commit.

3.4 **Rollback is truncation.** Nothing past the counters is ever read (the
    indexer scores `n_idx` rows, dense attention `n_comp`, raw windows are
    bounded), so rejected raw rows, comp rows and keys need no data restore.
    Host side: truncate `seq`/`compressed`, write drafter ring rows only for kept
    rows. Do NOT reuse the legacy `KvMark::advanced_by`: its accumulator restore
    is documented approximate when the kept prefix ends mid-segment (`state.rs`
    ~500-511), i.e. about half of all ratio-2 accepts. Do not use the
    process-global `SpeculativeAppend` flag either; it changes pager policy.

3.5 **A stream's rows stay in one lane.** Each lane's tables are built from
    pre-step arena state, `sd.kv_cur` is shared scratch, and `ready_first` gives no
    cross-lane layer ordering, so a split stream is wrong. Pass explicit
    stream-aligned cut points into all three drivers (pipelined `b.div_ceil(2)`
    ~2900; lanes/ready_first `offs` ~3087 / ~3238) and back to the server's logits
    slicing (`multistream.rs` ~913-941), balancing rows.

3.6 **Head.** `head_rows` returns full logits per row. `forward_head_batch` covers
    <= 16 rows per lane (`HEAD_BATCH_MAX`); above that each row falls back to its
    own head plus a blocking D2H (~1.1 ms each; S=8, K=5 ~55 ms). Chunk the batched
    head by 16. ~0.5 MB of logits per row to the host.

3.7 **Numerics, stated honestly.** Kernel families switch on lane row count
    (`mhc_pre_scaled_for` <= 8, `attn_dec_score_for` / dp4a / head <= 16,
    `prefill_f32_matvec` <= 64, and the indexer's gathered path for the whole lane
    once any row passes 512 comp rows). A verify row is bit-identical to a one-row
    step only while its lane stays in the one-row regime; otherwise it differs at
    LSB level, the same difference a stream already sees today between running
    alone and beside co-rows. Rejection sampling is exact with respect to the `p`
    the step computes; the gates are bit-exact where regimes match and KL-level
    (existing G5b bars) where they do not.

3.8 **Cache prior.** `V41_SUB` substitution applies to arena rows, so verify rows
    route like decode rows, consistent with production decode, which is already
    residency-dependent. Fidelity runs keep it unset. Fix the stale comment at
    `forward_prefill.rs` ~1085 ("DSpark verify is Contiguous").

## 4. Drafter in the arena

What exists (`het/mtp.rs`, legacy driver in `engine_worker.rs`): a 3-layer drafter
(`mtp.0-2`, 7.93 GB MXFP4/Q8_0, layers + entry on the iGPU, exit + tied head on the
dGPU) that drafts a 5-token block in ONE bidirectional B=5 pass from
`main_hidden` = mean-over-hc residuals ENTERING layers 37/38/39 at the last kept
position plus the embedding of the next token. It computes full `[5, N_VOCAB]`
logits on the dGPU, then a serial markov loop adds a bias from the previous draft
and takes argmax (`mtp.rs` ~1628). Cost: device ~17 ms (attention 8.5, MoE 7.3,
mHC 1.0), wall `19.8 + 2.55 n` ms. It is single-sequence (`MtpCtx` on
`WorkerState`) and loading it today turns multistream off for every request
(`multistream.rs` ~347: `is_legacy = move |_p| mtp_on`).

4.1 **Per-stream state is the ring.** Each drafter layer keeps a `[133 x 512]` f16
    ring (128 window + 5 transient block rows); `embed()` overwrites the hidden
    and the markov carry on every forward, so nothing else persists. The ring row
    for position `t` is written from the entry output of `main_hidden(t)`.
    ACTION: one A/B confirming `ring_write_only` == `advance_ring` state (the
    legacy code still spends a full forward on the last accepted row "for the
    carry"); if it holds, every ring update is a cheap row write. Arena layout:
    one `[n_slots x 133 x 512]` buffer per drafter layer (~3.4 MB for 8 slots),
    per-stream base + `ring_writes`, in `KvArena` next to the stream's KV.

4.2 **One drafter launch for all speculating streams.** B = 5 x S_spec rows.
    Dense GEMVs already B-pack (`GEMV_BPACK_MAX` = 16, so 3 streams per chunk;
    raise or chunk beyond), rope takes per-row positions, attention needs a
    per-stream ring base and `n_kv`, MoE/router/mHC are per-row loops, and the
    markov loop runs position-by-position in lockstep across streams.

4.3 **Exit: sample on device, export `q` compactly.** Replace the argmax markov
    loop with: bias, temperature `tau_d`, top-M (M ~ 64) truncation, renormalize,
    draw with a host-supplied uniform, feed the draw to the next position's bias.
    Export per position the M (id, prob) pairs, the draw, and `conf`; one D2H for
    all streams. Drop the diagnostic `plain` drafts from the hot path (5 of the 10
    blocking syncs per draft today). Point-mass mode = `tau_d -> 0`, no q export.

4.4 **Residual capture on arena rows.** The per-row `mtp_src` capture kernel
    exists in `forward_layer_pre_moe_v2` (`forward_prefill.rs` ~3561-3625) but the
    arena drivers refuse it (~2800, 2904, 3097, 3248, 3520). Capture into an
    ARENA-ROW-indexed buffer `[R x 3 x 5120]` f32 (~61 KB/row) so no lane mapping
    is needed (the legacy `mtp_lane_cut` knows only 2 lanes and was the source of
    two bugs, 271d751 / cb25715). After acceptance, the kept rows' captures feed
    ring writes and the last kept row feeds the next draft.

4.5 **Seed on every prefill, both lanes, ring-only.** `PrefillJob` captures the
    last <= 128 rows' `mtp_src` and writes them as ring rows. Legacy seeding read
    ONE lane (seeded ~63 of 128) and ran a full drafter forward per row (up to
    ~1.4 s of TTFT). Measured value of seeding: E 1.035 cold vs 1.649 seeded.

4.6 **Continuations.** Multistream restores a snapshot and prefills the suffix;
    seeding from the suffix fills the ring when the suffix is >= 128 tokens. For
    shorter suffixes persist the ring (0.4 MB + counter) with the snapshot and
    restore it before seeding the suffix rows. Without this an agent's short
    follow-up turn drafts from a cold ring (E ~1).

4.7 **Placement and memory.** The drafter shares `igpu.compute` with box 1's
    routed MoE, the ~17 ms/row term of the arena step; for a lone stream it runs
    after the step (no overlap), with 2+ lanes it can overlap the other lane but
    will contend — measure in M4. Its 7.93 GB comes out of box 1's expert pool:
    ~420 slots, ~9.5% of 4,450 (pool 78 -> ~70 GiB); M4's shadow run prices the
    extra misses for every stream. dGPU cost ~40 MB.

4.8 **Drafter latency levers** (section 7.2): its attention is a per-query loop
    because the batched attention kernels compile out the weighted sum on gfx1151;
    a 133-key B=5 attention should cost well under 1 ms, not 8.5. That alone takes
    the draft from ~20 to ~12 ms.

## 5. Scheduler integration (`multistream.rs`)

5.1 **Routing.** Delete `is_legacy = mtp_on`; DSpark becomes a per-stream property
    on the arena path (`V41_MS_DSPARK=off|shadow|accept`), so a DSpark server keeps
    multistream. Requests keep the legacy path only for what still needs it (none,
    once M8 lands).
5.2 **`decode_step` row build** (~822): per stream, rows `[next, d_0..d_{K_s-1}]`
    contiguous and in ONE lane; Engram hashes for draft rows from `s.seq` + the
    drafts; tables with per-row positions; lane split balances ROWS not streams.
5.3 **After the forward**: `head_rows` (already full logits) -> per-stream block
    procedure (section 2.3) -> emit each token with the stop check -> per-stream
    rollback of rows past `keep` -> `advance` by `keep` -> ring writes for kept
    rows -> one batched drafter launch for streams that will speculate next step.
5.4 **Admission**: `admit_stream` samples the first token from the prefill
    logits; the prefill capture supplies `main_hidden` for the first draft, so a
    stream can speculate from its first decode step.
5.5 **Lifecycle**: cancel/finish drop pending drafts; prefill bursts leave drafts
    valid (they depend only on the stream's own state); snapshot save happens
    after rollback, never with speculative rows in KV (the legacy driver has
    exactly this bug on a mid-block stop — do not port it).
5.6 **Trap**: a legacy prefill leaves `mtp_capture_rows` = 128 on the scratch;
    only the verify clears it, and the arena guard would then fail a step.

## 6. K policy

Per step, choose `K_s` for every stream to maximize expected tokens per ms:

    tokens(step) = sum_s (1 + sum_{k<=K_s} P_s(k))      time(step) = c(R) + t_draft(S)
    R = sum_s (1 + K_s)

`c(R)` is the live step-cost curve (EWMA of `ms.step` by rows, already logged).
`P_s(k)` = probability the first k drafts are all accepted, from the drafter's
confidence head, calibrated ONLINE at the production temperature (bucket conf ->
observed acceptance, EWMA; the 09-17 temp-0 calibration was [4,inf) 0.97,
[0,1) 0.46). Greedy: add candidate rows (s, k) in decreasing `P_s(k)` order while
the step's tokens/ms rises. Consequences: a lone stream gets deep blocks; as
streams arrive K shrinks toward 0, so speculation only ever fills spare batch
capacity; an unconfident stream gets K=0 while a confident one next to it gets 5.
Choosing K from anything known before the verify is lossless.

Hard caps: K <= 5 (block size), rows per lane request <= `PIN_DECODE_MAX_ROWS` =
16 (above it box 2 treats the request as prefill-shaped: staging band, no pins —
`remote_experts.rs:443`), and the arena's per-stream KV headroom.

## 7. Levers beyond parity ("and more")

7.1 **Hide the drafter behind the other lane** when >= 2 lanes exist.
7.2 **Shrink the drafter**: quantize its 7.93 GB (3 layers, 128 experts each,
    iGPU-resident) — free under rule 3 — to hand ~300-400 slots back to box 1's
    pool; graph-capture and fuse its B=5 chain.
7.3 **Per-row cost**: box 1's iGPU MoE grows ~17 ms per added row, while ~1.3 GB
    of added box-1 expert bytes at the measured 182 GB/s decode-kernel bandwidth is
    ~7 ms. Price the gap (catch-all share? the kwide chain at small b?) before
    counting on it; it also speeds up plain multi-stream decode.
7.4 **Issue draft rows' Engram gathers at draft time** (inputs known one step early).
7.5 Later: several candidate drafts per position with recursive rejection sampling
    (still exact); only worth it once misses are near zero.

## 8. Fidelity gates

* **G-RS1** (host, no GPU): the block procedure over synthetic conditional `p`/`q`
  families (point-mass q, disjoint supports, p == q, truncated q, K = 1..5, stops
  mid-block) — the emitted-sequence distribution matches direct sampling from `p`
  (chi-square over >= 1e6 draws per case).
* **G-RS2**: refactored plain sampler bit-identical to today's `sample_row`.
* **G5f** (GPU, server down; extends `tests/multistream_step.rs`): same-stream
  block invariance in three runs: `spec` (each step feeds `[next, d_1..d_K]`,
  drafts = the forced continuation), `alone` (one-row steps), `spec-reject` (rows
  past a scheduled `keep` are wrong tokens, then commit + truncate). Row j of
  `spec` equals `alone` at that position (bit-exact within a kernel regime,
  KL-level across one, section 3.7), and every step after a partial accept still
  equals `alone`, which proves rejected raw/comp rows, keys and stash are
  discarded. Cases: blocks starting at both ratio-2 parities (L2/8/14 fire every
  other row, L20 every row), rejections at both parities, `n_raw` below and at W,
  a block crossing `KV_CACHE_ROWS` (compaction), positions crossing 512 comp rows
  (pos 512 at ratio 1, 1024 at ratio 2) so the indexer fires mid-stream, the CED
  decoder window, every index-source layer, mixed steps (A at K=3 beside B, C at
  K=0) equal to their alone runs, all three lane drivers with streams whole.
  Plus a CPU unit test of the multi-row tables.
* **Golden gate** (`tests/v41_golden_gate.rs`, arena path): a speculative block
  with teacher-forced drafts is the golden transcript, so K>0 must report the same
  KL as K=0.
* **End to end**: temperature-0 byte identity DSpark on/off with deterministic
  residency (cache prior `V41_SUB` off, deterministic catch-all), run in a
  harness — NOT `scripts/v41_dspark_gate4.sh`, which pkills and relaunches the
  production server; at temperature 1, a suite comparing emitted-token NLL under
  the model, spec vs plain. Closes KNOWN_BUGS #7 for the arena path.

## 9. Milestones

Each milestone ends in a measurement or a gate. GPU gates load the model, so they
need the hub down (`tests/v41_golden_gate.rs`, `tests/multistream_step.rs`).

* **M0: acceptance at the production recipe** (parallel with M1; one server-down
  window; the existing legacy driver plus a small hook). Agentic suite at
  T=1.0/top_p 0.95 and at T=0; per depth, log `p(d)` (point-mass acceptance) and
  `sum min(p, q_tau)` for a `tau_d` sweep from the on-device exit logits; seeded
  vs cold ring. Also the one A/B of section 4.1 (`ring_write_only` ==
  `advance_ring`). Output: the E(K) distribution that says how far M5-M7 are
  worth taking. Skippable if the owner's acceptance data already covers T=1.
* **M1: rejection-sampling core** (host only): `target_dist` + `draw` refactor of
  `sample_row`, the block procedure of 2.3, G-RS1 and G-RS2.
* **M2: arena multi-row streams, no drafter**: 3.1-3.6; G5f, the tables unit
  test, the golden gate with teacher-forced blocks (K>0 must equal K=0 KL).
* **M3: drafter in the arena**: rings in `KvArena`, batched multi-stream drafter,
  capture on arena rows, device-side sampling exit exporting top-M `q`, seeding in
  `PrefillJob` (both lanes, ring-only), ring in snapshots. Tests: batched drafter
  == per-stream drafter; point-mass drafts == the legacy drafter's for identical
  inputs.
* **M4: shadow in production** (`V41_MS_DSPARK=shadow`): drafts and `q` computed,
  nothing acted on. Acceptance on REAL traffic at the production temperature from
  an unbiased estimator: with emitted tokens `y_i` and drafts `d_i`,
  `X_k = prod_{i<k-1} [1{y_i = d_i} * a_i / p_i(d_i)] * a_{k-1}`, where
  `a_i = min(1, p_i(d_i)/q_i(d_i))` (point-mass: `prod_{i<k-1} 1{y_i = d_i} *
  p_{k-1}(d_{k-1})`); `E[X_k]` = P(first k drafts accepted). Also measures the
  drafter's wall cost, iGPU contention, and what the smaller box-1 pool costs
  every stream. Go/no-go for M5.
* **M5: accept mode, simple K**: point-mass drafts; K=5 for a lone stream and a
  fixed row budget otherwise; A/B against off per `feedback_e2e_tokps_noise_floor`
  (suites at production temperature, the distribution of E, never one prompt).
* **M6: adaptive K and sampled drafts**: the section-6 policy, online confidence
  calibration, `tau_d` from M0/M4, stream-aligned lane balancing, the 16-row pin
  cap.
* **M7: cost levers** (section 7): drafter attention kernel for gfx1151, drafter
  quantization, drafter under the other lane, early Engram for draft rows, the
  iGPU per-row cost investigation.
* **M8: retire the legacy driver**: `finish_decode`'s DSpark path, the
  `V41_VERIFY_*` drivers, `mtp_lane_cut`, and `SpeculativeAppend`'s DSpark uses.

## 10. Risks and open questions

* **R1 Acceptance at T=1 on real agent traffic is unmeasured** (M0/M4). Every
  speedup in section 1 moves with it.
* **R2 The drafter costs every stream ~9.5% of box 1's pool.** Net value depends
  on the traffic mix (how much time one stream runs alone). M4 prices it; drafter
  quantization (7.2) shrinks it.
* **R3 iGPU contention** between the drafter and box 1's routed MoE.
* **R4 Kernel-regime switches with larger lanes** (3.7): LSB numerics, plus perf
  cliffs above 16 rows per lane in the head (3.6) and in box 2's pin eligibility
  (`PIN_DECODE_MAX_ROWS`, section 6).
* **R5 VRAM**: the dGPU is near full at `V41_MS_CTX_ROWS=844800`; `rows_cap`
  growth, the stash, and per-lane scratch sized for more rows must be counted
  before M2.
* **R6 Snapshot format** gains the drafter ring (versioned).
* **R7 Wasted paging**: rejected rows still page their experts and take cache-prior
  swaps; only the K policy's cost curve sees it.
* **Open**: DSpark on by default for every request, or per-request opt-out? Does
  the drafter's markov bias make `q_i` depend on `d_{i-1}` only (assumed in 2.2)?
