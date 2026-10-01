# DSpark on the multistream arena — build plan

rev 3, 2026-09-27. Base: production branch `worktree-b2-pin-deploy` @ 04c00f3.
Status: PLAN. Review round 1: APPROVE WITH CHANGES (15 findings, addressed in
rev 1; one pushed back: the production row mix, 1.3). Round 2: APPROVE WITH
CHANGES (N1-N7, addressed in rev 2). Round 3: APPROVE WITH CHANGES (R3-1..R3-5,
addressed in rev 3: paired gate + more reference positions, what 4.382 means,
A4 CED seeding, test hardening, the drafter KV quantizer FIXED in 6ab2c43, CPU
bisect first). **Round 4: APPROVE** (the new drafter KV chain verified step for
step against the main model's V4.1 chain and the reference quantizer; minor
R4-1..R4-4 folded in: one numeric pass rule, JSON check no longer silent, iGPU
kernel exercise, fuse the KV chain after parity). Supersedes the verify-path economics in
`DSPARK_VERIFY_ECONOMICS.md` / `DSPARK_WHAT_IS_LEFT.md` (they priced a
prefill-shaped verify that no longer needs to exist).

**Owner directive (2026-09-27): the engine's drafter must reach the SEEDED
REFERENCE acceptance, E 4.382 at K=5 on the gen2 agentic transcript (temperature
0, positions 256-344), before anything else counts** (bar ~4.2: a numerics-level
shortfall is fine). That is milestone M-A (section 9).

**What 4.382 is and is not** (review R3-2): it describes the DRAFTER ON gen2, a
transcript that is the reference model's own greedy continuation (greedy equals
the text at 100% of scored positions; T=1 rejection-sampling E 4.13 on it). On
`prose` (also the model's own greedy continuation, greedy == text at 100%) the
same reference drafter scores E 1.99 (RS at T=1: 1.82), BUT that run could seed
only 74 rows (the whole prompt) against gen2's 128, and seeding alone is worth
3.28 -> 4.38 on gen2; each figure is one 89-step transcript (effective n ~19).
The reference's own CONFIDENCE HEAD settles most of it (`scripts/v41_oracle/dspark_conf_look.py` over
the recorded `conf`/`hit`): within the prose run, steps with the window still
filling (i < 127, 53 steps) score E 1.74 and steps with a full window (i >= 127,
36 steps) 2.36, against gen2's 4.38; the chain of sigmoid(conf) predicts E 1.96
on prose (actual 1.99) and 4.57 on gen2 (actual 4.38); mean conf at d1-d5 is
+0.8 .. -1.4 on prose vs +8.8 .. +4.0 on gen2. The main model's entropy is NOT
lower on gen2 (0.55 vs 0.45 nats), so the gap is what the drafter can predict
(tool-call structure), not main-model certainty. So: the content effect is real
and large, the thin window costs part of prose's number. Rejection sampling at
T=1 costs little on either (RS-acc ~= greedy hit per depth). Unseeded, the head
is OVERCONFIDENT (gen2 noseed: predicts 4.34, actual 3.28), so realized E far
below the conf-predicted E is a cheap production alarm for a broken or cold
ring (M4 observability). M-A is a PARITY bar
(our drafter vs the reference drafter on the same input); production acceptance
is lower for the reference drafter too, and M0 prices production with the
realized E of the drafter that passed parity, measured on production-like,
reasoning-heavy traffic, never at 4.38.

rev 2 changes: M-A drafter parity first, with its test written
(`tests/dspark_parity.rs`); depth decisions as a stopping rule for sampled drafts
(6, N1); the point-mass test is exact for ANY draft value (2.4, N2); fairness
bound priced both ways (6, N3); M0 made feasible (N4); M4 sampled-q replay (N5);
the ratio-1 store's path in 3.2 (N6); nits (N7).

rev 1 changes: per-store stash indexed (slot, j) (3.2); the rule for what may
decide K (2.4); M0 is a hard go/no-go with traffic-weighted pricing (9); drafter
gap vs the reference as M0b (9); exact renewal replay replaces the M4 estimator
(9); ring-write ownership (4.1); dp4a regime at 8 rows and a default 8-row lane
cap (3.7); K clamps, lane count from streams, fairness, observability (5, 6, 10);
the acceptance-rate curve on the current step costs (1.2); the prompt-lookup
drafter measured on production outputs (7.6).

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
5. **A verify block is multistream rows that share one stream's attention
   context** (owner 2026-10-01: "dspark verify should basically be 'multistream
   but the attention is shared'"; share code as much as possible without hurting
   performance or making things convoluted). Concretely:
   * **One step.** Draft rows ride the same `forward_step_arena*` drivers, row
     tables, MoE/router/head and lane machinery as every decode row. DSpark adds
     per-row offsets inside a stream (3.1) and the per-store stash (3.2); there is
     no DSpark-specific forward.
   * **Shared attention.** A stream's rows read that stream's KV (raw window,
     compressed rows, index keys) with per-row counts. First implementation: the
     existing per-row kernels (attention is ~3-8 ms of a 154 ms 6-row step).
     Later, only if measured worthwhile: read the stream's KV once for all of its
     rows, as prefill's batched attention already does for one sequence.
   * **One sampler.** `spec_sample::TargetDist` is the single definition of `p`
     for plain sampling and verification (M1, done).
   * **One emit path.** Each emitted token goes through the stream's existing
     emit / stop / snapshot / cancel handling, once per token.
   * **DSpark-only code** is the drafter and its rings, the accept rule
     (`spec_sample::verify_block`), and the K policy.
   * **Performance guard:** with zero draft rows, the multi-row tables must equal
     today's (unit test) and the step must cost what it does today (A/B at the
     noise floor), so plain multistream pays nothing for the sharing.

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

### 1.1 The step-cost curve today

After the 09-27 05:37 restart (log 05:44-05:54) the arena step by rows is, p10 /
p50 ms: 1: 50 / 68, 2: 79 / 100, 3: 83 / 101, 4: 94 / 112; the 04:32-05:31 window
adds 5 rows 114 / 125. p10 is the zero-paging step
(`project_v41_step_variance_capacity_floor`). The owner's "44 tok/s at 4 rows" is
a 91 ms step = the warm (p10) ladder. Note the flat 2-4 row stretch on it: +12 ms
for two more rows.

A lone stream's block could run nearer the warm ladder than 4 separate streams
do (its rows share one warm working set in the pool), even though it touches as
many distinct experts. Unmeasured; M0(d) measures a real single-stream multi-row
step.

### 1.2 Per-stream decode speed against acceptance rate

One stream, i.i.d. per-draft acceptance `alpha`, `E = (1 - alpha^(K+1)) /
(1 - alpha)`, tok/s = `E / (t(1+K) + t_draft)`, best K per cell
(tmp `curve.py`). Ladders: "warm" = 1:50 2:79 3:83 4:91 (the owner's 44 tok/s
anchor; the log's p10 is 94) 5:110 6:124 (6 rows EXTRAPOLATED); "p50" = 1:68
2:100 3:101 4:112 5:125 6:140 (6 extrapolated). Cells whose best K is 5 (alpha
>= 0.95 warm, >= 0.85-0.9 p50) rest on the extrapolated t(6). The p50 column's
1-row baseline carries paging variance a same-stream block may not share, so it
is the less reliable column.

| alpha | warm, drafter hidden | warm, 20 ms drafter | p50, drafter hidden | p50, 20 ms drafter |
|---|---|---|---|---|
| plain decode | 20.0 | 20.0 | 14.7 | 14.7 |
| 0.6 | 23.9 (1.20x) | 19.6 (0.98x) | 19.4 (1.32x) | 16.5 (1.12x) |
| 0.7 | 27.8 (1.39x) | 22.8 (1.14x) | 22.6 (1.54x) | 19.2 (1.30x) |
| 0.8 | 32.4 (1.62x) | 26.6 (1.33x) | 26.9 (1.83x) | 23.2 (1.58x) |
| 0.9 | 37.8 (1.89x) | 32.5 (1.63x) | 33.5 (2.28x) | 29.3 (1.99x) |
| 0.95 | 42.7 (2.14x) | 36.8 (1.84x) | 37.8 (2.57x) | 33.1 (2.25x) |
| 1.0 | 48.4 (2.42x) | 41.7 (2.08x) | 42.9 (2.91x) | 37.5 (2.55x) |

Read it as: **a lone stream's ceiling is the multi-stream aggregate at 1+K rows**
(alpha = 1 at K=3 on the warm ladder IS the 44 tok/s), scaled by the efficiency
`E / (1+K)`. Break-even alpha on the warm ladder: ~0.45-0.5 with the drafter
hidden, ~0.6 with a 20 ms serial drafter. Where we are: in-engine E 2.35-2.91 at
K=5 (temp 0) = alpha 0.59-0.70; the seeded reference drafter's 4.38 = alpha 0.87.
Acceptance is clustered in practice, so realized E sits below the i.i.d. curve
at the same mean alpha (review finding 5); M0/M4 measure realized E directly.

So there are exactly three levers, and the plan pulls all three: **acceptance**
(the drafter gap to the reference, seeding, drafter temperature, K policy),
**drafter latency** (a 20 ms serial drafter costs 15-20% at every alpha: batch,
hide, shrink), **per-row cost** (7.3).

### 1.3 Several streams

With S streams already in the step, a draft row pays only if its acceptance
probability beats its marginal cost relative to the tokens the step already
yields: `P(accept) > dc(R) * S / c(R)` for the next row. Measured row mix
(share of steps): 20:57-00:37 1 row 41%, 3 rows 31%, 4 rows 19%, 5+ 7%;
04:32-05:31 3 rows 59%, 4 rows 34%, 5 rows 3%; 05:44-05:54 1 row 15%, 4 rows 57%.
Production is 1-4 streams almost all the time (the review's "5-8 most of the
time" does not match these logs). M0(b) turns this into time share by live-stream
count from the `hub_step` evtrace.

UNMEASURED: E at the production recipe (T=1.0, top_p 0.95) on real agent traffic.
M0 measures it before anything expensive is built.

## 2. Acceptance: the exact rule and what it must consume

### 2.1 Target distribution `p`
`p_j` is exactly the distribution `multistream::sample_row` draws from for row j:
temperature, then top-p over the tempered weights, then min-p. Checked: the OpenAI
request path accepts temperature (default 1.0), top_p (default 0.95) and seed;
`min_p_rel` is hard-coded 0.0; no penalties, logit_bias or top_k exist. Three
implementations of this distribution exist today (device kernel, legacy f64
`row_sample`, arena f32 `sample_row`); the arena one is the reference. Refactor `sample_row` into
`TargetDist::from_logits(row, mode)` (the survivors with their unnormalised
weights) plus `TargetDist::draw(u)`, and make BOTH plain sampling and rejection
sampling use it (done in M1, `spec_sample.rs`). Gate: the refactored plain sampler emits bit-identical tokens
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

Implementation details that the exactness depends on:
* **Zero-mass residual.** When `p ~= q`, `p/q` can round below 1, a reject can then
  meet `sum (p - q)+ ~= 0`; fall back to drawing from `p` (never normalize ~0).
* **Bit-identity of the plain path.** `TargetDist` keeps `sample_row`'s
  unnormalized f32 weights, its `z` summed in sorted order, and its vocab-order
  cumulative walk (now `spec_sample.rs`, `TargetDist`), so `draw` reproduces today's
  sampler exactly (G-RS2); f64 probabilities are derived only for the RS ratio
  and the residual.
* **Device-side draft sampling is tested**: a chi-square that the exported
  `(id, q)` pairs match the device's empirical draw frequencies (G-RS3).

### 2.4 What may decide K (and what may not)

Choosing how many drafts to verify is lossless only under this rule: whether
`d_j` is verified may depend on the prefix, on `d_{<j}`, on drafter state that is
not derived from the draw of `d_j`, and on independent randomness, never on the
drawn value of `d_j` itself. Counter-example (review finding 2): `p = q`, verify
`d` only when `d = a`; then `P(emit a) = p(a)(2 - p(a)) != p(a)`.

* Allowed: `P_s(k)` from calibrated confidence (conf_i reads the markov row of
  `d_{i-1}`, not `d_i`), the stream's acceptance history, load, stop/length caps.
* Forbidden for sampled drafts: gating on `q_j(d_j)`, dropping unlikely draws,
  "don't verify d_j because it is EOS".
* **The point-mass test is exact for ANY draft value** (accept with probability
  `p(d)`, else draw from `p` without `d`): conditional on every value of `d` the
  emitted token is distributed as `p`. It needs only that the draft is
  independent of the verifier's uniforms, not that it is deterministic (review
  N2). Consequences: under point-mass tests K may depend on any draft values,
  including later ones, so a global search over R is exact (M5); a SAMPLED draft
  can always be checked with the point-mass test instead of `min(1, p/q)` (exact,
  lower acceptance), which is the safe fallback for any position whose inclusion
  looked ahead; and a hybrid drafter chosen per block from the prefix (n-gram when
  the suffix match is >= 12 tokens, DSpark otherwise) is exact.
* "Draft-value-blind policy code" cannot hold literally: `P_s(k)` reads the markov
  row of `d_{k-2}` through conf. The rule that matters is the stopping rule of
  section 6.
* G-RS1 carries a negative control: a draft-value-dependent K rule must FAIL the
  chi-square (proves the test has power).

### 2.5 Invariant: `p_j` must not depend on `d_j`

It holds today: a stream's rows stay in one lane, the cache prior's held set
comes from earlier replies plus other lanes' pending overlay (the running Delta
in `b2_mirror.rs` ~239-262 is not from this call's rows), and nothing in the step
reads a stream's own later rows. The one LSB-level exception is the kernel
regime set by the lane's row count (3.7), which depends on K. Rule for future
work: no batch-level decision (catch-all, substitution overlays, pin grants) may
read a stream's own later rows in the same step.

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

3.1 **Multi-row tables** (`KvArena::tables`, `kv_arena.rs` ~959; drop the
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
    (a) state-write into a stash, ONE PER RATIO-2 STORE (L2/8/14), owned by
    `KvArena` next to that store's `state_kv` and indexed by (slot, j):
    `base = (slot * (1 + K_max) + j) * ratio * width`. Not by lane-local row: lanes
    interleave per layer and both have a row 0, and a single shared stash would be
    overwritten by the next KV-source layer before the post-accept commit (review
    finding 1; the KNOWN_BUGS #27 class). The ratio-1 store (L20) needs no stash:
    its pool reads only the row itself;
    (b) build per-fire snapshots with the existing `compressor_snapshot_gather`
    (`compressor_state_snapshot.hip` ~49-85: slot r of boundary k takes the stash
    row when `p >= pos0`, else the carried block), as a `_rows` variant or one
    launch per speculating stream on sliced views;
    (c) pool in snapshot mode, exactly as the Contiguous prefill path (~4911);
    (d) after acceptance, per ratio-2 store, if the next position P is odd copy the
    stash row of P-1 into block slot 0 (P-1 is always in the step: row 0 is always
    kept). Ratio 1 needs no commit.
    **The ratio-1 store (L20)** has no stash and no commit, but it must also stop
    state-writing into the shared per-slot block: K+1 rows of one stream would race
    on block slot 0 in one launch, harmless only because nothing reads it, while
    `export_to_state` still copies it. Its pool reads the row itself from `kv_cur`
    (the gather with rows = 1, or a direct pool). The existing gather adds APE when
    it reads `kv_cur`; the stash-reading variant must NOT add it again, since the
    state-write already did (review N6).
    The block is never written during the step, so a rejected tail costs nothing
    to undo and a failed step can be retried. About one extra launch per KV-source
    layer. (The multistream plan's "K sequential launches with per-row snapshots"
    is not needed.)

3.3 **Advance moves out of the drivers** (`advance` calls at ~2852, 3045, 3180,
    3387): the caller runs `advance(slot)` `keep` times (already exact per
    position incl. the `n_comp`/`n_index_comp` lockstep), then the commit.
    `needs_compaction` checks `raw_off + n_raw + rows`, not +1; `RowTablesDev`
    `rows_cap` becomes `n_slots * (1 + K_max)` (the `RowTablesDev::alloc` calls,
    `multistream.rs` ~174-188); no
    `compact_*` between a step and its commit.

3.4 **Rollback is truncation.** Nothing past the counters is ever read (the
    indexer scores `n_idx` rows, dense attention `n_comp`, raw windows are
    bounded), so rejected raw rows, comp rows and keys need no data restore.
    Host side: truncate `seq`/`compressed`, write drafter ring rows only for kept
    rows. Do NOT reuse the legacy `KvMark::advanced_by`: its accumulator restore
    is documented approximate when the kept prefix ends mid-segment (`state.rs`
    ~500-511), i.e. about half of all ratio-2 accepts. Do not use the
    process-global `SpeculativeAppend` flag either; it changes pager policy.

3.5 **A stream's rows stay in one lane (default).** Per-row tables are pure
    functions of pre-step counters and the row's offset j, so a lane-B slice built
    with the global j would be correct; the real blockers are the gather reading
    the shared `sd.kv_cur` and `ready_first`'s missing per-layer ordering between
    lanes. Pass explicit stream-aligned cut points into all three drivers
    (pipelined `b.div_ceil(2)` ~2900; lanes/ready_first `offs` ~3087 / ~3238) and
    back to the server's logits slicing (`multistream.rs`, the lane-cut / logits
    slicing in `decode_step`), balancing rows.
    Choose the lane count from the number of STREAMS, not rows:
    `V41_MS_PIPELINE_MIN_ROWS` counts rows and would split a lone stream's block.
    Option, after M0(d) measures a single-lane 6-row step: if the gather reads the
    per-store stash instead of `kv_cur`, a lone stream's block can span two lanes
    in the lockstep `pipelined` driver, recovering the box-2 overlap exactly where
    DSpark is worth most.

3.6 **Head.** `head_rows` returns full logits per row. `forward_head_batch` covers
    <= 16 rows per lane (`HEAD_BATCH_MAX`); above that each row falls back to its
    own head plus a blocking D2H (~1.1 ms each; S=8, K=5 ~55 ms). Chunk the batched
    head by 16. ~0.5 MB of logits per row to the host.

3.7 **Numerics, stated honestly.** Kernel families switch on lane row count
    (`mhc_pre_scaled_for` <= 8; `small_b_dense_dp4a` <= 8 for q_a, the shared
    expert and Engram, which switch to non-bit-exact f16x WMMA above
    (`dispatch.rs` ~256-262); the q_b/wo_a/wo_b dp4a arm and `attn_dec_score_for`
    and the head <= 16; `prefill_f32_matvec` <= 64; the indexer's gathered path
    for the whole lane once any row passes 512 comp rows). Production lanes are
    <= 8 rows today, so 9-16-row lanes would be a numerics regime decode never
    runs: **cap rows per lane at 8 by default** (G5f covers 9 rows for when the cap
    is lifted). A verify row is bit-identical to a one-row
    step only while its lane stays in the one-row regime; otherwise it differs at
    LSB level, the same difference a stream already sees today between running
    alone and beside co-rows. Rejection sampling is exact with respect to the `p`
    the step computes; the gates are bit-exact where regimes match and KL-level
    (existing G5b bars) where they do not.

3.8 **Cache prior.** `V41_SUB` substitution applies to arena rows, so verify rows
    route like decode rows, consistent with production decode, which is already
    residency-dependent. Fidelity runs keep it unset. Fix the stale comment at
    `forward_prefill.rs` ~1114 ("DSpark verify is Contiguous").

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
(`multistream.rs` ~401: `is_legacy = move |_p| mtp_on`).

4.1 **Per-stream state is the ring.** Each drafter layer keeps a `[133 x 512]` f16
    ring (128 window + 5 transient block rows); `embed()` overwrites the hidden
    and the markov carry on every forward, so nothing else persists. The ring row
    for position `t` is written from the entry output of `main_hidden(t)`.
    The review confirmed from code that `embed()` resets h and the carry and that
    `ring_write_only` runs `entry()` first, so ring rows + `ring_writes` are the
    only persistent state; make it a deterministic bitwise test (ring after N
    steps == a pure `ring_write_only` replay), not an A/B.
    **Who writes which row** (else the last kept row is written twice: a duplicate
    key and a drifting slot, invisible in output, visible only as lower E): the
    draft forward itself writes the row for its own `pos` (`mtp.rs` ~1110) and
    bumps `ring_writes` (~935). So after a block with `keep` kept rows,
    `ring_write_only` writes rows 0..keep-2 and the next draft writes row keep-1;
    a stream that will NOT draft next step writes all `keep` rows. Seeding: in
    the ARENA flow the first draft is at the last prefill row, so seeding writes
    every captured row except the last, and if the new stream does NOT draft at
    its first decode step that last row is written by `ring_write_only` right
    after admission (review N7). (The legacy driver's first draft comes after a
    bootstrap decode step at `start_pos`, so its `seed_mtp_ring` writes every
    captured row INCLUDING the last; 06c90dd.) A ring write is not free: `entry()` runs
    `main_proj` (~83 MB of weights) per row, so batch ring writes across rows and
    streams with `matvec_bpack`.
    Layout: one `[n_slots x 133 x 512]` f16 buffer per drafter layer (~3.4 MB for
    8 slots), per-stream base + `ring_writes`. It lives on the iGPU with the
    drafter, not in `KvArena`'s dGPU allocation; `KvArena` only owns the
    per-stream indices.

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
    exists in `forward_layer_pre_moe_v2` (`forward_prefill.rs` ~3586-3654) but the
    arena drivers refuse it (~2831, 2935, 3128, 3279, 3551). Capture into an
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
    ~420 slots, ~9.5% of 4,450 (pool 78 -> ~70 GiB); M0(c) prices that with no
    drafter at all (a live A/B of a 420-slot smaller pool). dGPU cost: ~40 MB
    today, plus ~66 MB once the markov embedding (129,280 x 256 f16, host-side
    today) moves to the device for on-device draft sampling (4.3).

4.8 **Drafter KV quantization is unverified for V4.1.** The drafter's ring and
    block KV go through the V4-style `fp8.launch_kv_post_fused` (`mtp.rs` ~1085 /
    ~1122), flagged in the 09-24 architecture review and never checked against
    V4.1's `fp4kv.launch_fp8_window`. A mismatch would only lower acceptance
    (rule 3), which is exactly why it could have hidden. M0b checks it.

4.9 **Drafter latency levers** (section 7.2): its attention is a per-query loop
    because the batched attention kernels compile out the weighted sum on gfx1151;
    a 133-key B=5 attention should cost well under 1 ms, not 8.5. That alone takes
    the draft from ~20 to ~12 ms.

## 5. Scheduler integration (`multistream.rs`)

5.1 **Routing.** Delete `is_legacy = mtp_on`; DSpark becomes a per-stream property
    on the arena path (`V41_MS_DSPARK=off|shadow|accept`), so a DSpark server keeps
    multistream. Requests keep the legacy path only for what still needs it (none,
    once M8 lands). Also change main's legacy-state stub condition
    (`engine_worker.rs` ~1098: `multistream::enabled() && mtp.is_none()`) and the
    guard at `multistream.rs` ~431: otherwise a DSpark server on the arena keeps
    ~1.07 GB of idle full-context legacy state on the dGPU (count it under R5).
5.2 **`decode_step` row build** (~937): per stream, rows `[next, d_0..d_{K_s-1}]`
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
5.7 **Clamp K before `tables()`**: `K_s <= max_new - completion - 1` and <= the
    stream's region headroom (raw and every store). `tables()` returns `Err` on a
    full region, and a failed step aborts every stream (`multistream.rs` ~336).
    Main's `KvArena::can_step` (`kv_arena.rs` ~542) assumes ONE position per step
    (`pos + 1`), and `take_stalled` / `grow` size regions on that basis: both must
    learn `1 + K_s` positions per stream before draft rows exist.
5.8 **Failure domain**: a drafter error sets that stream's K to 0 (plain decode),
    never `abort_all`; a drafter that keeps failing is disabled with a log line.
5.9 **Snapshots without residuals**: checkpoint snapshots taken under CED carry
    no layer 37-39 residuals; save them as "no ring" and let the stream start
    cold (seeded by its next prefill).

## 6. K policy

Per step, choose `K_s` for every stream to maximize expected tokens per ms:

    tokens(step) = sum_s (1 + sum_{k<=K_s} P_s(k))      time(step) = c(R) + t_draft(S)
    R = sum_s (1 + K_s)

`c(R)` is the live step-cost curve (EWMA of `ms.step` by rows, already logged).
`P_s(k)` = probability the first k drafts are all accepted, from the drafter's
confidence head, calibrated ONLINE at the production temperature (bucket conf ->
observed acceptance, EWMA; the 09-17 temp-0 calibration was [4,inf) 0.97,
[0,1) 0.46). Every input obeys the rule of 2.4.

**The confidence head is the K policy, from the first accept-mode milestone
(owner 2026-10-01: "we should make sure to use the confidence head").** Replay
over the reference's recorded per-block confidences and T=1 rejection-sampling
acceptances (`scripts/v41_oracle/dspark_conf_policy.py`; step costs = the
09-30..10-01 p50 ladder; K chosen per block to maximize predicted tokens/time
with `P(accept through k) = prod sigmoid(conf_j)`, uncalibrated), tok/s on the
55/45 code/reasoning blend of production output:

| streams, drafter | plain | best fixed K | confidence-gated | oracle (true per-block acceptance) |
|---|---|---|---|---|
| 1, hidden | 16.7 | 23.7 (1.42x) | 25.6 (1.53x) | 27.8 (1.67x) |
| 1, 20 ms | 16.7 | 19.7 (1.18x) | 22.6 (1.35x) | 24.4 (1.46x) |
| 2, hidden | 24.7 | 32.6 (1.32x) | 34.9 (1.42x) | 37.6 (1.52x) |
| 4, hidden | 37.4 | 39.4 (1.06x) | 43.2 (1.15x) | 46.1 (1.23x) |
| 4, 20 ms | 37.4 | 35.6 (0.95x) | 41.0 (1.10x) | 43.6 (1.17x) |

The head ranks blocks well: realized E at K=5 by predicted tercile is 2.19 /
4.59 / 5.56 on code (predicted 2.75 / 4.93 / 5.98) and 1.28 / 1.66 / 2.51 on
prose (predicted 1.40 / 1.90 / 2.57): slightly optimistic, monotone, which online
calibration fixes. It is what turns "speculation loses at 4 streams" (fixed K,
20 ms drafter, 0.95x) into a small win (1.10x). Caveats: one 89-step transcript
per content type; streams simulated in lockstep on the same transcript.
Plumbing it needs: `conf` per draft position exported with the drafts (4.3; the
exit computes it today, `MtpExit::conf`), the online calibration table, and
per-stream K in the row build (5.2).

Two stages, because `t_draft` is paid before conf exists:
1. **Draft or not**, per stream, from its recent realized acceptance and the
   current load (a stream whose recent blocks accept little is not drafted, and
   saves its share of the drafter launch).
2. **How deep**, per drafted stream, from conf. `c(R)` has jumps (the lane
   split, the 8-row kernel switch) and flat stretches (warm ladder: 2-4 rows cost
   +12 ms), so a first-decrease greedy leaves value on the table; but with
   SAMPLED drafts a global argmax over K makes "verify d_{k-1}" depend on conf_k,
   which reads d_{k-1} (review N1). So:
   * point-mass tests (M5): global search over every candidate R; exact (2.4).
   * sampled drafts (M6): depth is a STOPPING RULE: the decision on (s, k) may use
     `P_s(<=k)` only; values beyond k come from a draft-independent forecast (the
     stream's historical conditional acceptance by depth). Any position admitted
     by looking past the stopping rule is tested with the point-mass test.
   G-RS1 runs the REAL policy code under sampled drafts with a synthetic conf that
   depends on drafts through the markov-prev channel, not only the toy control.

Consequences: a lone stream gets deep blocks; as streams arrive K shrinks toward
0; an unconfident stream gets K=0 beside a confident one at 5. **Fairness**: the
objective is aggregate tokens/ms, so one confident stream's K lengthens every
other stream's step. Option: bound it so no stream's expected per-token latency
rises more than a set fraction over the step with EVERY stream at K=0. On the
warm ladder a 10% bound nearly disables speculation beside other streams (S=3:
one draft row +9.6% allowed, two +33% not; S=4: one row +21% not), so DSpark
would run only in lone-stream time (15-41% of steps). That is the owner's
throughput-versus-latency call: M0(e) prices it with and without the bound
(review N3). Always report the per-stream tok/s distribution, not only the
aggregate.

Hard caps: K <= 5 (block size); rows per lane <= 8 by default (3.7), which also
keeps every lane request under `PIN_DECODE_MAX_ROWS` = 16 (above it box 2 treats
the request as prefill-shaped: staging band, no pins, `remote_experts.rs:443`);
the arena's per-stream KV headroom.

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
7.6 **A prompt-lookup (n-gram) drafter, MEASURED on production outputs: weak.**
    Point-mass drafts that are a deterministic function of the prefix can be
    replayed exactly over sampled outputs (a block accepts exactly the leading
    drafts that equal the realized tokens). Replay over the 254 production
    snapshots (32 conversations, 912 unique assistant spans, 830K generated
    tokens including reasoning; tmp `ngram_replay.py`, `ngram_cond.py`), priced on
    the 20:57-00:37 mean ladder: fixed K gives E 1.2-1.7 and at best 1.08x on a
    lone stream; conditioning K on the suffix-match length (none 51% of positions,
    3-5 tokens 31% with P(first accepted) 0.49, >= 24 tokens 3.8% with 0.93) gives
    at best E 1.35 / 1.10x. So it is a zero-cost TEST VEHICLE for M2 on real
    traffic (M2.5), not a fallback worth shipping for speed.

## 8. Fidelity gates

* **G-RS1** (host, no GPU): the block procedure over synthetic conditional `p`/`q`
  families (point-mass q, disjoint supports, p == q, truncated q, K = 1..5, stops
  mid-block) — the emitted-sequence distribution matches direct sampling from `p`
  (chi-square over >= 1e6 draws per case).
  Includes the 2.4 negative control (a draft-value-dependent K rule must fail)
  and the zero-mass residual case.
* **G-RS2**: refactored plain sampler bit-identical to today's `sample_row`.
* **G-RS3** (GPU): device-side draft sampling, exported `(id, q)` vs empirical
  draw frequencies, chi-square.
* **G-RING** (GPU): after N accept/reject steps the drafter rings are bitwise equal
  to a pure `ring_write_only` replay of the kept rows (catches the double write
  of 4.1).
* **G5f** (GPU, server down; extends `tests/multistream_step.rs`): same-stream
  block invariance in three runs: `spec` (each step feeds `[next, d_0..d_{K-1}]`,
  drafts = the forced continuation), `alone` (one-row steps), `spec-reject` (rows
  past a scheduled `keep` are wrong tokens, then commit + truncate). Row j of
  `spec` equals `alone` at that position (bit-exact within a kernel regime,
  KL-level across one, section 3.7), and every step after a partial accept still
  equals `alone`, which proves rejected raw/comp rows, keys and stash are
  discarded. Cases: blocks starting at both ratio-2 parities (L2/8/14 fire every
  other row, L20 every row), rejections at both parities, `n_raw` below and at W,
  a block crossing the arena's raw region (`ARENA_RAW_ROWS` = 128 + 128 since
  main's b6e5e10, compaction), positions crossing 512 comp rows
  (pos 512 at ratio 1, 1024 at ratio 2) so the indexer fires mid-stream, the CED
  decoder window, every index-source layer, mixed steps (A at K=3 beside B, C at
  K=0) equal to their alone runs, all three lane drivers with streams whole,
  **two lanes that both commit at odd P in the same step** (the stash-collision
  case of 3.2), and a 9-row lane (the regime above the 8-row cap).
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

* **M-A: drafter parity with the seeded reference (FIRST; owner directive).**
  Target: E 4.382 at K=5, prefix acceptance 0.843 / 0.730 / 0.674 / 0.596 /
  0.539, on the gen2 agentic transcript at positions 256-344, temperature 0, as
  scored by DeepSeek's unmodified drafter (`gen2/dspark_accept_base.json`).
  **Bar: E >= ~4.2** (owner: a numerics-level shortfall such as 4.2 is fine;
  today's in-engine 2.3-2.9 is not). `PARITY_ASSERT=1` fails below ref - 0.2.
  Why this was never closed: every earlier engine number scored the engine's
  drafts against its OWN decode on text it generated itself (acceptance varies
  ~2x with content), so the 09-15 "engine matches the reference" conclusion
  compared a 500-token self-generation (E 3.077) with the reference's unseeded
  run on a different span. Only the entry projection was ever checked against the
  reference (`mtp_entry_parity`, one position).
  * **A1, drafter only, reference inputs** (`tests/dspark_parity.rs`, WRITTEN):
    feeds our drafter the oracle's own main-model residuals (converted by
    `parity_convert.py`, validated bit-exact against `mtp_ref/main_hidden.bin`),
    the same seeding (positions 128-255 into the window) and the same steps;
    compares draft by draft with the reference's recorded drafts, and scores
    against the reference's greedy targets. Needs the iGPU and ~8.7 GB, so the hub
    must be down (box 1 has ~5 GB free beside production). Run:
    `scripts/v41_oracle/run_dspark_parity.sh` (base, base with the legacy ring
    quantizer, nomarkov). **Gate on PAIRED statistics** (review R3-1): 89
    overlapping blocks have lag-1 autocorrelation ~0.65 (effective n ~19; the
    reference's own E carries a ~0.45 SE), so the gate is draft agreement given
    identical earlier drafts per depth, plus the moving-block bootstrap interval
    of E(ours) - E(ref) (both printed by the test). **The pass rule**
    (`PARITY_ASSERT=1`): point gap E(ours) >= E(ref) - 0.2 (~4.2 for `base`) AND
    the paired 90% interval's lower bound >= -0.45 (about one reference SE).
    Before the window, if possible, exercise `fp8_act_quant_inplace` and
    `kv_cache_append_slotdev` on gfx1151 (they have only ever run on the dGPU;
    review R4-1); otherwise a broken iGPU kernel shows up as `base` far below
    `base:v4`. Record `draft_ms`: the V4.1 chain is 4 launches per KV row instead
    of 1 (+54 per draft), so once parity holds, fuse a V4.1 `kv_post_fused`
    (per-32 ue8m0 over all 512 dims, tail included; review R4-2). More positions, CPU only: rerun the fixed `base` reference drafter
    on `main` (1,006 tokens, ~740 steps; its current `dspark_accept.json` predates
    the ffn_norm fix), `gen` and `gen_dspark` (needs ~13+ GB of RAM for the
    oracle's expert cache, so also a hub-down item unless the cache is trimmed).
    **Fixed before the first run** (6ab2c43): the drafter's ring and block KV
    used the V4-era `kv_post_fused` (E4M3, power-of-two scale per 64 over the first 448
    dims, RoPE tail unquantized), a DIFFERENT quantizer from the reference's
    `act_quant` (E4M3, ue8m0 scale per 32 over all 512 dims); now it is the main
    model's V4.1 window quantizer, `V41_MTP_KV_QUANT=v4` for the A/B. Remaining
    suspects if drafts still differ: the markov head (Q8_0 weights and Q8
    activations on a 256-dim embedding, `hf_v41.rs` ~444, vs the reference's f32
    `F.linear`; engine +0.82 E vs reference +2.00), q8 activations in the dense
    layers, the exit/head. **Bisect on the CPU first** (review R3-5): add
    engine-numerics toggles to the oracle's drafter script ((a) V4-style KV
    quant, (b) Q8 markov head, (c) q8 dense activations); whichever drops the
    reference from 4.38 toward ~2.8 names the culprit with no server downtime.
    **A1 RESULT (2026-09-27 17:58-18:02 window, logs ~/logs/dspark_parity_20260927):**

    | run | E ours | E ref | 90% paired interval | d1 draft == ref |
    |---|---|---|---|---|
    | base, V4.1 ring quantizer | 3.764 | 4.382 | [-0.92, -0.27] | 0.854 |
    | base, legacy quantizer | 3.719 | 4.382 | [-0.99, -0.32] | 0.854 |
    | nomarkov | 1.831 | 2.382 | [-0.97, -0.20] | 0.573 |

    **A1 PASSED after the shared-expert fix (761e46f; window 20:02-20:04, hub
    only, logs ~/logs/dspark_parity_20260927_fix):**

    | run | E ours | E ref | 90% paired interval | d1 draft == ref | agreement given identical earlier drafts, d1..d5 |
    |---|---|---|---|---|---|
    | base | 4.348 | 4.382 | [-0.101, +0.000] | 0.989 | 0.989 0.966 0.976 0.988 1.000 |
    | nomarkov | 2.427 | 2.382 | [+0.011, +0.090] | 0.978 | 0.978 0.920 0.963 0.948 0.959 |

    |conf - ref| fell from 3.7 to 0.15-0.21 per depth. The bug: `MtpState::moe` fed
    the shared expert through `dense_matvec`, which for Q8_0 weights reads only the
    quantized (`xq`, `xscale`) pair, and nothing in `moe()` requantized it, so the
    shared expert of all three layers ran on `attn()`'s quantization of the
    ATTENTION input. Found by static review against `model.py` after the first run
    localised the gap inside the drafter's layers. The pre-fix results below are
    kept for the record.

    Pre-fix: FAILS the bar. The quantizer fix is neutral (+0.045, noise). The drafter's own
    transformer output diverges from the reference inside its three layers: the
    transformer-only first draft matches the reference's only 57% of the time, and
    our confidence is systematically LOW (ours - ref mean -3.33 at d1, -1.95 d2,
    -0.71 d3, -0.49 d4, -0.23 d5; ours d1 mean +5.4 vs +8.75). Agreement given
    identical earlier drafts is 0.80-0.85 per depth. The markov head (fed the real
    token at d1) carries d1 up to 0.854. So the next step is a LAYER BISECT of the
    drafter against reference intermediates for a few steps (per layer: attention
    output, ring rows, hc mixes, MoE output, pre-head x): dump them from the CPU
    oracle (drafter only, ~6 GB RAM without the expert cache) and from our drafter
    (`dspark_parity` dump mode, a one-minute GPU run). Suspects: window attention
    (sink, scale, inverse RoPE, the per-query loop on gfx1151), hc pre/post mixing,
    FP8 -> Q8_0 weight requantization and q8 activations.
  * **A2, engine seeding**: `seed_mtp_ring` seeds from ONE prefill lane and only
    the last chunk (`engine_worker.rs` ~2888), so a short prompt or suffix seeds
    about half its rows (the 09-18 log: `seeded=19 n=20` on a 40-token prompt).
    Fix: capture across both lanes and across chunk boundaries, the last <= 128
    positions; ring-only writes. This is the same code M3 needs in `PrefillJob`.
    **Legacy path DONE (2026-09-27, builds; unrun):** `seed_mtp_ring` merges BOTH
    lanes by absolute position, takes the contiguous run ending at the latest
    captured row (<= 128), writes every row with `ring_write_only` INCLUDING the
    last prompt position (the first draft is at `start_pos`, after the bootstrap
    decode step, so no draft writes prompt rows; the old code skipped that row
    for want of its next token), and no longer runs a full drafter forward per
    row (up to ~1.4 s of TTFT). Remaining limit: a final chunk shorter than 128
    rows still under-seeds, because each lane keeps only its last forward's rows;
    the arena path gets a position-indexed capture (4.5).
  * **A3, engine end to end**: teacher-force the gen2 transcript through the
    engine and score our drafter on OUR residuals. Route: export a gen2 golden case
    from the oracle's gen2 dump (`scripts/v41_oracle/export_golden.py`; the
    existing `agentic` golden is a different 1,006-token transcript that shares
    only its first 258 tokens with gen2), add a tap that dumps the per-position
    mean-over-hc residuals after layers 36/37/38 during `tests/v41_golden_gate.rs`,
    then run `dspark_parity` with `PARITY_MH=<that dump>` and
    `PARITY_MH_BF16=0` (production feeds the drafter f32 captures). Compare our
    residuals with the oracle's per position (cosine) at the same time, and score
    against the ENGINE's own greedy as well as the oracle's (main-model top-1
    disagreement compounds over five depths and must not be charged to the
    drafter). The exit then runs on the dGPU as in production. Bar: E >= ~4.2.
  * **A4, the seeding condition production actually runs under** (review R3-3):
    every oracle transcript has a 57-token first prompt, so A3's seeding rows all
    come from decode steps. Production prompts are long and their ring is seeded
    from the CED replay's last 128 rows, whose decoder windows start empty
    (KNOWN_BUGS #28), residuals the reference drafter never saw. Run an A3 variant
    on `main` with `GOLDEN_DECODE_FROM` ~400 (seeding rows from a truncated
    replay) and compare E seeded from replay residuals against E seeded from
    decode residuals. If replay seeding costs E, mitigate (e.g. a 256-row decoder
    replay when DSpark is on, so the last 128 rows have full windows); otherwise a
    low M0(a) would be blamed on the drafter.
  Parity is judged at temperature 0 because that is how the reference was scored;
  production-temperature acceptance follows from the same drafter (M0).
* **M0: go/no-go, before M2 or M3** (parallel with M1), evaluated at the
  acceptance M-A delivers:
  (a) **realized E(K)** at T=1.0/top_p 0.95 and at T=0 on an agentic suite, with
  seeding FIXED (M-A A2 is a prerequisite: the legacy seeding covers ~half the
  rows), from the existing legacy driver plus a small hook (one server-down
  window): per block the accepted count, `p(d)`, and `sum min(p, q_tau)` for a
  `tau_d` sweep from the on-device exit logits;
  (b) **the traffic mix**: time share by live-stream count, from the `hub_step`
  evtrace (no server change);
  (c) **the price of the drafter's memory, with no drafter built and no
  restart**: replay box 1's pick trace through the existing pool simulators
  (`scripts/belady_bound.py`, `scripts/policy_holdout.py`) at ~4450 vs ~4030
  slots and convert misses to ms; a live A/B only to confirm (an effect near 9.5%
  sits at the 8% end-to-end noise floor and is dominated by box-2 warming);
  (d) **the single-lane cost of 5-6-row steps**: a synthetic load of 6+ streams in
  the window with `V41_MS_PIPELINE_MIN_ROWS=7` (live traffic has only 3-7% such
  steps, from different streams); the true same-stream cost comes from M2's G5f
  `spec` runs, which gate M2.5/M3;
  (e) the traffic-weighted expected gain from (a)-(d), with and without the
  section-6 fairness bound, against a stated threshold (start: >= 1.2x on
  one-stream time, no loss at the measured mix). Never decide no-go on a drafter
  gap M-A can close.
* **M0c: price the per-row cost** (7.3): pure measurement from `ms.stage` and a
  microbench; it also speeds plain multi-stream decode, so it is not DSpark-only.
* **M1: rejection-sampling core** (host only): `target_dist` + `draw` refactor of
  `sample_row`, the block procedure of 2.3, G-RS1 (with its negative control),
  G-RS2. **DONE 2026-10-01:** `crates/deepstrix-server/src/spec_sample.rs`
  (`TargetDist`, `DraftDist::{PointMass, Sampled}`, `verify_position`,
  `verify_block`; `multistream::sample_row` now calls `TargetDist`). Tests
  (`cargo test -p deepstrix-server --features v41 --release --lib spec_sample`,
  host only): G-RS2 bit-identical tokens and RNG consumption vs the old sampler
  over >10,000 row x mode cases; G-RS1 chi-square of the first 3 emitted tokens
  of a speculative toy chain vs plain sampling, 120,000 runs each, all under the
  p = 1e-4 critical value (point-mass K=1 28.8 / 54.2, K=4 26.6 / 54.2; sampled
  K=3 18.3 / 54.2, K=4 28.2 / 54.2; q outside supp(p) 8.1 / 28.4; q == p K=4
  140.3 / 215.9; random K 16.6 and 19.4 / 54.2); the negative control
  (draft-value-dependent K) scores 38,483 vs 215.9; temperature 0 emits exactly
  the greedy chain. G-RS3 (device-side draft sampling) waits for the drafter's
  device sampler (M3).
* **M2: arena multi-row streams, no drafter**: 3.1-3.6; G5f, the tables unit
  test, the golden gate with teacher-forced blocks (K>0 must equal K=0 KL).
* **M2.5: prompt-lookup drafter in production** (7.6): point-mass, zero drafter
  cost, exact under rule 1. Validates the multi-row path end to end on real
  traffic before the drafter build; expected ~1.1x, so it is a test, not a win.
* **M3: drafter in the arena**: rings on the iGPU (indices in `KvArena`), batched multi-stream drafter,
  capture on arena rows, device-side sampling exit exporting top-M `q`, seeding in
  `PrefillJob` (both lanes, ring-only), ring in snapshots. Tests: batched drafter
  == per-stream drafter; point-mass drafts == the legacy drafter's for identical
  inputs.
* **M4: shadow in production** (`V41_MS_DSPARK=shadow`, time-boxed or sampled,
  since shadow costs throughput): drafts computed, nothing acted on. **Exact
  renewal replay** for point-mass drafts: along the realized path `y`, a block
  started at t accepts exactly the leading drafts equal to `y`, so the shadow runs
  the drafter only at the block starts accept mode would have used (next start =
  t + accepted + 1) and reports realized E and the K policy's step times with zero
  variance given the path. Per-position estimators overstate realized E when
  acceptance is clustered (review finding 5: 50% easy / 50% hard gives 3.5 per
  position vs 1.71 realized). For sampled `q`: simulate ONE renewal path, drawing
  each acceptance from `min(1, q_i(y_i | y_{i-1}) / p_i(y_i))` with a
  teacher-forced markov prev and drafting only at the simulated starts (unbiased
  for E[blocks]); it must keep each block's pre-markov logits until its positions
  are realized (up to 5 steps, ~2.6 MB per stream) and recompute the markov bias
  per position (review N5). The per-position `X_k` estimator is kept only for
  confidence calibration. Also measures the drafter's wall cost and iGPU contention.
  Go/no-go for M5.
* **M5: accept mode with confidence-gated K**: point-mass drafts (so a global
  search over K is exact, 2.4) and K per stream per block chosen from the
  confidence head against the live step-cost curve (section 6), starting from the
  head's own sigmoid calibration; A/B against off per
  `feedback_e2e_tokps_noise_floor` (suites at production temperature, the
  distribution of E, never one prompt), and against fixed K to confirm the
  head's gain on real traffic.
* **M6: sampled drafts and the full policy**: online confidence calibration at the
  production temperature, the stopping rule for sampled drafts, `tau_d` from
  M0/M4, the draft-or-not stage, stream-aligned lane balancing, the 8-row lane
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
  on the traffic mix (how much time one stream runs alone). M0(c) prices it before
  anything is built; drafter quantization (7.2) shrinks it.
* **R3 iGPU contention** between the drafter and box 1's routed MoE.
* **R4 Kernel-regime switches with larger lanes** (3.7): LSB numerics, plus perf
  cliffs above 16 rows per lane in the head (3.6) and in box 2's pin eligibility
  (`PIN_DECODE_MAX_ROWS`, section 6).
* **R5 VRAM**: the dGPU is near full at `V41_MS_CTX_ROWS=844800`; `rows_cap`
  growth, the stash, and per-lane scratch sized for more rows must be counted
  before M2.
* **R6 Snapshot format** gains the drafter ring (versioned).
* **R7 Wasted paging**: rejected rows still page their experts, take cache-prior
  swaps and count toward box-2 pins, which pollutes later steps; `c(R)` does not
  capture that, so the shadow replay (M4) must report pool churn with and
  without speculation.
* **R8 Host RS cost**: each row is a ~0.5 MB D2H plus a survivor scan; at 30-48
  rows that can reach 10+ ms. Measure `sample_ms` against rows in M2; move the
  survivor extraction on device if it binds.
* **R9 Graph cache**: keyed by (layer, rows, lane) (`forward_prefill.rs` ~1199), it
  grows with the new range of row counts; count its memory with R5.
* **Observability** (needed by M4 anyway): new `HUB_STEP` fields spec_streams,
  spec_rows, accepted, emitted, draft_ms, ring_ms, commit_ms, rs_host_ms; a per-
  block event kind with K, accepted, conf, `p(d)`, `q(d)`.
* **Resolved**: `q_i` depends on `d_{i-1}` only, through the markov row; the
  transformer logits are computed once per block (review).
* **Open**: DSpark on by default for every request, or per-request opt-out?
