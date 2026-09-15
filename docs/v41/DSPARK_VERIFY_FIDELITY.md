# The verify's fidelity, measured (2026-09-15)

`V41_DSPARK_XCHECK` compares row 0 of the batched verify against the decode
forward that immediately follows it: same token, same position, same prefix. Its
logits must match. This is the DIRECT correctness test — acceptance is a proxy,
and a bad one, because changing the split changes the generated text and
acceptance is content-dependent.

## Two confounds had to be removed before any number meant anything

1. **`rollback_kv` never restored the compressor store.** It carried only
   per-layer `(n_raw, raw_off)`, so a speculative batch's ~b/ratio compressor
   boundaries stayed in the compressed KV permanently, cumulatively, for decode
   to attend to. The accept path's partial rollback had the same hole, so
   REJECTED drafts polluted the compressed KV — a DSpark correctness bug, not
   just an instrument artifact. Fixed: `KvMark` now snapshots `n_comp`,
   `n_index_comp`, `state_kv`, `state_score` for the 4 owning layers (~32 KB).

2. **The probe perturbed decode through PAGER RESIDENCY.** A content fingerprint
   of every component a rollback must restore reported "restored all components"
   on every step — and the probe STILL changed what decode generated at
   temperature 0 (coherent English vs a hallucinated Chinese request). Under
   `V41_T2_CATCHALL=1` the box1/box2 split is decided by `pg.is_resident()`, so
   it is a function of request history; the probe pages experts and repartitions
   decode's own MoE, and f32 addition is not associative. `V41_T2_CATCHALL=2`
   makes the split a constant. `forward_layer.rs` documents this for decode; it
   had never been applied to a fidelity measurement.

## Result

Verify row 0 vs decode, mean logit cosine:

    B    original   +comp rollback   +deterministic split
    1      0.709        0.979              0.9988
    2      0.758        0.986              0.9989
    3      0.631        0.687              0.821
    6      0.603        0.710              0.867

**At B<=2 the batched verify reproduces decode essentially exactly.** The B>=3
gap is real and unexplained; it is the remaining fidelity defect.

## Refuted, each by measurement rather than by reading

| Hypothesis | Instrument | Result |
|---|---|---|
| Prefill MoE kernels differ from decode's | `V41_VERIFY_DECODE_MOE=1` | no change (0.673 vs 0.710 at B=6) |
| Prefill attention kernels differ | `V41_VERIFY_DECODE_ATTN=1` | small (0.771 vs 0.710) |
| `n_comp_after` wrong on the own-compressor branch | `V41_COMP_POSITIONAL=2` | ZERO disagreements with the positional formula |
| Two-lane split | `V41_SINGLE_LANE_AB=6` | 0.694 vs 0.679 — both arms broken, because the cliff is where a LANE first holds >1 row (lane_split = b/2 rounded up), not where a second lane appears |
| Rows 1.. leak into row 0 | poison rows 1.. (tokens AND carries, so the mHC pre-mix is covered) | row 0 does not move |
| Expert paging | same poison (poisoned rows reroute, changing the paged union) | row 0 does not move |
| Small-B catch-all, deterministic | `V41_SMALL_B_CATCHALL_DET=1` | fidelity 0.688 -> 0.519 for 1277 -> 1147 ms. Fifth variant of "move the verify's expert paging" to fail, and the first scored on fidelity instead of wall time. |

## Where the throughput actually stands

Warm, host tuning restored (it is NON-PERSISTENT and had been lost — restoring
it took baseline 4.26 -> 5.52 tok/s):

    baseline decode          181 ms/tok   = 5.5 tok/s
    DSpark accept, mode 1    890 ms/tok   = 1.1 tok/s   E 1.553
    DSpark accept, mode 2    649 ms/tok   = 1.5 tok/s   E 2.070
    verify(B=6) alone        1150-1280 ms

The deterministic split is worth real acceptance (E 1.553 -> 2.070), which
REFUTES the earlier "mode 2 tanks acceptance (E 1.07-1.39)" — that was measured
under the compressor bug.

**DSpark is currently a 4x net LOSS.** Break-even needs
`verify(B=6) < E x 181 ms` ~ 375 ms at E=2.07, against 1200 ms today: a 3.2x
verify cost reduction just to reach parity, and ~12x for 20 tok/s. E also swings
1.24-2.07 run to run on identical config, so acceptance is not yet a stable
quantity to optimise against.

## The verify CAN be made 2.9x faster — the blocker is correctness, not cost

Measured with `V41_T2_CATCHALL=2` (deterministic split, so the numbers mean
something) and the compressor rollback fixed:

| verify arm | prefill_misses | read_ms | verify total | cos |
|---|---|---|---|---|
| catch-all off | 298 | 724 | **1250 ms** | 0.814 |
| `_DET=1` (all picks to box 2) | **0** | **0** | **429 ms** | 0.521 |
| residency-based | 0 | 0 | 1370 ms | 0.512 |
| residency, decoder half only | 9 | 130 | 1423 ms | 0.569 |

So the offload DOES eliminate box 1's expert misses and takes the verify from
1250 ms to 429 ms — a 2.9x cut, and the miss elimination is real
(`prefill_misses` 298 -> 0). **Every variant loses fidelity the same way**
(cos ~0.52 against 0.81 with it off), so the remaining blocker is correctness,
not cost.

### What it is NOT

- **Not the static HELLO mask dropping picks.** `V41_MASK_DBG=1` reports zero
  masked-live picks: `remote_exclude()` is on by default, so `sel_for_remote` is
  built and the submit goes out UNMASKED.
- **Not a double-count from a stale remap.** The decode path needs
  `mark_remote_after_ensure` because a filtered-out id keeps `-(e)-1` ("ours at
  slot e"). The prefill path does not: `set_remote_exclusion` rewrites all
  `N_EXPERT` entries authoritatively from `slot_of` and marks anything not
  resident in the window as 0. Adding the decode guard here changed nothing
  (tested), and was reverted as redundant.

### The live hypothesis

At 429 ms with box 1 paging nothing, box 2 cannot have paged ~18 experts/layer
x 40 layers from its own disk (that would be seconds at 7.14 ms/miss). So box 2
is most likely returning partials only for the experts it already holds and
silently contributing nothing for the rest — the "computed by NOBODY" failure,
one level below the hub's own `verify_routing_exactly_once`, which validates the
hub's `owns_eff` and not box 2's behaviour. Next step is to instrument box 2's
side: count, per request, picks received vs experts actually computed.

## The verify cost curve — why speculation cannot pay here

One run, `V41_VERIFY_PROBE=1,2,3,4,6`, deterministic split, compressor rollback
fixed, ZERO box-1 expert misses. Cost and fidelity per batch width:

| B | verify ms | cos |
|---|---|---|
| 1 | 489.9 | 0.9983 |
| 2 | 797.0 | 0.9987 |
| 3 | 991.7 | 0.804 |
| 4 | 1083.6 | 0.797 |
| 6 | 1275.5 | 0.793 |

Least squares over B=1..6:

    verify(B) ~ 333 + 157*B ms        baseline decode token = 192 ms

Two independent reasons speculation loses, both from this fit:

1. **The driver costs 2.5x decode for the SAME work.** At B=1 — one token, one
   position, no misses — the batched prefill path takes 490 ms where decode's
   own forward takes 192 ms. That 333 ms intercept is pure path overhead, and it
   alone exceeds a whole decode token. No drafter quality can repay it.

2. **The marginal row costs more than it can return.** Adding one draft row
   costs 157 ms. The first draft is accepted with p=0.744, so it is worth at
   most 0.744 * 192 = 143 ms. The FIRST speculative row already loses, before
   any acceptance decay. Every deeper row is worse.

Cross-check against the end-to-end measurement, which is what makes this a model
rather than a curve fit: at B=6 with the measured E=2.07, the fit predicts
1275 / 2.07 = 616 ms per accepted token; DSpark accept measured 649 ms/tok. The
cost curve explains the observed 4x loss.

### What any future attempt has to hit

For 20 tok/s (50 ms/token) at a generous E=3, the verify at B=6 must cost
150 ms total. Against today's `333 + 157*B`:

    intercept   333 ms -> ~30 ms   (11x)
    per row     157 ms -> ~20 ms   (8x)

The per-row target is roughly box 2's own model (`srv = 105 + 20*B + 87*D`
us/layer, ~10 ms/token-row summed over 40 layers for 3 new distinct experts), so
it is not obviously unreachable — but it cannot be reached from the prefill
driver, whose B=1 cost is already 2.5x decode's. This is the quantitative form of
the earlier conclusion that the batched verify belongs INSIDE the engine as a
batched DECODE entry point.

### The other wall, independent of speculation

Decode itself is 192 ms/token and 63% of that is box 2's expert service
(121 ms), which is dominated by box 2's own 6.14% miss rate at 7.14 ms/miss
(~53 ms/token). Box 2 is at RAM capacity (118 of 124 GB), box 1 cannot take the
work (cost is max(box1, box2), measured three times), and the link is 3% of the
token. So even a FREE verify leaves decode at 192 ms/token = 5.2 tok/s. Reaching
20 tok/s needs the expert FORMAT lever (fitting all 15,360 experts in RAM), not
a better verify.

## Box 2 slot rebalance: REFUTED (2026-09-15)

Decode is 63% box-2 expert service, and box 2's service is almost entirely its
own page misses. Measured over a 400-token run on the shipped placement
(260 encoder / 68 decoder slots per layer, 6560 total):

    requests +97,239   misses +7,469   =  18.7 misses/token x 7.14 ms ~ 133 ms/token

and it does NOT warm out (first 80 tokens 228 ms, last 80 210 ms). The obvious
move is to shift slots from the encoder half — which box 1 already covers with
21 pinned dense windows — to the decoder half. `box2_placement_164_164.txt` is
exactly that: same 6560 total slots, 164/164 instead of 260/68, so identical RAM
and no change to box 1's stride constraint (residency comes from the placement
file, ownership from `--experts`).

WORSE, on the same prompt with box 2's LRU warmed to convergence:

    260/68    177.95 ms/tok     aggregate hit 0.9386
    164/164   263.37 ms/tok     aggregate hit 0.9449   (+48% slower)

The aggregate hit rate IMPROVED and decode still got much worse, because the
aggregate is dominated by prefill traffic: moving slots off the encoder half
hurts the decode-time working set more than the extra decoder slots help. Keep
260/68. (Care needed reading this: the first attempt looked favourable only
because box 2's LRU was cold after the restart — 48k requests against the
8.5M the incumbent had accumulated. Always warm box 2 to convergence, ~300k
requests, before comparing placements.)

Box 2's miss rate is therefore a genuine capacity wall at 124 GB, not a
placement-policy problem, which points back at the expert FORMAT lever.

## Leg balance: `V41_LOCAL_PICKS=5` REFUTED (2026-09-15)

Box 2 is saturated and box 1 looks idle, so forcing box 1 to claim picks should
help — the knob's own doc models the optimum at n~5 for -17 ms/token. Measured,
with a repeated baseline to bracket drift:

    base    253.90 ms/tok   pager_ensure 2.7 ms    pager_misses 0
    n=5     772.36 ms/tok   pager_ensure 683 ms    pager_misses 173
    base    248.44 ms/tok   pager_ensure 3.3 ms    pager_misses 0

3x WORSE. This is the confound the knob's own doc warns about: it claims picks
REGARDLESS of residency, and box 1's decode LRU is 25 slots, so ~200 forced
picks/token become ~173 misses on box 1's dm-crypt NVMe, on the critical path via
`ensure`. Box 1 cannot take decode work it does not already hold, and the pool
sweep already showed that giving it more slots is itself a loss.

Note `igpu_busy_us=0` in decode summaries is an UNHARVESTED counter under the
graph path, not proof box 1 is idle. Do not read it as headroom.

## Placement comparison, and a warmth confound worth knowing

Same prompt, box 2 warmed before each:

    260/68 (running)   177.95 ms/tok   hit 0.9386 (8.5M requests, fully warm)
    164/164            263.37 ms/tok   hit 0.9449 (~340k requests)
    268/40 (script default) 237.12 ms/tok   hit 0.9001 (~200k requests)

260/68 wins, and is what is restored. But box 2's hit rate is a MOVING TARGET
that converges over hundreds of thousands of requests, and every arm above sits
somewhere different on that curve, so the gaps are confounded in the direction
that favours the incumbent. A clean placement comparison needs each arm warmed to
convergence — hours, not minutes. Treat the ordering as provisional.

The recorded 57.46 ms/token (17.4 tok/s) for this config was NOT reproduced by
any placement tried today; today's best is 177.95. That gap is unexplained and is
the most valuable open thread: it is 3x, and it is in NON-speculative decode,
where the goal's arithmetic actually lives.

## Methodology: box 2's LRU warming dominates every decode A/B on this box

The single most important practical finding from the decode work. Box 2's page
hit rate converges over HUNDREDS OF THOUSANDS of requests, and every arm of a
restart-based A/B sits at a different point on that curve. Measured today, same
prompt, same binary, arms run in sequence with a repeated baseline:

    base0    229.15 ms/tok    box2 misses/100 tok = 2063
    <arm>    221.49 ms/tok                          1944
    base1    207.54 ms/tok                          1758

The two BASELINES differ by 10.4% and the miss counts fall monotonically with
wall-clock, not with the arm. Any decode delta below ~10% measured this way is
warming, not the change. This retroactively weakens several comparisons made
earlier today (the placement ordering especially) and explains contradictory
results across the session.

**Rule: for a decode A/B on this cluster, either warm box 2 to convergence in
EVERY arm (~300k+ requests, tens of minutes each) or interleave the arms inside
one process. Restart-per-arm with a short warm-up cannot see a 10% effect.**

## Adaptive victim admission: UNTESTED, code reverted

The GEOMETRY memo's best modelled option is box 1 as a ~960-slot VICTIM cache
(12.0 miss/tok vs today's 20.4, ~54 ms/token). Victim policy is already the
default, but the fill budget is `lru_free_slots()`, a ONE-TIME warm-up budget —
so membership FREEZES at first fill and the LRU becomes a stale snapshot, which
is why a 1396-slot pool measured slower than a 25-slot one. The code says as
much: "NOT adaptive ... windowed-LFU promotion is the follow-up."

Implemented admission-with-eviction capped per token (~10 admissions = ~42 ms,
just under the decode plan's 45 ms cap) and measured: no resolvable effect, AND
`decode_misses=7` for a whole run — the adaptive path fired SEVEN times, not
10/token. The `BOX2_MISSED` marks are too sparse at decision time to drive it.
So the idea is untested rather than refuted, and the code was reverted rather
than shipped inert and unvalidated. Anyone retrying it must FIRST instrument how
often `box2_missed` is actually true at the split decision.
