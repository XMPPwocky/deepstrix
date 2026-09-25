# Box-2 miss substitution — design sketch
### 2026-09-24 · status: SKETCH, nothing built · quality validation in progress

## The idea

When a box-2 decode pick is not resident, do not block on the NVMe read. Compute
the next-ranked expert that *is* resident in its place, give it the missing
pick's weight, and (optionally) read the missing expert in the background so that
later tokens get it. Decode only (DSpark verify included), never prefill.

It trades bit-determinism for paging: output now depends on what box 2 has
resident. That is the whole cost, and it is why everything below is a knob that
defaults off.

## Why it is worth building (measured)

**Box 2's paging sets the step.** Server log, medians over 1,317 four-row
windows (2026-09-24): wall 266 ms; box-2 round trip 256 ms, of which server time
241 = service 160 (**page 124** + compute 36) + ~81 queued behind the other lane;
link 15. Box 1's own paging is ~0.75 misses per *step* and 4 ms: negligible. The
live fit is `step = 137 + 0.93 x box2.page_ms` (r = 0.99), so zero box-2 paging
is ~1.9x at 4 rows.

**Misses sit at the bottom of the ranking.** Replaying
`~/logs/picks-20260918-1501.trace` (its `D` lines are rank-ordered:
`router_topk` sorts descending) with box 1 = top-103/layer and a 6,160-slot
box-2 LRU: box-2 misses by rank 1..6 = 5 / 7 / 11 / 17 / 25 / **35%**. 94% of
layers that miss at all miss exactly one expert; **34% miss only the 6th pick.**
The balancing bias (`noaux_tc`) is what pushes cold experts into the last slot.

**The model tolerates it.** Golden CPU reference (DeepSeek's `model.py`), one
1,006-token agentic transcript, teacher-forced, swaps at the 668 generated
positions only, prompt KV exact:

| run | swaps/token | KL p50 | KL p90 | KL p99 | top-1 |
|---|---|---|---|---|---|
| cold rank-6 -> 7th, rate-matched | 1.36 | 0.0030 | 0.023 | 0.066 | 96.7% |
| cold rank-6 -> 7th, every one | 15.6 | 0.0035 | 0.030 | 0.102 | 96.9% |
| cold rank-6 dropped, renormalized | 1.38 | 0.0030 | 0.025 | 0.067 | 97.3% |
| rank-**1** -> 7th (positive control) | 1.37 | 0.0050 | 0.040 | 0.105 | 96.1% |

| **v1: cold rank-6 -> 7th, INHERITED weight** | 1.37 | 0.0030 | 0.026 | 0.085 | 97.0% |
| null: bf16-round the 6th's output, same sites | 1.39 | 0.0026 | 0.023 | 0.059 | 97.2% |
| null: random rank-6 swaps | 0.14 | 0.0011 | 0.017 | 0.045 | 97.9% |
| any rank -> next unused, rate ~2.5/token | 2.66 | 0.0038 | 0.033 | 0.137 | 95.5% |
| any rank, ~5/token | 5.05 | 0.0047 | 0.038 | 0.147 | 96.4% |

Controls: zero swaps is bit-identical to the baseline; re-runs are bit-identical;
every swap changes that layer's routed output (median 21% for rank 6, 66% for
rank 1). **KL saturates:** any perturbation at ~1-2 sites/token, even a 0.4%
bf16 rounding, compounds through later tokens to mean ~0.008. So a policy's cost
is its EXCESS over a count-matched null. Mean excess: rank-6 refgate +0.0012,
**rank-6 inherit (v1) +0.0019**, drop +0.0018 (with worse NLL: +0.012 nats/token,
+0.041 on the final turn), rank-1 +0.0064. The inherited weight differs from the
ref.Gate weight by a median 0.007 (p90 0.028). **v1 passes.**

These runs are teacher-forced over the whole transcript, which OVERSTATES
production: there a turn is re-prefilled exactly (invariant 6), so a swap only
affects the rest of its own turn. The any-rank tail maxima (1.2-1.9 nats) were
bimodal sharpenings with top-1 unchanged, or prompt positions after a perturbed
turn, which production re-prefills.

**Excess over each run's count-matched null** (the number that matters):

| policy | regime | sites/token | mean-KL ratio to null |
|---|---|---|---|
| cold rank-6 -> 7th | teacher-forced, whole transcript | 1.4 | 1.15x |
| cold any rank | teacher-forced, whole transcript | 2.6 | 1.88x |
| **cold any rank** | **turn-local (production-faithful)** | 2.6 | **1.20x** (paired 95% CI 0.97-1.50) |

Turn-local = exact context through turn 3, swaps and KL confined to that turn's
217 predictions. That's what production does (invariant 6). There, any-rank:
mean 0.0070 vs null 0.0058, p50 0.0023 vs 0.0020, p99 / max / top-1 no worse
than the null (top-1 98.2% both). Paired bootstrap: excess +0.0012, CI
[-0.0002, +0.0025]. The 1.88x was perturbed KV carried across turns. **Any rank
is viable on this evidence.** Caveats: one transcript, one turn, tokens still
fixed within the turn (free-running unmeasured), hot-set proxy rather than
box 2's live LRU. Hence the staged `sub_min_rank` rollout: 6, then 5, then 1,
with a free-running quality check at each step.

## Revision 2026-09-25: box 1 decides, from a mirror of box 2's pool

The box-2-side policy below can only substitute a **box-2-owned, box-2-resident**
alternative. Box 1 has already dispatched its own experts by the time box 2
looks. About 60% of 7th picks are box-1-owned (hot), so that caps coverage at
~1 - 0.6^m (~78% at m = 3). The validated policy was "best-ranked unused
expert, whoever owns it". So box 1 should make the decision itself:

- **Mirror.** Each box-2 response carries the pool changes that request caused:
  admitted and evicted `layer << 16 | expert` words (box 2 knows them in
  `ensure`; about 2 words per miss). Box 1 keeps the set. It lags by at most the
  requests in flight (two lanes).
- **Decide at route time.** In `pre_moe_route`, right after the pick readback and
  BEFORE the ownership split, box 1 rewrites each box-2 pick the mirror says is
  missing to that row's best-ranked unused alternative. A box-1-owned substitute
  that box 1 holds becomes an ordinary local pick: the pager ensures it and the
  local MoE computes it. A box-2-owned one the mirror holds goes to box 2 as
  usual. **Weights are renormalized exactly (ref.Gate), not inherited.** Box 1
  hasn't dispatched anything yet, so it can rewrite the row's `d_ew` too. The
  router's `alt_w` output puts each alternative on `d_ew`'s scale, so swapping
  pick j for alternative k divides the row's weights, and `alt_w[k]`, by
  `1 - w_j/1.5 + alt_w[k]/1.5`. Any-rank inherit was measured +8% mean / +23%
  p99 over refgate, with per-site weight errors up to 1.2 when a rank-1 weight
  goes to a weak substitute. Box 1 rewrites `d_selected` and `d_ew` with a
  blocking H2D before `pre_moe_prep`'s peer push carries them to the iGPU
  (`forward_prefill.rs` ~7138). Inherit remains only for the box-2-side
  fallback, rank 6.
- **No round trip and no second MoE pass on box 1.** The round-trip alternative
  (box 2 replies "compute X") arrives after box 1's local MoE is launched, and
  would need a deferred pass plus accumulate on box 1's graph-captured chain.
- **When the mirror is wrong.** Thinks resident, but evicted: box 2 misses and
  reads, as today (or applies the box-2-side policy below as a fallback). Thinks
  missing, but admitted: an unneeded substitution, a small quality cost and no
  time cost.
- **Coverage** is then nearly every box-2 miss: the 7th is resident on one box
  or the other almost always.

The box-2-side policy below stays as the fallback for mirror errors.

## Box-2-side policy (fallback)

Per request, per layer, on box 2:

1. `resident_mask(sel)` as today, giving the distinct missing experts.
2. For each missing expert `e`, take every row slot `(r, k)` with `sel[r*nu+k] == e`.
   `e` is **substitutable** iff *every* such slot has `k + 1 >= sub_min_rank`
   and its row has an alternative `a` (in rank order) that is resident on box 2,
   is not already one of that row's picks, and has not been used by an earlier
   substitution in that row.
3. If substitutable, rewrite each such slot `sel[r*nu+k] = a`. `ew` is unchanged
   (**inherit the weight**, see below). `e` leaves the missing set.
   If not, touch nothing: the read happens anyway, so substituting only some of
   its rows would cost quality and save no time.
4. If the missing set is now empty, the request is single-pass with no read.
   Otherwise the existing hits-first / PARK path runs on the smaller set.
5. `sub_admit`: queue every substituted-away `e` for a background read
   (`ExpertShard::prefetch_words_ex`), so a cold expert still enters the pool.
   A missed box-2 expert gets a mean 6.5 hits before eviction, and its first reuse
   is a median 95 tokens later, so the read has plenty of slack.

`sub_min_rank` starts at 6. It drops to 1 (any rank) only if the any-rank
reference runs pass.

## Weights: inherit (v1) vs exact (v2)

`ref.Gate` renormalizes the six unbiased scores over the chosen set. Swapping one
member changes *every* weight by S/S'. That includes weights of box-1 picks, which
box 1 has already applied by the time box 2 decides. Exactness therefore needs
box 2 to return a per-row scale and `ffn_combine` to multiply box 1's local
partial by it. That's a combine-kernel change plus a response-field change.

**v1 inherits instead:** the substitute takes the missing pick's final weight
verbatim, so the sum stays 1.5, box 1's combine is untouched, and box 2 needs no
scores at all, only ranked candidate ids. In the reference, a swap moves the
6th's weight by a median of 0.009 (p90 0.030) out of 1.5. The inherit rule is
being validated on its own (`anyrank25_inherit`). Build v2 only if that run is
clearly worse.

## Wire protocol (VERSION 4 -> 5)

New request flag `REQ_FLAG_ALTS = 64`. After the prefetch block: `u32 m`, then
`b * m` `i32` alternative ids, per row in rank order (ranks 7..6+m), `NO_PICK`
for an unusable slot. The frame length changes when the flag is set, so both boxes
rebuild together (the convention since VERSION 2). `decode_request` gains
`alts: &[i32]`.

Response: widen the fixed fields by one `u64` (keeps the clock triple 8-aligned):
`n_subs:u16 | n_reads_avoided:u16 | n_sub_admits:u16 | reserved:u16`, so box 1's
profile can show `box2.subs_per_step` next to `box2.page_ms`.

## Box 1 (hub) changes

- **Router emits candidates.** `router_topk` insertion-sorts into `n_used + m`
  slots and writes ranks 7..6+m to a new optional `alts` output. Weights are still
  computed over the first `n_used` only. A top-6 prefix of an insertion sort is the
  same whatever the array length (each element's position depends only on the
  comparisons ahead of it), so `sel`/`ew` must stay **bit-identical**: test it.
  `ROUTER_MAX_USED` is 8, so m <= 2 fits as is; m = 3 needs 9.
  Call sites: `forward_prefill.rs` ~5922 and ~6019 (arena decode lanes), `:859`.
- **Readback** with `sel` in the existing sync (`lh.sel_d2h`): b*m*4 bytes.
- **Mask** each alternative to box-2-owned (`owns_eff[a]`), else `NO_PICK`. A 7th
  that box 1 owns (hot set) cannot be computed on box 2 in v1. The reference says
  about 21% of near-ties are that case, so m = 2-3 buys some headroom.
- **Set `REQ_FLAG_ALTS` only on decode/verify submits** (the arena decode path at
  `forward_prefill.rs` ~6672 and single-stream `forward_layer.rs` ~2724), never
  prefill chunks. Env `V41_B2_SUB_ALTS=m` (default 0 = off).
- **Pick trace**: write `DA <layer> <ids x6> | <alts x m>` when the trace is on.
  That is the production data (rank x gap x residency) the offline estimate
  below needs, and this step alone changes no output.

## Box 2 (daemon) changes

- **Decision** in `ExpertExecutor::run` (`remote_experts.rs` ~3336), before the
  `path_decode` / hits-first split, on a local copy `self.sel_sub` (b x nu). Guard:
  alternatives present && `b <= 16` (the existing "verify is still decode" line)
  && `knobs::sub()`. Everything downstream (`resident_mask`, pass A/B, reduce,
  `ensure_layer_*`) then sees `sel_sub` in place of `sel`. Merged lanes
  (`b2_merge`, ~4110) must concatenate `alts` exactly as they concatenate `sel`.
- **Pure function, unit-tested:**
  `plan_substitutions(sel: &mut [i32], alts: &[i32], nu, m, resident: impl Fn(i32) -> bool, min_rank) -> SubPlan { subs: Vec<(slot, from, to)>, avoided: Vec<i32> }`.
  Cases: all rows substitutable; one row not (so none); an alternative duplicates
  a pick; an alternative not resident; two missing picks in one row (take the 7th,
  then the 8th); `NO_PICK` slots; the rank gate.
- **Knobs** (knobs file + SIGUSR2, as `park`): `sub` (default 0),
  `sub_min_rank` (6), `sub_admit` (1). Env mirrors `V41_B2_SUB*`.
- **Counters** into `ExecTiming` + the daemon stats line + the new response word.
  Emit a trace event per substituted slot (layer, row, rank, from, to).

## Invariants and tests

1. `V41_B2_SUB_ALTS=0` or `sub=0`: **bit-identical** to today (the existing
   `deepstrix-expert-bench --check-*` and determinism recipe).
2. Router with m > 0: `sel`/`ew` bit-identical to m = 0.
3. Substitution on but everything resident: bit-identical (the plan is empty).
4. Test daemon (`:7432`, small pool, `--catchall`), `sub=1`: substituted requests
   do no read (`n_reads_avoided` > 0, pread count drops), and the result equals the
   local reference computed with the *rewritten* `sel`. The bench needs the plan
   echoed back for that (debug flag).
5. Every determinism gate runs with `sub=0`, and says so.
7. **Never under the fidelity pin.** The golden gate's routing pin
   (`het::fidelity_tap::pin_device_rows`, branch `worktree-architecture-review`)
   rewrites rows to the CPU reference's picks right after the router launch in
   `pre_moe_chain`. Box-1 substitution runs later (`pre_moe_route`, after the
   readback), so it is ordered after the pin. It must also be skipped entirely
   when `fidelity_tap::pin_on()`. Otherwise the pinned gate hard-fails reading
   a substitution as "pin did not hold".
8. **Renormalize by dividing.** With `c = 1 - w_j/1.5 + alt_w/1.5` (= S'/S),
   every kept weight becomes `w_i / c` and the substitute gets `alt_w / c`.
   Clamp `c` so S' stays at or above `ROUTER_WEIGHT_EPS` (the kernel's own floor
   on S). It never binds in practice.
6. **Substituted KV must not outlive its turn.** Today it cannot. The multistream
   path snapshots only at the prompt end (`multistream.rs:657-669`, at
   `<｜Assistant｜>`), `finish()` just releases the slot (turn-end snapshots are
   the unbuilt "M2", `:1221-1229`), and the next request re-prefills the previous
   assistant turn exactly. Prefill never substitutes. If turn-end or RESIDENT
   snapshots are built, a stream that substituted must not snapshot past its
   first substitution. Otherwise the downstream-prompt tail the reference measured
   (KL 0.8-1.3 at tool-result openings after a perturbed turn) becomes part of
   the price.

## Rollout (revised 2026-09-25)

1. **Router alternatives + trace.** DONE on branch `worktree-b2-miss-substitution`:
   `V41_ROUTER_ALTS=m` (0..=4, default 0). `router_topk{,_par}` emit ranks
   7..6+m into `bd.d_alts`, with their weights on `d_ew`'s scale in `bd.d_alt_w`.
   Both are read back with the picks and written as `A <layer> <b> <alts> /
   <owner chars> / <alt_w>` after each `P` trace row. The test
   `tests/router_topk_alts.rs` shows picks and weights bit-identical for
   m = 1..4, the alternatives are the next ranks (up to float near-ties),
   `alt_w` matches a host f64 computation, and the renormalized weights sum to
   1.5.
   Cost when on: m extra argmax passes per token-layer.
2. **Mirror, dry run.** Pool changes in box-2 responses (VERSION 5) plus the
   box-1 mirror. Log would-substitute counts, and mirror accuracy against box 2's
   actual misses. No output change.
3. **Box-1 route-time substitution** behind a knob, `sub_min_rank` staged
   6 -> 5 -> 1. ABBA on a warm pool (`box2.page_ms`, step wall, subs/step), with
   free-running quality at each step: echo2 at 1.5K/6K under >= 4 rows, needle,
   and the golden fidelity gate once it exists.
4. **Box-2-side fallback** (policy above, `REQ_FLAG_ALTS`), only if mirror errors
   leave material paging. `sub_admit` background reads can come with step 3 or 4.

## Expected gain

`sub_min_rank = 6`: ~1 swap per token, removing ~40% of box-2 misses: roughly
**+15-25% at 4 rows** (less once PARK removes the queueing half). Any rank, if
validated: every box-2 miss whose row has a resident box-2 alternative. The
ceiling is ~1.9x at zero box-2 paging, cut by the all-rows rule, by box-1-owned
7ths and by non-resident alternatives. Rollout step 1 measures that fraction.

## Open questions

- m = 2 or 3? It depends on how often the 7th is box-1-owned or also cold.
  The `DA` trace answers it.
- A miss nobody can substitute still blocks. The low-bit on-disk fallback (read a
  ~half-size IQ2 copy now, the MXFP4 copy in the background) is the v2 for that
  remainder. It differs from `MIXED_PRECISION_BY_RESIDENCY.md`, which kept IQ2
  resident and verified speculatively.
- `sub_admit` background reads vs foreground misses on the same drives: they need
  foreground-first scheduling (chunked background reads) if step 3 shows
  contention.
- Rates rise with streams (misses/token 1.24 -> 1.62 from S = 1 to 32, 09-18 model),
  so the anyrank ~5/token run is the headroom check.
