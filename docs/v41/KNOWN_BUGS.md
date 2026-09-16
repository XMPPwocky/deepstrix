# V4.1 known bugs and open correctness questions

Living list. Add here the moment something is found, even when it is not being
fixed right now, and delete only when it is fixed AND has a regression test.
Ranked by risk of SILENT WRONGNESS (produces wrong numbers rather than an error).

Status key: **OPEN** / *MITIGATED* / ~~FIXED~~

---

## Silent wrongness

### 0b. OPEN — verify vs decode ~1.9 nats, **SEEDED AT LAYER 0 by a different `hc_mixes` kernel**

**LOCALISED 2026-09-16.** Per-layer residual diff (both paths now dump; see
`92257cc`, `3e4a17c`), same position, B=1:

    entering layer 0   relRMSE 0.000e+00   BIT-IDENTICAL
    entering layer 1   relRMSE 6.504e-03   <- FIRST DIVERGENCE
    entering layer 2   8.753e-02
    entering layer 5   2.773e-01
    entering layer 39  5.602e-01           -> the ~1.9 nats at the head

Both paths enter layer 0 identically and diverge leaving it. Layer 0 is the
SEED; everything after is amplification.

**Character of the seed:** position-INDEPENDENT (~6e-3 at positions 42,43,44,46),
so not attention/KV, which would scale with context. Spread over 20380/20480
channels and all four HC copies (copy 2 is 10x cleaner than the rest), so not a
few wrong experts or a slot bug -- it is the same math from a DIFFERENT KERNEL.

**Mechanism:** the two paths compute `hc_mixes` with different kernels.

    decode   forward_layer.rs:609  matvec_narrow_ksplit_pre_scaled
                                   (K split into 20 chunks + reduce, RMS folded IN)
             forward_layer.rs:615  matvec_pre_scaled  (non-split, RMS folded IN)
    prefill  forward_prefill.rs    matvec_narrow_batched
                                   (single-pass warp reduction, RMS applied SEPARATELY)

Different reduction order AND a different point of applying the RMS scalar. Then
`hc_split_sinkhorn` runs **20 doubly-stochastic iterations** on those coefficients
(`ARCH_SPEC:45`) before they scale the residual -- an iterative amplifier that
turns f32-level mix differences into 6e-3 on the residual.

**Fix:** give prefill a BATCHED form of decode's exact kernel
(`matvec_narrow_ksplit_pre_scaled` / `matvec_pre_scaled`, i.e. pre-scaled and
K-split the same way), so the verify reproduces decode bit-for-bit at b<=64.
Swapping to `matvec_narrow_batched` (`21cb203`) fixed the f16 SPEC violation and
was worth +83% and E 2.185->3.077, but it is still not decode's kernel.

Earlier framing (superseded by the localisation above):
QUANTIFIED 2026-09-16 with the in-tree cross-check
(`V41_VERIFY_PROBE=6,6 V41_VERIFY_BATCHED=1` -> `dspark.xcheck`), 648-token
prompt, deterministic split, box 2 fixed:

    verify vs decode, f32 matvecs : agree 62/111 (55.9%)  cos 0.778  KLD 2.21 nats
    verify vs decode, all-WMMA    : agree 66/111 (59.5%)  cos 0.781  KLD 2.06 nats

    for scale: decode vs CPU ORACLE = 7.52e-04 nats

So the verify is ~3000x further from decode than decode is from the fp32
reference, and disagrees outright on ~42% of positions. **This is structural,
not numeric** -- the f16->f32 matvec fixes moved accept rate a lot (E 2.185 ->
3.077) but barely moved this (2.06 -> 2.21), so precision was a second-order
term on top of a path computing something different.

DSpark cannot emit correct tokens at any speed until this closes. Decode is the
right reference to iterate against (it is near-exact vs the oracle), so no CPU
oracle is needed in the loop -- just drive `dspark.xcheck` KLD toward ~0.

Candidates not yet separated: the speculative KV append's window addressing
(`SpeculativeAppend`, the raw_off slide), the compressor rollback, CED mode
(the verify runs `last_only=false` => `CedMode::Exact` while decode does not),
and the batched prefill attention at B=6 generally.

Earlier framing (still true, now explained):
Now cleanly reproducible, and no longer maskable as noise (#0 is fixed, runs are
bit-identical). Same box 2, same 648-token prompt, same config, temp 0:

    decode only : "# Maintaining a Lighthouse Through a Winter Storm: A Keeper's
                   Night..."                                        COHERENT
    DSpark      : "# Maintaining Lamp, Lens and the signal apparatus, and the the
                   the the"                                         DEGENERATE

The verify decides which tokens are emitted, so the verify is producing them.
Short prompts stay coherent; long ones degenerate, so suspect state that only a
multi-chunk / large-B prefill establishes (KV window addressing across the
speculative append, the compressor rollback, or the ring seeding's interaction
with either). E is ~2.2 either way, so this is not simply low acceptance.


### 0. FIXED (2026-09-16, `7f89090`) — was box 2's batched group-id bound
**ROOT CAUSE + FIX.** Box 2's remap encodes ABSOLUTE pool slots (since
2026-09-14) but the batched MoE passed `N_EXPERT` (384) as the group-id bound;
`moe_group_builder.hip:118` dropped every `g >= 384` while the reducer still
counted those picks as ours and summed their ZEROED partial rows. Every routed
expert above slot 383 contributed exactly 0.0 -- essentially all of layers 2..39
with the shipped 6160-slot assignment. B=1 decode was spared (no group builder);
every prefill chunk and every DSpark verify batch was hit.

Discrete because each pick is binary; bimodal because the pool settles into a few
slot layouts; survived box-1 restarts because the ShardPool LRU is never reset.

VERIFIED after the fix, two DSpark runs in separate processes, 648-token prompt:

    relRMSE 0.000e+00   KLD 0.000e+00   BIT-IDENTICAL   (was relRMSE 0.42)

The determinism gate is trustworthy again. **Every two-box measurement taken
before `7f89090` ran against a model silently missing most of its experts** --
including everything in this session and any earlier A/B that used the batched
path. Treat those numbers as void.

Original report follows.

This was the highest-priority bug in this file and it invalidated a methodology
the project depends on. `scripts/v41_determinism_gate.sh` scores expert-cache
changes by requiring `T2_CATCHALL=2` + temp 0 + same prompt to reproduce. It does
not. Three runs of one fixed config (catchall=2, WINDOWS=21, 648-token prompt,
temp 0), first-token logits:

    a vs b : relRMSE 0.170   KLD 0.00196
    a vs c : relRMSE 0.408   KLD 0.567
    b vs c : relRMSE 0.428   KLD 0.545

argmax was stable (5) in all three, and a is close to b while c is far -- BIMODAL,
not drift, which points at a discrete state difference rather than accumulating
float error.

**LOCALISED TO BOX 2 (2026-09-16, measured).** Box 1 run SOLO with no remote at
all (`V41_REMOTE_SPLIT=0`, `V41_PAGER_STRIDE=384` so the full 384-expert union
fits a window) is **BIT-IDENTICAL across two separate server processes**:

    box 1 solo        relRMSE 0.000e+00   KLD 0.000e+00   BIT-IDENTICAL
    box 1 + box 2     relRMSE 0.42        KLD 1.3-1.5     argmax 35 or 5

So box 1's whole pipeline -- prefill, pager, two-lane driver, KV, mHC, MoE -- is
deterministic. **100% of the nondeterminism is box 2.** That also explains the
6.02-vs-14.21 tok/s swing on an identical config, the apparent geometry
sensitivity (#1, retracted), and the long-prompt DSpark degeneracy.

Ruled out inside box 1: float atomics (none in the V4.1 path); HashMap iteration
order (every map is lookup-only, LRU order comes from a VecDeque); the two-lane
pager/iGPU race (`V41_PAGER_SYNC_IGPU` guard changed nothing: 0.416 vs 0.406).

**Next:** box 2's `ShardPool` assigns experts to slots by LRU history, which
persists across box-1 restarts. If its partial-sum reduction iterates or groups
by SLOT, f32 summation order changes with residency history -- changing results
without changing which experts are computed, and bimodally if the pool settles
into one of a few states. Test cheaply by restarting `deepstrix-expertd` between
two box-1 runs.

**Prior hypothesis (superseded by the solo measurement above):** box 2. Its LRU persists across box-1
restarts, and with `V41_B2_POOL_FLOOR=0` its global victim search assigns experts
to different SLOTS depending on history. If its MoE groups/reduces by slot order,
the f32 summation order changes with residency -- residency changing VALUES, on
box 2, even under the deterministic split. Test: restart `deepstrix-expertd`
fresh before each run and see whether the spread collapses.

**Consequence for everything else in this file and in memory:** any logit- or
sha-based comparison taken without a same-config control is uninterpretable if
its effect is below ~0.43 relRMSE. That includes the geometry results in #1 and
the mHC kernel comparison. Re-measure with a noise floor, or interleave arms
inside ONE process.


### 1. RETRACTED — the geometry 'corruption' was NOISE. See #0.
> **RETRACTED 2026-09-16 (same day).** The 14% relRMSE below is INSIDE the
> run-to-run noise floor, which I had never measured. Three runs of an IDENTICAL
> config differ by up to relRMSE 0.43 / KLD 0.57 (see #0). The geometry deltas
> (0.136, 0.145) are smaller than that. There is no evidence `V41_PAGER_WINDOWS`
> corrupts anything. I called this bug resolved, then confirmed, then retracted
> it in one session -- each time by changing one variable and attributing the
> difference without establishing what "no change" looks like. **Measure the
> noise floor before attributing any delta.**

**SUPERSEDED — the measurement below is real but is noise-dominated (2026-09-16).** sha comparison cannot
distinguish 1e-7 from corruption -- greedy decoding amplifies any difference into
a different token and then totally different text. First-token logits
(`V41_DUMP_FIRST_LOGITS`) have no trajectory amplification. Deterministic split
(`T2_CATCHALL=2`), 648-token prompt, ONLY the window geometry varying:

    WINDOWS=21 (reference)   argmax 35    logit rms 4.08
    WINDOWS=8    max|d|=3.27  RMSE=0.589  relRMSE=14.5%  KLD=0.057 nats  argmax 35
    WINDOWS=4    max|d|=2.82  RMSE=0.556  relRMSE=13.6%  KLD=0.140 nats  argmax  5  DIFF

**relRMSE ~14%.** f32 re-association is ~1e-6. This is six orders of magnitude
too large to be numerics: the pager geometry CORRUPTS THE MODEL, and WINDOWS=4
changes the argmax of the FIRST generated token.

Consequences:
- Every throughput number taken at a non-default WINDOWS is measured on a
  different (wrong) model. The 14.21 tok/s decode result at WINDOWS=4 is void.
- WHICH geometry is correct is still unknown -- 21 is only the reference here,
  not a verified truth. Needs `scripts/v41_oracle`.
- This is the highest-priority bug in this file. It is upstream of all perf work.

**Ruled out as the cause:** residency-driven split / f32 re-association (this test
holds the split deterministic); prefill lane racing (single-lane prefill diverges
too -- 1-lane W21 sha a03322848dd6 vs 1-lane W4 d2ee0b0f07e3).

**PARTIAL RESOLUTION, THEN REOPENED (2026-09-16).** The "resolved" claim below
held only for a SHORT prompt and is WRONG in general. With a ~648-token prompt,
still under the deterministic split, geometry changes the output again:

    LONG prompt, catchall=2, two-lane prefill
      WINDOWS=21 -> sha 328d5151f9de
      WINDOWS=4  -> sha 2a27d105ff60      DIFFERENT

So f32 re-association from a residency-driven split does NOT explain it. A short
prompt is a single chunk; a long one is multi-chunk and TWO-LANE, which is the
discriminator.

**Leading hypothesis (from the architecture review, not yet confirmed):** the
prefill steady state is `post_A(L), pre_A(L+1), post_B(L), pre_B(L+1)`
(`forward_prefill.rs:731-760`), so `ensure_layer_union(L+1)` for lane A runs on
the host BEFORE lane B's layer-L MoE has been waited on. When `window_of(L)` ==
`window_of(L+1)` the union path clears `slot_key` for the whole window and
reassigns from `next_free = 0` (`expert_pager.rs:988-996`), overwriting bytes
lane B's queued kernels will read AND the single shared `remap_dev`. Both
survive only on an accidental null-stream drain. The failure is MONOTONE IN
WINDOW COUNT exactly as observed (more windows -> fewer L/L+1 collisions;
WINDOWS=1 -> all 40 layers in one window -> garbage).

**Decisive test:** single-lane prefill (`V41_PREFILL_SINGLE_LANE_MAX` large) at
two geometries. If the shas then agree, the lane race is the cause.

The short-prompt observation below is still true and still useful, but it is NOT
a resolution.

Under the DETERMINISTIC split (`V41_T2_CATCHALL=2`) SHORT-prompt output
is geometry-INDEPENDENT. Same prompt, temp 0, 648-token prompt:

    catchall=2, WINDOWS=21 -> sha 0a9a37457b53   167 ms/tok   6.00 tok/s
    catchall=2, WINDOWS=4  -> sha 0a9a37457b53    70 ms/tok  14.21 tok/s   IDENTICAL

So `V41_PAGER_WINDOWS` does NOT mis-address weights. The geometry-dependence
seen under `catchall=1` is the documented mode-1 behaviour: the split is decided
by `pg.is_resident`, so geometry changes WHICH BOX computes which experts,
changing the f32 partial-sum grouping (`forward_layer.rs:2296-2320`). That is
exactly what mode 2 exists to remove.

Consequences:
- **E is not a correctness probe** under mode 1. E moved with geometry because
  the SPLIT moved, not because the main model computes differently. An earlier
  note in this file claimed otherwise; it was wrong.
- **WINDOWS is safe to tune under mode 2**, and there it is worth 2.4x on
  decode at byte-identical output. Tune it there, never under mode 1.
- Any A/B that varies residency MUST run under `V41_T2_CATCHALL=2`, per
  `scripts/v41_determinism_gate.sh`.

Original report follows.

Residency must never change numerics; a miss is a page-in, not a different
answer. LRU size is correctly neutral (25 -> 1510 slots: byte-identical output).
But the DENSE packed-window geometry is not. Same prompt, temp 0, 120 tokens:

    WINDOWS=21  sha (A)  E 2.380      <- production default
    WINDOWS=8   sha (B)  E 2.000
    WINDOWS=4   sha (C)  E 1.595
    WINDOWS=1   GARBAGE  E 1.000      <- see #3

E falls monotonically with `dense_windows`. **E is a CORRECTNESS PROBE here,
not a tuning knob**: the MTP drafter has its own resident weights and never
touches the expert pager (`mtp.rs` has zero pager references), so its proposals
are invariant under geometry. E is the agreement between that fixed drafter and
the VERIFY, and the verify IS the main model -- so E moving means the main
model's logits moved, i.e. the pager fed it wrong weights. Use E as a cheap
correctness signal; the CPU oracle is only needed to say WHICH geometry is right.

**Affects PLAIN DECODE too**, so this is not DSpark-specific and no path is
safe: same prompt, temp 0, no DSpark --

    WINDOWS=21  sha 93317dccd2e6   124 ms/tok   8.07 tok/s
    WINDOWS=4   sha 5c9e906bedcc    92 ms/tok  10.91 tok/s

so the faster geometry is computing a different model. Any throughput number
taken at a non-default WINDOWS is untrustworthy until this is fixed.

Ruled out so far: #2 (`remote_split_on=true` makes `set_remote_exclusion`
rebuild the remap afterwards); union OVERFLOW (errors loudly at
`expert_pager.rs:1068`, never truncates); window RECLAIM (clears `slot_key` and
`slot_of` together, `:990`); LRU/dense REGION overlap (`ensure` skips to
`lru_lo = dense_slots()` and only evicts above it, `:1382`). Root cause NOT yet
found. #2 was the leading suspect and
is ruled out for the default config (`remote_split_on=true` makes
`set_remote_exclusion` rebuild the remap afterwards). Needs `scripts/v41_oracle`
to say which geometry is even correct.

### 2. OPEN — `ensure_layer_dense` never writes `self.remap`
`expert_pager.rs:532`. `ensure_layer_union` publishes via `write_window_remap`
("the remap IS the slot table now") but the dense twin does not, yet
`pg.remap_dev` is handed to the dispatch unconditionally
(`forward_prefill.rs`). Reachable with a STALE (previous layer's) remap when no
exclusion builder runs afterwards: the >=90% union->dense arm with
`remote_split_on == false`, and `V41_PAGER_UNION=0`. On the first call
`remap_dev` is raw `hipMalloc` memory. Neither path runs
`verify_routing_exactly_once`. Not active in the default config.

### 3. OPEN — `V41_PAGER_WINDOWS=1` is degenerate and silently accepted
`dense_windows=1` => `pinned_windows=0` => `window_base(w)=0` for EVERY layer,
so all 40 layers share one 384-slot window and overwrite each other. Emits CJK
garbage, E=1.000. Should be rejected at startup. Do NOT use as an "unpacked
reference" -- it is not one.

### 4. OPEN — `remap_dev` carries three incompatible slot spaces, untyped
`ensure` -> ABSOLUTE pool slot (whole-pool view); `ensure_layer_union`/`_dense`
-> WINDOW-RELATIVE (window view); box 2's `LayerShard` -> ABSOLUTE shard slot.
The allocator, the exclusion builder and the WEIGHTS VIEW must be chosen as one
decision; two of three right is still silently wrong. No bound check that
`-r-1 < view.n_slots` (only `set_remote_exclusion` checks its own case).
Fix: `ExpertPlan { view, remap }` from `plan_decode()` / `plan_prefill()`, the
five primitives made private. See memory `project-v41-remap-slot-spaces`.

### 5. OPEN — remote wire mask is applied against the static HELLO bitmap
`remote_experts.rs:3319`. `submit_inner` masks picks by what box 2 STATICALLY
advertises, not by the hub's `owns_eff`, so a pick the hub reassigned to box 2
can be replaced by `NO_PICK` and computed by nobody. Its own comment says
`verify_routing_exactly_once` cannot see it (that validates the HUB's view).
`submit_inner` can also return `Ok(None)` after the local exclusion remap is
already published.

### 6. OPEN — `V41_REMOTE_SPLIT=2` is documented as an arithmetically identical
control, and is not. With `dry=true`, `ids` still skips box-2-owned experts but
`owns_eff` only carries `extra_remote`, so under `mark_remote_after_ensure`
those experts keep `ensure`'s default `-(e)-1` = "ours at pool slot e",
pointing at whatever occupies slot `e`. The exactly-once audit PASSES.

### 7. OPEN — the DRAFTER changes the accepted output at temperature 0
With a correct verify, which tokens are ACCEPTED must not depend on what the
drafter proposed -- the verify decides, the drafter only decides how many are
won per step. Measured 2026-09-16, same prompt/temp 0/120 tokens, changing only
`V41_DSPARK_DENSE_RING`:

    ring off  sha 0a0b617de9cf  E 2.380
    ring on   sha e63bf762b1bd  E 2.553

Output DIFFERS. Consistent with the known "verify is exact only at B<=2" result
(f32 non-associativity between a B=6 batched verify and a B=1 decode), so this
is probably a fidelity limit rather than a logic error -- but it has not been
confirmed against `scripts/v41_oracle`, and until it is, any A/B that changes
the drafter is also changing the answer.

### 8. OPEN — `V41_RESET_ZERO` parks an open correctness question behind a flag
`state.rs`. A 3.9 GB memset that exists to test whether "kernels never read past
`n_comp`" is false. Resolve it or make it a bounds check in the kernels.

### 9. MITIGATED — compressor lend/return pairs are not exception-safe
`engine.rs:840`/`:1000` (28 `?` between) and `forward_prefill.rs:727`/`:783`.
Any `?` in between leaks the store, so every LATER request fails with
"L{src}: missing compressor state" naming the WRONG layer.
`HetModelState::restore_compressor_lending` now repairs ownership at forward
entry -- but NOT state: if the forward failed after a compressor boundary
fired, `n_comp`/`state_kv` are advanced relative to rolled-back KV and the next
request inherits that silently. Real fix is a `CompressorLoan` RAII guard;
`with_kv_source` is the model.

### 15. OPEN — the expert pool is statically partitioned BY PHASE
`dense_windows` reserves N windows for PREFILL's encoder layers; decode cannot
use them even when idle. At the default (pool 78 GB, WINDOWS=21) that is 2944
of 4454 slots withheld from decode, leaving it 1510. Measured 2026-09-16,
plain decode, per-token:

    1510 LRU slots   219 ms/tok   4.57 tok/s   box2 srv 158.7 ms
    3174 LRU slots   236 ms/tok   4.23 tok/s   box2 srv  84.2 ms
    3686 LRU slots   185 ms/tok   5.40 tok/s   box2 srv  83.9 ms

Giving box 1 more slots HALVES box 2's server time, because the hub's
residency catch-all then keeps the picks locally instead of shipping them --
the max(t_box1, t_box2 + rtt) rebalance. The partition should be dynamic
(phase-aware pool), not a startup constant.

For contrast, box 2 does NOT have this problem: its `--experts` spec and
placement file are only the INITIAL LOAD, after which `enable_paging()` sets
`owned = true` on every layer and runs a shard-wide global LRU with no
per-layer floor. Box 1 is the statically partitioned, non-evicting side (see
#16), not box 2.

NOTE: the windows are NOT id-indexed. `ensure_layer_union` assigns slots
"densely by arrival order, not by expert id", so a window holds the DEMANDED
union. The `slot == id` dense twin refuses on a packed window and is
unreachable at STRIDE<384. Do not repeat the claim that windows pin "each
layer's first 128 ids" -- that was wrong.

### 16. OPEN — box 1's decode "LRU" never evicts (fill-once-then-freeze)
`forward_layer.rs:2331` takes `budget = pg.lru_free_slots()`, and
`lru_free_slots` (`expert_pager.rs:1538`) counts slots that are **empty**, not
evictable. So under the T2 catch-all box 1 admits a new expert only while the
decode LRU still has VIRGIN slots; once it is full `budget == 0` on every
subsequent call and box 1 never admits another expert for the life of the
process. Whatever arrived first is frozen in.

This makes the in-tree note "a 1396-slot LRU filled that way was SLOWER than a
25-slot one (igpu.routed_moe +89%)" a symptom, not a tuning result: a bigger
frozen cache just locks in a worse set. Contrast box 2, which pages dynamically
across its whole pool with no per-layer floor (V41_B2_POOL_FLOOR=0) and reaches
a 96.9% hit rate.

Note the admission GATE itself is correct and wanted: `victim_cache()` (default
ON, `expert_pager.rs:219`) only admits experts box 2 reported missing
(`box2_missed`), so box 1 is a victim cache for box 2 rather than a duplicate of
its hot set. The bug is the freeze, not the gate.

## Structural / ergonomic

### 10. OPEN — two sources of truth for the current HIP device
`DeviceGuard` is correct and `pub(crate)` (invisible to the crate that needs
it); the engine keeps a `current_device: AtomicI32` mirroring a THREAD-LOCAL
HIP property. 91 `set_current*` calls in `het/`, zero guards. One site is
hand-patched with a comment explaining the cache goes stale.

### 11. OPEN — env-var sprawl: 292 `std::env::var` reads in production `src/`
113 distinct names in `het/` alone. 21 per layer in the prefill hot path
(~840/step) -- measured at ~42 us/layer, i.e. NOT a bottleneck, but several
select different NUMERICS from a free integer rather than a validated enum.
Should be one `EngineConfig` parsed once, with a `validate()` rejecting the
contradictory combinations, printed in the startup banner and in run
fingerprints.

### 12. OPEN — `KvMark.slid: bool` changes the meaning of `per_layer` and
disables the wrap check in `rollback_kv`. Two mark kinds in one struct; should
be an enum so `rollback_kv` cannot be handed a mark whose provenance it must
trust.

### 13. OPEN — doc-comment drift attaches `///` blocks to the WRONG item
Verified at `expert_pager.rs:154`, `:257`, `:1188`, `forward_prefill.rs:61`,
`:81`, `state.rs:425`. In a codebase where comments ARE the invariant
documentation, a comment on the wrong item is worse than none.

### 14. OPEN — `CompKvStore::fp8_enabled()`/`e2m1_enabled()` read env at every
allocation, so one process can hold layers that disagree on storage format.

---

## Fixed 2026-09-16 (kept briefly for cross-reference)

- ~~sparse verify allocator paired with `set_remote_exclusion`~~ -> picks
  computed by nobody; `86808af`, tests in `v41_sparse_verify_remap_pairing.rs`
- ~~sparse verify used `routed_window` with ABSOLUTE slots~~ -> silently read
  another expert's weights; worth E 1.804 -> 2.261; `dea8525`
- ~~chat "extend" path never called `normalize_raw_windows`~~ -> after any turn
  over 128 tokens the suffix prefill overwrote live KV; `76ce146`
- ~~`V41_B2_POOL_FLOOR` defaulted to 0.90~~ against its own doc-comment's
  measurement; `148044b`
