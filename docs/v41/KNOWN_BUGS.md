# V4.1 known bugs and open correctness questions

Living list. Add here the moment something is found, even when it is not being
fixed right now, and delete only when it is fixed AND has a regression test.
Ranked by risk of SILENT WRONGNESS (produces wrong numbers rather than an error).

Status key: **OPEN** / *MITIGATED* / ~~FIXED~~

---

## Silent wrongness

### 1. OPEN — temperature-0 output depends on `V41_PAGER_WINDOWS`
Residency must never change numerics; a miss is a page-in, not a different
answer. LRU size is correctly neutral (25 -> 1510 slots: byte-identical output).
But the DENSE packed-window geometry is not. Same prompt, temp 0, 120 tokens:

    WINDOWS=21  sha (A)  E 2.380      <- production default
    WINDOWS=8   sha (B)  E 2.000
    WINDOWS=4   sha (C)  E 1.595
    WINDOWS=1   GARBAGE  E 1.000      <- see #3

E falls monotonically with `dense_windows`, so the PROMPT PREFILL is producing
geometry-dependent KV. Root cause NOT yet found. #2 was the leading suspect and
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

### 7. OPEN — `V41_RESET_ZERO` parks an open correctness question behind a flag
`state.rs`. A 3.9 GB memset that exists to test whether "kernels never read past
`n_comp`" is false. Resolve it or make it a bounds check in the kernels.

### 8. MITIGATED — compressor lend/return pairs are not exception-safe
`engine.rs:840`/`:1000` (28 `?` between) and `forward_prefill.rs:727`/`:783`.
Any `?` in between leaks the store, so every LATER request fails with
"L{src}: missing compressor state" naming the WRONG layer.
`HetModelState::restore_compressor_lending` now repairs ownership at forward
entry -- but NOT state: if the forward failed after a compressor boundary
fired, `n_comp`/`state_kv` are advanced relative to rolled-back KV and the next
request inherits that silently. Real fix is a `CompressorLoan` RAII guard;
`with_kv_source` is the model.

## Structural / ergonomic

### 9. OPEN — two sources of truth for the current HIP device
`DeviceGuard` is correct and `pub(crate)` (invisible to the crate that needs
it); the engine keeps a `current_device: AtomicI32` mirroring a THREAD-LOCAL
HIP property. 91 `set_current*` calls in `het/`, zero guards. One site is
hand-patched with a comment explaining the cache goes stale.

### 10. OPEN — env-var sprawl: 292 `std::env::var` reads in production `src/`
113 distinct names in `het/` alone. 21 per layer in the prefill hot path
(~840/step) -- measured at ~42 us/layer, i.e. NOT a bottleneck, but several
select different NUMERICS from a free integer rather than a validated enum.
Should be one `EngineConfig` parsed once, with a `validate()` rejecting the
contradictory combinations, printed in the startup banner and in run
fingerprints.

### 11. OPEN — `KvMark.slid: bool` changes the meaning of `per_layer` and
disables the wrap check in `rollback_kv`. Two mark kinds in one struct; should
be an enum so `rollback_kv` cannot be handed a mark whose provenance it must
trust.

### 12. OPEN — doc-comment drift attaches `///` blocks to the WRONG item
Verified at `expert_pager.rs:154`, `:257`, `:1188`, `forward_prefill.rs:61`,
`:81`, `state.rs:425`. In a codebase where comments ARE the invariant
documentation, a comment on the wrong item is worse than none.

### 13. OPEN — `CompKvStore::fp8_enabled()`/`e2m1_enabled()` read env at every
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
