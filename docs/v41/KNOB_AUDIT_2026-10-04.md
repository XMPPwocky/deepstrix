# Knob audit, 2026-10-04: settings that bypass `crate::knobs`

Branch `worktree-lm-prefill-prod` @ fc35700. The audit started at 777a8e2, and
fc35700 (`V41_SUB_DEFER_ACCEPTED`) landed partway through. It adds no env reads
but shifts lines in `forward_prefill.rs`, `b2_mirror.rs` and `multistream.rs`;
every line number below has been re-checked against fc35700. I only read the
code: nothing was built or run, and no server was touched. Box 2's launch env comes from
`~/scratch-ms/deploy_long_20261003.sh:35,129` (read locally, no ssh). The hub's
env comes from `~/scratch-ms/hub_env_lm_20261001.txt`, which the deploy scripts
pass through `env -i`, so it is the whole environment. The deploy re-sets
`V41_PICK_TRACE` to a fresh path each time.

Path prefixes: `k/` = `crates/v4flash-kernels/src/`, `kh/` = `k/het/`,
`srv/` = `crates/deepstrix-server/src/`, `core/` = `crates/v4flash-core/src/`.
Line numbers point at the `env::var` call.

## Summary

The audit found **333 env names** read outside a `Knob`:
- **293 V4.1 names in 291 table rows.** One row covers the three
  `DEEPSTRIX_DUMP_SUBTENSOR_*` names.
- **40 Laguna names**, in one grouped row per file.

It does not count `V41_KNOBS_FILE` or `V41_B2_KNOBS`, which bootstrap the
framework, or system variables (`HOME`, `XDG_CACHE_HOME`, `OUT_DIR`, ...).

| priority | V4.1 rows | rows with a value in the hub or box-2 prod env |
|---|---|---|
| HIGH | 50 | 45 (the other 5 are A/B targets nobody sets) |
| MED  | 159 | 0 |
| LOW  | 82 | 0 |

Suggested kind (V4.1 rows): flag 158, int 69, choice 29, text 26, real 9.
Laguna: 40 more names, MED/LOW, mostly flag/choice/int.

Mechanisms in use today:
- `LazyLock`/`OnceLock` env caches: most rows.
- Uncached `std::env::var` per call: about 160 call sites outside Laguna. Most
  run at startup or allocation, but 28 table rows are hot-path (YES). About 50
  of those sites sit in the multistream per-lane-layer chain. See
  [Hot path](#hot-path-uncached-env-reads-perf-bugs).
- Hand-rolled live mechanisms that duplicate what `Knob::live` does:
  - `V41_B2_HITS_FIRST`: AtomicBool plus a SIGUSR1 xor toggle.
  - `V41_SMALL_B_CATCHALL_MAX`, `V41_PREFILL_SINGLE_LANE_MAX`: atomics with setters.
  - `V41_B2_PIN`: atomic plus `set_pin_wanted`.
  - `V41_PARTITION_BOX1_SHARE`: atomic.
  - `V41_EXPERT_MIRROR_FRAC`: a seed in v4flash-core, plus the knob hook.
  - The B1 hot-set family: env re-read on every refresh.

Two production settings are probably inert on the multistream path. Both are
read only by the single-stream driver (`forward_layer_impl_inner`):
- **`V41_PARTITION_BOX1_SHARE=0.15`**
- **`V41_REMOTE_SPLIT_DECODE=1`**

With `V41_MULTISTREAM=1` plus `V41_MS_DSPARK=accept`, no request goes to that
driver: `srv/multistream.rs:860-861` makes `is_legacy` false, so nothing is
legacy.

Many production `=1` settings are default-off flags that prod always turns on:
`V41_CANDIDATE_POOL`, `V41_INDEX_K`, `V41_PUSH_XQ`, `V41_PREFILL_UNIFIED_POOL`,
`V41_PAGED_EXPERTS`, `V41_B1_PAGE_MISSES`, `V41_B1_PREFETCH`, `V41_SUB_ADMIT_GATE`,
`V41_T2_PARTITION`, `V41_T2_CATCHALL`, `V41_B2_PIN`, `V41_MTP_MOE_GROUPED`. When
they become knobs, the knob default is the place to flip them.

### Convert first (top 10)

1. **`V41_SUB_PROTECT`** → live int 0..6. Prod sets 1, and the owner wants a live
   fidelity A/B. Each router launch passes it as a kernel argument
   (`kh/forward_prefill.rs:7902`) and nothing is sized by it. This is a one-line
   swap of the `LazyLock` at `kh/b2_mirror.rs:211`.
2. **`V41_B2_HITS_FIRST`** → live flag (b2=1). It is already live through the
   SIGUSR1 xor toggle (`kh/remote_experts.rs:3766-3790`). The flips never reach
   the knob log or the `.effective` file, so an A/B can't be reconstructed afterwards.
3. **`V41_B2_MISS_PAR`** → remove the duplicate. A `LazyLock` (`:3808`) sizes the
   staging, while the live knob `MISS_PAR` (`miss_par`) caps each request
   (`:5172`). One knob should do both: its startup value sizes the staging and
   its live value caps the request. Today a file value above the env value is
   silently clamped.
4. **`V41_B2_EARLY_PAGE`** → live flag. Each request checks it (`:7190`), and it
   is a known A/B target.
5. **`V41_B2_RESTORE`** → live flag. Re-read it in `me_switch`, the same way
   `prefill_budget` already is (`:3066-3069`). The field is only consulted at
   phase switches (`:3054,:3073,:4365`).
6. **B1 hot-set family** → live: `V41_B1_HOT_PER_LAYER` (103), `V41_B1_HOT_HYST`
   (100), `V41_B1_HOT_MAX_CHANGE` (3), plus `_MIN_PICKS` and `_SLACK`. All are
   prod-set and re-read from env on every refresh (`kh/expert_pager.rs:587-650`),
   so they are live in effect already, but they can't be changed.
7. **`V41_SMALL_B_CATCHALL_MAX`** (8), plus `V41_PREFILL_SINGLE_LANE_MAX` → live
   int. They were built runtime-settable for in-process A/B
   (`kh/forward_prefill.rs:11317-11370`), so the knob replaces the hand-rolled
   atomic and setter.
8. **`V41_B1_PREFETCH_ADMIT`** (24), plus `V41_B1_PREFETCH_MIN_TOUCH` → live int.
   They are read on each drain and each hint (`kh/expert_pager.rs:1230,1206`).
9. **Link busy-poll windows** → live int. On the hub: `V41_DECODE_BUSY_POLL_US`
   (5000) and `V41_BATCH_BUSY_POLL_US`, applied with setsockopt at every phase
   switch (`kh/engine.rs:567-582`). On box 2: `V41_B2_DECODE_BUSY_POLL_US`,
   applied when the wanted window changes (`kh/remote_experts.rs:196-210`).
10. **Substitution switches.**
    - Live flags: `V41_SUB_PENDING` (0) and `V41_SUB_ADMIT_GATE` (1), each read
      per lookup or per want.
    - Static choice: `V41_SUB` (3).
    - Static choices that remove duplicate caches: `V41_REMOTE_SPLIT` (4
      `LazyLock`s) and `V41_T2_CATCHALL` (2 `LazyLock`s).

Two framework gaps you will hit while converting:
- **`.alias()` and `.legacy()` both force `live`** (`k/knobs.rs:183-195`). That
  means a STATIC knob can't keep an old env name. Unprefixed names (`IQ2_VARIANT`,
  `QB_WMMA`, `DGPU_HOT_CAP`, `PAIR_VARIANT`, `COMP_KV_FP8`, ...) and every
  `DEEPSTRIX_*` and `LAGUNA_*` name need one of:
  - a static env alias, or
  - a relaxed `V41_` prefix assertion in
    `srv/knobs.rs:the_table_is_sane_and_only_the_legacy_file_knobs_are_live`.
- **v4flash-core can't depend on the knob crate.** Its settings
  (`V41_EXPERT_MIRROR_FRAC`, `V41_EXPERT_ODIRECT`, `V41_EXPERT_PREAD_THREADS`,
  `V41_EXPERT_MIRROR_DIR`) need the hook-into-atomic pattern that `MIRROR_FRAC`
  already uses.

## Hot path: uncached env reads (perf bugs)

**Production multistream decode and prefill.** `forward_step_arena*`
(`kh/forward_prefill.rs:3665-4400`) calls the following for every layer of every
lane, and LM/chunked prefill uses the same chain:
- `pre_moe_chain`
- `pre_moe_route`
- `pre_moe_prep`
- `pre_moe_launch`
- `forward_layer_post_moe_v2`

These functions hold about 50 uncached env lookup sites. Every one takes Rust's
env lock and does a glibc `getenv` linear scan over the 115-var production
environment. Many also allocate a `String`, for example
`unwrap_or_else(|| "kwide".into())`.

| function | uncached sites (lines) |
|---|---|
| `pre_moe_chain` (4401-8341) | `QB_WMMA` 4982<br>`Q8_GROUPED_VARIANT` 7399<br>`Q8_OUT_VARIANT` 7489<br>`V41_ROUTER_WMMA` 7781<br>`V41_INDEXER_FORCE` 6320<br>`ATTN_FUSED` 7036<br>`INDEXER_TOPK_SELECT` 6887<br>`DEEPSTRIX_COMP_GEMM`, `DEEPSTRIX_COMP_TILED`, `DEEPSTRIX_COMP_GATHER` 5453/5527/5613<br>`V41_WINDOW_DBG` 5314 (SWA layers, row 0)<br>`V41_COMP_POS_DBG` 6069 (mismatch only)<br>helpers: `prefill_f32_matvec` ×6 and `prefill_f32_matvec_qb_wo` ×3 (`V41_PREFILL_F32_MATVEC`), `mhc_pre_scaled_for` ×5 (`V41_MHC_PRE_SCALED`), `mhc_narrow_fallback_for` ×5 (`V41_MHC_NARROW`), `engram_gemv_fallback` ×1, `swa_via_mixed` ×1, `attn_scores_stride` ×2 (`DEEPSTRIX_ATTN_LEGACY_STRIDE`, `k/attention.rs:308`) |
| `pre_moe_route` (8342-9343) | `V41_GROUP_AUDIT_VERBOSE` 8400, 8573, 9054, 9080<br>`V41_REMOTE_DBG` 9038 (short-circuited), 9101, 9193<br>`V41_SMALL_B_CATCHALL_HALF` 8899 (catch-all branch) |
| `pre_moe_prep` (9344-9876) | `V41_GROUP_AUDIT_VERBOSE` 9367, 9412, 9573<br>`V41_PAGER_SYNC_IGPU` 9407 (when ensure may evict)<br>`V41_SPARSE_REMAP_SYNC` 9451<br>`V41_PAGER_SYNC_AFTER_ENSURE` 9494<br>`IQ2_VARIANT` 9689<br>`IGPU_MOE_WMMA` 9690 (`kh/dispatch.rs:363`) |
| `pre_moe_launch` (9877-10763) | `IQ2_VARIANT` 9955<br>`IGPU_MOE_WMMA` 9961<br>`IQ2_HYBRID_THRESHOLD` 10174<br>`V41_GROUP_AUDIT_VERBOSE` 10116<br>`Q2K_VARIANT` 10489 (Q2_K down only)<br>`PAIR_VARIANT` via `moe_wi_devcount_supported`, `pair_prefill_stage_rows`, `moe_gate_up_chunked_rows` (`kh/dispatch.rs:369` → 376/480/585/624) |
| `forward_layer_post_moe_v2` (10764-11136) | `V41_REMOTE_DBG` 10906, 10967, 11012 |

Estimate (not measured): roughly 30-40 of these sites run per lane-layer. That is
40 layers × 2 lanes, about 2,400-3,200 getenv calls per decode step, or roughly
0.25-1 ms of host enqueue time per step at 0.1-0.3 µs each.

The fix is mechanical. Each read becomes a knob (an atomic load), or at least a
`LazyLock`. Every one of these settings is a rollback, diagnostic or kernel-arm
switch that nobody sets in production.

Other uncached reads on production per-step or per-request paths. Each costs
little; they are listed for completeness:
- `V41_DSPARK_DRAFT_TIMING`: once per drafted step (`kh/engine.rs:841`, reached
  from `srv/ms_dspark.rs:515`).
- `V41_CED`: several times per prefill start (`kh/forward_prefill.rs:354`).
- `DEEPSTRIX_SNAPSHOT_REUSE`: once per prefill start (`srv/multistream.rs:1119`).
- `V41_RESET_ZERO`: once per `reset_in_place` (`kh/state.rs:746`).
- `V41_MASK_DBG`: per remote submit, but only when picks were masked
  (`kh/remote_experts.rs:8475,8483`).

**Legacy single-stream path (not production).** `forward_layer_impl_inner`
(`kh/forward_layer.rs:510-`) has 22 uncached sites per layer per token:
- `MHC_FUSED` 592, 2035
- `RMS_NW_MW` 681, 2090, 3154
- `RMS_W_MW` 725, 2127, 3196
- `F16_KSPLIT` 628, 695, 2047, 2102, 3166
- `V41_WINDOW_DBG` 1366, 1870
- `V41_INDEXER_FORCE` 1457
- `DECODE_INDEXER` 1464
- `V41_INDEXER_DBG` 1716
- `V41_INDEXER_DUMP` 1741
- `DECODE_SCORE` 1884
- `DECODE_SMWSUM` 1919
- `V41_REMOTE_SPLIT_DECODE` 2328
- `V41_REMOTE_NOMASK` 2717
- `V41_PAGER_DBG` 2814, 3086
- `V41_PAGER_RESIDENT` 2846
- `V41_PAGER_NOGRAPH` 2871

The same path also calls:
- `set_partition_share` once per layer (`kh/expert_pager.rs:494`).
- `F16_KSPLIT_V8` once per narrow-ksplit launch (`k/f16.rs:390`). The only
  callers are in `forward_layer.rs`.

The legacy DSpark verify loop reads these once per verify step:
- `V41_DSPARK_K` (`srv/engine_worker.rs:3698`)
- `V41_DSPARK_CONF_MIN` (`:3709`)
- `V41_DUMP_VERIFY_DIST` (`:3987`)
- `V41_DSPARK_FORCE_N0` (`:4019`)
- `V41_DSPARK_RING_FAST` (`:4271`)
- `V41_DSPARK_DENSE_RING` (`:4272`)
- `V41_DSPARK_STEP_TIMING` (`:4314`)

**Laguna path.** These read env per layer or per launch:
- `k/laguna_het.rs`: `layer` 1003, `attn_batched` 1691-1779 (11 reads), `moe_batched` 1882-1987 (4)
- `k/laguna_moe_tiled.rs`: per kernel launch, 227-549
- `k/laguna.rs:166`: per router call
- `k/gqa_attention.rs:49`: per flash launch

## Duplicates and inconsistencies

- **`V41_B2_MISS_PAR`.** Two parses of the same name:
  - A `LazyLock` at `kh/remote_experts.rs:3808` sizes the staging (`:4073`).
  - The live knob `MISS_PAR` (`:1955`) caps each request (`:5172`).

  A knob-file value above the env value is silently clamped to the staging.
  The test at `:8782-8800` documents the env/file split.
- **`V41_EXPERT_MIRROR_FRAC`.** Two sources:
  - The box-2 knob `MIRROR_FRAC` (`:1980`), whose hook pushes the value into core.
  - `core/hf_v41.rs:191`, which seeds the same atomic from env on first read.

  The defaults agree (0.6), so this is harmless. It is still a second parse.
- **`V41_INDEX_K`** has four independent `LazyLock`s: `kh/forward_prefill.rs:103`,
  `kh/forward_layer.rs:155`, `k/attention.rs:161`, `k/attention.rs:207`.
- **`V41_REMOTE_SPLIT`** has four `LazyLock`s, each accepting a different value
  set (`kh/forward_prefill.rs:76,110,178,189`). It should be one choice knob.
- **`V41_T2_CATCHALL`** has two `LazyLock`s (`kh/expert_pager.rs:3104` `=="2"`,
  `:3111` `!="0"`). It should be one choice knob.
- **`V41_PICK_TRACE`** has two `LazyLock`s (`kh/expert_pager.rs:463,475`).
  `V41_XCHECK_ROWS` has two `OnceLock`s with different values
  (`srv/engine_worker.rs:5697,5702`).
- **`V41_REMOTE_ADDR`** is read 9 times with `is_ok()`, once for each scratch
  buffer (`kh/batch_scratch.rs:988,1278,1289,1300,1308,1315,1320`;
  `kh/scratch.rs:426`; `kh/engine.rs:481`).
- **`DGPU_HOT_CAP`** is read in three places: `kh/weights.rs:1336`
  (`LazyLock`), `:1393` (uncached validate), `kh/forward_prefill.rs:203`.
  `DGPU_HOT_PREFILL` is read at `kh/forward_prefill.rs:57` and again at
  `kh/weights.rs:1405`.
- **`V41_PREFILL_F32_MATVEC`** is parsed twice, uncached, in
  `prefill_f32_matvec` and `prefill_f32_matvec_qb_wo`
  (`kh/forward_prefill.rs:490,515`).
- **`V41_DSPARK`** is read at `srv/engine_worker.rs:991` and `:3183`.
  `V41_DSPARK_SEED_RING` at `:3068` and `:5213`. `DEEPSTRIX_SNAPSHOT_REUSE` at
  `srv/multistream.rs:1119` and `srv/engine_worker.rs:2356`.
- **Env used as a side channel.** At startup, `srv/engine_worker.rs:964` calls
  `std::env::set_var("DGPU_HOT_EXPERTS_FILE", ..)` so the loader picks the value
  up later. Calling `set_var` in a threaded process is unsound (Rust 2024 makes it
  `unsafe`). A static text knob, or passing the path explicitly, removes it.
- **`V41_B1_HOT_MASS`** is documented as a cap ("also capped at
  `V41_B1_HOT_MASS`", `kh/expert_pager.rs:525-526`). The code only reports it:
  `let _ = mass_cap; // ranks decide; the mass is reported, not enforced`, at
  `:639`.
- **`V41_B2_ODIRECT`** has stacked docs. The first paragraph says "OFF by default"
  (`kh/remote_experts.rs:3545`). The second says "DEFAULT ON since 2026-09-15"
  (`:3574`). The code defaults ON (`:3826`).
- **`srv/knobs.rs:4-5`** says "Only the eight knobs that were live before are
  live". The test in the same file pins 15 live knobs (with fc35700's
  `V41_SUB_DEFER_ACCEPTED`).
- **`reload()`** prints a summary (`kh/remote_experts.rs:2089-2094`) that omits
  `prefill_budget_long` and `V41_B2_EVTRACE_DEV`.
- **A stray doc line.** "`V41_B2_EARLY_PAGE=0` ..." sits on `b2_merge_wait_us`
  (`kh/remote_experts.rs:1905-1906`), not on `b2_early_page` (`:2119`).
- **Documented knob-file keys that are not wired:**
  - `docs/v41/BOX2_MISS_SUBSTITUTION.md:309-310` lists keys `sub`,
    `sub_min_rank`, `sub_admit` ("as `park`") and env `V41_B2_SUB*`. None of them
    exist: the box-2 table has no such aliases, and substitution became hub-side
    `V41_SUB*` env vars.
  - Hits-first has no knob-file key at all: only SIGUSR1. A `hits_first=` line in
    `~/expertd-knobs.txt` only gets an "unknown key" warning.

## Table

Sorted by priority, then file. "LIVE-capable" means it is safe to change between
reads, judged from the code I read. A value under "set in prod?" comes from the
hub env file (hub=) or box 2's launch line (b2=).

| file:line | env name | current mechanism | suggested kind | live/static + reason | hot-path? | priority | set in prod? |
|---|---|---|---|---|---|---|---|
| kh/b2_mirror.rs:211 | V41_SUB_PROTECT | LazyLock u32, default 2, min 6 | int 0..6 | LIVE: passed as `n_protect` to each router launch (forward_prefill.rs:7902); sizes nothing | no | HIGH | hub=1 |
| kh/b2_mirror.rs:150 | V41_SUB | LazyLock u32 0..3 (+ startup log) | choice off/dry/host/prior | STATIC: read several times per lane-layer (forward_prefill.rs:7832,7840,8780,8795) and per submit (remote_experts.rs:8404); a flip mid-step mixes modes | no | HIGH | hub=3 |
| kh/b2_mirror.rs:251 | V41_SUB_PENDING | LazyLock flag, default on | flag | LIVE: consulted per residency lookup (:357,:1453); no state | no | HIGH | hub=0 |
| kh/b2_mirror.rs:1239 | V41_SUB_ADMIT_GATE | LazyLock flag, default off | flag | LIVE: per-want gate (:1249) | no | HIGH | hub=1 |
| kh/b2_mirror.rs:398 | V41_B2_PIN | atomic seeded by an uncached read on first call, + `set_pin_wanted` | flag | STATIC: takes effect at connect (:421) | no | HIGH | hub=1 |
| kh/engine.rs:569 | V41_DECODE_BUSY_POLL_US | LazyLock, default 3000 (0 = off) | int | LIVE: setsockopt at each phase switch (engine.rs:621,646,981; forward_prefill.rs:1364...) | no | HIGH | hub=5000 |
| kh/engine.rs:572 | V41_BATCH_BUSY_POLL_US | LazyLock, default 50 | int | LIVE: same mechanism | no | HIGH (link A/B) | - |
| kh/engine.rs:481 (+ batch_scratch.rs:988,1278,1289,1300,1308,1315,1320; scratch.rs:426) | V41_REMOTE_ADDR | 9 uncached reads (connect + `is_ok()` per scratch buffer) | text | STATIC: connect and allocation | alloc-time only | HIGH | hub=10.99.0.2:7431 |
| kh/evtrace.rs:378 | V41_EVTRACE_DIR | uncached at init | text | STATIC: opens the trace | no | HIGH | hub, b2 |
| kh/expert_pager.rs:587 | V41_B1_HOT_PER_LAYER | uncached `env_u` on every refresh | int | LIVE: read per refresh (multistream.rs:1583), clamped to capacity; only ranks | per refresh | HIGH | hub=103 |
| kh/expert_pager.rs:612 | V41_B1_HOT_HYST | uncached, per refresh | int | LIVE: same | per refresh | HIGH | hub=100 |
| kh/expert_pager.rs:650 | V41_B1_HOT_MAX_CHANGE | uncached, per layer per refresh (40x) | int | LIVE: same | per refresh | HIGH | hub=3 |
| kh/expert_pager.rs:884 | V41_B1_PAGE_MISSES | LazyLock flag | flag | LIVE-capable: per-pick routing choice (forward_prefill.rs:8973); miss staging always allocated (:1533) | no | HIGH | hub=1 |
| kh/expert_pager.rs:890 | V41_B1_PREFETCH | LazyLock flag | flag | STATIC: starts the prefetcher thread (srv/engine_worker.rs:1238) | no | HIGH | hub=1 |
| kh/expert_pager.rs:898 | V41_B1_PREFETCH_ADMIT | LazyLock, default 8 | int | LIVE: cap per drain (:1230) | no | HIGH | hub=24 |
| kh/expert_pager.rs:494 | V41_PARTITION_BOX1_SHARE | uncached in `set_partition_share`, stored in an atomic | real 0..1 | live-capable (atomic), but the only caller is single-stream forward_layer.rs:2572, so it is never set on the multistream path (falls back to 420 per mille until the hot set warms, :511-512) | legacy: per layer per token | HIGH | hub=0.15 (likely inert) |
| kh/expert_pager.rs:487 | V41_T2_PARTITION | LazyLock flag | flag | STATIC: id ownership must agree across lanes, in-flight requests and box 2 | no | HIGH | hub=1 |
| kh/expert_pager.rs:3104,3111 | V41_T2_CATCHALL | two LazyLocks (`=="2"`, `!="0"`) | choice 0/1/2 | STATIC: ownership policy; box 2 must be `--paged` | no | HIGH | hub=1 |
| kh/expert_pager.rs:834 | V41_PAGER_MISS_PAR | LazyLock, default 4, clamp 1..16 | int | STATIC: sizes pinned miss staging (:1533) | no | HIGH | hub=8 |
| kh/expert_pager.rs:841 | V41_PAGER_MISS_THREADS | LazyLock, default 1 | int | LIVE-capable: per-miss scoped threads (:2662) | no | HIGH | hub=3 |
| kh/expert_pager.rs:853 | V41_PAGER_UNION | LazyLock flag, default on | flag | STATIC: paging path | no | HIGH | hub=1 |
| kh/expert_pager.rs:1359 | V41_PAGER_POOL_GB | uncached in `ExpertPager::new` | real | STATIC: pool size | no | HIGH | hub=95 |
| kh/expert_pager.rs:1437 | V41_PAGER_STRIDE | uncached in `new` | int 1..384 | STATIC: window layout | no | HIGH | hub=384 |
| kh/expert_pager.rs:1461 | V41_PAGER_WINDOWS | uncached in `new` | int | STATIC: dense windows | no | HIGH | hub=0 |
| kh/expert_pager.rs:1513 | V41_MISS_HIST | uncached in `new` | flag | STATIC: allocates the histogram | no | HIGH | hub=1 |
| kh/expert_pager.rs:463,475 | V41_PICK_TRACE | two LazyLocks | text | STATIC: opens the file | no | HIGH | hub=path |
| kh/forward_layer.rs:148 | V41_CANDIDATE_POOL | LazyLock flag | flag | STATIC: sizes batch scratch (batch_scratch.rs:1216,1225) | no | HIGH | hub=1 |
| kh/forward_layer.rs:2328 | V41_REMOTE_SPLIT_DECODE | uncached, per layer per token | flag | STATIC; only the single-stream path reads it, so it is inert under multistream | legacy: per layer | HIGH | hub=1 (inert) |
| kh/forward_prefill.rs:76,110,178,189 | V41_REMOTE_SPLIT | four LazyLocks, each with its own value set | choice off/1/2/3/4 | STATIC: exclusion and combine must match for in-flight requests | no | HIGH | hub=1 |
| kh/forward_prefill.rs:103 (+ forward_layer.rs:155, attention.rs:161,207) | V41_INDEX_K | four LazyLocks | flag | STATIC: sizes indexer scratch (attention.rs:203) | no | HIGH | hub=1 |
| kh/forward_prefill.rs:354 | V41_CED | uncached on every call | flag | STATIC: pager layout (expert_pager.rs:1449), scratch (batch_scratch.rs:1352), job shape | per prefill start (multistream.rs:1136; :965,968) | HIGH | hub=1 |
| kh/forward_prefill.rs:631 | V41_PREFILL_F16_REPLIES | OnceLock flag, default on (+ "UNTESTED FIDELITY" warning) | flag | STATIC until reply buffers are checked; fidelity A/B target | no | HIGH | - |
| kh/forward_prefill.rs:1572 | V41_MS_MHC_SPLIT | LazyLock | flag | STATIC: kernel path (graphs may bake it in) | no | HIGH | hub=0 |
| kh/forward_prefill.rs:11335 | V41_SMALL_B_CATCHALL_MAX | atomic seeded from env + `set_small_b_catchall_max` | int | LIVE: built runtime-settable for in-process A/B (:11317-11324) | no | HIGH | hub=8 |
| kh/forward_prefill.rs:11413 | V41_PREFILL_UNIFIED_POOL | LazyLock | flag | STATIC: pool layout | no | HIGH | hub=1 |
| kh/forward_prefill.rs:11518 | V41_PUSH_XQ | OnceLock | flag | STATIC: transfer path (not checked for live safety) | no | HIGH | hub=1 |
| kh/mtp.rs:479 | V41_MTP_EXPERT_STATS | OnceLock | flag | STATIC | no | HIGH | hub=1 |
| kh/mtp.rs:495 | V41_MTP_MOE_GROUPED | OnceLock | flag | STATIC: drafter kernel path | no | HIGH | hub=1 |
| kh/remote_experts.rs:3768 | V41_B2_HITS_FIRST | OnceLock seed into AtomicBool, SIGUSR1 xor toggle (:3774-3790) | flag | LIVE (already, by hand): per request (:6041); flips invisible to the knob log | no | HIGH | b2=1 |
| kh/remote_experts.rs:3808 | V41_B2_MISS_PAR | LazyLock, clamp 1..16; duplicate of live knob `MISS_PAR` (:1955) | int | STATIC part sizes staging (:4073), live part caps each request (:5172); merge them | no | HIGH | b2=4 |
| kh/remote_experts.rs:2147 | V41_B2_PREFETCH_SETS | uncached, read at shard alloc (:4090) | int 0..32 | STATIC: sizes pinned staging sets | no | HIGH | b2=16 |
| kh/remote_experts.rs:2120 | V41_B2_EARLY_PAGE | LazyLock flag, default on | flag | LIVE: per-request check (:7190) | no | HIGH | - |
| kh/remote_experts.rs:2441 | V41_B2_RESTORE | LazyLock flag, default off | flag | LIVE-capable: copied once into `me.restore_on` (:4742) but only consulted at phase switches (:3054,:3073,:4365); refresh it in `me_switch` | no | HIGH | - (needs MODE_EVICT) |
| kh/remote_experts.rs:2449 | V41_B2_MODE_EVICT | LazyLock flag | flag | STATIC: `enable_mode_evict_live` allocates per-slot state at enable_paging (:4739-4741) | no | HIGH | b2=1 |
| kh/remote_experts.rs:2630 | V41_B2_PREFILL_STAGE | LazyLock, default 384 | int | STATIC: staging band set once (:4738) | no | HIGH | b2=0 |
| kh/remote_experts.rs:198 | V41_B2_DECODE_BUSY_POLL_US | LazyLock, default 5000 | int | LIVE: setsockopt when the wanted window changes (:206-210) | no | HIGH (link A/B) | - |
| kh/remote_experts.rs:4706 (+ core/safetensors.rs:139) | V41_EXPERT_MIRROR_DIR | uncached at open / enable_paging | text | STATIC: opens mirror files | no | HIGH | b2=/weights2/dsv4.1f |
| core/hf_v41.rs:191 | V41_EXPERT_MIRROR_FRAC | env seed of an atomic; duplicate of box-2 knob `MIRROR_FRAC` | real 0..1 | LIVE already via the knob hook; drop the env seed | no | HIGH | b2=0.70 |
| srv/engine_worker.rs:991 (+3183) | V41_DSPARK | uncached at init; legacy finish_decode reads it again per request | choice off/1/shadow/accept | STATIC: loads the 7.93 GB drafter | no | HIGH | hub=0 |
| srv/main.rs:192 | DEEPSTRIX_HANG_DEADLINE_MS | uncached at startup | int | STATIC: watchdog armed once | no | HIGH | hub=120000 |
| k/dense_gemm.rs:77 | KQ_GEMM | LazyLock | choice wmma/dp4a | STATIC kernel arm | no | MED | - |
| k/f16.rs:48 | V41_ROUTER_MV_H20 | LazyLock, default on | flag | STATIC kernel rollback | no | MED | - |
| k/f16.rs:66 | V41_MHC_GEMM_NARROW | LazyLock, default on | flag | STATIC kernel rollback | no | MED | - |
| k/f16.rs:91 | V41_F16_MV_Z16 | LazyLock, default on | flag | STATIC kernel rollback | no | MED | - |
| k/f16.rs:390 | F16_KSPLIT_V8 | uncached per launch | flag | STATIC kernel arm | legacy: per launch (callers only in forward_layer.rs) | MED | - |
| k/indexer.rs:237 | V41_IDX_SCORE_QREG | LazyLock | flag | STATIC kernel rollback | no | MED | - |
| k/indexer.rs:704 | V41_TOPK_SELECT_ILP | LazyLock | flag | STATIC kernel rollback | no | MED | - |
| k/indexer.rs:722 | V41_IDX_TOPK_HYBRID | LazyLock | flag | STATIC kernel rollback | no | MED | - |
| k/indexer.rs:1158 | V41_IDX_GATHER_B128 | LazyLock | flag | STATIC kernel rollback | no | MED | - |
| k/iq2_xxs.rs:344 | DECODE_IQ2 | LazyLock | choice | STATIC kernel arm (IQ2-era) | no | MED | - |
| k/lib.rs:107 | V41_GRID_PAD | LazyLock | flag | STATIC kernel rollback | no | MED | - |
| k/mxfp4_pair.rs:48 | V41_MXFP4_PAIR_WARPS | LazyLock | int | STATIC launch geometry | no | MED | - |
| k/q8_0.rs:54 | V41_GEMV_BPACK | OnceLock | flag | STATIC kernel rollback | no | MED | - |
| k/q8_0.rs:72 | V41_GEMV_TB | LazyLock | flag | STATIC kernel rollback | no | MED | - |
| k/q8_0.rs:126 | V41_Q8_QUANT_WAVE | LazyLock | flag | STATIC kernel rollback | no | MED | - |
| k/q8_0.rs:147 | V41_Q8_QUANT_GRID_PAD | LazyLock | flag | STATIC kernel rollback | no | MED | - |
| k/q8_0.rs:177 | V41_GEMV_BPACK_Z16 | LazyLock | flag | STATIC kernel rollback | no | MED | - |
| k/q8_0.rs:200 | V41_ENGRAM_I8X | LazyLock | flag | STATIC kernel rollback | no | MED | - |
| k/q8_0.rs:228 | V41_F16X_DB_BN64 | LazyLock | flag | STATIC kernel rollback | no | MED | - |
| k/q8_0.rs:242 | V41_F16X_256 | LazyLock | flag | STATIC kernel rollback | no | MED | - |
| k/rms_norm.rs:37 | V41_RMS_FAST | LazyLock | flag | STATIC kernel rollback | no | MED | - |
| k/router_topk.rs:69 | V41_TOPK_WFRED | LazyLock | flag | STATIC kernel rollback | no | MED | - |
| k/candidate_blocks.rs:137 | V41_CAND_THRESH_ILP | LazyLock | flag | STATIC kernel rollback | no | MED | - |
| k/weights.rs:58 | DEEPSTRIX_FAST_LOAD | LazyLock | flag | STATIC loader path | no | MED | - |
| kh/b2_mirror.rs:184 | V41_SUB_MIN_RANK | LazyLock, default 6 | int 1..6 | LIVE: planner built per pass (:1519) | no | MED | - |
| kh/b2_mirror.rs:203 | V41_SUB_DRY | LazyLock flag | flag | STATIC unless snapshotted per step: read 5x per layer (forward_prefill.rs:7903,8123,8657,8670,8918) | no | MED | - |
| kh/b2_mirror.rs:259 | V41_SUB_ADMIT | LazyLock flag, default on | flag | LIVE: per swapped pick (forward_prefill.rs:8719,8807) | no | MED | - |
| kh/b2_mirror.rs:266 | V41_SUB_MAX_W | LazyLock Option<f32> | real (0 = none) | LIVE: planner per pass | no | MED | - |
| kh/b2_mirror.rs:275 | V41_SUB_INCOMING | LazyLock, default 2 | int | LIVE: read at mark/lookup; a change only re-times outstanding marks | no | MED | - |
| kh/b2_mirror.rs:432 | V41_B2_PIN_HEADROOM | LazyLock, default 256 | int | LIVE: release threshold per step | no | MED | - |
| kh/b2_mirror.rs:441 | V41_B2_PIN_DECAY_STEPS | LazyLock, default 256 | int | LIVE: decay cadence | no | MED | - |
| kh/b2_mirror.rs:464 | V41_B2_PIN_WANTS | LazyLock flag, default on | flag | STATIC: changes what the pick counts mean | no | MED | - |
| kh/b2_mirror.rs:497 | V41_B2_PIN_RESTORE | LazyLock flag, default on | flag | LIVE: read at decode-phase entry | no | MED | - |
| kh/b2_mirror.rs:507 | V41_B2_PIN_RESTORE_PER_REQ | LazyLock, default 16 | int | LIVE: per request | no | MED | - |
| kh/batch_scratch.rs:79 | DEEPSTRIX_F32_SCORES | LazyLock | flag | STATIC: scratch sizing | no | MED | - |
| kh/dispatch.rs:259 | V41_SMALL_B_DENSE_DP4A | LazyLock | flag | STATIC kernel arm | no | MED | - |
| kh/dispatch.rs:262 | V41_SMALL_B_DENSE_MAX | LazyLock, default 8, max 16 | int | STATIC kernel arm | no | MED | - |
| kh/dispatch.rs:363 | IGPU_MOE_WMMA | uncached | flag | STATIC rollback | YES: 2 per lane-layer (forward_prefill.rs:9690,9961) | MED | - |
| kh/dispatch.rs:369 | PAIR_VARIANT | uncached | choice chunked/kwide | STATIC rollback | YES: per MoE dispatch (:376,480,585,624) | MED | - |
| kh/dispatch.rs:449 | V41_MOE_DOWN_DN2 | LazyLock | flag | STATIC kernel rollback | no | MED | - |
| kh/dispatch.rs:462 | V41_MOE_WMMA_GATEUP | LazyLock | flag | STATIC kernel rollback | no | MED | - |
| kh/dispatch.rs:472 | V41_MOE_WMMA_DOWN | LazyLock | flag | STATIC kernel rollback | no | MED | - |
| kh/engine.rs:487 | V41_REMOTE_BUSY_POLL_US | uncached at connect | int | STATIC (could go live via setsockopt) | no | MED | - |
| kh/engine.rs:490 | V41_REMOTE_QUICKACK | uncached at connect | flag | STATIC socket option | no | MED | - |
| kh/engine.rs:1086 | DECODE_PREISSUE | LazyLock (legacy forward_token_impl) | flag | STATIC | no | MED | - |
| kh/expert_pager.rs:540 | V41_B1_HOT | LazyLock flag, default on | flag | STATIC: WARM/OWN state persists if turned off | no | MED | - |
| kh/expert_pager.rs:589 | V41_B1_HOT_SLACK | uncached, per refresh | int | LIVE | per refresh | MED | - |
| kh/expert_pager.rs:602 | V41_B1_HOT_MIN_PICKS | uncached, per refresh | int | LIVE | per refresh | MED | - |
| kh/expert_pager.rs:947 | V41_B1_PREFETCH_MIN_TOUCH | LazyLock, default 2 | int | LIVE: per hint (:1206) | no | MED | - |
| kh/expert_pager.rs:771 | V41_VICTIM_CACHE | LazyLock, default on | flag | STATIC; only legacy forward_layer.rs:2613 reads it | no | MED | - |
| kh/expert_pager.rs:778 | V41_PAGER_GPU_REPACK | LazyLock, default on | flag | STATIC: allocates repack | no | MED | - |
| kh/expert_pager.rs:785 | V41_PAGER_BATCH_MISS | LazyLock, default on | flag | STATIC; only legacy forward_layer.rs:2785 reads it | no | MED | - |
| kh/expert_pager.rs:792 | V41_PAGER_READ_THREADS | LazyLock, default 4 | int | STATIC | no | MED | - |
| kh/expert_pager.rs:809 | V41_PAGER_COALESCE | LazyLock | flag | STATIC: sizes stages (:1521-1526) | no | MED | - |
| kh/expert_pager.rs:860 | V41_PAGER_READ_BATCH | LazyLock, default 32 | int | STATIC: sizes par buffers (:1471) | no | MED | - |
| kh/expert_pager.rs:1356 | V41_PAGER_SLOTS | uncached in `new` | int | STATIC sizing | no | MED | - |
| kh/expert_pager.rs:1456 | V41_PAGER_DECODE_FRAC | uncached in `new` | real 0..0.95 | STATIC layout | no | MED | - |
| kh/expert_pager.rs:991 | V41_PREFILL_READAHEAD | LazyLock | flag | STATIC: starts readahead (srv/engine_worker.rs:1241) | no | MED | - |
| kh/expert_pager.rs:1000 | V41_PREFILL_READAHEAD_DEPTH | LazyLock, default 1 | int | STATIC | no | MED | - |
| kh/forward_layer.rs:108 | V41_LOCAL_PICKS | LazyLock (legacy only) | int | STATIC | no | MED | - |
| kh/forward_layer.rs:125 | V41_LOCAL_CLAIM_MAX | LazyLock (legacy only) | int | STATIC | no | MED | - |
| kh/forward_layer.rs:162 | V41_DECODE_PRESUBMIT | LazyLock (legacy only) | flag | STATIC | no | MED | - |
| kh/forward_layer.rs:235,592,2035 | MHC_FUSED | LazyLock + 2 uncached per layer | flag | STATIC kernel path | legacy: per layer | MED | - |
| kh/forward_layer.rs:236 | V41_MHC_SPLIT | LazyLock | flag | STATIC kernel path | no | MED | - |
| kh/forward_layer.rs:237,681,2090,3154 | RMS_NW_MW | LazyLock + 3 uncached per layer | choice | STATIC kernel arm | legacy: per layer | MED | - |
| kh/forward_layer.rs:725,2127,3196 | RMS_W_MW | uncached per layer | choice | STATIC kernel arm | legacy: per layer | MED | - |
| kh/forward_layer.rs:628,695,2047,2102,3166 | F16_KSPLIT | uncached per layer | choice/int | STATIC kernel arm | legacy: per layer | MED | - |
| kh/forward_layer.rs:774 | DECODE_QKV_GRAPH | LazyLock | flag | STATIC (graph capture) | no | MED | - |
| kh/forward_layer.rs:1464 | DECODE_INDEXER | uncached per layer | choice | STATIC kernel arm | legacy: per layer | MED | - |
| kh/forward_layer.rs:1536 | INDEXER_DECODE | LazyLock | choice | STATIC kernel arm | no | MED | - |
| kh/forward_layer.rs:1884 | DECODE_SCORE | uncached per layer | choice | STATIC kernel arm | legacy: per layer | MED | - |
| kh/forward_layer.rs:1919 | DECODE_SMWSUM | uncached per layer | choice | STATIC kernel arm | legacy: per layer | MED | - |
| kh/forward_prefill.rs:57 (+ weights.rs:1405) | DGPU_HOT_PREFILL | LazyLock + uncached validate | flag | STATIC | no | MED | - |
| kh/forward_prefill.rs:201,203 | DGPU_HOT_CAP_PREFILL | LazyLock | int | STATIC | no | MED | - |
| kh/forward_prefill.rs:93 | V41_REPLAY_OFFLOAD | LazyLock | flag | STATIC: box-2 capacity precondition | no | MED | - |
| kh/forward_prefill.rs:377 | V41_ENGRAM_GEMV | uncached | flag | STATIC rollback | YES: per prefill engram pass (:4646) | MED | - |
| kh/forward_prefill.rs:389 | V41_SWA_MIXED | uncached | flag | STATIC rollback | YES: per ratio-0 attention (:6383) | MED | - |
| kh/forward_prefill.rs:461 | V41_PREFILL_PRESUBMIT | LazyLock | flag | STATIC | no | MED | - |
| kh/forward_prefill.rs:471 | V41_RB_PACK | LazyLock, default on | flag | STATIC rollback | no | MED | - |
| kh/forward_prefill.rs:484 | V41_RB_PACK_MAX_ROWS | LazyLock, default 16 | int | STATIC | no | MED | - |
| kh/forward_prefill.rs:490,515 | V41_PREFILL_F32_MATVEC | uncached, 2 parsers | choice auto/0/1 | STATIC kernel arm | YES: 9 sites per lane-layer | MED | - |
| kh/forward_prefill.rs:514 | V41_REPLAY_F16X | LazyLock, default on | flag | STATIC kernel arm | no | MED | - |
| kh/forward_prefill.rs:534 | V41_ENGRAM_CHUNK128 | LazyLock, default on | flag | STATIC: sizes engram scratch | no | MED | - |
| kh/forward_prefill.rs:561 | V41_MHC_PRE_SCALED | uncached | choice auto/0/1 | STATIC kernel arm | YES: 5 sites per lane-layer | MED | - |
| kh/forward_prefill.rs:569 | V41_MHC_NARROW | uncached | choice auto/0/1 | STATIC kernel arm | YES: 5 sites per lane-layer | MED | - |
| kh/forward_prefill.rs:737 | V41_LM_ROWS | OnceLock, default 4096 | int | STATIC: device store capacity | no | MED | - |
| kh/forward_prefill.rs:1538 | V41_LOOKAHEAD_DEPTH | LazyLock, default 2 | int 1..2 | STATIC | no | MED | - |
| kh/forward_prefill.rs:1545 | V41_LOOKAHEAD_TOPK | LazyLock, default 5 | int 1..8 | STATIC | no | MED | - |
| kh/forward_prefill.rs:1584 | V41_MHC_ARENA_FUSED | LazyLock, default on | flag | STATIC kernel rollback | no | MED | - |
| kh/forward_prefill.rs:1606 | V41_MHC_FAST | LazyLock, default on | flag | STATIC kernel rollback | no | MED | - |
| kh/forward_prefill.rs:1632 | V41_SHARED_FUSED | LazyLock, default on | flag | STATIC kernel rollback | no | MED | - |
| kh/forward_prefill.rs:1645 | V41_ATTN_META_FILL | LazyLock, default on | flag | STATIC kernel rollback | no | MED | - |
| kh/forward_prefill.rs:1660 | V41_ATTN_DEC_SCORE | LazyLock, default on | flag | STATIC kernel rollback | no | MED | - |
| kh/forward_prefill.rs:1679 | V41_ATTN_DEC_FUSED | LazyLock, default on | flag | STATIC kernel rollback | no | MED | - |
| kh/forward_prefill.rs:1693 | V41_ATTN_DEC_SCORE_BLK128 | LazyLock, default on | flag | STATIC kernel rollback | no | MED | - |
| kh/forward_prefill.rs:1729 | V41_DEC_SKIP_DEAD | LazyLock, default on | flag | STATIC kernel rollback | no | MED | - |
| kh/forward_prefill.rs:1748 | V41_DEC_FUSE | LazyLock, default on | flag | STATIC kernel rollback | no | MED | - |
| kh/forward_prefill.rs:1764 | V41_MHC_FFN_LATE | LazyLock, default on | flag | STATIC kernel rollback | no | MED | - |
| kh/forward_prefill.rs:1776 | V41_MOE_WI_DEVCOUNT | LazyLock, default on | flag | STATIC kernel rollback | no | MED | - |
| kh/forward_prefill.rs:1795 | V41_ROUTER_ALTS | LazyLock | int | STATIC: alternative buffers | no | MED | - |
| kh/forward_prefill.rs:1831 | V41_LOOKAHEAD_PREFETCH | LazyLock | flag | STATIC | no | MED | - |
| kh/forward_prefill.rs:1837 | V41_MS_GRAPHS | LazyLock, default on | flag | STATIC: graph capture | no | MED | - |
| kh/forward_prefill.rs:4982 | QB_WMMA | uncached, allocates its default | choice dp4a/f16x | STATIC kernel arm | YES: per lane-layer | MED | - |
| kh/forward_prefill.rs:5453 | DEEPSTRIX_COMP_GEMM | uncached | flag | STATIC | YES: compressor path | MED | - |
| kh/forward_prefill.rs:5527 | DEEPSTRIX_COMP_TILED | uncached | flag | STATIC rollback | YES: compressor path | MED | - |
| kh/forward_prefill.rs:5613 | DEEPSTRIX_COMP_GATHER | uncached | flag | STATIC rollback | YES: compressor path | MED | - |
| kh/forward_prefill.rs:6731,6739 | INDEXER_SCORE_VARIANT | LazyLock | choice | STATIC kernel arm | no | MED | - |
| kh/forward_prefill.rs:6887 | INDEXER_TOPK_SELECT | uncached | flag | STATIC rollback | YES: per indexer layer | MED | - |
| kh/forward_prefill.rs:7036 | ATTN_FUSED | uncached `var_os` | flag | STATIC kernel arm | YES: per attention | MED | - |
| kh/forward_prefill.rs:7399 | Q8_GROUPED_VARIANT | uncached, allocates its default | choice dp4a/f16x | STATIC kernel arm | YES: per lane-layer | MED | - |
| kh/forward_prefill.rs:7489 | Q8_OUT_VARIANT | uncached, allocates its default | choice dp4a/f16x | STATIC kernel arm | YES: per lane-layer | MED | - |
| kh/forward_prefill.rs:7781 | V41_ROUTER_WMMA | uncached | choice auto/0/1 | STATIC kernel arm | YES: per lane-layer | MED | - |
| kh/forward_prefill.rs:8899 | V41_SMALL_B_CATCHALL_HALF | uncached | choice 0/1/2 | LIVE-capable policy | YES: catch-all branch | MED | - |
| kh/forward_prefill.rs:9407 | V41_PAGER_SYNC_IGPU | uncached | flag | STATIC race guard | YES: when ensure may evict | MED | - |
| kh/forward_prefill.rs:9689,9955 | IQ2_VARIANT | uncached x2, allocates its default | choice | STATIC kernel arm | YES: 2 per lane-layer | MED | - |
| kh/forward_prefill.rs:10174 | IQ2_HYBRID_THRESHOLD | uncached | int | STATIC kernel arm | YES: per lane-layer | MED | - |
| kh/forward_prefill.rs:10489 | Q2K_VARIANT | uncached | choice | STATIC kernel arm | only with a Q2_K down weight | MED | - |
| kh/forward_prefill.rs:11359 | V41_PREFILL_SINGLE_LANE_MAX | atomic seeded from env + setter | int | LIVE: built runtime-settable (:11348-11370) | no | MED | - |
| kh/forward_prefill.rs:11433 | V41_PREFILL_SCAN_SLOTS | LazyLock | int | STATIC | no | MED | - |
| kh/forward_prefill.rs:11459 | V41_SMALL_B_CATCHALL_DET | OnceLock | flag | STATIC | no | MED | - |
| kh/forward_prefill.rs:11467 | V41_COMP_POSITIONAL | LazyLock | int | STATIC | no | MED | - |
| kh/mtp.rs:429 | V41_MTP_KV_QUANT | OnceLock | choice | STATIC: drafter KV layout | no | MED | - |
| kh/mtp.rs:468 | V41_MTP_GRAPH | OnceLock | flag | STATIC: graph capture | no | MED | - |
| kh/remote_experts.rs:1708 | V41_B2_SPEC_CHUNK_KB | uncached, at reader start (:4210), stored in io_throttle | int | LIVE-capable through a hook: `io_throttle::set_chunk_bytes` is an atomic store (core/io_throttle.rs:41) | no | MED | - |
| kh/remote_experts.rs:2125 | V41_B2_PREFETCH_PAR | uncached, at reader start (:4207,:4222) | int 1..16 | STATIC: reader threads | no | MED | - |
| kh/remote_experts.rs:2140 | V41_B2_PREFETCH_RESERVE | LazyLock, default 1 | int 0..8 | STATIC: PfQueue readers (:4208); also read per word (:4337) | no | MED | - |
| kh/remote_experts.rs:2643 | V41_B2_PIN_RESERVE | LazyLock Option | int | STATIC: pin budget per connection | no | MED | - |
| kh/remote_experts.rs:3434 | V41_B2_GLOBAL_POOL | LazyLock, default on | flag | STATIC | no | MED | - |
| kh/remote_experts.rs:3474 | V41_B2_POOL_FLOOR | LazyLock, default 0 | real 0..1 | STATIC: `ShardPool::seeded` | no | MED | - |
| kh/remote_experts.rs:3494 | V41_B2_GPU_REPACK | LazyLock, default on | flag | STATIC: allocates repack | no | MED | - |
| kh/remote_experts.rs:3530 | V41_REMOTE_BATCHED_MULTI | LazyLock, default on (hub side) | flag | LIVE-capable: per-submit flag (:8445) | no | MED | - |
| kh/remote_experts.rs:3540 | V41_REMOTE_BATCHED_B1 | LazyLock, default on (hub side) | flag | LIVE-capable: per submit (~0.3% numerics shift) | no | MED | - |
| kh/remote_experts.rs:3816 | V41_B2_SCAN_CLASS | LazyLock, default on | flag | STATIC | no | MED | - |
| kh/remote_experts.rs:3826 | V41_B2_ODIRECT | LazyLock, default on | flag | STATIC: staging alignment at alloc | no | MED | - |
| kh/state.rs:59 | COMP_KV_FP8 | uncached at state alloc (:208) | flag | STATIC: KV layout | no | MED | - |
| kh/state.rs:68 | INDEXER_KEYS_E2M1 | uncached at state alloc (:213) | flag | STATIC: KV layout | no | MED | - |
| kh/state.rs:440 | V41_COMP_ROLLBACK | OnceLock, default on | flag | STATIC | no | MED | - |
| kh/weights.rs:976 | DEEPSTRIX_FAST_EXPERT_LOAD | LazyLock | int | STATIC loader | no | MED | - |
| kh/weights.rs:1002 | DEEPSTRIX_EXPERT_READ_THREADS | LazyLock | int | STATIC loader | no | MED | - |
| kh/weights.rs:1304 | IGPU_DEDUP_HOT | uncached (load + init) | flag | STATIC | no | MED | - |
| kh/weights.rs:1336 (+1393, forward_prefill.rs:203) | DGPU_HOT_CAP | LazyLock + uncached validate | int | STATIC | no | MED | - |
| kh/weights.rs:1432 (+ srv/engine_worker.rs:960-964 `set_var`) | DGPU_HOT_EXPERTS_FILE | uncached at load; the server writes it with `set_var` | text | STATIC | no | MED | - |
| kh/weights.rs:1459 | DGPU_HOT_EXPERTS | uncached at load | int | STATIC | no | MED | - |
| core/engram_table.rs:62 | ENGRAM_CACHE_ROWS | uncached at open | int | STATIC sizing | no | MED | - |
| core/hf_v41.rs:170 | V41_EXPERT_ODIRECT | LazyLock | flag | STATIC | no | MED | - |
| core/hf_v41.rs:241 | V41_EXPERT_PREAD_THREADS | LazyLock, default 8 | int | STATIC | no | MED | - |
| core/hf_v41.rs:264 | DEEPSTRIX_HF_THREADS | uncached at open | int | STATIC | no | MED | - |
| srv/engine_worker.rs:1117 | V41_MS_LANE_C_ROWS | uncached at startup | int | STATIC: sizes lane-C scratch | no | MED | - |
| srv/engine_worker.rs:1258 | V41_ENGRAM_DIR | uncached at startup | text | STATIC | no | MED | - |
| srv/engine_worker.rs:3068,5213 | V41_DSPARK_SEED_RING | uncached (legacy) | flag | STATIC | legacy: per request | MED | - |
| srv/engine_worker.rs:3698 | V41_DSPARK_K | uncached (legacy verify loop) | int | LIVE-capable | legacy: per verify step | MED | - |
| srv/engine_worker.rs:3709 | V41_DSPARK_CONF_MIN | uncached (legacy verify loop) | real | LIVE-capable | legacy: per verify step | MED | - |
| srv/engine_worker.rs:4271 | V41_DSPARK_RING_FAST | uncached (legacy) | flag | STATIC rollback | legacy: per verify step | MED | - |
| srv/engine_worker.rs:4272 | V41_DSPARK_DENSE_RING | uncached (legacy) | flag | STATIC rollback | legacy: per verify step | MED | - |
| srv/main.rs:153 | DEEPSTRIX_MMPROJ | uncached at startup | text | STATIC deploy config (could stay env or move to CLI) | no | MED | - |
| srv/main.rs:158 | DEEPSTRIX_ALLOW_IMAGE_DIRS | uncached at startup | text | STATIC deploy config | no | MED | - |
| srv/multistream.rs:1119 (+ engine_worker.rs:2356) | DEEPSTRIX_SNAPSHOT_REUSE | uncached | flag | LIVE-capable: per-request lookup decision | per prefill start | MED | - |
| srv/snapshot.rs:373 | DEEPSTRIX_SNAPSHOT_KEEP_PER_SESSION | uncached at store construction | int | STATIC (could be live: eviction policy) | no | MED | - |
| crates/deepstrix-expertd/src/main.rs:87 | V41_MODEL | uncached at startup (also bench.rs:143) | text | STATIC deploy config | no | MED | - |
| k/attention.rs:308 | DEEPSTRIX_ATTN_LEGACY_STRIDE | uncached | flag | STATIC: scratch sizing (batch_scratch.rs:1340) | YES: per attention launch (forward_prefill.rs:6390,7044) | LOW | - |
| kh/b2_mirror.rs:520 | V41_B2_ASSERT_NO_SURPRISE | LazyLock | flag | STATIC (verification runs) | no | LOW | - |
| kh/engine.rs:841 | V41_DSPARK_DRAFT_TIMING | uncached | flag | LIVE diagnostic | YES: per drafted step (ms_dspark.rs:515) | LOW | - |
| kh/engine.rs:1315,1448 | DEEPSTRIX_EXPERT_STATS | LazyLock + uncached per 64 tokens (legacy) | text | STATIC | legacy | LOW | - |
| kh/engine.rs:1351,1369 | DEEPSTRIX_EXPERT_TRACE | LazyLock + uncached per 64 tokens (legacy) | text | STATIC | legacy | LOW | - |
| kh/engine.rs:1515 | V41_DECODE_LOGITS_DUMP | uncached (legacy) | text | STATIC | legacy: per token | LOW | - |
| kh/engine.rs:1788 | DEEPSTRIX_SEL_STATS | LazyLock | flag | STATIC | no | LOW | - |
| kh/engine.rs:2088 | DEEPSTRIX_SUBSTITUTE_RESIDUAL | OnceLock | text | STATIC debug | no | LOW | - |
| kh/engine.rs:2146 | DEEPSTRIX_DUMP_RESIDUAL_DIR | OnceLock | text | STATIC debug | no | LOW | - |
| kh/engine.rs:2192,2194,2195 | DEEPSTRIX_DUMP_SUBTENSOR_LAYERS / _LAYER / _DIR | OnceLock | text | STATIC debug | no | LOW | - |
| kh/evtrace.rs:409 | V41_EVTRACE_MAX_MB | uncached at init | int | STATIC | no | LOW | - |
| kh/evtrace.rs:410 | V41_EVTRACE_KEEP | uncached at init | int | STATIC | no | LOW | - |
| kh/evtrace.rs:424 | V41_EVTRACE_SYS_MS | uncached at init | int | STATIC: sampler thread | no | LOW | - |
| kh/evtrace_ring.rs:265 | V41_EVTRACE_RING_MB | uncached at init | int | STATIC: ring size | no | LOW | - |
| kh/evtrace_ring.rs:270 | V41_EVTRACE_RING_DIR | uncached at init | text | STATIC | no | LOW | - |
| kh/evtrace_ring.rs:282 | V41_EVTRACE_RING_KEEP_MB | uncached at init | int | STATIC | no | LOW | - |
| kh/expert_pager.rs:606 | V41_B1_HOT_MASS | uncached, per refresh | real | LIVE, but only reported, never enforced (:639) | per refresh | LOW | - |
| kh/expert_pager.rs:821 | V41_PAGER_COALESCE_CHECK | LazyLock | flag | STATIC diagnostic | no | LOW | - |
| kh/forward_layer.rs:257 | V41_ATTN_COMP_CAP | LazyLock | int | STATIC ablation (output invalid) | no | LOW | - |
| kh/forward_layer.rs:1716 | V41_INDEXER_DBG | uncached | flag | LIVE diagnostic | legacy: per layer | LOW | - |
| kh/forward_layer.rs:1741 | V41_INDEXER_DUMP | uncached | text | STATIC diagnostic | legacy: per layer | LOW | - |
| kh/forward_layer.rs:2717 | V41_REMOTE_NOMASK | uncached | flag | STATIC rollback of a correctness fix | legacy: per layer | LOW | - |
| kh/forward_layer.rs:2814,3086 | V41_PAGER_DBG | uncached | flag | LIVE diagnostic | legacy: per layer | LOW | - |
| kh/forward_layer.rs:2846 | V41_PAGER_RESIDENT | uncached | flag | STATIC diagnostic | legacy: per layer | LOW | - |
| kh/forward_layer.rs:2871 | V41_PAGER_NOGRAPH | uncached | flag | STATIC diagnostic | legacy: per layer | LOW | - |
| kh/forward_prefill.rs:868 | V41_PREFILL_LOGITS_DUMP | uncached | text | STATIC diagnostic | per call | LOW | - |
| kh/forward_prefill.rs:1711 | V41_KV_F16_ROUNDTRIP | LazyLock | flag | STATIC ablation | no | LOW | - |
| kh/forward_prefill.rs:2370 | V41_PREFILL_LANE_DEBUG | uncached | flag | LIVE diagnostic | per pipelined chunk | LOW | - |
| kh/forward_prefill.rs:5314 (+ forward_layer.rs:1366,1870) | V41_WINDOW_DBG | uncached | flag | LIVE diagnostic | YES: SWA layers | LOW | - |
| kh/forward_prefill.rs:6069 | V41_COMP_POS_DBG | uncached | flag | LIVE diagnostic | only on mismatch | LOW | - |
| kh/forward_prefill.rs:6320 (+ forward_layer.rs:1457) | V41_INDEXER_FORCE | uncached | flag | STATIC diagnostic | YES: per lane-layer | LOW | - |
| kh/forward_prefill.rs:8400,8573,9054,9080,9367,9412,9573,10116 | V41_GROUP_AUDIT_VERBOSE | uncached x8 | flag | LIVE diagnostic | YES: about 5 unconditional per lane-layer | LOW | - |
| kh/forward_prefill.rs:9038,9101,9193,10906,10967,11012 | V41_REMOTE_DBG | uncached x6 | flag | LIVE diagnostic | YES: per remote lane-layer | LOW | - |
| kh/forward_prefill.rs:9451 | V41_SPARSE_REMAP_SYNC | uncached | flag | STATIC diagnostic | YES | LOW | - |
| kh/forward_prefill.rs:9494 | V41_PAGER_SYNC_AFTER_ENSURE | uncached | flag | STATIC diagnostic | YES | LOW | - |
| kh/forward_prefill.rs:11183 | V41_LAYER_HOST_TIMING | LazyLock | flag | STATIC | no | LOW | - |
| kh/forward_prefill.rs:11443 | V41_SPARSE_VERIFY_RESIDENCY | LazyLock | flag | STATIC | no | LOW | - |
| kh/forward_prefill.rs:11451 | V41_GROUP_AUDIT | OnceLock | flag | STATIC | no | LOW | - |
| kh/forward_prefill.rs:11481 | V41_LAYER_MISS_HIST | OnceLock | flag | STATIC | no | LOW | - |
| kh/forward_prefill.rs:11525 | V41_VERIFY_DECODE_MOE | OnceLock | flag | STATIC parity diagnostic | no | LOW | - |
| kh/forward_prefill.rs:11532 | V41_VERIFY_DECODE_ATTN | OnceLock | flag | STATIC parity diagnostic | no | LOW | - |
| kh/mtp.rs:126 | V41_DSPARK_LAYER_TIMING | OnceLock | choice 0/1/2 | STATIC | no | LOW | - |
| kh/mtp.rs:213 | V41_SLACK_PROBE | OnceLock | text | STATIC | no | LOW | - |
| kh/mtp.rs:404 | V41_DSPARK_NO_ATTN | OnceLock | flag | STATIC ablation | no | LOW | - |
| kh/mtp.rs:408 | V41_DSPARK_DEBUG_MOE | OnceLock | flag | STATIC | no | LOW | - |
| kh/mtp.rs:414 | V41_DSPARK_NO_ROUTED | OnceLock | flag | STATIC ablation | no | LOW | - |
| kh/probe_dump.rs:55 | V41_PROBE_DUMP | OnceLock | text | STATIC | no | LOW | - |
| kh/probe_dump.rs:67 | V41_PROBE_SRC | OnceLock | text | STATIC | no | LOW | - |
| kh/remote_experts.rs:2665 | V41_B2_ASSERT_PINNED | LazyLock | flag | STATIC (verification runs) | no | LOW | - |
| kh/remote_experts.rs:3657 | V41_B2_COALESCE_CHECK | LazyLock | flag | STATIC diagnostic | no | LOW | - |
| kh/remote_experts.rs:3796 | V41_B2_DECODE_DOWN | LazyLock | flag | LIVE-capable: per request path | no | LOW | - |
| kh/remote_experts.rs:7255 | V41_B2_PARK_LOG | LazyLock | flag | LIVE diagnostic | no | LOW | - |
| kh/remote_experts.rs:7495 | V41_B2_DBG | LazyLock | flag | LIVE diagnostic | no | LOW | - |
| kh/remote_experts.rs:8475,8483 | V41_MASK_DBG | uncached (hub submit) | flag | LIVE diagnostic | only when picks are masked | LOW | - |
| kh/route_probe.rs:40 | V41_ROUTE_PROBE | OnceLock | text | STATIC | no | LOW | - |
| kh/state.rs:746 | V41_RESET_ZERO | uncached | flag | LIVE diagnostic | per `reset_in_place` (per request) | LOW | - |
| kh/trace.rs:679 | V41_PROFILE_KERNEL_STAGES | LazyLock | flag | STATIC | no | LOW | - |
| kh/trace.rs:688 | DEEPSTRIX_TOKEN_PROFILE | LazyLock | flag | STATIC | no | LOW | - |
| kh/trace.rs:878 | DEEPSTRIX_PREFILL_PROFILE | LazyLock | flag | STATIC | no | LOW | - |
| kh/weights.rs:725 | DEEPSTRIX_BIAS_VL_FILE | uncached (V4-Flash GGUF vision sidecar) | text | STATIC | no | LOW | - |
| kh/weights.rs:969 | DEEPSTRIX_EXPERT_LOAD_PROFILE | LazyLock | flag | STATIC | no | LOW | - |
| crates/v4flash-hip/src/buffer.rs:44 | DEEPSTRIX_ALLOC_TRACE | LazyLock | flag | STATIC | no | LOW | - |
| crates/v4flash-vision/src/kernels.rs:51 | VIT_GEMM | uncached at load | choice | STATIC | no | LOW | - |
| crates/v4flash-vision/src/tower.rs:340 | VIT_PROFILE | uncached per tower construction | flag | STATIC | no | LOW | - |
| srv/engine_worker.rs:287 | V41_SINGLE_LANE_AB | OnceLock (legacy A/B harness) | int | STATIC | no | LOW | - |
| srv/engine_worker.rs:294 | V41_XCHECK_POISON | OnceLock | flag | STATIC | no | LOW | - |
| srv/engine_worker.rs:302 | V41_SMALL_B_CATCHALL_AB | OnceLock (legacy A/B harness) | int | STATIC | no | LOW | - |
| srv/engine_worker.rs:1042 | V41_PERFETTO_OUT | uncached at startup | text | STATIC | no | LOW | - |
| srv/engine_worker.rs:1882 | V41_MISS_HIST_EVERY | uncached (legacy worker loop) | int | LIVE | legacy: per request | LOW | - |
| srv/engine_worker.rs:2140 | DGPU_HOT_ALPHA | uncached per stats flush | real | LIVE | per flush | LOW | - |
| srv/engine_worker.rs:3115 | V41_DUMP_FIRST_LOGITS | uncached (legacy) | text | STATIC | legacy: per request | LOW | - |
| srv/engine_worker.rs:3152 | DEEPSTRIX_TRACE_TOKENS | uncached (legacy) | flag | LIVE diagnostic | legacy: per request | LOW | - |
| srv/engine_worker.rs:3170 | V41_VERIFY_PROBE | uncached (legacy) | text | STATIC | legacy: per request | LOW | - |
| srv/engine_worker.rs:3187 | V41_VERIFY_DECODE_PATH | OnceLock | flag | STATIC | no | LOW | - |
| srv/engine_worker.rs:3191 | V41_VERIFY_BATCHED | uncached (legacy) | flag | STATIC | legacy: per request | LOW | - |
| srv/engine_worker.rs:3192 | DEEPSTRIX_HEARTBEAT_TOKENS | uncached (legacy) | int | LIVE | legacy: per request | LOW | - |
| srv/engine_worker.rs:3224 | V41_PROBE_FPRINT | uncached (legacy) | flag | STATIC | legacy: per request | LOW | - |
| srv/engine_worker.rs:3987 | V41_DUMP_VERIFY_DIST | uncached (legacy) | flag | STATIC | legacy: per verify step | LOW | - |
| srv/engine_worker.rs:4019 | V41_DSPARK_FORCE_N0 | uncached (legacy) | flag | STATIC | legacy: per verify step | LOW | - |
| srv/engine_worker.rs:4314 | V41_DSPARK_STEP_TIMING | uncached (legacy) | flag | LIVE diagnostic | legacy: per verify step | LOW | - |
| srv/engine_worker.rs:5697,5702 | V41_XCHECK_ROWS | two OnceLocks (`=="2"`, `=="1"`) | choice | STATIC | no | LOW | - |
| srv/main.rs:272 | V41_EXIT_GRACE_S | uncached at shutdown | int | STATIC | no | LOW | - |
| k/laguna_het.rs:71,443,626,628,635,645,646,647,822,825,826,830,1003,1691,1697,1728,1730,1731,1732,1741,1747,1761,1779,1882,1916,1931,1987,2237,2245,2250,2338,2488,2493,2568 | LAGUNA_PROJ_LDS_TILED, LAGUNA_HOT_EXPERTS_DGPU, LAGUNA_FP8_KV, LAGUNA_FP8_FAKE, LAGUNA_FP8_FAKE_BLK, LAGUNA_FP8_FAKE_FMT_K, LAGUNA_FP8_FAKE_FMT_V, LAGUNA_FP8_FAKE_LAYERS, LAGUNA_HET_DIAG, LAGUNA_SHEXP_DGPU, LAGUNA_EXPERT_HIST, LAGUNA_PREFILL_HOT_CAP, LAGUNA_SWA_OFF (1003,1691,1728), LAGUNA_ATTN_HG (1697,1761), LAGUNA_ATTN_WMMA, LAGUNA_ATTN_FLASH, LAGUNA_ATTN_NAIVE, LAGUNA_ATTN_WMMA_LEGACY, LAGUNA_ATTN_HG_PACKED, LAGUNA_ATTN_KVFIRST, LAGUNA_GATEUP, LAGUNA_DOWN_PART, LAGUNA_DOWN, LAGUNA_SHEXP, LAGUNA_PREFILL_HET, LAGUNA_PIPELINE (2245,2488), LAGUNA_PREFILL_BMAX (2250,2338,2493,2568) | uncached except :71 (LazyLock); load-time reads in `load` (443-830) | flag / choice / int | STATIC: kernel arms, load config | YES: `layer`, `attn_batched`, `moe_batched` read per layer | MED (Laguna, not V4.1 prod) | - |
| k/laguna_moe_tiled.rs:227,242,277,280,282,405,406,408,498,549 | LAGUNA_GU_ABL, LAGUNA_GU_RPW, LAGUNA_DOWN_LDS, LAGUNA_DOWN_ROWS, LAGUNA_DOWN_ABL, LAGUNA_DENSE_WIDE | uncached | int / flag | STATIC launch geometry | YES: per kernel launch | LOW | - |
| k/laguna.rs:166,732 | LAGUNA_ROUTER_WRO, LAGUNA_DECODE_GRAPH | uncached | flag | STATIC | :166 per router call | LOW | - |
| k/gqa_attention.rs:49,69,94,116,131,151 | LAGUNA_HG_G, LAGUNA_DECODE_ATTN_NAIVE, LAGUNA_DECODE_ATTN, LAGUNA_DECODE_FLASH_MIN_KV, LAGUNA_DECODE_KV_SPLITS (131,151) | :49 uncached, the rest LazyLock | int / choice | STATIC | :49 per flash launch | LOW | - |

Framework bootstrap, left as env on purpose and not counted:
- `V41_KNOBS_FILE` (`k/knobs.rs:377`)
- `V41_B2_KNOBS` (`kh/remote_experts.rs:2081`, b2 set)

## Test-only env vars

178 names appear only in `crates/*/tests`, mostly `BENCH_*`, `WMMA_*`,
`MS_*`, `LAGUNA_*`, `GOLDEN_*`, `PARITY_*`, `FP8_AB_*`, `V41_FIXTURE_*`,
`V41_LAYER_MAJOR*`, `V41_HF_*`, `V41_REQUIRE_FIXTURES`, `TOKENIZER_JSON`.

`#[cfg(test)]` blocks in src also read some:
- `V41_REQUIRE_FIXTURES`, `V41_MODEL`, `ENGRAM_GATHER_THREADS` (`core/engram_table.rs:215-263`, `core/engram_hash.rs:236`)
- `TOKENIZER_JSON`, `SYNTH_ROUNDS` (`srv/prompt_v41.rs:843-871`, `core/tokenizer.rs:1080`)
- `DEEPSTRIX_IQ3S_BLOB` (`core/iq3_s_ref.rs:421`)

Some src tests read knob names as raw env, but only to skip or verify:
- `srv/multistream.rs:2595`
- `kh/b2_mirror.rs:2563`
- `kh/weights.rs:1698`

`kh/expert_pager.rs:691-715` sets `V41_B1_HOT_*` with `set_var`. Once those
settings become knobs, that test has to switch to `Knob::set`.
