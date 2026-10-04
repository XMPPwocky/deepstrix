# Tuning reference — every knob touched or ruled out
### First written 2026-09-14; corrected against the code and the live hub on 2026-10-04 (docs audit). Host settings are persisted in the NixOS flake; `scripts/apply_host_tuning.sh` is the fallback / verifier.

**Read §0 first.** Most "defaults" this doc used to quote were values the launch
script sets, not code defaults. The code reads ~340 `V41_*` names; how a knob is
resolved (default < env < legacy one-value file < `V41_KNOBS_FILE`, live vs
static, `<knob file>.effective`) is documented at the top of
`crates/v4flash-kernels/src/knobs.rs`. The full inventory of knobs still read
outside that framework is `KNOB_AUDIT_2026-10-04.md`.

## 0. Production vs code default (hub process environment, 2026-10-04)

Captured read-only from the running hub (`/proc/<pid>/environ`), launched by
`~/run_v41_server.sh` (not in the repo) plus `~/scratch-ms/` env files. Live
knob file `/dev/shm/deepstrix-knobs.txt` at the same time: `V41_LM_PREFETCH=1`,
`V41_SUB_LAMBDA=0.15`, `V41_SUB_DEFER_ACCEPTED=0`, `V41_MS_DSPARK_STREAMS=1`
(the file wins over the env for live knobs, so `SUB_LAMBDA` is 0.15, not the
env's 0.25). Box 2's environment is not captured here; its 10-03 launch line is
in §3. Only knobs whose production value differs from the code default, plus the
ones that define the topology:

| knob | code default | production | what it does |
|---|---|---|---|
| `V41_PAGED_EXPERTS` | 0 | 1 | LRU-paged routed experts (the only supported V4.1 mode) |
| `V41_REMOTE_ADDR` | unset | `10.99.0.2:7431` | box 2 over USB4 / `thunderbolt0` |
| `V41_REMOTE_SPLIT` / `_DECODE` | off / off | 1 / 1 | two-box MoE split in prefill / decode |
| `V41_T2_CATCHALL` | 0 | 1 | box 2 is a catch-all LRU tier; needs expertd `--paged`. `=2` = constant split (§3) |
| `V41_T2_PARTITION` | 0 | 1 | expert ids split between the boxes; with the hot set warm, box 1 owns each layer's hottest ids |
| `V41_PARTITION_BOX1_SHARE` | from slot counts | 0.15 | hash share before the hot set warms |
| `V41_B1_HOT_PER_LAYER` / `_HYST` / `_MAX_CHANGE` | 90 / 40 / 0 (unlimited) | 103 / 100 / 3 | live hot-set size, rank hysteresis, newcomers per layer per refresh |
| `V41_PAGER_POOL_GB` | 60 | 95 | box 1 expert pool (raised with the 128 GB machine, 2026-10-03) |
| `V41_PAGER_WINDOWS` / `V41_PAGER_STRIDE` | derived / 384 | 0 (= 1) / 384 | see §3 |
| `V41_PAGER_UNION` | on | 1 | — |
| `V41_PAGER_MISS_THREADS` / `_MISS_PAR` | 1 / 4 | 3 / 8 | box 1 miss readers |
| `V41_PREFILL_UNIFIED_POOL` | 0 | 1 | prefill pages through the decode LRU (no phase partition) |
| `V41_B1_PAGE_MISSES` / `V41_B1_PREFETCH` | 0 / 0 | 1 / 1 | box 1 pages its own misses / prefetches; `_PREFETCH_ADMIT` 8 -> 24 |
| `V41_INDEX_K` / `V41_CANDIDATE_POOL` | off / off | 1 / 1 | CSA2 sparse indexer + candidate pool (S1-S3). Correctness-relevant at long context |
| `V41_CED` | on | 1 | — |
| `V41_LM_PREFILL` | off | 1 | layer-major prefill windows |
| `V41_LM_PREFETCH` | off | 1 (knob file) | box 2 staging prefetch for layer-major prefill |
| `V41_MULTISTREAM` | off | 1 | the arena scheduler (`multistream.rs`); everything `V41_MS_*` needs it |
| `V41_MS_PREFILL_JOBS` | 2 | 1 | prefill jobs in flight |
| `V41_MS_CTX_ROWS` | twice the model's KV maximum | 1300000 | arena rows per stream |
| `V41_MS_STAGGER` | 0 | 2 (ready-first) | lane driver; two-lane DSpark verify needs 2 |
| `V41_MS_PIPELINE_MIN_ROWS` | 6 | 4 | two lanes from this many rows |
| `V41_MS_LANES_LEARNED` | off | 1 | pick one or two lanes per row count from live cost cells |
| `V41_MS_DSPARK` | off | accept | arena DSpark |
| `V41_MS_DSPARK_STREAMS` | 1 | 1 | streams that may speculate at once (max 2; `MS_DSPARK_STREAMS_DESIGN.md`) |
| `V41_MS_DSPARK_RING` | all | solo | drafter ring writes only while speculation is possible |
| `V41_MS_DSPARK_DRAFT_MS` / `_COST` | 20 / unset | 12 / 8-point one-lane ladder (ms for 1..8 rows) | starting estimates for the live cost cells (`_COST_LIVE`, default on) |
| `V41_MTP_MOE_GROUPED` / `V41_MTP_EXPERT_STATS` | 0 / 0 | 1 / 1 | drafter MoE grouping / stats |
| `V41_SUB` | 0 | 3 | cache-prior router bias (`BOX2_MISS_SUBSTITUTION.md`) |
| `V41_SUB_LAMBDA` | 0.1 | 0.15 (knob file) | cache-prior strength |
| `V41_SUB_PROTECT` / `_ADMIT_GATE` / `_PENDING` | 2 / off / on | 1 / 1 / 0 | — |
| `V41_B2_PIN` / `V41_B2_PIN_PREFILL_BAND` | off / 2048 | 1 / 4096 | box-2 pinning |
| `V41_REMOTE_PARTIAL_ASYNC` | off | 1 | async upload of box 2's partial (`HOT_SPLIT_SIM.md`) |
| `V41_PUSH_XQ` | 0 | 1 | Q8_K push dGPU -> iGPU |
| `V41_SMALL_B_CATCHALL_MAX` | 0 | 8 | — |
| `V41_DECODE_BUSY_POLL_US` | 3000 | 5000 | hub socket spin in decode (§1) |
| `V41_EVTRACE_DEV` | **on** | **0** | Tier B device timing. Off since 2026-10-02 after a measured decode regression (two-lane DSpark ~+100 ms), cause not isolated. A launch without the env/knob file turns it back on |
| `V41_DSPARK` | 0 | 0 | legacy serial DSpark driver (arena DSpark replaces it) |
| `V41_MS_MHC_SPLIT` | 0 | 0 | measured neutral |
| `DEEPSTRIX_HANG_DEADLINE_MS` | CLI flag | 120000 | forward-progress watchdog |

Not set in production, but worth knowing: `V41_PREFILL_F16_REPLIES` is **on by
default** — box 2's prefill partials travel as f16. **No KLD / golden gate has
been run on that** (the code logs `UNTESTED FIDELITY`); the gate is still owed
(owner, 2026-10-04).

## 1. Host settings — both boxes, PERSISTED in the NixOS flake

**Persisted (corrected 2026-10-04):** the authoritative flake is box 1's
`/home/claude-code/lumi-flake` (main e73b702; box 2's `~/lumi-flake` mirrors it).
`modules/host-tuning.nix` (`lumi.tuning.enable`, per-host `linkCpus`: governor
`performance`, a `deepstrix-cpuidle` oneshot turning C2/C3 off everywhere and C1
off on the link CCX) and `modules/interconnect.nix` (`net.core.busy_read` /
`busy_poll` = 5000, `ethtool -K thunderbolt0 tso off gso off gro off`, udev
`power/control=on` for thunderbolt devices). Verified live on box 1 on
2026-10-04: busy_read/busy_poll 5000, governor `performance`,
`deepstrix-cpuidle` active, C1 disabled on cpu0 and enabled on cpu8. Not
persisted: the expertd CPU pin (box 2's launcher runs `taskset -c 8-15,24-31`).
`scripts/apply_host_tuning.sh` re-applies and verifies at runtime.

Together: **decode 4.07 -> 4.94 tok/s (+21%)**, measured n=3 each side (2026-09-14).

| knob | was | set to | measured effect |
|---|---|---|---|
| `/sys/class/net/thunderbolt0/device/power/control` | `auto` | `on` | rtt at 500us idle 1255 -> 592 us (-53%); only -2% at the current 5 ms duty cycle, but grows as per-layer work shrinks |
| `/sys/devices/system/cpu/cpu*/cpuidle/state3/disable` | `0` | `1` | rtt at 5 ms idle 978 -> 643 us; raw ICMP idle 0.788 -> 0.203 ms |

C3's exit latency is **350 us** (POLL/C1/C2/C3 = 0/1/18/350, governor `menu`). It
is paid twice per layer: once waking the link, once waking the host out of the
HIP stream sync in `sel_sync` (22-23 ms/token). That is why the e2e gain
(~44 ms/token) is 3x what the link measurement alone predicted (14 ms).

**Tidier form, not yet built:** a PM QoS request — a process holding
`/dev/cpu_dma_latency` open with 0 written to it — bounds exit latency without
disabling a state machine-wide. Needs a udev rule (`crw------- root root`) and
should be held by `deepstrix-expertd` and `deepstrix-server` for their lifetimes.

### Added 2026-09-19 .. 09-21 (measured in LINK_IDLE_LATENCY.md)

| knob | set to | measured effect (warm token) |
|---|---|---|
| `cpufreq scaling_governor` (all cpus, both boxes) | `performance` | link -1 ms/token: idle cores no longer wake at 1.8 GHz |
| `cpuidle/state2/disable` (all cpus, both boxes) | `1` | link -1 ms/token (C1 kept elsewhere) |
| `cpuidle/state1/disable` on the LINK CCX only (box 1: 0-7,16-23; box 2: 8-15,24-31) | `1` | POLL-only idle: sel_sync -0.7..-1.7 ms/token (no IPI on the HIP-sync wake) |
| expertd `taskset` 8-15,24-31 (box 2) | pinned | srv -0.8 ms/token (reader->compute handoff stays on one L3) |
| `net.core.busy_read` / `busy_poll` (both boxes) | `5000` | REQUIRED for the hub's per-phase SO_BUSY_POLL: -6 ms/token exposed link |
| `ethtool -K thunderbolt0 tso off gso off` (both boxes; the flake also turns `gro` off) | off | removes the ~1 ms sender-side hold on every link reply over one 65,520-B segment (f32 >= 4 rows, f16 >= 8, every prefill/verify chunk). 2026-09-21 |

The SO_BUSY_POLL windows themselves are engine knobs, set per phase by
`HetEngine::remote_set_phase_busy_poll`: `V41_DECODE_BUSY_POLL_US` (code default
3000, production 5000) in decode, `V41_BATCH_BUSY_POLL_US` (default 50 since
2026-09-21; was 500 — the receiver-side hold on multi-segment replies scales
with the window, 2.1 ms at 3000 vs 0.36 at 20) for batched phases. Box 2 adapts
its own reader per request: a decode request (`REQ_FLAG_DECODE`) whose request
and reply fit one ~64 KB segment spins `V41_B2_DECODE_BUSY_POLL_US` (default
5000, 0 = never adapt), otherwise the daemon's base `--busy-poll` window.

## 2. Engine defaults CHANGED in code (2026-09-14 .. 09-18)

| env | code default | why |
|---|---|---|
| `V41_B2_GPU_REPACK` | **on** | box 2's MXFP4 HF->engine repack on its iGPU (0.04 ms) instead of its CPU. With zero-copy pinned staging: miss 10.57 -> 6.60 ms |
| `V41_VICTIM_CACHE` | **on** | box 1's decode LRU fills only from box 2's reported misses, so its residency is exclusive. Repairs the big-pool regression (3.60 -> ~4.3 tok/s) |
| `V41_REMOTE_BATCHED_MULTI` | **on** | hub sets `REQ_FLAG_BATCHED` on multi-token submits so a verify batch takes box 2's by-expert chain (+20 us/token) not its decode chain (+333 us/token). -11..-13% at B=2..4. Since 2026-10-03 `V41_REMOTE_BATCHED_B1` (default on) does the same for B=1, so box 2's per-token decode chain is effectively unused |
| `V41_B2_ODIRECT` | **on** (corrected 2026-10-04) | Box 2's zero-copy O_DIRECT page-ins have been ON by default since 2026-09-15 (e1b7b65; `b2_odirect()` in `remote_experts.rs`; `=0` reverts). The three O_DIRECT rejections (box 1 -16% decode; box 2 +6%; +2% with the bounce buffer removed) predate the zero-copy path. Box 1's own `V41_EXPERT_ODIRECT` is still default off |
| `V41_B2_POOL_FLOOR` | **0** (2026-09-16) | with `V41_B2_GLOBAL_POOL` (default on since 2026-09-18) box 2 runs one shard-wide LRU; floor 0 is bit-identical to 0.90 with coalescing off. The earlier "floor 0 is numerically unsound" was `V41_B2_COALESCE` (now default off) |

## 3. Engine envs worth knowing

| env | code default | note |
|---|---|---|
| `V41_T2_CATCHALL` | **0** (corrected 2026-10-04) | production sets 1. `=2` makes the split a constant (everything routed goes to box 2) for reproducibility. The old note "DSpark REQUIRES 2: mode 1's history-dependent f32 summation order diverges" was WRONG: the divergence was decode double-counting box 1's resident picks on box 2 (fedbea2, KNOWN_BUGS 0c). See the CORRECTED 2026-09-18 note in `forward_layer.rs` |
| `V41_T2_PARTITION` | 0 | production sets 1: fixed split of the expert id space, then live hot-set ownership (box 1 owns each layer's top ids, box 2 is the cold tier). `expert_pager.rs` `t2_partition` / `hot_set` |
| `V41_PAGER_POOL_GB` | **60** (corrected 2026-10-04) | production 95 since 2026-10-03 (78 from 2026-09-16, when pool 52 with 21 windows left decode's LRU only 25 slots: +27% DSpark at 78, commit 148044b). The 2026-09-14 "76 measured slower, leave at 52" was retracted by that commit |
| `V41_PAGER_WINDOWS` / `V41_PAGER_STRIDE` | derived / `N_EXPERT` (384) (corrected 2026-10-04) | With `WINDOWS` unset the dense prefill window count is `floor(total * (1 - V41_PAGER_DECODE_FRAC))` (frac 0.75), capped at `CED_DECODER_START + 2` under CED. `WINDOWS=0` means 1. Production: 0 / 384, with `V41_PREFILL_UNIFIED_POOL=1`. The old 21 / 128 was a launch setting, and 148044b names it as what starved decode |
| `V41_LOCAL_CLAIM_MAX` | unset | caps box 1's local picks/layer. `=0` measured a NO-OP at pool 52 (box 2's picks/token identical to 4 s.f.) because box 1's decode LRU was only ~25 slots |
| `V41_VERIFY_PROBE` / `V41_VERIFY_BATCHED` | 0 | speculative-ingest + rollback probe; accepts a comma list to sweep B in one run. ~2.9x decode cost when on. With `V41_VERIFY_BATCHED=1` it also runs the verify-vs-decode cross-check (there is no `V41_DSPARK_XCHECK` switch) |
| `V41_CED` | 1 | no material effect on verify-step cost (B=5: 791 vs 775 ms) |
| `ENGRAM_FADV_RANDOM` | — | REMOVED. Measured inert; Engram does ~no disk I/O in steady-state decode |
| `GLIBC_TUNABLES` | `glibc.malloc.arena_max=2` | from the RSS-retention fix |

Box 2 daemon, as launched on 2026-10-03 (`~/scratch-ms/deploy_hw_20261003.sh`):
`--experts L0-L39:278-383 --paged --decode-max-b 1 --load-threads 8`, under
`taskset -c 8-15,24-31`, with `V41_B2_HITS_FIRST=1 V41_B2_MISS_PAR=4
V41_B2_PREFETCH_SETS=16` and a mirror on `/weights2` (4240 owned experts, 79.7 GB
pool on the 96 GB machine). The 2026-09-14 placement file
(`--experts-file box2_placement_260_68.txt --experts-k 268`) is history. Box 2
MUST be `--paged` and span all 40 layers.

## 4. RULED OUT — do not re-try without new evidence

| knob | state | why not |
|---|---|---|
| `power_dpm_force_performance_level` (box 2 GPU) | `auto` | **Not a lever.** Box 2's `srv` is flat at ~407 us across a 0-5000 us idle sweep — its iGPU never downclocks. The idle penalty was entirely link + CPU |
| `net.ipv4.tcp_slow_start_after_idle` | 1 | Fires on an RTO (~200 ms). The observed cliff is at 500 us of idle |
| ~~`--busy-poll` / `net.core.busy_poll` 500~~ | — | (corrected 2026-10-04) Only the 500 us setting was ruled out (it bought 10% of the idle penalty). Busy polling at 5000 us plus the hub's per-phase SO_BUSY_POLL is REQUIRED — §1 |
| O_DIRECT for **box 1** expert reads | off (`V41_EXPERT_ODIRECT`) | Rejected on box 1. Box 2's zero-copy O_DIRECT path is ON — §2 |
| Predictive routing prefetch | — | Recall on MISSES is 0.08-0.47; catching 47% costs 278 MB/token against 23.6 ms saved. (Look-ahead prefetch was later built anyway as an opt-in: `REQ_FLAG_PREFETCH`, `V41_LOOKAHEAD_PREFETCH`, default off) |

## 5. Still on the table (2026-09-14 list)

  * **231 us of link at 5 ms idle** vs 44 us warm — ~187 us/layer, ~7.5 ms/token
    still idle-related after both fixes. C2 exit is 18 us so it is elsewhere;
    likely remaining Thunderbolt/PCIe link states.
  * **The WARM link cost**, now the larger half: 44 us link + ~380 us `srv` per
    5.9 KB round trip, 40x per token.
  * ~~Racing the picks to box 2 before `pg.ensure`~~ — DONE: decode ships the
    token's activations to box 2 before the pager runs (`forward_layer.rs`,
    "Ship this token's activations to box 2 BEFORE the pager runs").
    `sel_sync` was 22-23 ms/token (~570 us/layer) when this was written.

## 6. What box 1's page cache is actually buying (measured 2026-09-14)

Prompted by "we're just leaving all this page cache on the machine for no good
reason". It is not idle — but what it serves is PREFILL, not decode.

Per-generated-token disk slope (same prompt, gen 64 vs gen 512, warm, from
`/proc/<pid>/io read_bytes`):

    pool 52 GB   page cache 32 GB   slope  +0.993 MB/token   request: 26 / 471 MB
    pool 76 GB   page cache  9 GB   slope  -1.180 MB/token   request: 5421 / 4893 MB

**The slope is ~0 either way** — decode itself reads ~1 MB per generated token, so
the page cache is NOT serving the decode path. Engram's 96 random rows/token are
absorbed by its own row cache.

**The INTERCEPT is what moves**: ~5 GB of disk per REQUEST at pool 76 against
0.03-0.5 GB at pool 52. That is prefill's expert paging plus the CED replay
(~41 GB working set, fixed per request) falling out of a 9 GB cache. So the 32 GB
of page cache at pool 52 is buying ~5 GB/request of avoided prefill I/O.

At 4.93 GB/s that is ~1 s per request — real, but it does NOT account for the
20-30 s regression on a 512-token generation, so **the pool-76 slowdown still had
no confirmed cause** on 2026-09-14. See `WHY_THE_BIG_POOL_REGRESSED.md`, whose
kernel attribution was retracted; the pool was raised to 78 on 2026-09-16 once
the window split was fixed (§3).

Method note: an earlier pass dismissed the page-cache hypothesis using a slope
measured at pool 52 (32 GB of cache) and applied it to pool 76 (9 GB). That does
not follow — the hypothesis is *about* the smaller cache. Measure the hypothesis
in the regime it describes.
