# Tuning reference — every knob touched or ruled out
### as of 2026-09-14. Re-apply the host settings after any reboot: `scripts/apply_host_tuning.sh`

## 1. Host settings — NON-PERSISTENT, both boxes

Together: **decode 4.07 -> 4.94 tok/s (+21%)**, measured n=3 each side.

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

### Added 2026-09-19 (all in `scripts/apply_host_tuning.sh`, measured in LINK_IDLE_LATENCY.md)

| knob | set to | measured effect (warm token) |
|---|---|---|
| `cpufreq scaling_governor` (all cpus, both boxes) | `performance` | link -1 ms/token: idle cores no longer wake at 1.8 GHz |
| `cpuidle/state2/disable` (all cpus, both boxes) | `1` | link -1 ms/token (C1 kept elsewhere) |
| `cpuidle/state1/disable` on the LINK CCX only (box 1: 0-7,16-23; box 2: 8-15,24-31) | `1` | POLL-only idle: sel_sync -0.7..-1.7 ms/token (no IPI on the HIP-sync wake) |
| expertd `taskset` 8-15,24-31 (box 2) | pinned | srv -0.8 ms/token (reader->compute handoff stays on one L3) |
| `net.core.busy_read` / `busy_poll` (both boxes) | `5000` | REQUIRED for the server's per-phase SO_BUSY_POLL (3000 us in decode): -6 ms/token exposed link |

## 2. Engine defaults CHANGED in code this session

| env | default now | why |
|---|---|---|
| `V41_B2_GPU_REPACK` | **1** | box 2's MXFP4 HF->ggml permute on its iGPU (0.04 ms) instead of its CPU. With zero-copy pinned staging: miss 10.57 -> 6.60 ms |
| `V41_VICTIM_CACHE` | **1** | box 1's decode LRU fills only from box 2's reported misses, so its residency is exclusive. Repairs the big-pool regression (3.60 -> ~4.3 tok/s); does not beat pool 52 |
| `V41_REMOTE_BATCHED_MULTI` | **1** | hub sets `REQ_FLAG_BATCHED` on multi-token submits so a verify batch takes box 2's by-expert chain (+20 us/token) not its decode chain (+333 us/token). -11..-13% at B=2..4, inert at B=1 |
| `V41_B2_ODIRECT` | **0** | O_DIRECT REJECTED three times (box 1 -16% decode; box 2 +6%; and +2% even with the bounce buffer fully removed). Cause is read concurrency, not the copy |

## 3. Engine envs worth knowing, defaults UNCHANGED

| env | default | note |
|---|---|---|
| `V41_T2_CATCHALL` | 1 | **DSpark REQUIRES 2.** Mode 1's residency-based partition is history-dependent, so f32 summation order changes and speculative rollback diverges. Mode 2 measured byte-identical |
| `V41_PAGER_POOL_GB` | 52 | 76 allocates fine (4340 slots, 81.6 GB) — the old "76 OOMs" note was WRONG — but measured SLOWER (3.26-3.60 vs 4.06-4.09). Leave at 52 |
| `V41_LOCAL_CLAIM_MAX` | unset | caps box 1's local picks/layer. `=0` measured a NO-OP at pool 52 (box 2's picks/token identical to 4 s.f.) because box 1's decode LRU is only ~25 slots |
| `V41_VERIFY_PROBE` / `V41_VERIFY_BATCHED` | 0 | speculative-ingest + rollback probe; accepts a comma list to sweep B in one run. ~2.9x decode cost when on |
| `V41_CED` | 1 | no material effect on verify-step cost (B=5: 791 vs 775 ms) |
| `ENGRAM_FADV_RANDOM` | — | REMOVED. Measured inert; Engram does ~no disk I/O in steady-state decode |
| `V41_PAGER_WINDOWS` / `V41_PAGER_STRIDE` | 21 / 128 | 21 packed windows pin all 20 CED encoder layers; stride must be >= box 1's owned count per layer |
| `GLIBC_TUNABLES` | `glibc.malloc.arena_max=2` | from the RSS-retention fix |

Box 2 daemon: `--experts-file box2_placement_260_68.txt --experts-k 268 --paged`
(260 slots on encoder layers, 68 on decoder layers; adopted, +13% decode).

## 4. RULED OUT — do not re-try without new evidence

| knob | state | why not |
|---|---|---|
| `power_dpm_force_performance_level` (box 2 GPU) | `auto` | **Not a lever.** Box 2's `srv` is flat at ~407 us across a 0-5000 us idle sweep — its iGPU never downclocks. The idle penalty was entirely link + CPU |
| `net.ipv4.tcp_slow_start_after_idle` | 1 | Fires on an RTO (~200 ms). The observed cliff is at 500 us of idle |
| `--busy-poll` / `net.core.busy_poll` | 500 | Bought only 10% of the idle penalty; not receive-side polling |
| O_DIRECT for expert reads | off | Three independent rejections, see above |
| Predictive routing prefetch | — | Recall on MISSES is 0.08-0.47; catching 47% costs 278 MB/token against 23.6 ms saved |

## 5. Still on the table

  * **231 us of link at 5 ms idle** vs 44 us warm — ~187 us/layer, ~7.5 ms/token
    still idle-related after both fixes. C2 exit is 18 us so it is elsewhere;
    likely remaining Thunderbolt/PCIe link states.
  * **The WARM link cost**, now the larger half: 44 us link + ~380 us `srv` per
    5.9 KB round trip, 40x per token.
  * **`sel_sync` = 22-23 ms/token** (~570 us/layer) waiting to learn six expert
    ids. Racing the picks to box 2 before `pg.ensure` would let box 2 start its
    reads earlier every layer.

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
20-30 s regression on a 512-token generation, so **the pool-76 slowdown still has
no confirmed cause**. See `WHY_THE_BIG_POOL_REGRESSED.md`, whose kernel
attribution was retracted.

Method note: an earlier pass dismissed the page-cache hypothesis using a slope
measured at pool 52 (32 GB of cache) and applied it to pool 76 (9 GB). That does
not follow — the hypothesis is *about* the smaller cache. Measure the hypothesis
in the regime it describes.
