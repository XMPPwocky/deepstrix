#!/usr/bin/env bash
# V4.1-Flash hub server: THE production launch (checked in 2026-09-24; was
# ~/run_v41_server.sh plus ~16 env vars passed inline, recorded nowhere but a
# memory note). Every setting below is `${VAR:-default}`, so a bare run of this
# script IS production and any var can still be overridden from the caller.
# Restart: deploy/restart-hub.sh. Box 2 first: deploy/restart-expertd-b2.sh.
#
# Historical header (config as first validated, 2026-09-13):
#
# Measured with this exact config (6.0k prefill probe + 2048-tok decode,
# back-to-back A/B, routing verifier clean):
#     prefill 255 tok/s       decode 15.8 tok/s (zero misses, decode_hit 1.0000)
# versus the morning's 131 / 4.5. (143 / 14.33 before the Engram gather batching
# in EngramCtx::rows_for_chunk, which was worth +17 tok/s of prefill.)
#
# Pair with box 2 (10.99.0.2) running deploy/restart-expertd-b2.sh (historically
# `~/b2_run_expertd.sh`, which defaulted to
#     --experts L0-L19:116-383,L20-L39:344-383 --paged; now L0-L39:230-383)
# Box 2 MUST be `--paged` and MUST span all 40 layers, or decode dies on layer 20
# (prefill survives — CED only runs layers 0-19 — so the failure looks like a
# decode-only bug).
set -x

# ---- standard log destination -------------------------------------------
# Every restart used to invent its own filename at the call site, so "where is
# the logfile" had a different answer each time and older runs were easy to
# lose. The script now owns it: one known path, and the previous run is archived
# under a TIMESTAMP rather than a fixed .1 (a fixed .1 once ate a 153 MB
# production log when a relaunch rotated twice in a row).
#   LOG=/some/path  to override.   V41_LOG_ATTACHED=1  to keep the caller's fds.
LOG=${LOG:-$HOME/logs/v41-server.log}
if [ -z "${V41_LOG_ATTACHED:-}" ]; then
  mkdir -p "$(dirname "$LOG")"
  [ -f "$LOG" ] && mv -f "$LOG" "$LOG.$(date +%Y%m%d-%H%M%S)"
  # Keep the last 10 archives; they are the only record of a crashed run.
  ls -1t "$LOG".2* 2>/dev/null | tail -n +11 | xargs -r rm -f
  export V41_LOG_ATTACHED=1
  exec >>"$LOG" 2>&1
fi
echo "=== deepstrix-server starting $(date -uIs) ==="
# -------------------------------------------------------------------------

export V41_PAGED_EXPERTS=1
export V41_CED=${V41_CED:-1}

# Pool. **"76 OOMs" was WRONG — RETESTED 2026-09-14: 76 allocates 4340 slots =
# 81.6 GB and serves fine.** The iGPU's GTT ceiling is the whole 100.33 GB of
# system RAM, so the driver was never the limit; the old OOM was most likely
# measured on a box still holding ~58 GB in the amdgpu page pool from a previous
# run (that memory IS reclaimable by the next HIP process — a 55.8 GB pool
# allocates from exactly that state).
#
# *** The "DO NOT raise it / leg balance" argument that stood here is RETRACTED
# (2026-09-16). It read:
#     pool 55.8 GB (2969 slots, decode LRU   25) -> 4.06 tok/s
#     pool 81.6 GB (4340 slots, decode LRU 1396) -> 3.1-3.6 tok/s, WARM
# and concluded capacity was not the lever. The real cause is that box 1's
# decode LRU NEVER EVICTS: `budget = pg.lru_free_slots()` counts EMPTY slots
# (expert_pager.rs), so once full it admits nothing for the life of the process.
# A 1396-slot cache frozen on whatever arrived first is worse than a 25-slot
# one -- that is a symptom, not evidence against big pools. See
# docs/v41/KNOWN_BUGS.md #16.
#
# Measured at 78 GB with the pool floor at 0 (commit 148044b), DSpark accept,
# 120 tokens, E unchanged at 2.380:
#     pool 52, floor 0.90   1.78 tok/s   ensure_ms 837   misses 17196/31276
#     pool 78, floor 0      2.27 tok/s   ensure_ms 130   misses  3387/31276
# 86 GB OOMs (hipErrorOutOfMemory) on this 93 GB box, so 78 is near the ceiling.
export V41_PAGER_POOL_GB=${V41_PAGER_POOL_GB:-78}

# Dense prefill windows. The `V41_PAGER_DECODE_FRAC` default (0.75) gives prefill
# only 1 window, which was correct when decode was miss-bound at 11 misses/token.
# Under T2 catch-all decode misses ~1/token and its LRU is oversized, so windows
# are worth more to prefill than slots are to decode: w1 -> w4 measured
# prefill 134 -> 143 AND decode 14.10 -> 14.33 (box 1 pages less, box 2 absorbs it).
# NOTE: 0 means exactly ONE dense window (expert_pager.rs: `Some(0) => 1`), not
# "derive from stride and pool" as this comment used to say. With
# V41_PREFILL_UNIFIED_POOL=1 prefill pages through the shared pool anyway, which
# is what the live server has run since the unified pool landed.
export V41_PAGER_WINDOWS=${V41_PAGER_WINDOWS:-0}

# PACKED windows. A window only needs to be as wide as the union it holds, and
# box 1 owns just 384-268=116 experts per encoder layer, so 384-wide windows ran
# at ~11% occupancy. At stride 128 the same pool holds 21 windows instead of 4,
# pinning ALL 20 CED encoder layers: prefill_hit 0.478 -> 0.727, prefill
# 159 -> 255 tok/s, decode 14.8 -> 15.8. Bit-identical to stride 384 when the
# device partition is held fixed (V41_T2_CATCHALL=0).
#
# The stride MUST be >= box 1's owned count per layer (N_EXPERT - box2 owned).
# Exceeding it is a hard error naming the layer, not a silent eviction. If box 2's
# --experts changes so it owns LESS, raise this.
# STALE DEFAULT FIXED 2026-09-17: 128 assumed box 2 owned 268/layer (box 1: 116).
# Box 2's own default is now --experts L0-L39:230-383 = 154 owned, so box 1 owns
# 230/layer and a 128 stride is a HARD ERROR at load ("L0 union needs > 128
# slots"). That mismatch took the server down mid-session once already, because
# the stride was only ever passed on the command line. Keep this >= 384 minus
# box 2's owned count.
export V41_PAGER_STRIDE=${V41_PAGER_STRIDE:-384}

# Two-box split + T2 catch-all: box 1 computes only what it already holds and
# reassigns every miss to box 2, which pages from its OWN disk. Took box 1's
# paging from 144 ms/token to 2.9 ms and decode 7.0 -> 14.3.
export V41_REMOTE_ADDR=${V41_REMOTE_ADDR:-10.99.0.2:7431}
# Overridable (2026-09-16): these were hardcoded, so `env V41_REMOTE_SPLIT=0 ...`
# was silently ignored and a "box 1 solo" control could not be run at all.
export V41_REMOTE_SPLIT=${V41_REMOTE_SPLIT:-1}
export V41_REMOTE_SPLIT_DECODE=${V41_REMOTE_SPLIT_DECODE:-1}
export V41_T2_CATCHALL=${V41_T2_CATCHALL:-1}
export V41_PAGER_UNION=${V41_PAGER_UNION:-1}

# Live-config defaults that previously existed only on the command line, so a
# restart from this script silently ran a different engine than the one measured.
# Sparse indexer top-k (decode + prefill), one shared pager pool for prefill
# instead of a private window set, and deterministic expert partitioning.
export V41_INDEX_K=${V41_INDEX_K:-1}
export V41_PREFILL_UNIFIED_POOL=${V41_PREFILL_UNIFIED_POOL:-1}
export V41_T2_PARTITION=${V41_T2_PARTITION:-1}

# Box 1 services a decode miss by reading its THREE role tensors (gate/up/down,
# ~6.3 MB each). Default 1 = serially, which the code comment notes runs well
# under the NVMe ceiling and leaves the single-core repack un-overlapped. Box 2's
# shard pool already spawns a thread per role; box 1 did not.
# MEASURED 2026-09-18 on live traffic: decode_ms_per_miss 13.0 -> 9.3 (-28%).
# (pread_gbps drops 2.4 -> 2.2 because three streams share the device, but the
# read finishes in less WALL time, which is what the miss costs.)
# Warm repeated-prompt throughput is unchanged at 16.3 tok/s -- that regime has
# decode_misses=0, so it cannot show this either way.
export V41_PAGER_MISS_THREADS=${V41_PAGER_MISS_THREADS:-3}

# ---- settings that used to be passed only on the command line -------------
# (the production launch line since 2026-09-20; see the memory note
# "V4.1 server restart procedure" for the history of each)
# Multi-stream scheduler: the production decode path. OFF in code, and a launch
# without it silently runs the serial driver (a different code path).
export V41_MULTISTREAM=${V41_MULTISTREAM:-1}
# KV arena rows: 2.75x --ctx fits three ~269K agents (3x left 190 MiB dGPU).
export V41_MS_CTX_ROWS=${V41_MS_CTX_ROWS:-844800}
export V41_MS_PREFILL_JOBS=${V41_MS_PREFILL_JOBS:-1}
export V41_MS_PIPELINE_MIN_ROWS=${V41_MS_PIPELINE_MIN_ROWS:-4}
export V41_MS_STAGGER=${V41_MS_STAGGER:-2}
# Per-stage GPU busy + host timers ("ms.stage") every 20 steps; cost unmeasurable.
export V41_MS_PROFILE=${V41_MS_PROFILE:-1}
export V41_MS_PROFILE_EVERY=${V41_MS_PROFILE_EVERY:-20}
# Long-context candidate pool (sparse indexer). Changes long-context output.
export V41_CANDIDATE_POOL=${V41_CANDIDATE_POOL:-1}
export V41_MISS_HIST=${V41_MISS_HIST:-1}
# Decode's T2 catch-all for <=8-row batched steps: box 1 computes only resident
# experts, misses go to box 2 (else every arena step pages from box 1's NVMe).
export V41_SMALL_B_CATCHALL_MAX=${V41_SMALL_B_CATCHALL_MAX:-8}
export V41_B1_HOT_PER_LAYER=${V41_B1_HOT_PER_LAYER:-103}
export V41_B1_PREFETCH=${V41_B1_PREFETCH:-1}
export V41_B1_PREFETCH_ADMIT=${V41_B1_PREFETCH_ADMIT:-24}
export V41_B1_PAGE_MISSES=${V41_B1_PAGE_MISSES:-1}
export V41_PARTITION_BOX1_SHARE=${V41_PARTITION_BOX1_SHARE:-0.15}
export V41_PAGER_MISS_PAR=${V41_PAGER_MISS_PAR:-8}
# ---------------------------------------------------------------------------

# DSpark speculative decoding. *** IT IS OFF (V41_DSPARK=0, below) *** -- the
# "PRODUCTION DEFAULT" claim that used to open this block was two lines above a
# line disabling it, which reads as if the +15% is live. It is not, and roughly
# 900 lines of accept/shadow code in engine_worker.rs are inert. Reason it is
# off: KNOWN_BUGS "DSpark accept CORRUPTS output" / "the DRAFTER changes the
# accepted output at temperature 0". Turn it on only with that resolved.
#
# The historical measurement, for the record. Costs 7.93 GB
# of iGPU residency for the drafter. Measured same-binary/same-prompt A/B:
# plain 17.12 tok/s -> DSpark 19.70 (+15%), E = 2.909 tokens per verify step.
# Temp-0 output differs slightly from plain, but plain decode itself is not
# temp-0 stable across a cold vs warm pager (the pager-geometry bug), so that
# difference is not attributable to DSpark. V41_DSPARK=0 to fall back.
export V41_DSPARK=${V41_DSPARK:-0}

# ON by default since 2026-09-14 (lever 1, pre-submit reorder). Rollback:
#   V41_DECODE_PRESUBMIT=0
# Moves the shared-expert graph AFTER `submit` and pre-quantizes moe_xq before the
# pick-readback sync (killing the second per-layer stream sync). Measured
# back-to-back in ONE binary: decode 10.78 -> 11.32 tok/s (+5.0%), sel_sync_us
# 27,313 -> 24,258, output byte-identical (sha b7f58f53b529).

# OFF by default, both measured losses / not affordable — see the code comments:
#   V41_MHC_SPLIT=1      mHC mixes on a side stream: decode 6.95 -> 6.22
#   V41_REPLAY_OFFLOAD=1 CED replay to box 2: works (6.9 -> 0.2 s) but needs
#                        >=170 decoder slots/layer, which costs more encoder
#                        capacity than the replay is worth at 124 GB.

export DEEPSTRIX_HANG_DEADLINE_MS=1800000
export GLIBC_TUNABLES=${GLIBC_TUNABLES:-glibc.malloc.arena_max=2}
# Second listen address: the tailnet IP, same as run_deepstrix.sh, so other
# tailscale devices can reach this server. The server binds each --addr
# independently and only WARNS on one it cannot bind. Set ADDR2= to disable.
# NOTE: no auth -- anything routing to this host can use the model.
# --ctx since 2026-09-18. CAUTION, read this before touching --ctx: the same
# change that freed the attention scratch also removed the ceiling that was
# refusing --ctx > 131_072, and for a day the indexer SILENTLY scored only the
# first 131,200 compressed positions — everything after was invisible to all 38
# compressed layers. Fixed b154602: ATTN_MIXED_MAX_KEYS now covers V41_MAX_CTX +
# IMAGE_RAW_WINDOW_MAX and the admission check derives from the env-INDEPENDENT
# `indexer_max_scored_keys`. Raising --ctx past V41_MAX_CTX (368640 since
# 2026-09-22, was 307200) now fails at
# startup naming both bounds; raise the constant in attention.rs to go higher
# (costs ~1.7 MB dGPU per 1K of ctx at lane_rows 512).
#
# PRODUCTION defaults (fixed 2026-09-17): the live server the agent talks to has
# always run on 18080 with --ctx 65536, but only ever via explicit command-line
# overrides -- the script still defaulted to the 18141/8192 dev pair, so a restart
# "from the script" silently brought up a different server on a port nothing
# connects to. Override ADDR/ADDR2/CTX for a scratch instance.
ADDR2=${ADDR2-100.79.4.101:18080}
MODEL=$(readlink -f ~/.cache/deepstrix/models/dsv4.1f)
MMPROJ=${MMPROJ-$MODEL}
exec ${DEEPSTRIX_BIN:-/home/claude-code/deepstrix/target-v41/release/deepstrix-server} \
  --gguf "$MODEL" --addr "${ADDR:-127.0.0.1:18080}" \
  ${ADDR2:+--addr "$ADDR2"} --ctx ${CTX:-368640} \
  --model-name deepseek-v4.1-flash \
  ${MMPROJ:+--mmproj "$MMPROJ"} \
  --snapshot-dir /home/claude-code/.cache/deepstrix/snapshots-v41
