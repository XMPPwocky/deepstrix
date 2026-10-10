#!/usr/bin/env bash
# A perfetto timeline of the last N seconds of production (both boxes, from the
# always-on evtrace; host time).
#   scripts/trace_now.sh [SECONDS=30] [OUT=~/traces/decode-<utc>.json.gz]
#   LONE=1: a window inside the latest run of lone-stream (DSpark) steps
#   TWO=1: a window inside the latest run of 2-stream DSpark steps (live == 2
#   and spec_streams >= 2; short interruptions tolerated) lasting >= SECONDS,
#   else the longest; default OUT ~/traces/dspark2-<utc>.json.gz. (LONE and
#   TWO are exclusive.) The trace has, besides the hub / box-2 request tracks,
#   "box 1 disks + system", "box 2 disks + system" (per NVMe: MB/s, IOPS,
#   latency, queue depth, util; PSI, procs, dirty) and "Thunderbolt link"
#   (per-request wire legs, link MB/s, NIC counters, flows); disk models come
#   from `lsblk` on each box (an extra ssh call, no load).
#   HUB=<hub .evt>: that file instead of the newest
#   DUMP=0: no hub Tier B dump (default 1, but 0 with TWO=1: production runs
#   V41_EVTRACE_DEV=0 so no ring answers; DUMP=1 forces one, for the
#   device intervals; RING=<ring dir>, default ~/logs/evtrace-ring); a window
#   older than DUMP_MAX_S (default 150: ~the 128 MB ring) gets none
#   TMPDIR=<dir>: scratch space on box 1 (default ~/.cache, not /tmp)
#   B2DUMP=1: also box 2's Tier B ring (its device intervals, with
#   `V41_B2_EVTRACE_DEV` on; /dev/shm/evtrace-ring there); older than
#   B2DUMP_MAX_S (default 150; box 2's ring fills at its own rate) gets none
# Open OUT in https://ui.perfetto.dev (drag and drop).
# Gentle on production: Python parses at tens of MB/s, box 2's cut reads only
# the files that can hold the window and stops just past it, runs off
# expertd's cores (8-15,24-31) at nice 19, and only a gzipped slice crosses the
# expert link, rate-limited. (ionice is set but both boxes' NVMe use the `none`
# scheduler, which ignores it.)
set -euo pipefail
# A hard cap: a bug here must fail with MemoryError, not push the hub (tens of
# GB resident, ~3-5 GB free) toward the OOM killer.
ulimit -v "${VMEM_KB:-2000000}"
S=${1:-30}
if [ -n "${LONE:-}" ] && [ -n "${TWO:-}" ]; then echo "LONE and TWO are exclusive" >&2; exit 2; fi
PFX=decode; [ -n "${TWO:-}" ] && PFX=dspark2
OUT=${2:-$HOME/traces/$PFX-$(date -u +%Y%m%d-%H%M%S).json.gz}
B2=${B2:-10.43.0.95}
B2_CPUS=${B2_CPUS:-0-7,16-23}
LINK_KBIT=${LINK_KBIT:-80000}
HERE=$(cd "$(dirname "$0")" && pwd)
LOW=(nice -n 19 ionice -c3)
# Tier B device timing is OFF in production (V41_EVTRACE_DEV=0): no ring answers, so TWO=1 defaults to no dump.
DUMP=${DUMP:-$([ -n "${TWO:-}" ] && echo 0 || echo 1)}
TMPROOT=${TMPDIR:-$HOME/.cache}; mkdir -p "$TMPROOT"
TMP=$(mktemp -d "$TMPROOT/trace_now.XXXXXX")
RDIR=$(ssh "$B2" 'mktemp -d /tmp/evt2perfetto.XXXXXX')
cleanup() {
  rm -rf "$TMP"
  # shellcheck disable=SC2029
  ssh "$B2" "rm -rf '$RDIR'" || true
}
trap cleanup EXIT
mkdir -p "$(dirname "$OUT")"
if [ -n "${HUB:-}" ]; then
  W=$("${LOW[@]}" python3 "$HERE/evt2perfetto.py" window "$HUB" --last "$S" ${LONE:+--lone} ${TWO:+--two})
else
  # The newest hub file that yields a window: right after a hub restart the
  # newest has no steps yet, or (LONE / TWO) no run as long as S -- so try the
  # next (3 at most) with --full, then settle for the newest's best run.
  W=
  FILES=$(ls -t "$HOME"/logs/evtrace/hub-*.evt | head -3)
  FULL=${LONE:+--full}${TWO:+--full}
  for HUB in $FILES; do
    if W=$("${LOW[@]}" python3 "$HERE/evt2perfetto.py" window "$HUB" --last "$S" ${LONE:+--lone} ${TWO:+--two} ${FULL:+--full}); then
      break
    fi
    echo "no full window in $HUB; trying the next older hub file" >&2
    W=
  done
  if [ -z "$W" ] && [ -n "$FULL" ]; then
    for HUB in $FILES; do
      if W=$("${LOW[@]}" python3 "$HERE/evt2perfetto.py" window "$HUB" --last "$S" ${LONE:+--lone} ${TWO:+--two}); then
        echo "no run >= $S s in the 3 newest hub files: the best run of $HUB" >&2
        break
      fi
      W=
    done
  fi
  [ -n "$W" ] || { echo "no window in the 3 newest hub files" >&2; exit 1; }
fi
get() { python3 -c "import json,sys; print(repr(json.loads(sys.argv[1])['$1']))" "$W"; }
HF=$(get hub_from); HT=$(get hub_to); BF=$(get b2_from); BT=$(get b2_to)
# The hub's device intervals live in its Tier B ring: dump back to the window's
# start (+2 s) and wait for the dump (written by the hub, paced).
RING=${RING:-$HOME/logs/evtrace-ring}
HUB_B=()
AGE=$(python3 -c "import sys,time; print(int((time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - float(sys.argv[1])) / 1e9))" "$HF")
if [ "$DUMP" = 1 ] && [ "$AGE" -gt "${DUMP_MAX_S:-150}" ]; then
  echo "window starts ${AGE} s ago, older than the hub ring holds (DUMP_MAX_S=${DUMP_MAX_S:-150}): no device intervals" >&2
elif [ "$DUMP" = 1 ]; then
  D=$(bash "$HERE/evt_dump_now.sh" "$RING" hub "$HF")
  if [ -n "$D" ]; then HUB_B=("$D"); else echo "no hub Tier B dump (device intervals left out)" >&2; fi
fi
scp -q -l "$LINK_KBIT" "$HERE/evt2perfetto.py" "$HERE/evt_dump_now.sh" "$B2:$RDIR/"
# shellcheck disable=SC2029
B2_FILES='~/logs/evtrace/b2-*.evt'
B2AGE=0
if [ "${B2DUMP:-0}" = 1 ]; then
  B2AGE=$(ssh "$B2" "python3 -c 'import sys,time; print(int((time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - float(sys.argv[1])) / 1e9))' $BF")
fi
if [ "${B2DUMP:-0}" = 1 ] && [ "$B2AGE" -gt "${B2DUMP_MAX_S:-150}" ]; then
  echo "window starts ${B2AGE} s ago on box 2, older than its ring holds (B2DUMP_MAX_S=${B2DUMP_MAX_S:-150}): no box-2 device intervals" >&2
elif [ "${B2DUMP:-0}" = 1 ]; then
  # Box 2's ring back to the window's start (its RAW clock) + 2 s; its
  # dumps are in tmpfs and cut like the rest.
  # shellcheck disable=SC2029
  D2=$(ssh "$B2" "bash $RDIR/evt_dump_now.sh /dev/shm/evtrace-ring b2 $BF")
  [ -n "$D2" ] && B2_FILES="$B2_FILES $D2" || echo "no box-2 Tier B dump" >&2
fi
CUTS=$(ssh "$B2" "taskset -c $B2_CPUS ionice -c3 nice -n 19 python3 $RDIR/evt2perfetto.py cut $B2_FILES --from $BF --to $BT -o $RDIR/b2cut >/dev/null && taskset -c $B2_CPUS nice -n 19 gzip -1 -f $RDIR/b2cut.*.evt && ls $RDIR/b2cut.*.evt.gz")
LOCAL=()
for c in $CUTS; do
  scp -q -l "$LINK_KBIT" "$B2:$c" "$TMP/"
  gunzip -f "$TMP/$(basename "$c")"
  LOCAL+=("$TMP/$(basename "$c" .gz)")
done
# Disk models (names -> models) of both boxes, for the disk track labels.
lsblk -d -n -o NAME,MODEL > "$TMP/models.hub" 2>/dev/null || : > "$TMP/models.hub"
ssh "$B2" 'lsblk -d -n -o NAME,MODEL' > "$TMP/models.b2" 2>/dev/null || : > "$TMP/models.b2"
"${LOW[@]}" python3 "$HERE/evt2perfetto.py" trace "$HUB" "${HUB_B[@]}" "${LOCAL[@]}" --from "$HF" --to "$HT" -o "$OUT" \
  --disk-models-hub "$TMP/models.hub" --disk-models-b2 "$TMP/models.b2"
echo "open in https://ui.perfetto.dev : $OUT"
