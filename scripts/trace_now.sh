#!/usr/bin/env bash
# A perfetto timeline of the last N seconds of production (both boxes, from the
# always-on evtrace; host time).
#   scripts/trace_now.sh [SECONDS=30] [OUT=~/traces/decode-<utc>.json.gz]
#   LONE=1: a window inside the latest run of lone-stream (DSpark) steps
#   HUB=<hub .evt>: that file instead of the newest
#   DUMP=0: no hub Tier B dump (default: one covering the window, for the
#   device intervals; RING=<ring dir>, default ~/logs/evtrace-ring); a window
#   older than DUMP_MAX_S (default 150: ~the 128 MB ring) gets none
#   B2DUMP=1: also box 2's Tier B ring (its device intervals, with
#   `V41_B2_EVTRACE_DEV` on; /dev/shm/evtrace-ring there)
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
OUT=${2:-$HOME/traces/decode-$(date -u +%Y%m%d-%H%M%S).json.gz}
B2=${B2:-10.43.0.95}
B2_CPUS=${B2_CPUS:-0-7,16-23}
LINK_KBIT=${LINK_KBIT:-80000}
HERE=$(cd "$(dirname "$0")" && pwd)
LOW=(nice -n 19 ionice -c3)
TMP=$(mktemp -d)
RDIR=$(ssh "$B2" 'mktemp -d /tmp/evt2perfetto.XXXXXX')
cleanup() {
  rm -rf "$TMP"
  # shellcheck disable=SC2029
  ssh "$B2" "rm -rf '$RDIR'" || true
}
trap cleanup EXIT
mkdir -p "$(dirname "$OUT")"
HUB=${HUB:-$(ls -t "$HOME"/logs/evtrace/hub-*.evt | head -1)}
W=$("${LOW[@]}" python3 "$HERE/evt2perfetto.py" window "$HUB" --last "$S" ${LONE:+--lone})
get() { python3 -c "import json,sys; print(repr(json.loads(sys.argv[1])['$1']))" "$W"; }
HF=$(get hub_from); HT=$(get hub_to); BF=$(get b2_from); BT=$(get b2_to)
# The hub's device intervals live in its Tier B ring: dump back to the window's
# start (+2 s) and wait for the dump (written by the hub, paced).
RING=${RING:-$HOME/logs/evtrace-ring}
HUB_B=()
AGE=$(python3 -c "import sys,time; print(int((time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - float(sys.argv[1])) / 1e9))" "$HF")
if [ "${DUMP:-1}" = 1 ] && [ "$AGE" -gt "${DUMP_MAX_S:-150}" ]; then
  echo "window starts ${AGE} s ago, older than the hub ring holds (DUMP_MAX_S=${DUMP_MAX_S:-150}): no device intervals" >&2
elif [ "${DUMP:-1}" = 1 ]; then
  D=$(bash "$HERE/evt_dump_now.sh" "$RING" hub "$HF")
  if [ -n "$D" ]; then HUB_B=("$D"); else echo "no hub Tier B dump (device intervals left out)" >&2; fi
fi
scp -q -l "$LINK_KBIT" "$HERE/evt2perfetto.py" "$HERE/evt_dump_now.sh" "$B2:$RDIR/"
# shellcheck disable=SC2029
B2_FILES='~/logs/evtrace/b2-*.evt'
if [ "${B2DUMP:-0}" = 1 ]; then
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
"${LOW[@]}" python3 "$HERE/evt2perfetto.py" trace "$HUB" "${HUB_B[@]}" "${LOCAL[@]}" --from "$HF" --to "$HT" -o "$OUT"
echo "open in https://ui.perfetto.dev : $OUT"
