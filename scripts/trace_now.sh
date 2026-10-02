#!/usr/bin/env bash
# A perfetto timeline of the last N seconds of production (both boxes, from the
# always-on evtrace; host time).
#   scripts/trace_now.sh [SECONDS=30] [OUT=~/traces/decode-<utc>.json.gz]
#   LONE=1: a window inside the latest run of lone-stream (DSpark) steps
#   HUB=<hub .evt>: that file instead of the newest
#   DUMP=0: no hub Tier B dump (default: one covering the window, for the
#   device intervals; RING=<ring dir>, default ~/logs/evtrace-ring)
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
if [ "${DUMP:-1}" = 1 ] && [ -d "$RING" ]; then
  SECS=$(python3 -c "import sys,time; print(int((time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - float(sys.argv[1])) / 1e9) + 2)" "$HF")
  BEFORE=$(ls "$RING"/hub-dump-*.evt 2>/dev/null | wc -l)
  echo "$SECS" > "$RING/.dump-request.tmp" && mv "$RING/.dump-request.tmp" "$RING/dump-request"
  for _ in $(seq 1 120); do
    if [ ! -e "$RING/dump-request" ] && [ "$(ls "$RING"/hub-dump-*.evt 2>/dev/null | wc -l)" -gt "$BEFORE" ]; then
      break
    fi
    sleep 0.25
  done
  # The newest dump, once its size settles (the writer streams it).
  D=$(ls -t "$RING"/hub-dump-*.evt 2>/dev/null | head -1 || true)
  if [ -n "$D" ]; then
    S0=-1
    while [ "$(stat -c %s "$D")" != "$S0" ]; do S0=$(stat -c %s "$D"); sleep 0.5; done
    HUB_B=("$D")
  else
    echo "no hub Tier B dump (device intervals left out)" >&2
  fi
fi
scp -q -l "$LINK_KBIT" "$HERE/evt2perfetto.py" "$B2:$RDIR/evt2perfetto.py"
# shellcheck disable=SC2029
CUTS=$(ssh "$B2" "taskset -c $B2_CPUS ionice -c3 nice -n 19 python3 $RDIR/evt2perfetto.py cut ~/logs/evtrace/b2-*.evt --from $BF --to $BT -o $RDIR/b2cut >/dev/null && taskset -c $B2_CPUS nice -n 19 gzip -1 -f $RDIR/b2cut.*.evt && ls $RDIR/b2cut.*.evt.gz")
LOCAL=()
for c in $CUTS; do
  scp -q -l "$LINK_KBIT" "$B2:$c" "$TMP/"
  gunzip -f "$TMP/$(basename "$c")"
  LOCAL+=("$TMP/$(basename "$c" .gz)")
done
"${LOW[@]}" python3 "$HERE/evt2perfetto.py" trace "$HUB" "${HUB_B[@]}" "${LOCAL[@]}" --from "$HF" --to "$HT" -o "$OUT"
echo "open in https://ui.perfetto.dev : $OUT"
