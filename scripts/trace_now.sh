#!/usr/bin/env bash
# A perfetto timeline of the last N seconds of production (both boxes, from the
# always-on evtrace; host time). Only a time slice of box 2's file crosses the
# expert link.
#   scripts/trace_now.sh [SECONDS=30] [OUT=~/traces/decode-<utc>.json.gz]
#   LONE=1: end at the hub file's last DSpark (lone-stream) block instead
#   HUB=<hub .evt>: that file instead of the newest
# Open OUT in https://ui.perfetto.dev (drag and drop).
set -euo pipefail
S=${1:-30}
OUT=${2:-$HOME/traces/decode-$(date -u +%Y%m%d-%H%M%S).json.gz}
B2=${B2:-10.43.0.95}
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$(dirname "$OUT")"
HUB=${HUB:-$(ls -t "$HOME"/logs/evtrace/hub-*.evt | head -1)}
W=$(python3 "$HERE/evt2perfetto.py" window "$HUB" --last "$S" ${LONE:+--lone})
get() { python3 -c "import json,sys; print(repr(json.loads(sys.argv[1])['$1']))" "$W"; }
HF=$(get hub_from); HT=$(get hub_to); BF=$(get b2_from); BT=$(get b2_to)
scp -q "$HERE/evt2perfetto.py" "$B2:/tmp/evt2perfetto.py"
B2FILES=$(ssh "$B2" 'ls -t ~/logs/evtrace/b2-*.evt | head -4 | tr "\n" " "')
# shellcheck disable=SC2029
ssh "$B2" "python3 /tmp/evt2perfetto.py cut $B2FILES --from $BF --to $BT -o /tmp/b2cut.evt" >/dev/null
scp -q "$B2:/tmp/b2cut.evt" "$TMP/b2cut.evt"
python3 "$HERE/evt2perfetto.py" trace "$HUB" "$TMP/b2cut.evt" --from "$HF" --to "$HT" -o "$OUT"
echo "open in https://ui.perfetto.dev : $OUT"
