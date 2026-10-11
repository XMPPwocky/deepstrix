#!/usr/bin/env bash
# Read-ahead lane A/B (owner 10-11 "go"): box 2 08cf447e (readahead class + prompt landing) + hub e610996c
# (V41_LM_PREFETCH_LAST_ONLY=1 default). Alternates the hub's LIVE V41_LM_PREFETCH_AHEAD between 0 and 2 every SLOT_S
# seconds for N_SLOTS slots, logging "UTC_ISO V41_LM_PREFETCH_AHEAD=<v>" per switch to $OUT; restores 0 (production)
# at the end (trap: always; SIGKILL script + sleep to stop WITHOUT the trap).
#   ab_ra.sh [SLOT_S=900] [N_SLOTS=24] > ~/ab_ra.log 2>&1
set -u
SLOT=${1:-900}; N=${2:-24}
K=/dev/shm/deepstrix-knobs.txt
OUT=/home/claude-code/logs/ab_ra_20261011.txt
setv() {
  grep -v -E '^V41_LM_PREFETCH_AHEAD=' "$K" > "$K.tmp.$$" && echo "V41_LM_PREFETCH_AHEAD=$1" >> "$K.tmp.$$" && cat "$K.tmp.$$" > "$K" && rm -f "$K.tmp.$$" \
    && echo "$(date -u +%FT%T) V41_LM_PREFETCH_AHEAD=$1" | tee -a "$OUT"
}
trap 'setv 0; echo "$(date -u +%FT%T) END (restored 0)" | tee -a "$OUT"' EXIT
trap 'exit 130' INT TERM HUP
echo "$(date -u +%FT%T) START slot=${SLOT}s n=$N arms 2/0 (readahead lane, last-only)" | tee -a "$OUT"
for i in $(seq 0 $((N - 1))); do
  if [ $((i % 2)) = 0 ]; then setv 2 || exit 1; else setv 0 || exit 1; fi
  sleep "$SLOT"
done
