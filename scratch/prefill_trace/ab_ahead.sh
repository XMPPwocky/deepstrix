#!/usr/bin/env bash
# Layer-ahead streaming A/B (owner 10-11 "sure"): alternate the hub's LIVE V41_LM_PREFETCH_AHEAD between 2 and 0 every
# SLOT_S seconds for N_SLOTS slots (after WAIT_FOR's process exits, e.g. the mirror_frac A/B), logging
# "UTC_ISO V41_LM_PREFETCH_AHEAD=<v>" per switch to $OUT; restores 2 (production) at the end (trap: always).
#   ab_ahead.sh [SLOT_S=900] [N_SLOTS=32] > ~/ab_ahead.log 2>&1
set -u
SLOT=${1:-900}; N=${2:-32}
K=/dev/shm/deepstrix-knobs.txt
OUT=/home/claude-code/logs/ab_ahead_20261011.txt
while pgrep -f '^bash /home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe/scratch/prefill_trace/ab_mirror.sh' >/dev/null; do sleep 10; done
setv() {
  grep -v -E '^V41_LM_PREFETCH_AHEAD=' "$K" > "$K.tmp.$$" && echo "V41_LM_PREFETCH_AHEAD=$1" >> "$K.tmp.$$" && cat "$K.tmp.$$" > "$K" && rm -f "$K.tmp.$$" \
    && echo "$(date -u +%FT%T) V41_LM_PREFETCH_AHEAD=$1" | tee -a "$OUT"
}
trap 'setv 2; echo "$(date -u +%FT%T) END (restored 2)" | tee -a "$OUT"' EXIT
trap 'exit 130' INT TERM HUP
echo "$(date -u +%FT%T) START slot=${SLOT}s n=$N arms 0/2" | tee -a "$OUT"
for i in $(seq 0 $((N - 1))); do
  if [ $((i % 2)) = 0 ]; then setv 0 || exit 1; else setv 2 || exit 1; fi
  sleep "$SLOT"
done
