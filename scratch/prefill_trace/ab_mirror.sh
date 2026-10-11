#!/usr/bin/env bash
# mirror_frac A/B (owner 10-11 "ok"): alternate box 2's LIVE mirror_frac between ARM_A and ARM_B every SLOT_S seconds
# for N_SLOTS slots, logging "UTC_ISO mirror_frac=<v>" per switch to $OUT; restores 0.70 at the end (trap: always).
#   ab_mirror.sh [SLOT_S=900] [N_SLOTS=8] > ~/ab_mirror.log 2>&1
set -u
SLOT=${1:-900}; N=${2:-8}
ARM_A=${ARM_A:-1.0}; ARM_B=${ARM_B:-0.70}
OUT=${OUT:-/home/claude-code/logs/ab_mirror_20261011.txt}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 10.43.0.95"
setv() {
  $SSH "K=~/expertd-knobs.txt; sed 's/^mirror_frac=.*/mirror_frac=$1/' \$K > \$K.tmp.\$\$ && mv \$K.tmp.\$\$ \$K && grep -q '^mirror_frac=$1\$' \$K" \
    && echo "$(date -u +%FT%T) mirror_frac=$1" | tee -a "$OUT"
}
trap 'setv 0.70; echo "$(date -u +%FT%T) END (restored 0.70)" | tee -a "$OUT"' EXIT
trap 'exit 130' INT TERM HUP
echo "$(date -u +%FT%T) START slot=${SLOT}s n=$N arms $ARM_A/$ARM_B" | tee -a "$OUT"
for i in $(seq 0 $((N - 1))); do
  if [ $((i % 2)) = 0 ]; then setv $ARM_A || exit 1; else setv $ARM_B || exit 1; fi
  sleep "$SLOT"
done
