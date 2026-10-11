#!/usr/bin/env bash
# miss_par A/B (owner 10-11 "run it"): alternate box 2's LIVE miss_par (parallel demand-miss reads; <= the startup
# V41_B2_MISS_PAR=4) between 4 and 1 (production) every SLOT_S seconds for N_SLOTS slots, logging
# "UTC_ISO miss_par=<v>" to $OUT. Target: cold FIRST windows (20-50 misses/request; box 2's drives run shallow: median
# second 1.6 / 0.84 GB/s vs ~10 GB/s combined when queued deep). Ends on 1 (trap; stop with SIGKILL + its sleep).
#   ab_misspar.sh [SLOT_S=1200] [N_SLOTS=12] > ~/ab_misspar.log 2>&1
set -u
SLOT=${1:-1200}; N=${2:-12}
OUT=/home/claude-code/logs/ab_misspar_20261011.txt
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 10.43.0.95"
setv() {
  $SSH "K=~/expertd-knobs.txt; sed 's/^miss_par=.*/miss_par=$1/' \$K > \$K.tmp.\$\$ && mv \$K.tmp.\$\$ \$K && grep -q '^miss_par=$1\$' \$K" \
    && echo "$(date -u +%FT%T) miss_par=$1" | tee -a "$OUT"
}
trap 'setv 1; echo "$(date -u +%FT%T) END (restored 1)" | tee -a "$OUT"' EXIT
trap 'exit 130' INT TERM HUP
echo "$(date -u +%FT%T) START slot=${SLOT}s n=$N arms 4/1" | tee -a "$OUT"
for i in $(seq 0 $((N - 1))); do
  if [ $((i % 2)) = 0 ]; then setv 4 || exit 1; else setv 1 || exit 1; fi
  sleep "$SLOT"
done
