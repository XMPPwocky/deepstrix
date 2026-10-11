#!/usr/bin/env bash
# Last-unit evict-behind A/B (docs/v41/B2_LAST_WINDOW_EVICT_BEHIND_DESIGN.md sec 4): alternate box 2's LIVE
# prefill_budget_last between 0 (ON) and unset (OFF = unlimited) every SLOT_S seconds for N_SLOTS slots; log
# "UTC_ISO budget_last=0" / "UTC_ISO budget_last=999999" (999999 = the OFF arm, line removed) to $OUT. 20-min slots so
# the arms decorrelate from the hub's 15-min streaming A/B. Ends OFF (trap: always).
#   ab_last.sh [SLOT_S=1200] [N_SLOTS=24] > ~/ab_last.log 2>&1
set -u
SLOT=${1:-1200}; N=${2:-24}
OUT=/home/claude-code/logs/ab_last_20261011.txt
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 10.43.0.95"
setv() {  # $1 = on|off
  if [ "$1" = on ]; then
    $SSH "K=~/expertd-knobs.txt; { grep -v '^prefill_budget_last=' \$K; echo prefill_budget_last=0; } > \$K.tmp.\$\$ && mv \$K.tmp.\$\$ \$K && grep -q '^prefill_budget_last=0\$' \$K" \
      && echo "$(date -u +%FT%T) budget_last=0" | tee -a "$OUT"
  else
    $SSH "K=~/expertd-knobs.txt; grep -v '^prefill_budget_last=' \$K > \$K.tmp.\$\$ && mv \$K.tmp.\$\$ \$K && ! grep -q '^prefill_budget_last=' \$K" \
      && echo "$(date -u +%FT%T) budget_last=999999" | tee -a "$OUT"
  fi
}
trap 'setv off; echo "$(date -u +%FT%T) END (OFF)" | tee -a "$OUT"' EXIT
trap 'exit 130' INT TERM HUP
echo "$(date -u +%FT%T) START slot=${SLOT}s n=$N arms ON(0)/OFF" | tee -a "$OUT"
for i in $(seq 0 $((N - 1))); do
  if [ $((i % 2)) = 0 ]; then setv on || exit 1; else setv off || exit 1; fi
  sleep "$SLOT"
done
