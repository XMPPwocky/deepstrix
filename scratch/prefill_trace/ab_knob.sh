#!/usr/bin/env bash
# Generic live-knob A/B on the hub: alternate KNOB between V1 and V0 (V1 first) every SLOT_S seconds for N slots via the
# hub knob file, logging "UTC_ISO KNOB=<v>" per switch to OUT; restores V0 (production) at the end (trap: always;
# SIGKILL the script + its sleep to stop WITHOUT the trap).
#   ab_knob.sh KNOB V0 V1 [SLOT_S=1200] [N=12] OUT > ~/ab_knob.log 2>&1
set -u
KNOB=$1; V0=$2; V1=$3; SLOT=${4:-1200}; N=${5:-12}; OUT=$6
K=/dev/shm/deepstrix-knobs.txt
setv() {
  grep -v -E "^$KNOB=" "$K" > "$K.tmp.$$" && echo "$KNOB=$1" >> "$K.tmp.$$" && cat "$K.tmp.$$" > "$K" && rm -f "$K.tmp.$$" \
    && echo "$(date -u +%FT%T) $KNOB=$1" | tee -a "$OUT"
}
trap 'setv '"$V0"'; echo "$(date -u +%FT%T) END (restored '"$V0"')" | tee -a "$OUT"' EXIT
trap 'exit 130' INT TERM HUP
echo "$(date -u +%FT%T) START $KNOB slot=${SLOT}s n=$N arms $V1/$V0" | tee -a "$OUT"
for i in $(seq 0 $((N - 1))); do
  if [ $((i % 2)) = 0 ]; then setv "$V1" || exit 1; else setv "$V0" || exit 1; fi
  sleep "$SLOT"
done
