#!/usr/bin/env bash
# Ask a process's evtrace Tier B ring for a dump back to RAW time SINCE (+2 s),
# wait for it, print its path (nothing if none came).
#   scripts/evt_dump_now.sh RING_DIR ROLE SINCE_RAW_NS
# RING_DIR: the hub's ~/logs/evtrace-ring, box 2's /dev/shm/evtrace-ring.
# The request is written temp-then-rename; the dump is complete once its size
# stops changing (the writer streams it in synced 4 MB pieces).
set -u
R=$1; ROLE=$2; SINCE=$3
[ -d "$R" ] || exit 0
S=$(python3 -c "import sys,time; print(int((time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - float(sys.argv[1])) / 1e9) + 2)" "$SINCE")
N=$(ls "$R"/"$ROLE"-dump-*.evt 2>/dev/null | wc -l)
echo "$S" > "$R/.dump-request.tmp" && mv "$R/.dump-request.tmp" "$R/dump-request"
for _ in $(seq 1 120); do
  if [ ! -e "$R/dump-request" ] && [ "$(ls "$R"/"$ROLE"-dump-*.evt 2>/dev/null | wc -l)" -gt "$N" ]; then
    break
  fi
  sleep 0.25
done
F=$(ls -t "$R"/"$ROLE"-dump-*.evt 2>/dev/null | head -1)
[ -n "$F" ] || exit 0
S0=-1
while [ "$(stat -c %s "$F")" != "$S0" ]; do S0=$(stat -c %s "$F"); sleep 0.5; done
echo "$F"
