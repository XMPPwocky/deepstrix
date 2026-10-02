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
# The newest dump before the request: a NEW name is ours (a count can stall
# when pruning removes an old dump as the new one lands).
OLD=$(ls -t "$R"/"$ROLE"-dump-*.evt 2>/dev/null | head -1)
echo "$S" > "$R/.dump-request.tmp" && mv "$R/.dump-request.tmp" "$R/dump-request"
F=
for _ in $(seq 1 120); do
  NEWEST=$(ls -t "$R"/"$ROLE"-dump-*.evt 2>/dev/null | head -1)
  if [ ! -e "$R/dump-request" ] && [ -n "$NEWEST" ] && [ "$NEWEST" != "$OLD" ]; then
    F=$NEWEST
    break
  fi
  sleep 0.25
done
if [ -z "$F" ]; then
  # No ring acted on it (Tier B off, a stale dir): leave no request behind
  # for a ring started later, and no stale dump.
  rm -f "$R/dump-request"
  exit 0
fi
S0=-1
while [ "$(stat -c %s "$F")" != "$S0" ]; do S0=$(stat -c %s "$F"); sleep 0.5; done
echo "$F"
