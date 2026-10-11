#!/usr/bin/env bash
# Box-1 swap activity at 1 Hz: "<unix_s> <pswpin> <pswpout> <SwapFree_kB> <MemAvailable_kB>" to OUT for DUR_S seconds.
#   swapsample.sh OUT [DUR_S=14400]
OUT=$1; DUR=${2:-14400}
end=$(( $(date +%s) + DUR ))
while [ "$(date +%s)" -lt "$end" ]; do
  i=$(awk '/^pswpin /{print $2}' /proc/vmstat); o=$(awk '/^pswpout /{print $2}' /proc/vmstat)
  sf=$(awk '/^SwapFree:/{print $2}' /proc/meminfo); ma=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  echo "$(date +%s.%N) $i $o $sf $ma" >> "$OUT"
  sleep 1
done
