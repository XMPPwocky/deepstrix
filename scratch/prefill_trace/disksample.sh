#!/usr/bin/env bash
# Sample /proc/diskstats for whole nvme devices every second: "<unix_s> <dev> <sectors_read> <ms_reading> <reads>" lines
# to OUT, for DUR_S seconds. Light (one awk per second). Runs on either box.
#   disksample.sh OUT [DUR_S=28800]
OUT=$1; DUR=${2:-28800}
end=$(( $(date +%s) + DUR ))
while [ "$(date +%s)" -lt "$end" ]; do
  awk -v t="$(date +%s.%N)" '$3 ~ /^nvme[0-9]+n1$/ {print t, $3, $6, $7, $4}' /proc/diskstats >> "$OUT"
  sleep 1
done
