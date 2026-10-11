#!/usr/bin/env bash
# 1 Hz NVMe thermal + read-throughput logger (box 2): "<unix_s> <dev> <sectors_read> <ms_reading> <reads> <composite_C>
# <sensor1_C> <sensor2_C>" per whole nvme device, for DUR_S seconds. Is a drive's throughput capped when it is hot?
#   nvme_thermal.sh OUT [DUR_S=7200]
OUT=$1; DUR=${2:-7200}
end=$(( $(date +%s) + DUR ))
while [ "$(date +%s)" -lt "$end" ]; do
  t=$(date +%s.%N)
  for n in /sys/class/nvme/nvme*; do
    b=$(basename "$n")n1
    temps=$(for i in 1 2 3; do f=$(ls "$n"/hwmon*/temp${i}_input 2>/dev/null | head -1); [ -n "$f" ] && echo -n "$(( $(cat "$f") / 1000 )) " || echo -n "- "; done)
    awk -v t="$t" -v b="$b" -v tp="$temps" '$3 == b {print t, $3, $6, $7, $4, tp}' /proc/diskstats
  done >> "$OUT"
  sleep 1
done
