#!/usr/bin/env bash
# passive: sample thunderbolt0 counters every 2 s for N samples (no load added)
N=${1:-15}
prev=""
for i in $(seq 1 "$N"); do
  rxb=$(cat /sys/class/net/thunderbolt0/statistics/rx_bytes)
  txb=$(cat /sys/class/net/thunderbolt0/statistics/tx_bytes)
  rxp=$(cat /sys/class/net/thunderbolt0/statistics/rx_packets)
  txp=$(cat /sys/class/net/thunderbolt0/statistics/tx_packets)
  rxm=$(cat /sys/class/net/thunderbolt0/statistics/rx_missed_errors)
  rxe=$(cat /sys/class/net/thunderbolt0/statistics/rx_errors)
  rxl=$(cat /sys/class/net/thunderbolt0/statistics/rx_length_errors)
  now=$(date +%s.%N)
  if [ -n "$prev" ]; then
    set -- $prev
    awk -v n="$now" -v o="$1" -v rxb="$rxb" -v txb="$txb" -v rxp="$rxp" -v txp="$txp" -v rxm="$rxm" -v rxe="$rxe" -v rxl="$rxl" \
        -v prxb="$2" -v ptxb="$3" -v prxp="$4" -v ptxp="$5" -v prxm="$6" -v prxe="$7" -v prxl="$8" \
        'BEGIN{dt=n-o; printf "rx %7.1f MB/s  tx %7.1f MB/s  rx_pkts %8.0f/s tx_pkts %8.0f/s  missed %5.1f/s errs %5.1f/s len %4.1f/s\n",
         (rxb-prxb)/dt/1e6,(txb-ptxb)/dt/1e6,(rxp-prxp)/dt,(txp-ptxp)/dt,(rxm-prxm)/dt,(rxe-prxe)/dt,(rxl-prxl)/dt}'
  fi
  prev="$now $rxb $txb $rxp $txp $rxm $rxe $rxl"
  sleep 2
done
