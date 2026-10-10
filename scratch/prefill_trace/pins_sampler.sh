#!/usr/bin/env bash
# runs ON box 2: timestamped samples of the expertd page-stats pin figures (read-only, tail of the log)
LOG=/home/claude-code/logs/expertd-sliceD-20261006.log
OUT=${1:-/tmp/b8ee_pins.log}
N=${2:-1200}
i=0
while [ "$i" -lt "$N" ]; do
  line=$(tail -c 4000 "$LOG" | grep -a "page stats" | tail -1)
  pins=$(echo "$line" | grep -a -o "pins [0-9]*/[0-9]* released=[0-9]* new=[0-9]* denied=[0-9]*")
  req=$(echo "$line" | grep -a -o "requests=[0-9]*")
  echo "$(date -u +%FT%TZ) $req $pins" >> "$OUT"
  i=$((i+1))
  sleep 2
done
