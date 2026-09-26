#!/usr/bin/env bash
# wait_all.sh NAME TICKET [NAME TICKET ...] — block on each ticket in turn, save full output to results/NAME.txt,
# print the quantize summary lines.
set -u
cd "$(dirname "$0")"
while [ $# -ge 2 ]; do
  NAME=$1; T=$2; shift 2
  bash ../../../_infra/gpu_wait.sh "$T" > "results/$NAME.txt" 2>&1
  echo "#### $NAME rc=$? ($T)"
  grep -E "^== quantize|^base q8_0_quantize|^cand q8_0_quantize|^null 1-WG|^CMP quant|^CASE|^ODD SUMMARY|first diff|refused|error|Segm" "results/$NAME.txt" | awk '/^== quantize|^CASE|^ODD/{p=1} p' | cut -c1-150
done
