#!/usr/bin/env bash
# wait_to.sh TICKET OUTFILE — block on a scheduler ticket, save its full output to results/OUTFILE,
# print the summary lines (mode headers, CMP, table rows, null floor, errors).
set -u
cd "$(dirname "$0")"
T=$1; OUT=results/$2
bash ../_infra/gpu_wait.sh "$T" > "$OUT" 2>&1
echo "rc=$? -> $OUT"
grep -E "^####|^CMP|^== |^base|^cand|^null|shared b=|error|Segm|gpu_|refused|rc=" "$OUT" | cut -c1-160
