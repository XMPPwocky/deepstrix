#!/usr/bin/env bash
# wait_to.sh TICKET OUTFILE — block on a scheduler ticket, save full output to results/OUTFILE, print summary.
set -u
cd "$(dirname "$0")"
T=$1; OUT=results/$2
bash ../../../_infra/gpu_wait.sh "$T" > "$OUT" 2>&1
echo "rc=$? -> $OUT"
grep -E "^####|^CMP|^== |^base|^cand|^null|^FAIL|^MISSING|^ODD|error|Segm|gpu_|refused|rc=" "$OUT" | cut -c1-150
