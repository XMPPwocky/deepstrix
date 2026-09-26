#!/usr/bin/env bash
# Collect the five review tickets (order as submitted) into results/.
set -u
cd "$(dirname "$0")"
mapfile -t T < results/tickets.txt
names=(run1.txt run2.txt run3.txt corr_xs0.01.txt corr_xs0.003.txt)
for i in 0 1 2 3 4; do
    bash ../../../_infra/gpu_wait.sh "${T[$i]}" > "results/${names[$i]}" 2>&1
    echo "rc=$? -> results/${names[$i]}"
done
for f in run1 run2 run3; do
    echo "=== $f"
    grep -E "^== |^base chain|cand B r1 \+ tB|cand B r1: |cand tB chain|^null|REVIEW|int8_diff=[1-9]|bitexact=NO|error|Segm|refused" "results/$f.txt" | cut -c1-150
done
for f in corr_xs0.01 corr_xs0.003; do
    echo "=== $f"
    grep -E "^#### |REVIEW|int8_diff|bitexact|error|Segm|refused" "results/$f.txt" | cut -c1-170
done
