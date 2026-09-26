#!/usr/bin/env bash
# Reviewer (gateup): A/B with the EXACT grid (no empty tail) vs the bound grid, 96 experts,
# to price how much of the measured gap is the cheaper empty-tail of the 18-WG-wide candidate.
# Usage: bash run_exact.sh RUN_ID
set -u
cd "$(dirname "$0")"
RUN=${1:-1}
export KB_CAND=cand_wmma
export KB_GU_RPW=128
for B in 1024 512; do
    KB_EXACT_GRID=1 ./harness_gu ab $B 96 $RUN 20 > results/exact_ab_B${B}_run${RUN}.txt 2>&1
    echo "exact ab B=$B rc=$?"
    KB_EXACT_GRID=0 ./harness_gu ab $B 96 $RUN 20 > results/bound_ab_B${B}_run${RUN}.txt 2>&1
    echo "bound ab B=$B rc=$?"
done
grep -h "review\|setup\|KBJSON" results/exact_ab_B*_run${RUN}.txt results/bound_ab_B*_run${RUN}.txt
