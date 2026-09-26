#!/usr/bin/env bash
# Reviewer (gateup): the REAL per-layer shape: all 384 experts physically present (7.2 GB of
# weights, every pick lands), production bound grid + device count, B=1024 and B=512.
# Usage: bash run_full.sh RUN_ID
set -u
cd "$(dirname "$0")"
RUN=${1:-1}
export KB_CAND=cand_wmma
export KB_GU_RPW=128
for B in 1024 512; do
    ./harness_gu ab $B 384 $RUN 10 > results/full384_ab_B${B}_run${RUN}.txt 2>&1
    echo "full384 ab B=$B rc=$?"
done
grep -h "review\|setup\|KBJSON" results/full384_ab_B*_run${RUN}.txt
