#!/usr/bin/env bash
# Reviewer repro of the engineer's job (run_cand.sh cand_wmma <run> all 128 1024 512 128) from
# review/ builds, into review/results/. Usage: bash run_review.sh RUN_ID [shapes...]
set -u
cd "$(dirname "$0")"
mkdir -p results
RUN=${1:-1}; shift || true
SHAPES=${*:-1024 512 128}
export KB_CAND=cand_wmma
export KB_GU_RPW=128
for B in $SHAPES; do
    ./harness cmp $B 96 $RUN 5 > results/cmp_B${B}_run${RUN}.txt 2>&1
    echo "cmp B=$B rc=$?"
done
for B in $SHAPES; do
    ./harness ab $B 96 $RUN 20 > results/ab_B${B}_run${RUN}.txt 2>&1
    echo "ab B=$B rc=$?"
done
grep -h "KBJSON" results/ab_B*_run${RUN}.txt | grep down
