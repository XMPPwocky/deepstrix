#!/usr/bin/env bash
# WMMA candidates: correctness vs production at B=1024/512/128 (+ odd tail shapes), then
# interleaved A/B at B=1024 and 512. Usage: bash run_wmma.sh RUN_ID [cmp|ab|all]
set -u
cd "$(dirname "$0")"
RUN=${1:-1}; WHAT=${2:-all}
if [ "$WHAT" = cmp ] || [ "$WHAT" = all ]; then
    for B in 1024 512 128 37; do
        ./harness cmp $B 96 $RUN 5 > results/wmma_cmp_B${B}_run${RUN}.txt 2>&1
        echo "cmp B=$B rc=$?"
    done
fi
if [ "$WHAT" = ab ] || [ "$WHAT" = all ]; then
    for B in 1024 512; do
        ./harness ab $B 96 $RUN 20 > results/wmma_ab_B${B}_run${RUN}.txt 2>&1
        echo "ab B=$B rc=$?"
    done
fi
