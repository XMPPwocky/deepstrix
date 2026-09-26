#!/usr/bin/env bash
# WMMA candidates: extra separate-process A/B runs (B=1024/512/128) + chunked-members
# correctness at B=2048 (>32 members per expert -> 2 work items). Usage: bash run_wmma_confirm.sh RUN_ID
set -u
cd "$(dirname "$0")"
RUN=${1:-2}
for B in 1024 512 128; do
    ./harness ab $B 96 $RUN 20 > results/wmma_ab_B${B}_run${RUN}.txt 2>&1
    echo "ab B=$B rc=$?"
done
./harness cmp 2048 96 $RUN 5 > results/wmma_cmp_B2048_run${RUN}.txt 2>&1
echo "cmp B=2048 rc=$?"
