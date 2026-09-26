#!/usr/bin/env bash
# kwide ablations at B=1024 and B=512 (96 experts). Usage: bash run_ablate.sh RUN_ID
set -u
cd "$(dirname "$0")"
RUN=${1:-1}
for B in 1024 512; do
    ./harness ablate $B 96 $RUN 20 > results/ablate_B${B}_run${RUN}.txt 2>&1
    echo "B=$B rc=$?"
done
