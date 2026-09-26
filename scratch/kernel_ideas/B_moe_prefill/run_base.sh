#!/usr/bin/env bash
# Clean baseline (hub down): production chain at B=128/512/1024, 96 expert copies (1.8 GB).
# Usage: bash run_base.sh RUN_ID   (run under gpu_submit.sh --dev igpu --mb 2100)
set -u
cd "$(dirname "$0")"
RUN=${1:-1}
for B in 128 512 1024; do
    ./harness base $B 96 $RUN 25 > results/base_B${B}_run${RUN}.txt 2>&1
    echo "B=$B rc=$?"
done
