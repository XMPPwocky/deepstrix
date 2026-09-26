#!/usr/bin/env bash
# Reviewer (gather_b128 claim) correctness checker (see gather_b128/gcheck.cpp). ~1.2 GB of device
# memory (b=1024 destinations). Output under review/gather_b128/results/gcheck.txt.
# Run under: _infra/gpu_run.sh --dev dgpu --mb 1300 --label review/E_indexer -- bash review/run_gcheck.sh
set -u
cd "$(dirname "$0")/gather_b128"
mkdir -p results
./gcheck_gfx1201 . > results/gcheck.txt 2>&1
rc=$?
cat results/gcheck.txt
echo "rc=$rc -> review/gather_b128/results/gcheck.txt"
