#!/usr/bin/env bash
# Reviewer (gather_b128 claim) extra timing probes (see gather_b128/gtime.cpp): 235K-row store (235 MB)
# + 2 x 268 MB destinations + 64 MB flush ~ 840 MB. usage: run_gtime.sh <run-id>
# Run under: _infra/gpu_run.sh --dev dgpu --mb 900 --label review/E_indexer -- bash review/run_gtime.sh N
set -u
cd "$(dirname "$0")/gather_b128"
run=${1:-1}
mkdir -p results
./gtime_gfx1201 . 30 > "results/gtime_run$run.txt" 2>&1
echo "rc=$? -> review/gather_b128/results/gtime_run$run.txt"
