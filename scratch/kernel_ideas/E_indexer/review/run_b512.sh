#!/usr/bin/env bash
# Reviewer (gather_b128 claim) repro of the engineer's b=512 (prefill lane shape) gather run.
# ~700 MB of device memory. Output under review/gather_b128/results/.
# Run under: _infra/gpu_run.sh --dev dgpu --mb 720 --label review/E_indexer -- bash review/run_b512.sh
set -u
cd "$(dirname "$0")/gather_b128"
mkdir -p results
export GATHER_B512=1
export GATHER_CANDS=gather_u4_r1,gather_u4_r4,gather_u4_r4_2ipt
./harness_gfx1201 gather . rounds=30 > results/gather_review_b512.txt 2>&1
echo "rc=$? -> review/gather_b128/results/gather_review_b512.txt"
