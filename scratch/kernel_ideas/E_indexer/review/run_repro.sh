#!/usr/bin/env bash
# Reviewer (gather_b128 claim) repro of the engineer's exact gather confirm run: harness section
# `gather`, rounds=50, the three confirm candidates; hsacos rebuilt into review/gather_b128/ and
# output under review/gather_b128/results/.  usage: run_repro.sh <run-id> [harness opts, e.g. b=1,4,8]
# Run under: _infra/gpu_run.sh --dev dgpu --mb 156 --label review/E_indexer -- bash review/run_repro.sh N
set -u
cd "$(dirname "$0")/gather_b128"
run=$1; shift
mkdir -p results
export GATHER_CANDS=gather_u4_r1,gather_u4_r4,gather_u4_r4_2ipt
./harness_gfx1201 gather . rounds=50 "$@" > "results/gather_review_run$run.txt" 2>&1
echo "rc=$? -> review/gather_b128/results/gather_review_run$run.txt"
