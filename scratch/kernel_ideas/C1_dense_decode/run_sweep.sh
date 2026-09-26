#!/usr/bin/env bash
# run_sweep.sh <run-tag> [shapes...]  — baseline + candidate A/B over the decode shapes and b in
# {1,2,4,5,8}; every GPU touch goes through gpu_run.sh (--dev dgpu). Results in results/.
set -u
cd "$(dirname "$0")"
TAG=${1:-r1}; shift || true
SHAPES=${*:-"qa qb kv wob gate down engram head"}
export C1_DIR="$PWD"
for sh in $SHAPES; do
    for b in 1 2 4 5 8; do
        ../_infra/gpu_run.sh --dev dgpu --mb 115 --label C1_dense_decode/gemv_${sh}_b${b} --timeout 240 -- \
            ./harness gemv $sh $b > results/gemv_${sh}_b${b}_${TAG}.txt 2>&1
        grep -E "^CMP|^== |^base|^cand|^null" results/gemv_${sh}_b${b}_${TAG}.txt
    done
done
