#!/usr/bin/env bash
# run_small.sh <run-tag> — grouped wo_a, quantize and swiglu baselines (+ candidates when present).
set -u
cd "$(dirname "$0")"
TAG=${1:-r1}
export C1_DIR="$PWD"
for b in 1 2 4 5 8; do
    ../_infra/gpu_run.sh --dev dgpu --mb 115 --label C1_dense_decode/grouped_b${b} --timeout 240 -- \
        ./harness grouped $b > results/grouped_b${b}_${TAG}.txt 2>&1
    grep -E "^CMP|^== |^base|^cand|^null" results/grouped_b${b}_${TAG}.txt
done
for K in 5120 1280 8192 32768 2304; do
    for b in 1 4 8; do
        ../_infra/gpu_run.sh --dev dgpu --mb 64 --label C1_dense_decode/quant_K${K}_b${b} --timeout 120 -- \
            ./harness quant $K $b > results/quant_K${K}_b${b}_${TAG}.txt 2>&1
        grep -E "^CMP|^== |^base|^cand|^null" results/quant_K${K}_b${b}_${TAG}.txt
    done
done
for b in 1 4 8; do
    ../_infra/gpu_run.sh --dev dgpu --mb 64 --label C1_dense_decode/swiglu_b${b} --timeout 120 -- \
        ./harness swiglu $b > results/swiglu_b${b}_${TAG}.txt 2>&1
    grep -E "^== |^base|^cand" results/swiglu_b${b}_${TAG}.txt
done
