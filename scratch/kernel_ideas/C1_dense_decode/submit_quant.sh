#!/usr/bin/env bash
# submit_quant.sh <tag> — the K=32768 quantize anomaly (wo_a input, FP:6126): b sweep + neighbours.
set -u
cd "$(dirname "$0")"
TAG=${1:-q1}
export C1_DIR="$PWD"
bash ../_infra/gpu_submit.sh --dev dgpu --mb 100 --label C1_dense_decode/quant32k_$TAG --timeout 200 -- \
    ./harness quant 32768 1 : quant 32768 2 : quant 32768 4 : quant 32768 5 : quant 32768 8 \
    : quant 16384 4 : quant 16384 8 : quant 65536 1 : quant 65536 2 : quant 8192 8 : quant 32768 4
