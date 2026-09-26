#!/usr/bin/env bash
# submit_quant_alias.sh <tag> — aliasing test for the 65536/131072-element quantize pathology:
# same K,b with the x / xq / xscale pointers shifted inside oversized allocations (harness_q).
set -u
cd "$(dirname "$0")"
TAG=${1:-a1}
export C1_DIR="$PWD"
bash ../_infra/gpu_submit.sh --dev dgpu --mb 100 --label C1_dense_decode/quant_alias_$TAG --timeout 200 -- \
    ./harness_q quant 32768 2 0 0 0 : quant 32768 2 65536 0 0 : quant 32768 2 0 65536 0 : quant 32768 2 0 0 4096 \
    : quant 32768 2 4096 8192 0 : quant 32768 2 131072 0 0 : quant 32768 2 0 4096 0 : quant 32768 2 0 256 0 \
    : quant 32768 4 0 0 0 : quant 32768 4 0 65536 0 : quant 32768 4 4096 8192 0 : quant 32768 4 0 256 0 \
    : quant 32768 5 0 0 0 : quant 32768 5 0 65536 0 : quant 32767 2 0 0 0 : quant 32800 2 0 0 0
