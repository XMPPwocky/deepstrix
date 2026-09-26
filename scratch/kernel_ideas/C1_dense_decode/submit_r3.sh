#!/usr/bin/env bash
# submit_r3.sh <tag> — enqueue the three C1 measurement jobs (shared chain, gemv sweep, small kernels)
# on the dGPU scheduler; prints one ticket per line into results/tickets_<tag>.txt.
# Each job = ONE harness process over many ':'-separated modes (README: batch, <= 300 s).
set -u
cd "$(dirname "$0")"
TAG=${1:-r3}
export C1_DIR="$PWD"
SUB="bash ../_infra/gpu_submit.sh"
: > "results/tickets_${TAG}.txt"
$SUB --dev dgpu --mb 250 --label C1_dense_decode/shared_$TAG --timeout 240 -- \
    ./harness shared 1 : shared 2 : shared 4 : shared 5 : shared 8 | tee -a "results/tickets_${TAG}.txt"
$SUB --dev dgpu --mb 1000 --label C1_dense_decode/gemv_sweep_$TAG --timeout 300 -- \
    ./harness gemv qa 1 : gemv qa 2 : gemv qa 4 : gemv qa 5 : gemv qa 8 \
    : gemv qb 1 : gemv qb 4 : gemv qb 5 : gemv qb 8 \
    : gemv kv 1 : gemv kv 4 : gemv kv 5 \
    : gemv wob 1 : gemv wob 4 : gemv wob 5 : gemv wob 8 \
    : gemv gate 1 : gemv gate 4 : gemv gate 5 \
    : gemv down 1 : gemv down 4 : gemv down 5 : gemv down 8 \
    : gemv engram_full 1 : gemv engram_full 4 : gemv engram_full 5 \
    : gemv head_full 1 : gemv head_full 4 : gemv head_full 5 : gemv head_full 8 | tee -a "results/tickets_${TAG}.txt"
$SUB --dev dgpu --mb 250 --label C1_dense_decode/small_$TAG --timeout 240 -- \
    ./harness grouped 1 : grouped 2 : grouped 4 : grouped 5 : grouped 8 \
    : quant 5120 1 : quant 5120 4 : quant 5120 8 : quant 1280 4 : quant 8192 4 : quant 32768 4 : quant 2304 4 \
    : swiglu 1 : swiglu 4 \
    : lanes qa 4 4 : lanes qa 5 5 : lanes wob 4 4 : lanes gate 3 3 : lanes down 4 4 : lanes qb 4 4 | tee -a "results/tickets_${TAG}.txt"
