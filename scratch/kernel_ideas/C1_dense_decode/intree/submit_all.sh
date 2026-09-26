#!/usr/bin/env bash
# Integration check jobs (dGPU, <= 250 MB each, serialized by the scheduler):
#   1. scratch harness correctness modes against the IN-TREE hsacos (intree/, build_intree.sh)
#   2. tests/q8_0_tb_bitexact (defaults: tB + wave + pad + fused)
#   3. rollback paths (every knob = 0) + the pre-existing q8_0_matvec_batched test
set -u
cd "$(dirname "$0")/.."
export C1_DIR="$PWD/intree"
SUB="bash ../_infra/gpu_submit.sh"
T=/home/claude-code/deepstrix/target-v41-ki/release/deps
: > intree/tickets.txt
$SUB --dev dgpu --mb 130 --label C1_dense_decode/intree_harness --timeout 300 -- env C1_DIR="$C1_DIR" \
    ./harness gemv qa 1 1 : gemv qa 2 1 : gemv qa 3 1 : gemv qa 4 1 : gemv qa 5 1 : gemv qa 8 1 \
    : gemv kv 4 1 : gemv gate 3 1 : gemv gate 5 1 : gemv down 4 1 : gemv wob 4 1 \
    : grouped 1 : grouped 2 : grouped 3 : grouped 4 : grouped 5 : grouped 8 \
    : quant 5120 4 : quant 32768 2 : quant 32768 4 : quant 2304 3 \
    : shared 1 : shared 2 : shared 3 : shared 4 : shared 5 : shared 8 | tee -a intree/tickets.txt
[ -n "${HARNESS_ONLY:-}" ] || $SUB --dev dgpu --mb 120 --label C1_dense_decode/intree_test_default --timeout 300 -- env C1_DIR="$C1_DIR" \
    $T/q8_0_tb_bitexact-673d25e4dcf40c03 --ignored --test-threads=1 --nocapture | tee -a intree/tickets.txt
[ -n "${HARNESS_ONLY:-}" ] || $SUB --dev dgpu --mb 120 --label C1_dense_decode/intree_test_knobs0 --timeout 300 -- env C1_DIR="$C1_DIR" \
    bash intree/run_tests_knobs0.sh | tee -a intree/tickets.txt
