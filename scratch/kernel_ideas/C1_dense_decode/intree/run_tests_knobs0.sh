#!/usr/bin/env bash
# Rollback paths: every C1 knob = 0 must restore the previous kernels (the wrapper comparisons in
# q8_0_tb_bitexact then pit the OLD path against the explicit old symbols), plus the pre-existing
# bpack test on both settings. One GPU process at a time (scheduler job).
set -u
T=/home/claude-code/deepstrix/target-v41-ki/release/deps
rc=0
echo "### q8_0_tb_bitexact with V41_GEMV_TB=0 V41_Q8_QUANT_WAVE=0 V41_Q8_QUANT_GRID_PAD=0 V41_SHARED_FUSED=0"
V41_GEMV_TB=0 V41_Q8_QUANT_WAVE=0 V41_Q8_QUANT_GRID_PAD=0 V41_SHARED_FUSED=0 \
    $T/q8_0_tb_bitexact-673d25e4dcf40c03 --ignored --test-threads=1 2>&1 | grep -E "^test |test result|PASS|panicked|FAIL" || rc=1
echo "### q8_0_matvec_batched (defaults)"
$T/q8_0_matvec_batched-a28708b3ee09561f --ignored --test-threads=1 --nocapture 2>&1 | grep -E "^test |test result|overall|PASS|panicked" || rc=1
echo "### q8_0_matvec_batched with V41_GEMV_TB=0"
V41_GEMV_TB=0 $T/q8_0_matvec_batched-a28708b3ee09561f --ignored --test-threads=1 --nocapture 2>&1 | grep -E "^test |test result|overall|PASS|panicked" || rc=1
exit $rc
