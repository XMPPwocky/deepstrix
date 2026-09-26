#!/usr/bin/env bash
# Repeat runs of the candidates that won in r1 (separate process per run tag).
# usage: bash jobs/winners.sh <run-tag>
set -u
cd "$(dirname "$0")/.."
tag=${1:-r2}
export HARNESS_DIR=$PWD
./harness_gfx1201 gemm 512        > results/gemm_512_warm_${tag}.txt 2>&1
./harness_gfx1201 gemm 512 flush  > results/gemm_512_flush_${tag}.txt 2>&1
./harness_gfx1201 rms 1           > results/rms_b1_${tag}.txt 2>&1
./harness_gfx1201 rms 4           > results/rms_b4_${tag}.txt 2>&1
./harness_gfx1201 rms 1 direct    > results/rms_b1_direct_${tag}.txt 2>&1
grep -h "^==\|^[a-zA-Z].*[0-9]\.[0-9][0-9] " results/gemm_512_warm_${tag}.txt results/gemm_512_flush_${tag}.txt results/rms_b1_${tag}.txt results/rms_b4_${tag}.txt results/rms_b1_direct_${tag}.txt
grep -h "^CMP" results/gemm_512_warm_${tag}.txt results/rms_b1_${tag}.txt results/rms_b4_${tag}.txt | grep -c "bitexact=YES"
grep -h "^CMP" results/gemm_512_warm_${tag}.txt results/rms_b1_${tag}.txt results/rms_b4_${tag}.txt | grep -c "bitexact=no"
