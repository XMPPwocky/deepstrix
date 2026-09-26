#!/usr/bin/env bash
# Reviewer repro of the engineer's exact commands (one process per b, as in run_engram.sh):
#   harness engram --m 25600 --b {64,128} --corr --cand q8_0_gemm_wmma_i8x_db
# run_timing.sh <tag>  -> results/t_<tag>_b64.txt, results/t_<tag>_b128.txt
set -u
cd "$(dirname "$0")"
TAG=${1:-r1}
./harness_gfx1201 gfx1201 . engram --m 25600 --b 64  --corr --cand q8_0_gemm_wmma_i8x_db > results/t_${TAG}_b64.txt  2> results/t_${TAG}_b64.err
./harness_gfx1201 gfx1201 . engram --m 25600 --b 128 --corr --cand q8_0_gemm_wmma_i8x_db > results/t_${TAG}_b128.txt 2> results/t_${TAG}_b128.err
grep -h "^==\|^base\|^cand\|^CMP\|rror" results/t_${TAG}_b64.txt results/t_${TAG}_b128.txt
