#!/usr/bin/env bash
# Module-composition test: the same kernels from the 40-kernel module (cand) vs an 8-kernel module
# (cand_small) vs the recovered r1 binary, in ONE process per shape. run_modtest.sh <tag>
set -u
cd "$(dirname "$0")"
TAG=${1:-r1}; R=results/modtest_$TAG; mkdir -p $R
./harness_gfx1201 gfx1201 . engram --m 25600 --corr --cand i8x_r1,r1c_i8x,small:r1c_i8x,q8_0_gemm_wmma_i8x,small:q8_0_gemm_wmma_i8x,q8_0_gemm_wmma_i8x_db,small:q8_0_gemm_wmma_i8x_db > $R/engram_b64.txt 2>$R/engram_b64.err
for sh in kv qb woa; do
  ./harness_gfx1201 gfx1201 . $sh --corr --cand q8_0_gemm_wmma_f16x_pf2,small:q8_0_gemm_wmma_f16x_pf2,q8_0_gemm_wmma_f16x_db_bn64,small:q8_0_gemm_wmma_f16x_db_bn64,q8_0_gemm_wmma_f16x_256x128,small:q8_0_gemm_wmma_f16x_256x128 > $R/${sh}_b512.txt 2>$R/${sh}_b512.err
done
grep -h "^==\|^base\|^cand\|rror" $R/*.txt; grep -h "^CMP" $R/*.txt | grep -c "bitexact=YES"; grep -h "^CMP" $R/*.txt | grep -c "bitexact=no"
