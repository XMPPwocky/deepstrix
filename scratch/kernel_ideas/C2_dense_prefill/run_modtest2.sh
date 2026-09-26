#!/usr/bin/env bash
# Module-size test on the r1 source: i8x from cand_r1 (8 kernels) vs cand_r1plus (8 + 22 dummies), one process.
set -u
cd "$(dirname "$0")"
R=results/modtest2; mkdir -p $R
./harness_gfx1201 gfx1201 . engram --m 25600 --corr --cand mod:cand_r1:q8_0_gemm_wmma_i8x,mod:cand_r1plus:q8_0_gemm_wmma_i8x,mod:cand_small:q8_0_gemm_wmma_i8x_db,q8_0_gemm_wmma_i8x_db > $R/engram_b64.txt 2>$R/engram_b64.err
./harness_gfx1201 gfx1201 . kv --corr --cand mod:cand_r1:q8_0_gemm_wmma_f16x_pf2,mod:cand_r1plus:q8_0_gemm_wmma_f16x_pf2,mod:cand_r1:q8_0_gemm_wmma_f16x_lb,mod:cand_r1plus:q8_0_gemm_wmma_f16x_lb > $R/kv_b512.txt 2>$R/kv_b512.err
grep -h "^==\|^base\|^cand\|rror" $R/*.txt; grep -h "^CMP" $R/*.txt | grep -c "bitexact=YES"; grep -h "^CMP" $R/*.txt | grep -c "bitexact=no"
