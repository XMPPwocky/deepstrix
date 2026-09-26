#!/usr/bin/env bash
# W0 test: explicit s_wait_loadcnt 0 before the WMMA block. run_w0.sh <tag>
set -u
cd "$(dirname "$0")"
TAG=${1:-r1}; R=results/w0_$TAG; mkdir -p $R
./harness_gfx1201 gfx1201 . engram --m 25600 --corr --cand i8x_r1,q8_0_gemm_wmma_i8x,q8_0_gemm_wmma_i8x_w0,q8_0_gemm_wmma_i8x_db,q8_0_gemm_wmma_i8x_db_w0 > $R/engram_b64.txt 2>$R/engram_b64.err
for sh in kv qb woa wob shg; do
  ./harness_gfx1201 gfx1201 . $sh --corr --cand q8_0_gemm_wmma_f16x_pf2,q8_0_gemm_wmma_f16x_pf2_w0,q8_0_gemm_wmma_f16x_db,q8_0_gemm_wmma_f16x_db_w0 > $R/${sh}_b512.txt 2>$R/${sh}_b512.err
done
grep -h "^==\|^base\|^cand\|rror" $R/*.txt; grep -h "^CMP" $R/*.txt | grep -c "bitexact=YES"; grep -h "^CMP" $R/*.txt | grep -c "bitexact=no"
