#!/usr/bin/env bash
# Warm-cache control (weights + activations resident, no flush): does the win survive?
set -u
cd "$(dirname "$0")"
C=q8_0_gemm_wmma_f16x_db_bn64
R=results/warm; mkdir -p $R
for sh in kv qa; do
  ./harness_review_gfx1201 gfx1201 . $sh --warm --cand $C > $R/${sh}_b512.txt 2>&1
done
grep -h "^==\|^base\|^cand\|rror" $R/*.txt
