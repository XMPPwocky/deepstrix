#!/usr/bin/env bash
# b=1024 check (the engineer's table shows q_a LOSING at 1024): kv and qa, cold, one process each.
set -u
cd "$(dirname "$0")"
C=q8_0_gemm_wmma_f16x_db_bn64
R=results/b1024; mkdir -p $R
for sh in kv qa; do
  ./harness_review_gfx1201 gfx1201 . $sh --b 1024 --cand $C > $R/${sh}_b1024.txt 2>&1
done
grep -h "^==\|^base\|^cand\|rror" $R/*.txt
