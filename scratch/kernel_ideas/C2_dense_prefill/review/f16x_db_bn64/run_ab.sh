#!/usr/bin/env bash
# run_ab.sh <tag>: one process per shape, same regime as the engineer (cold weights, 60 rounds,
# inner=1, flush 96 MB), candidate db_bn64 only. Runs INSIDE one scheduler job.
set -u
cd "$(dirname "$0")"
TAG=${1:-r1}
C=q8_0_gemm_wmma_f16x_db_bn64
R=results/ab_$TAG; mkdir -p $R
for sh in kv qa; do
  ./harness_review_gfx1201 gfx1201 . $sh --cand $C > $R/${sh}_b512.txt 2>&1
  ./harness_review_gfx1201 gfx1201 . $sh --b 256 --cand $C > $R/${sh}_b256.txt 2>&1
done
grep -h "^==\|^base\|^cand\|rror" $R/*.txt
