#!/usr/bin/env bash
# Reviewer's own harness: extra-shape correctness + cold-regime sensitivity (1 / 32 / 64 W copies, b=1/4/8).
set -u
cd "$(dirname "$0")"
tag=${1:-v1}
REVIEW_DIR=$PWD
export REVIEW_DIR
./review_mv_gfx1201 cmp > results/review_cmp_${tag}.txt 2>&1; echo "cmp rc=$?"
grep "REVIEW_CMP\|bitexact=no\|nonfinite=[1-9]" results/review_cmp_${tag}.txt
for b in 1 4 8; do for nc in 1 32 64; do
  ./review_mv_gfx1201 time $b $nc > results/review_time_b${b}_nc${nc}_${tag}.txt 2>&1
  grep -h "^==\|^prod\|^cand" results/review_time_b${b}_nc${nc}_${tag}.txt
done; done
true
