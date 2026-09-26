#!/usr/bin/env bash
# Reviewer A/B (f16x_256x128 claim): production f16x vs 256x128 on qb and wo_a (the claimed sites),
# cold weights (96 MB flush), 60 rounds, inner=1, direct launch = the engineer's regime.
# run_ab.sh <tag> [b]
set -u
cd "$(dirname "$0")"
TAG=${1:-r1}; B=${2:-512}
R=results/ab_$TAG; mkdir -p $R
C=q8_0_gemm_wmma_f16x_256x128
for sh in qb woa; do
  ./harness_gfx1201 gfx1201 . $sh --b $B --cand $C > $R/${sh}_b$B.txt 2>&1
done
grep -h "^==\|^base\|^cand\|rror" $R/*.txt
