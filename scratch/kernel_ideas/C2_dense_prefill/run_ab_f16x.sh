#!/usr/bin/env bash
# Interleaved A/B of the f16x candidates vs production f16x, every prefill shape, cold weights (flush).
# run_ab_f16x.sh <run-tag> [cand list] [extra harness args]
set -u
cd "$(dirname "$0")"
TAG=${1:-r1}; shift || true
C=${1:-q8_0_gemm_wmma_f16x_v2,q8_0_gemm_wmma_f16x_lb,q8_0_gemm_wmma_f16x_pf2,q8_0_gemm_wmma_f16x_bn64,q8_0_gemm_wmma_f16x_64x64}; shift || true
R=results/ab_f16x_$TAG; mkdir -p $R
for sh in qa qb kv woa wob shg shd; do
  ./harness_gfx1201 gfx1201 . $sh --cand $C "$@" > $R/${sh}_b512.txt 2>&1
done
for sh in qa kv shg; do
  ./harness_gfx1201 gfx1201 . $sh --b 256 --cand $C "$@" > $R/${sh}_b256.txt 2>&1
  ./harness_gfx1201 gfx1201 . $sh --b 1024 --cand $C "$@" > $R/${sh}_b1024.txt 2>&1
done
grep -h "^==\|^base\|^cand\|rror" $R/*.txt
