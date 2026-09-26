#!/usr/bin/env bash
# Correctness of the f16x candidates vs the production f16x on every production shape (incl. tail batches).
set -u
cd "$(dirname "$0")"
C=${1:-q8_0_gemm_wmma_f16x_v2,q8_0_gemm_wmma_f16x_lb,q8_0_gemm_wmma_f16x_pf2,q8_0_gemm_wmma_f16x_bn64,q8_0_gemm_wmma_f16x_64x64}
for sh in qa qb kv woa wob shg shd; do
  echo "=== $sh"
  ./harness_gfx1201 gfx1201 . $sh --corr --short --cand $C 2>/dev/null
done
