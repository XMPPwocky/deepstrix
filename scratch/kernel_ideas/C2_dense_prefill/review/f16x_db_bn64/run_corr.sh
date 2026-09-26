#!/usr/bin/env bash
# Correctness of db_bn64 vs the production f16x on all 7 shapes at b=512 (tails 1..511) and on
# kv/qa/shg/shd at b=1024 (tails incl. 513, 1000, 1023); wo_a exercises the 8 groups.
set -u
cd "$(dirname "$0")"
C=q8_0_gemm_wmma_f16x_db_bn64
for sh in qa qb kv woa wob shg shd; do
  echo "=== $sh b=512"
  ./harness_review_gfx1201 gfx1201 . $sh --corr --short --cand $C 2>/dev/null | grep "^CMP"
done
for sh in kv qa shg shd; do
  echo "=== $sh b=1024"
  ./harness_review_gfx1201 gfx1201 . $sh --b 1024 --corr --short --cand $C 2>/dev/null | grep "^CMP"
done
