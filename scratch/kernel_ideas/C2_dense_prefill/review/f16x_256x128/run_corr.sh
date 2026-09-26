#!/usr/bin/env bash
# Reviewer correctness (f16x_256x128 claim): vs production f16x on every f16x site, incl. shapes the
# engineer did not test: b=1024 (max production chunk) + its tails, and odd/tail batches 7, 257, 513, 1023.
set -u
cd "$(dirname "$0")"
mkdir -p results
C=q8_0_gemm_wmma_f16x_256x128
for sh in qb woa qa kv wob shg shd; do
  echo "=== $sh b=1024 (+ tails 1,3,17,100,129,255,256,500)"
  ./harness_gfx1201 gfx1201 . $sh --b 1024 --corr --short --cand $C 2>/dev/null | grep "^CMP"
  for b in 7 257 513 1023; do
    ./harness_gfx1201 gfx1201 . $sh --b $b --corr --short --cand $C 2>/dev/null | grep "^CMP" | grep "b=$b "
  done
done
