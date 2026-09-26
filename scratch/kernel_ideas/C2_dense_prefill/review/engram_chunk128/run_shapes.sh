#!/usr/bin/env bash
# Shapes the engineer did not test: odd tails spanning a 64-row production chunk boundary
# (65, 100, 127), grid.x > 1 (129, 300), max production lane (512), plus the timing at 256/512 rows
# (is 128 the right chunk, or does the per-WG chain make 256/512 nearly free too?).
# --corr also checks b in {1, 17, 63} below each requested b.
set -u
cd "$(dirname "$0")"
for b in 65 100 127 129 300; do
  ./harness_gfx1201 gfx1201 . engram --m 25600 --b $b --corr --short --cand q8_0_gemm_wmma_i8x_db > results/s_b${b}.txt 2> results/s_b${b}.err
done
for b in 256 512; do
  ./harness_gfx1201 gfx1201 . engram --m 25600 --b $b --corr --cand q8_0_gemm_wmma_i8x_db > results/s_b${b}.txt 2> results/s_b${b}.err
done
grep -h "^==\|^base\|^cand\|^CMP\|rror" results/s_b*.txt
