#!/usr/bin/env bash
# Reviewer extra shapes the engineer did not test: b=1 (single row -> one z-slice of 1), b=7 (partial slice),
# b=49 (3 full slices + a 1-row tail), b=128 (the whole SWA replay window, in case one lane ever gets it).
# --corr also compares the harness's built-in tails {1,5,17,33,63} below b. 20 rounds, cold.
set -u
cd "$(dirname "$0")"
R=results/extra; mkdir -p "$R"
for sh in qb64 kv64 wob64; do
  for b in 1 7 49 128; do
    ./harness_gfx1201 gfx1201 . $sh --b $b --rounds 20 --cand q8_0_gemv_bpack_z16 --corr > "$R/${sh}_b$b.txt" 2> "$R/${sh}_b$b.err"
  done
done
for b in 1 7 49 128; do
  ./harness_gfx1201 gfx1201 . woa64 --b $b --rounds 20 --cand q8_0_grouped_gemv_bpack_z16 --corr > "$R/woa64_b$b.txt" 2> "$R/woa64_b$b.err"
done
grep -h "^==\|^base\|^cand\|^CMP\|rror" "$R"/*.txt "$R"/*.err
