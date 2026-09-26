#!/usr/bin/env bash
# Reviewer's harness. usage: run_review.sh <tag> [harness args: rounds=N shapes=n:b,... nolarge extra]
set -u
cd "$(dirname "$0")"
tag=$1; shift
mkdir -p results
./review_harness_gfx1201 . "$@" > results/review_$tag.txt 2>&1
echo "== review_harness rc=$? -> results/review_$tag.txt"
grep -c MISMATCH results/review_$tag.txt
