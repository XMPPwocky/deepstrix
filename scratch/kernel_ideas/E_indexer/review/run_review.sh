#!/usr/bin/env bash
# Reviewer's harness: untested shapes (tails, vacuous edge, b=8, b=512/1024 prefill), adversarial
# inputs, CPU reference, graph AND direct-launch timing. usage: run_review.sh <tag> [rounds]
set -u
cd "$(dirname "$0")"
tag=${1:-run1}; rounds=${2:-50}
mkdir -p results
./review_harness_gfx1201 . rounds=$rounds > results/review_shapes_$tag.txt 2>&1
echo "== review_harness rc=$? -> results/review_shapes_$tag.txt"
grep -c MISMATCH results/review_shapes_$tag.txt
