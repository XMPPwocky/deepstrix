#!/usr/bin/env bash
# Reviewer's extra shapes + correctness (run under gpu_submit.sh --dev dgpu --mb 340).
set -u
cd "$(dirname "$0")"
./harness_review . 40 > results/review_extra.txt 2>&1
echo "== review_extra rc=$?"
