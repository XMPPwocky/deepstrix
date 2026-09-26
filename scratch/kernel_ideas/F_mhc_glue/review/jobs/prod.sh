#!/usr/bin/env bash
# Reviewer harness, production regime (rms_nw + GEMM prefix untimed before every single timed call).
# usage: bash review/jobs/prod.sh <tag>
set -u
cd "$(dirname "$0")/.."
tag=${1:-rv1}
export HARNESS_DIR=$PWD
mkdir -p results
./harness_review_gfx1201 prod > results/review_prod_${tag}.txt 2>&1
rc=$?
grep -h "^==\|TRIO\|CAND\|collapse-only\|copy alone\|hc_weighted\|rms_norm" results/review_prod_${tag}.txt | grep -v KBJSON
echo "harness rc=$rc"
exit $rc
