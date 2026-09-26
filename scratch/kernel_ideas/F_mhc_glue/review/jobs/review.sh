#!/usr/bin/env bash
# Reviewer harness: correctness (cur, norm, carry) on untested shapes + timing incl. the copy.
# usage: bash review/jobs/review.sh <tag>
set -u
cd "$(dirname "$0")/.."
tag=${1:-rv1}
export HARNESS_DIR=$PWD
mkdir -p results
./harness_review_gfx1201 all > results/review_all_${tag}.txt 2>&1
rc=$?
grep -h "^==\|CMP\|REF\|CORRECTNESS\|TRIO\|CAND\|collapse-only\|copy alone\|hc_weighted\|rms_norm" results/review_all_${tag}.txt
echo "harness rc=$rc"
exit $rc
