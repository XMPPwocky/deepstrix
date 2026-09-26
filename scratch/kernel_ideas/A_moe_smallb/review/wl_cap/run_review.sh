#!/usr/bin/env bash
# run_review.sh <mode:chain|parts> <run-tag> <cand-list> <b-list> [extra harness key=val ...]
# Reviewer's driver: same harness invocation as the engineer's run_chain.sh / run_parts.sh, but with the
# reviewer's own code objects (this dir) and outputs under this dir's results/. One scheduler job per call.
set -u
cd "$(dirname "$0")"
mode=$1; tag=$2; cands=$3; blist=$4
shift 4
mkdir -p results
for b in $blist; do
    ./harness_gfx1151 gfx1151 . $mode b=$b E=4 ppr=3 rounds=40 inner=8 cand=$cands "$@" > results/${mode}_${tag}_b${b}.txt 2>&1
    echo "== $mode $tag b=$b $*"
    grep -E "^CMP|^base out|^variant|us  |^roofline|\[h\] b=|error|Error|fault" results/${mode}_${tag}_b${b}.txt | grep -v KBJSON
done
