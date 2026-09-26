#!/usr/bin/env bash
# One GPU job: every non-score section of the harness (select, gather, cand, small).
# usage: run_rest.sh <tag> [rounds] [sections...]   (run under gpu_submit.sh --dev dgpu --mb 156)
set -u
cd "$(dirname "$0")"
tag=$1; rounds=${2:-30}; shift 2 || true
secs="$*"; [ -n "$secs" ] || secs="select gather cand small"
for sec in $secs; do
    ./harness_gfx1201 $sec . rounds=$rounds > results/${sec}_${tag}.txt 2>&1
    echo "== $sec rc=$? -> results/${sec}_${tag}.txt"
done
