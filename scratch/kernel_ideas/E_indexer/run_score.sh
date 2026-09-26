#!/usr/bin/env bash
# One GPU job: the score section (baseline + every cand_score_* kernel present).
# usage: run_score.sh <tag> [rounds] [extra harness opts...]   (run under gpu_submit.sh --dev dgpu --mb 156)
set -u
cd "$(dirname "$0")"
tag=$1; rounds=${2:-30}; shift 2 || true
./harness_gfx1201 score . rounds=$rounds "$@" > results/score_${tag}.txt 2>&1
echo "== score rc=$? -> results/score_${tag}.txt"
