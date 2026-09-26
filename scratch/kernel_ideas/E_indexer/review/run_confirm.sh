#!/usr/bin/env bash
# Reviewer's re-run of the engineer's exact repro (harness select section, rounds=50), N separate
# processes, using the code objects built in review/ (harness dir = review/). Output to review/results/.
# usage: run_confirm.sh <tag> [N=3]   (run under gpu_run.sh --dev dgpu --mb 60 --label review/E_indexer)
set -u
cd "$(dirname "$0")"
tag=$1; N=${2:-3}
for r in $(seq 1 $N); do
    ./harness_gfx1201 select . rounds=50 > results/select_${tag}_run$r.txt 2>&1
    echo "== select run$r rc=$? -> results/select_${tag}_run$r.txt"
done
