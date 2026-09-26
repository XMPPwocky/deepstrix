#!/usr/bin/env bash
# Reviewer runs: 3 separate process runs of the engineer's exact repro (f16rt mode, harness built in
# this dir from his unmodified source, baselines rebuilt from in-tree) + the reviewer's correctness harness.
# All via the scheduler. Usage: bash run_review.sh submit | wait
set -eu
cd "$(dirname "$0")"
mkdir -p results
if [ "${1:-submit}" = submit ]; then
    : > tickets.txt
    for r in 1 2 3; do
        bash ../../_infra/gpu_submit.sh --dev dgpu --mb 96 --label review/D_attention --timeout 300 \
            -- ./attn_harness2_gfx1201 gfx1201 . f16rt >> tickets.txt
    done
    bash ../../_infra/gpu_submit.sh --dev dgpu --mb 96 --label review/D_attention --timeout 300 \
        -- ./review_f16rt_gfx1201 gfx1201 . >> tickets.txt
    cat tickets.txt
else
    i=0
    while read -r t; do
        i=$((i + 1))
        if [ $i -le 3 ]; then out=results/f16rt_review_run$i.txt; else out=results/review_correctness_run1.txt; fi
        bash ../../_infra/gpu_wait.sh "$t" > "$out" 2>&1 || echo "job $t rc=$?"
        echo "== $out"; cat "$out"
    done < tickets.txt
fi
