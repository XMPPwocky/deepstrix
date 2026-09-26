#!/usr/bin/env bash
# Reviewer's re-run of the engineer's repro: 3 separate harness processes, score section, rounds=50.
# (= run_confirm.sh confirm 3 score, from the review copy; run under gpu_submit.sh --dev dgpu --mb 156)
set -u
cd "$(dirname "$0")"
export SCORE_CANDS=score_mw_pf_qreg,s2_w8n8_pf_hw
for r in 1 2 3; do
    ./harness_gfx1201 score . rounds=50 > results/score_confirm_run$r.txt 2>&1
    echo "== score run$r rc=$? -> results/score_confirm_run$r.txt"
done
