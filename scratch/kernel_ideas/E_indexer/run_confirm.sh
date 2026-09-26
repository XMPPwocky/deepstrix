#!/usr/bin/env bash
# Confirmation runs: N separate harness PROCESSES per section (winning candidates only), rounds=50.
# usage: run_confirm.sh <tag> [N=3] [sections...]   (run under gpu_submit.sh --dev dgpu --mb 156)
set -u
cd "$(dirname "$0")"
tag=$1; N=${2:-3}; shift 2 || true
secs="$*"; [ -n "$secs" ] || secs="score select gather cand"
export SCORE_CANDS=score_mw_pf_qreg,s2_w8n8_pf_hw
export GATHER_CANDS=gather_u4_r1,gather_u4_r4,gather_u4_r4_2ipt
for r in $(seq 1 $N); do
    for sec in $secs; do
        ./harness_gfx1201 $sec . rounds=50 > results/${sec}_${tag}_run$r.txt 2>&1
        echo "== $sec run$r rc=$? -> results/${sec}_${tag}_run$r.txt"
    done
done
