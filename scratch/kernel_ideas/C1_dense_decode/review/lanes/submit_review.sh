#!/usr/bin/env bash
# Reviewer measurements (harness_lanes, reviewer-built baseline = same hash as the engineer's):
#  lanes2 x3: production-shaped ONE-graph 2-lane baseline vs tB+tB vs merged, all engineer shapes
#             + kv 4+4, odd split 4+3 (tB7), 5+4 (tB9), gate 4+4
#  chain2 x2: 4-site chain per lane (100 MB) in the hub's lane order vs merged chain
#  corr  x1 : tails / odd M, 1+1, 1+7, 2+6, 5+3 correctness only (fast)
set -u
REV="$(cd "$(dirname "$0")" && pwd)"
INFRA="$(cd "$REV/../../../_infra" && pwd)"
SUB="bash $INFRA/gpu_submit.sh --dev dgpu --mb 300 --timeout 280 --label review/C1_dense_decode"
: > "$REV/tickets_review.txt"
sub() { local name=$1; shift; local t; t=$("$@" 2>&1 | tail -1); echo "$name $t" >> "$REV/tickets_review.txt"; echo "$name -> $t"; }
cd "$REV"
export C1_DIR="$REV"
L2="lanes2 qa 4 4 : lanes2 qa 5 5 : lanes2 wob 4 4 : lanes2 gate 3 3 : lanes2 down 4 4 : lanes2 qb 4 4 : lanes2 kv 4 4 : lanes2 qa 4 3 : lanes2 wob 4 3 : lanes2 wob 5 4 : lanes2 gate 4 4"
for r in 1 2 3; do
    sub "rev_lanes2_r$r" $SUB -- ./harness_lanes $L2
done
for r in 1 2; do
    sub "rev_chain2_r$r" $SUB -- ./harness_lanes chain2 4 4 : chain2 3 3 : chain2 4 3 : chain2 5 5
done
sub "rev_corr_r1" $SUB -- env C1_ROUNDS=5 ./harness_lanes tail 1004 5120 4 4 : tail 1000 5120 4 3 : tail 8 5120 1 1 : tail 2312 2304 5 3 : tail 1280 5120 1 7 : tail 1280 5120 2 6 : tail 1280 5120 1 1 : tail 520 5120 3 3
cat "$REV/tickets_review.txt"
