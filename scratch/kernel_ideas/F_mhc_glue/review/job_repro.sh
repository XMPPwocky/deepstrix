#!/usr/bin/env bash
# Replays the engineer's exact `mv` invocations (jobs/mv_topk.sh, mv subset) with the ENGINEER's
# harness binary + hsacos, output redirected into review/results/. usage: bash job_repro.sh <tag>
set -u
cd "$(dirname "$0")"
tag=${1:-v1}
HARNESS_DIR=$(cd .. && pwd)
export HARNESS_DIR
H=../harness_gfx1201
$H mv 1       > results/mv_b1_${tag}.txt 2>&1
$H mv 4       > results/mv_b4_${tag}.txt 2>&1
$H mv 1 cold  > results/mv_b1_cold_${tag}.txt 2>&1
$H mv 4 cold  > results/mv_b4_cold_${tag}.txt 2>&1
grep -h "^==\|^router\|^cand" results/mv_b1_${tag}.txt results/mv_b4_${tag}.txt results/mv_b1_cold_${tag}.txt results/mv_b4_cold_${tag}.txt
echo -n "CMP bitexact=YES: "; grep -h "^CMP" results/mv_b*_${tag}.txt | grep -c "bitexact=YES"
echo -n "CMP bitexact=no:  "; grep -h "^CMP" results/mv_b*_${tag}.txt | grep -c "bitexact=no"
true
