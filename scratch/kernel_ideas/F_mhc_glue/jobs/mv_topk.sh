#!/usr/bin/env bash
# One scheduler job: router matvec candidates (warm + cold-W rotation) + wfred top-k candidate
# (correctness + timing).  usage: bash jobs/mv_topk.sh <run-tag>
set -u
cd "$(dirname "$0")/.."
tag=${1:-r1}
export HARNESS_DIR=$PWD
./harness_gfx1201 mv 1            > results/mv_b1_${tag}.txt 2>&1
./harness_gfx1201 mv 4            > results/mv_b4_${tag}.txt 2>&1
./harness_gfx1201 mv 1 cold       > results/mv_b1_cold_${tag}.txt 2>&1
./harness_gfx1201 mv 4 cold       > results/mv_b4_cold_${tag}.txt 2>&1
./harness_gfx1201 mv 1 direct     > results/mv_b1_direct_${tag}.txt 2>&1
./harness_gfx1201 topk 1 prior    > results/topk_b1_prior_${tag}.txt 2>&1
./harness_gfx1201 topk 4 prior    > results/topk_b4_prior_${tag}.txt 2>&1
./harness_gfx1201 topk 1          > results/topk_b1_plain_${tag}.txt 2>&1
if [ "$tag" = r1 ]; then ./harness_gfx1201 topk 512 > results/topk_b512_plain_${tag}.txt 2>&1; fi
grep -h "^==\|^[a-zA-Z].*[0-9]\.[0-9][0-9] " results/mv_b1_${tag}.txt results/mv_b4_${tag}.txt results/mv_b1_cold_${tag}.txt results/mv_b4_cold_${tag}.txt results/mv_b1_direct_${tag}.txt results/topk_b1_prior_${tag}.txt results/topk_b4_prior_${tag}.txt results/topk_b1_plain_${tag}.txt results/topk_b512_plain_${tag}.txt 2>/dev/null
echo "bitexact YES / no / sel_diff!=0:"
grep -h "^CMP" results/mv_b*_${tag}.txt results/topk_b*_${tag}.txt | grep -c "bitexact=YES"
grep -h "^CMP" results/mv_b*_${tag}.txt results/topk_b*_${tag}.txt | grep -c "bitexact=no"
grep -h "^CMP" results/topk_b*_${tag}.txt | grep "sel_diff" | grep -vc "sel_diff=0 orig_diff=0"
true
