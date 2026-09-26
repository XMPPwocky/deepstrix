#!/usr/bin/env bash
# One scheduler job: (a) the engineer's exact harness invocations for the topk claim
# (jobs/mv_topk.sh lines for topk, same binary, same args), outputs into review/; (b) the
# reviewer harness timing sweep.  usage: bash job.sh <tag>   (tag=corr runs the correctness sweep)
set -u
cd "$(dirname "$0")"
tag=${1:-rev1}
export REV_DIR=$PWD
if [ "$tag" = corr ]; then
    ./rev_topk corr > results/corr.txt 2>&1
    echo "rc=$?"
    grep -c "^CORR .* OK " results/corr.txt
    grep "DIFF\|SUMMARY" results/corr.txt
    exit 0
fi
E=../..
( cd "$E" && HARNESS_DIR=$PWD ./harness_gfx1201 topk 1 prior ) > results/eng_topk_b1_prior_${tag}.txt 2>&1
( cd "$E" && HARNESS_DIR=$PWD ./harness_gfx1201 topk 4 prior ) > results/eng_topk_b4_prior_${tag}.txt 2>&1
( cd "$E" && HARNESS_DIR=$PWD ./harness_gfx1201 topk 1 )       > results/eng_topk_b1_plain_${tag}.txt 2>&1
./rev_topk time > results/rev_time_${tag}.txt 2>&1
grep -h "^==\|^router_topk_par\|^router_topk_wfred" results/eng_topk_b1_prior_${tag}.txt results/eng_topk_b4_prior_${tag}.txt results/eng_topk_b1_plain_${tag}.txt results/rev_time_${tag}.txt
echo "engineer-harness CMP bitexact YES/no: $(grep -h '^CMP' results/eng_topk_*_${tag}.txt | grep -c 'bitexact=YES') / $(grep -h '^CMP' results/eng_topk_*_${tag}.txt | grep -c 'bitexact=no'); sel_diff!=0: $(grep -h '^CMP' results/eng_topk_*_${tag}.txt | grep sel_diff | grep -vc 'sel_diff=0 orig_diff=0')"
