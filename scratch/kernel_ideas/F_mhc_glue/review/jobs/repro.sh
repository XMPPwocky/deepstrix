#!/usr/bin/env bash
# Engineer's exact repro (the prefill half of jobs/base_all.sh), engineer's binary, engineer's hsacos;
# only the output goes under review/results/.   usage: bash review/jobs/repro.sh <tag>
set -u
cd "$(dirname "$0")/../.."
tag=${1:-rv1}
export HARNESS_DIR=$PWD
mkdir -p review/results
./harness_gfx1201 prefill 512       > review/results/eng_prefill_512_warm_${tag}.txt 2>&1
./harness_gfx1201 prefill 512 flush > review/results/eng_prefill_512_flush_${tag}.txt 2>&1
grep -h "^==\|hc_weighted\|rms_norm_weighted\|collapse\|CMP" review/results/eng_prefill_512_warm_${tag}.txt review/results/eng_prefill_512_flush_${tag}.txt
