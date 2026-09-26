#!/usr/bin/env bash
# One scheduler job: every production baseline of family F, hub DOWN (idle GPU).
# usage: bash jobs/base_all.sh <run-tag>
set -u
cd "$(dirname "$0")/.."
tag=${1:-r2}
export HARNESS_DIR=$PWD
./harness_gfx1201 decode 1          > results/base_decode_b1_${tag}.txt 2>&1
./harness_gfx1201 decode 1 direct   > results/base_decode_b1_direct_${tag}.txt 2>&1
./harness_gfx1201 decode 4          > results/base_decode_b4_${tag}.txt 2>&1
./harness_gfx1201 decode 4 direct   > results/base_decode_b4_direct_${tag}.txt 2>&1
./harness_gfx1201 prefill 512       > results/base_prefill_512_warm_${tag}.txt 2>&1
./harness_gfx1201 prefill 512 flush > results/base_prefill_512_flush_${tag}.txt 2>&1
grep -h "^==\|^[a-zA-Z].*[0-9]\.[0-9][0-9] " results/base_decode_b1_${tag}.txt results/base_decode_b1_direct_${tag}.txt results/base_decode_b4_${tag}.txt results/base_decode_b4_direct_${tag}.txt results/base_prefill_512_warm_${tag}.txt results/base_prefill_512_flush_${tag}.txt
