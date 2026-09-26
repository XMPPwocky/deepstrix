#!/usr/bin/env bash
# Reviewer's job: the engineer's exact rms repro (rms 1 / rms 4 / rms 1 direct, as in jobs/winners.sh)
# plus extra shapes (b = 7 odd, b = 8 max production rows, B = 512 prefill direct) and the
# adversarial-input probe. One process per harness invocation, all inside ONE scheduler job.
# usage: bash jobs/review_rms.sh <run-tag>
set -u
cd "$(dirname "$0")/.."
tag=${1:-r1}
export HARNESS_DIR=$PWD
./harness_gfx1201 rms 1          > results/rms_b1_${tag}.txt 2>&1
./harness_gfx1201 rms 4          > results/rms_b4_${tag}.txt 2>&1
./harness_gfx1201 rms 1 direct   > results/rms_b1_direct_${tag}.txt 2>&1
./harness_gfx1201 rms 7          > results/rms_b7_${tag}.txt 2>&1
./harness_gfx1201 rms 8          > results/rms_b8_${tag}.txt 2>&1
./harness_gfx1201 rms 512 direct > results/rms_b512_direct_${tag}.txt 2>&1
./extra_gfx1201                  > results/extra_${tag}.txt 2>&1
echo "extra rc=$?"
grep -h "^==\|^[a-zA-Z].*[0-9]\.[0-9][0-9] " results/rms_b1_${tag}.txt results/rms_b4_${tag}.txt results/rms_b1_direct_${tag}.txt results/rms_b7_${tag}.txt results/rms_b8_${tag}.txt results/rms_b512_direct_${tag}.txt
echo -n "harness CMP bitexact=YES: "; grep -h "^CMP" results/rms_b*_${tag}.txt | grep -c "bitexact=YES"
echo -n "harness CMP bitexact=no:  "; grep -h "^CMP" results/rms_b*_${tag}.txt | grep -c "bitexact=no"
grep -h "^EXTRA" results/extra_${tag}.txt
grep -h "^CMP" results/extra_${tag}.txt | grep "bitexact=no" | head -20
