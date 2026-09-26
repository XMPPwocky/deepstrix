#!/usr/bin/env bash
# Reviewer's replay of the F_mhc_glue/gemm_narrow_m24 claim: one scheduler job per run tag.
# usage: bash jobs/review_gemm.sh <tag> [full]
set -u
cd "$(dirname "$0")/.."
tag=${1:-r1}
what=${2:-quick}
export HARNESS_DIR=$PWD
./harness_gfx1201 gemm 512        > results/gemm_512_warm_${tag}.txt 2>&1
./harness_gfx1201 gemm 512 flush  > results/gemm_512_flush_${tag}.txt 2>&1
./harness_gfx1201 gemm 512 coldfull > results/gemm_512_coldfull_${tag}.txt 2>&1
if [ "$what" = full ]; then
    # shapes the engineer did not test: b=1, tiny odd, sub-tile, odd tails around tile edges, max production b
    for B in 1 3 7 17 33 63 64 129 511 513 1000 1024; do
        ./harness_gfx1201 gemm $B > results/gemm_${B}_warm_${tag}.txt 2>&1
    done
fi
grep -h "^==\|^[a-zA-Z].*[0-9]\.[0-9][0-9] " results/*_${tag}.txt
echo "bitexact YES: $(grep -h '^CMP' results/*_${tag}.txt | grep -c 'bitexact=YES')  no: $(grep -h '^CMP' results/*_${tag}.txt | grep -c 'bitexact=no')"
grep -h "^CMP" results/*_${tag}.txt | grep -v "bitexact=YES"
