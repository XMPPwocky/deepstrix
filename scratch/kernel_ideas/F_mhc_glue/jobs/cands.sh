#!/usr/bin/env bash
# One scheduler job: all candidate A/Bs of family F (correctness + interleaved timing).
# usage: bash jobs/cands.sh <run-tag> [gemm|topk|rms|all]
set -u
cd "$(dirname "$0")/.."
tag=${1:-r1}
what=${2:-all}
export HARNESS_DIR=$PWD
if [ "$what" = all ] || [ "$what" = gemm ]; then
    ./harness_gfx1201 gemm 512        > results/gemm_512_warm_${tag}.txt 2>&1
    ./harness_gfx1201 gemm 512 flush  > results/gemm_512_flush_${tag}.txt 2>&1
    if [ "$tag" = r1 ]; then   # tails / other shapes: correctness (timing there is incidental)
        for B in 65 100 128 1024; do ./harness_gfx1201 gemm $B > results/gemm_${B}_warm_${tag}.txt 2>&1; done
    fi
fi
if [ "$what" = all ] || [ "$what" = topk ]; then
    ./harness_gfx1201 topk 1 prior    > results/topk_b1_prior_${tag}.txt 2>&1
    ./harness_gfx1201 topk 1          > results/topk_b1_plain_${tag}.txt 2>&1
    ./harness_gfx1201 topk 4 prior    > results/topk_b4_prior_${tag}.txt 2>&1
    if [ "$tag" = r1 ]; then ./harness_gfx1201 topk 512 > results/topk_b512_plain_${tag}.txt 2>&1; fi
fi
if [ "$what" = all ] || [ "$what" = rms ]; then
    ./harness_gfx1201 rms 1           > results/rms_b1_${tag}.txt 2>&1
    ./harness_gfx1201 rms 4           > results/rms_b4_${tag}.txt 2>&1
fi
grep -h "^==\|^[a-zA-Z].*[0-9]\.[0-9][0-9] \|^CMP" results/*_${tag}.txt
