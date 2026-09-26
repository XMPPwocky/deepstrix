#!/usr/bin/env bash
# Reviewer (gateup): correctness + one A/B in a REALISTIC scale regime (E8M0 bytes 118..124 =
# 2^-10..2^-4, the range the engineer's NOTES cite for real experts) so that the +-10 SwiGLU
# clamp no longer masks ~90% of the gate/up outputs. Also a no-clamp regime (118..122).
# Usage: bash run_realistic.sh RUN_ID
set -u
cd "$(dirname "$0")"
RUN=${1:-1}
export KB_CAND=cand_wmma
export KB_GU_RPW=128
export KB_SC_LO=118
export KB_SC_SPAN=7
for B in 1024 512 37; do
    ./harness_gu cmp $B 96 $RUN 5 > results/real_cmp_B${B}_run${RUN}.txt 2>&1
    echo "real cmp B=$B rc=$?"
done
./harness_gu ab 1024 96 $RUN 20 > results/real_ab_B1024_run${RUN}.txt 2>&1
echo "real ab B=1024 rc=$?"
export KB_SC_SPAN=5
./harness_gu cmp 1024 96 $RUN 5 > results/noclamp_cmp_B1024_run${RUN}.txt 2>&1
echo "noclamp cmp B=1024 rc=$?"
grep -h "CMP\|SPOTCHECK\|review\|setup" results/real_cmp_B*_run${RUN}.txt results/noclamp_cmp_B1024_run${RUN}.txt
grep -h "KBJSON" results/real_ab_B1024_run${RUN}.txt
