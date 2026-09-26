#!/usr/bin/env bash
# Reviewer (gateup) extra shapes for the cand_wmma correctness comparison (shapes the engineer
# did not test): b=1, odd/tail row counts, and the largest plausible pass, in the engineer's
# default scale regime. Also a second seed at 1024 and a small expert count (many members per
# expert -> chunked work items at B=1024).
# Usage: bash run_extra.sh RUN_ID
set -u
cd "$(dirname "$0")"
RUN=${1:-1}
export KB_CAND=cand_wmma
export KB_GU_RPW=128
for B in 1 7 33 1023 2048 4096; do
    ./harness_gu cmp $B 96 $((RUN + 40)) 5 > results/extra_cmp_B${B}_run${RUN}.txt 2>&1
    echo "cmp B=$B rc=$?"
done
./harness_gu cmp 1024 96 $((RUN + 77)) 5 > results/extra_cmp_B1024_seed$((RUN + 77))_run${RUN}.txt 2>&1
echo "cmp B=1024 seed rc=$?"
./harness_gu cmp 1024 8 $((RUN + 40)) 5 > results/extra_cmp_B1024_nexp8_run${RUN}.txt 2>&1
echo "cmp B=1024 n_exp=8 rc=$?"
grep -h "CMP\|SPOTCHECK\|setup\|device n_work" results/extra_cmp_B*_run${RUN}.txt
