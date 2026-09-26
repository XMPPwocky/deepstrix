#!/usr/bin/env bash
# Reviewer extras (one job): shapes the engineer did not test for the down kernel.
#   cmp at B=1, B=3 (odd, tiny groups), B=2048 (>32 members -> chunked work items);
#   tail mode (n_rows 16/5008/5104/5120, sentinel-filled partials, DC-offset activations);
#   full 384-expert layer A/B at B=1024 and B=512 (the real per-layer number, 7.2 GB weights).
set -u
cd "$(dirname "$0")"
mkdir -p results
export KB_CAND=cand_wmma
export KB_GU_RPW=128
for B in 1 3 2048; do
    ./harness cmp $B 96 201 5 > results/cmp_B${B}_run201.txt 2>&1
    echo "cmp B=$B rc=$?"
done
./harness_rev tail 1024 96 202 > results/tail_B1024_run202.txt 2>&1
echo "tail B=1024 rc=$?"
./harness_rev tail 37 96 203 > results/tail_B37_run203.txt 2>&1
echo "tail B=37 rc=$?"
for B in 1024 512; do
    ./harness ab $B 384 204 10 > results/ab_full384_B${B}_run204.txt 2>&1
    echo "ab full-layer B=$B rc=$?"
done
grep -h "CMP\|WRITTEN\|tail mode" results/cmp_B1_run201.txt results/cmp_B3_run201.txt results/cmp_B2048_run201.txt results/tail_B1024_run202.txt results/tail_B37_run203.txt
grep -h "KBJSON" results/ab_full384_B*_run204.txt
