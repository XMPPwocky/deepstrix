#!/usr/bin/env bash
# run_job.sh <tag> : one scheduler job = every (c) shape; outputs results/<shape>_<tag>.txt
set -u
cd "$(dirname "$0")"
tag=$1
./harness idxq 1 2 3 4 5 6 8 12 16 32 64 > results/idxq_$tag.txt 2>&1
./harness comp1 1 2 3 4 5 6 8 12 16 32 64 > results/comp1_$tag.txt 2>&1
./harness proj 1 4 16 64 > results/proj_$tag.txt 2>&1
./harness router 1 4 8 16 32 64 > results/router_$tag.txt 2>&1
./harness tail 1 3 17 > results/tail_$tag.txt 2>&1
grep -h 'MISMATCH\|RESULT' results/*_$tag.txt
