#!/usr/bin/env bash
# run_job.sh <tag> : every (e) shape at its production prefill batch (VRAM-capped where noted)
set -u
cd "$(dirname "$0")"
tag=$1
./harness qa 512 64 > results/qa_$tag.txt 2>&1
./harness kv 512 64 > results/kv_$tag.txt 2>&1
./harness shg 512 64 > results/shg_$tag.txt 2>&1
./harness shd 512 64 > results/shd_$tag.txt 2>&1
./harness wob 512 40 > results/wob_$tag.txt 2>&1
./harness qb 512 8 > results/qb_$tag.txt 2>&1        # 67 MB output: its writes + 8 MB flush evict the MALL
./harness woa 256 24 > results/woa_$tag.txt 2>&1      # 256 rows: the f32 heads input alone is 67 MB at 512
./harness qa 1024 64 > results/qa1024_$tag.txt 2>&1
grep -h 'HIP error' results/*_$tag.txt
echo done
