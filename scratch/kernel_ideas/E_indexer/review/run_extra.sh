#!/usr/bin/env bash
# Reviewer's extra-shape correctness run + the engineer's harness at shapes the engineer did not
# time (odd n, b=8). Run under gpu_run.sh --dev dgpu --mb 60 --label review/E_indexer.
set -u
cd "$(dirname "$0")"
./extra_gfx1201 . > results/extra_shapes.txt 2>&1
echo "== extra rc=$? -> results/extra_shapes.txt"
./harness_gfx1201 select . rounds=30 n=4097,100003,235001 b=1,8 > results/select_oddshapes.txt 2>&1
echo "== oddshapes rc=$? -> results/select_oddshapes.txt"
