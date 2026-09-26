#!/usr/bin/env bash
# Reviewer repro of the engineer's run_replay.sh: same four shapes at b=64, cold (96 MB flush), 60 rounds,
# baseline = production grid.z=b kernel, candidate = bpack_z16 only (the other candidates are not under review).
# run_review.sh <tag> [--corr]
set -u
cd "$(dirname "$0")"
TAG=${1:-rv1}; shift || true
R=results/replay_$TAG; mkdir -p "$R"
for sh in qb64 kv64 wob64; do
  ./harness_gfx1201 gfx1201 . $sh --cand q8_0_gemv_bpack_z16 "$@" > "$R/${sh}_b64.txt" 2> "$R/${sh}_b64.err"
done
./harness_gfx1201 gfx1201 . woa64 --cand q8_0_grouped_gemv_bpack_z16 "$@" > "$R/woa64_b64.txt" 2> "$R/woa64_b64.err"
grep -h "^==\|^base\|^cand\|^CMP\|rror" "$R"/*.txt "$R"/*.err
