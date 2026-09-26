#!/usr/bin/env bash
# Replay regime (b=64 per lane): production grid.z=64 GEMV vs bpack_z16 / bpack64 (bit-exact twins)
# and f16x / f16x_bn64 at b=64 (f16 activations: NOT bit-exact, needs the fidelity gate).
# run_replay.sh <run-tag> [--corr]
set -u
cd "$(dirname "$0")"
TAG=${1:-r1}; shift || true
R=results/replay_$TAG; mkdir -p $R
for sh in qb64 kv64 wob64; do
  ./harness_gfx1201 gfx1201 . $sh --cand q8_0_gemv_bpack_z16,q8_0_gemv_bpack64_warp8,f16x_at_b,f16x_bn64_at_b "$@" > $R/${sh}_b64.txt 2>$R/${sh}_b64.err
done
./harness_gfx1201 gfx1201 . woa64 --cand q8_0_grouped_gemv_bpack_z16,q8_0_grouped_gemv_bpack64,f16x_at_b,f16x_bn64_at_b "$@" > $R/woa64_b64.txt 2>$R/woa64_b64.err
grep -h "^==\|^base\|^cand\|^CMP\|rror" $R/*.txt
