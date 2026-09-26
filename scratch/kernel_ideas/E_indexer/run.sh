#!/usr/bin/env bash
# run.sh <section> <run-tag> [harness opts...]   — one gpu_run per section, raw output to results/.
# Memory per section (allocated by the harness; dGPU cap 256 MB, hub headroom ~438 MB - 270):
#   score  : keys 4 x 235000 x 80 B = 75 MB + scores 2 x 6 MB + flush 64 MB          -> --mb 156
#   select : keys 19 MB + scores 6 MB + scratch + flush none                         -> --mb 40
#   gather : store 48 MB + dst 2 x 16 MB + flush 64 MB                               -> --mb 150
#   cand   : keys 19 MB + scores 2 x 6 MB + block scores                             -> --mb 48
#   small  : weights 11 MB + flush 64 MB                                             -> --mb 90
set -eu
cd "$(dirname "$0")"
sec=$1; tag=$2; shift 2
case "$sec" in
    score) mb=156 ;; select) mb=40 ;; gather) mb=150 ;; cand) mb=48 ;; small) mb=90 ;;
    *) echo "unknown section $sec" >&2; exit 64 ;;
esac
mkdir -p results
out=results/${sec}_${tag}.txt
../_infra/gpu_run.sh --dev dgpu --mb $mb --label E_indexer/$sec --timeout 600 -- ./harness_gfx1201 "$sec" . "$@" 2>&1 | tee "$out"
echo "saved $out"
