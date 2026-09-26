#!/usr/bin/env bash
# run_parts.sh <run-tag> <cand-list|-> [b-list] [extra key=val]: per-kernel breakdown at every b (one job)
set -u
cd "$(dirname "$0")"
tag=$1; cands=$2; blist=${3:-"1 2 4 8"}
shift 3 2>/dev/null || shift $#
[ "$cands" = "-" ] && cands=""
for b in $blist; do
    ./harness_gfx1151 gfx1151 . parts b=$b E=4 ppr=3 rounds=40 inner=8 cand=$cands "$@" > results/parts_${tag}_b${b}.txt 2>&1
    echo "== b=$b"
    grep -B1 -A30 "^variant" results/parts_${tag}_b${b}.txt | grep -v KBJSON
done
