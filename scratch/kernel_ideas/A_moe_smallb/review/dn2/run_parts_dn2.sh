#!/usr/bin/env bash
# run_parts_dn2.sh <tag> <cand-list> <b-list> [extra key=val ...]  -- per-kernel breakdown (parts mode)
set -u
cd "$(dirname "$0")"
tag=$1; cands=$2; blist=$3; shift 3
for b in $blist; do
    ./harness_gfx1151 gfx1151 . parts b=$b E=4 ppr=3 rounds=40 inner=8 cand=$cands "$@" > results/parts_${tag}_b${b}.txt 2>&1
    echo "== b=$b"
    grep -E "^variant|us  |^roofline|error|Error|fault|\[h\]" results/parts_${tag}_b${b}.txt | grep -v KBJSON
done
