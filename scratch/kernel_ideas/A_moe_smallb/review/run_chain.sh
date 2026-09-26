#!/usr/bin/env bash
# run_chain.sh <run-tag> <cand-list> [b-list] [extra harness key=val ...]
# Reviewer copy of the engineer's run_chain.sh: same harness invocation, code objects and outputs under review/.
set -u
cd "$(dirname "$0")"
mkdir -p results
tag=$1; cands=$2; blist=${3:-"2 4 8"}
shift 3 2>/dev/null || shift $#
for b in $blist; do
    ./harness_gfx1151 gfx1151 . chain b=$b E=4 ppr=3 rounds=40 inner=8 cand=$cands "$@" > results/chain_${tag}_b${b}.txt 2>&1
    echo "== b=$b rc=$?"
    grep -E "^CMP|^base out|^variant|us  |^roofline|error|Error|fault|^\[h\]" results/chain_${tag}_b${b}.txt | grep -v KBJSON
done
