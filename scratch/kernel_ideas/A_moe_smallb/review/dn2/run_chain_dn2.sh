#!/usr/bin/env bash
# run_chain_dn2.sh <tag> <cand-list> <b-list> [extra key=val ...]  -- chain A/B, hsacos from review/dn2
# Same harness invocation as the engineer's run_chain.sh (E=4 ppr=3 rounds=40 inner=8) unless overridden.
set -u
cd "$(dirname "$0")"
tag=$1; cands=$2; blist=$3; shift 3
for b in $blist; do
    ./harness_gfx1151 gfx1151 . chain b=$b E=4 ppr=3 rounds=40 inner=8 cand=$cands "$@" > results/chain_${tag}_b${b}.txt 2>&1
    echo "== b=$b"
    grep -E "^CMP|^base out|^variant|us  |^roofline|error|Error|fault|\[h\]" results/chain_${tag}_b${b}.txt | grep -v KBJSON
done
