#!/usr/bin/env bash
# Generic candidate driver: correctness (cmp) + interleaved A/B for one candidate module.
# Usage: bash run_cand.sh CAND RUN_ID [cmp|ab|all] [GU_ROWS_PER_WG] [shapes...]
#   CAND = cand_wmma | cand_wmma2 | ... (module exporting the two *_wmma symbols)
#   GU_ROWS_PER_WG = rows per WG of the gate+up candidate (128 for cand_wmma/2, 64 for cand_wmma3)
set -u
cd "$(dirname "$0")"
CAND=${1:-cand_wmma}; RUN=${2:-1}; WHAT=${3:-all}; RPW=${4:-128}; shift 4 || true
SHAPES=${*:-1024 512 128}
export KB_CAND=$CAND
export KB_GU_RPW=$RPW
if [ "$WHAT" = cmp ] || [ "$WHAT" = all ]; then
    for B in $SHAPES; do
        ./harness cmp $B 96 $RUN 5 > results/${CAND}_cmp_B${B}_run${RUN}.txt 2>&1
        echo "cmp $CAND B=$B rc=$?"
    done
fi
if [ "$WHAT" = ab ] || [ "$WHAT" = all ]; then
    for B in $SHAPES; do
        ./harness ab $B 96 $RUN 20 > results/${CAND}_ab_B${B}_run${RUN}.txt 2>&1
        echo "ab $CAND B=$B rc=$?"
    done
fi
