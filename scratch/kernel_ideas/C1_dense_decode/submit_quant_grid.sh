#!/usr/bin/env bash
# submit_quant_grid.sh <tag> — grid-size hypothesis for the quantize pathology: same shapes with
# (a) the exact grid, (b) one extra idle WG (C1_GPAD=1), (c) 3 extra, (d) plain launches (C1_NOGRAPH).
set -u
cd "$(dirname "$0")"
TAG=${1:-g1}
export C1_DIR="$PWD"
SHAPES="quant 32768 2 : quant 32768 4 : quant 32768 1 : quant 65536 1 : quant 32800 2 : quant 16384 4 : quant 5120 4 : quant 8192 4"
SUB="bash ../_infra/gpu_submit.sh --dev dgpu --mb 100 --timeout 200"
$SUB --label C1_dense_decode/quant_grid0_$TAG -- ./harness $SHAPES
$SUB --label C1_dense_decode/quant_grid1_$TAG -- env C1_GPAD=1 ./harness $SHAPES
$SUB --label C1_dense_decode/quant_grid3_$TAG -- env C1_GPAD=3 ./harness $SHAPES
$SUB --label C1_dense_decode/quant_nograph_$TAG -- env C1_NOGRAPH=1 ./harness $SHAPES
