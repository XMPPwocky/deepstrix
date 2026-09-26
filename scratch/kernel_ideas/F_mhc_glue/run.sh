#!/usr/bin/env bash
# run.sh <label> <mb> <harness args...>   -> results/<label>.txt (stdout+stderr), through gpu_run.sh (dGPU)
set -eu
cd "$(dirname "$0")"
label=$1; mb=$2; shift 2
export HARNESS_DIR=$PWD
../_infra/gpu_run.sh --dev dgpu --mb "$mb" --label "F_mhc_glue/$label" --timeout 600 -- ./harness_gfx1201 "$@" 2>&1 | tee "results/$label.txt"
