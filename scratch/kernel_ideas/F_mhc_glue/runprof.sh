#!/usr/bin/env bash
# runprof.sh <label> <mb> <kernel-regex> <outdir> <harness args...>  -> ATT trace on the dGPU
set -eu
cd "$(dirname "$0")"
label=$1; mb=$2; re=$3; out=$4; shift 4
export HARNESS_DIR=$PWD
../_infra/gpu_run.sh --dev dgpu --mb "$mb" --label "F_mhc_glue/$label" --timeout 600 -- bash ../_infra/prof.sh att dgpu "$re" "$out" -- ./harness_gfx1201 "$@" 2>&1 | tail -5
