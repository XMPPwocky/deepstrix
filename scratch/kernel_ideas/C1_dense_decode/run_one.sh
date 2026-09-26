#!/usr/bin/env bash
# run_one.sh <tag> <mode> <args...>  — one harness invocation through gpu_run.sh, result saved.
# env: C1_WARM=1 (warm, no flush), C1_ROUNDS=N, C1_MB (default 160)
set -u
cd "$(dirname "$0")"
TAG=$1; shift
export C1_DIR="$PWD"
name=$(echo "$*" | tr ' ' '_')
../_infra/gpu_run.sh --dev dgpu --mb "${C1_MB:-115}" --label "C1_dense_decode/$name" --timeout 300 -- \
    ./harness "$@" > "results/${name}_${TAG}.txt" 2>&1
grep -E "^CMP|^== |^base|^cand|^null|Segm|error|gpu_run" "results/${name}_${TAG}.txt"
