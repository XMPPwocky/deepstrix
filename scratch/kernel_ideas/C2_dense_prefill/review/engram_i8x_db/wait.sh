#!/usr/bin/env bash
# wait.sh TICKET results/name.txt : block on a scheduler job, save its output, print the summary lines.
set -u
cd "$(dirname "$0")"
bash ../../../_infra/gpu_wait.sh "$1" > "$2" 2>&1
rc=$?
echo "rc=$rc -> $2"
grep "^==\|^base\|^cand\|^CMP\|rror\|\[kb\] shape\|\[kb\] flush" "$2"
