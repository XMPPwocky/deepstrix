#!/usr/bin/env bash
# bash w.sh TICKET OUTFILE  -> wait for the job, save its full output, print the summary lines
set -u
cd "$(dirname "$0")"
t=$1; out=$2
bash ../_infra/gpu_wait.sh "$t" > "$out" 2>&1
echo "rc=$? saved $out"
grep -E "^== |^variant|^base |^attention|^pair|^CMP|^REF|^####|^f16rt|^KBJSON \{\"cmp\":\"f16|error|Error|fault|abort" "$out"
