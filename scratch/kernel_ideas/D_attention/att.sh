#!/usr/bin/env bash
# bash att.sh <label> <kernel-regex> <outdir> <harness args...>
# ATT-profile matching kernels on the dGPU, then rank stalls.
set -u
cd "$(dirname "$0")"
label=$1; re=$2; out=$3; shift 3
../_infra/gpu_run.sh --dev dgpu --mb 48 --label "D_attention/$label" --timeout 800 -- \
    bash ../_infra/prof.sh att dgpu "$re" "$out" -- "$@" > "$out.log" 2>&1
echo "rc=$?"
for f in "$out"/stats_ui_output_*.csv; do
    [ -f "$f" ] || continue
    echo "== $f"
    python3 ~/scripts/att_top.py "$f" --by stall --top 25
done
