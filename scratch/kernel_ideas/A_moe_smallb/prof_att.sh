#!/usr/bin/env bash
# prof_att.sh <kernel-regex> <outname> [harness args...]  — ATT on the iGPU for the chain (prof mode)
set -u
cd "$(dirname "$0")"
re=$1; out=$2; shift 2
../_infra/gpu_run.sh --dev igpu --mb 400 --label A_moe_smallb/att-$out --timeout 800 -- \
    bash ../_infra/prof.sh att igpu "$re" prof/att_$out -- ./harness_gfx1151 gfx1151 . prof "$@"
ls prof/att_$out/stats_ui_output_*.csv 2>/dev/null | head -3
for f in prof/att_$out/stats_ui_output_*.csv; do
    [ -s "$f" ] || continue
    echo "== $f"
    python3 ~/scripts/att_top.py "$f" --total
    python3 ~/scripts/att_top.py "$f" --by stall --top 30
done
