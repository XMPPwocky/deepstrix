#!/usr/bin/env bash
# bash attsum.sh OUTDIR [top]  -> rank stalls of every ATT stats csv in OUTDIR
set -u
cd "$(dirname "$0")"
out=$1; top=${2:-25}
for f in "$out"/stats_ui_output_*.csv; do
    [ -f "$f" ] || continue
    echo "== $f"
    python3 ~/scripts/att_top.py "$f" --by stall --top "$top"
done
