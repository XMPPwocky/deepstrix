#!/usr/bin/env bash
# att_summary.sh OUTDIR [top] — rank per-instruction stalls of an ATT trace directory (prof.sh att).
set -u
cd "$(dirname "$0")"
OUT=$1; TOP=${2:-25}
ls "$OUT" 2>/dev/null | head -20
for f in "$OUT"/stats_ui_output_*.csv; do
    [ -e "$f" ] || { echo "no stats csv in $OUT"; continue; }
    echo "== $f"
    python3 ~/scripts/att_top.py "$f" --by stall --top "$TOP"
done
