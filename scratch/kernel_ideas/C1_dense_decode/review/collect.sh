#!/usr/bin/env bash
# Wait on every ticket in tickets.txt, save output to results/<name>.txt, print a summary.
set -u
REV="$(cd "$(dirname "$0")" && pwd)"
INFRA="$(cd "$REV/../../_infra" && pwd)"
while read -r name t; do
    [ -n "$name" ] || continue
    out="$REV/results/$name.txt"
    if [ ! -s "$out" ] || ! grep -q "^== .*rc=" "$out"; then
        bash "$INFRA/gpu_wait.sh" "$t" > "$out" 2>&1
        echo "rc=$? $name -> $out"
    fi
done < "$REV/tickets.txt"
echo "=== summary (med_us per node): name | shape | base grid0 med | (cand rows)"
for f in "$REV"/results/eng_*.txt "$REV"/results/rebuilt_*.txt; do
    echo "--- $(basename "$f") $(grep -m1 '^== .*rc=' "$f" | cut -c1-80)"
    grep -E "^quant grid|^== quantize|^base q8_0|^null" "$f" | sed -E 's/ +/ /g' | cut -c1-110
done
for f in "$REV"/results/check_*.txt; do
    echo "--- $(basename "$f") $(grep -m1 '^== .*rc=' "$f" | cut -c1-80)"
    grep -E "^SHAPE|^CMP|^== quantize|^q8_0_quantize_f32 grid|^RESULT" "$f" | sed -E 's/ +/ /g' | cut -c1-120
done
