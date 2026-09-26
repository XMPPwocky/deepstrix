#!/usr/bin/env bash
# Wait on every ticket in tickets_repro.txt + tickets_review.txt, save to results/<name>.txt, summarise.
set -u
REV="$(cd "$(dirname "$0")" && pwd)"
INFRA="$(cd "$REV/../../../_infra" && pwd)"
for tf in "$REV/tickets_repro.txt" "$REV/tickets_review.txt"; do
    [ -s "$tf" ] || continue
    while read -r name t; do
        [ -n "$name" ] || continue
        out="$REV/results/$name.txt"
        if [ ! -s "$out" ] || ! grep -q "^RESULT\|^== .*rc=\|rc=" "$out"; then
            bash "$INFRA/gpu_wait.sh" "$t" > "$out" 2>&1
            echo "rc=$? $name -> $out"
        fi
    done < "$tf"
done
for f in "$REV"/results/*.txt; do
    echo "--- $(basename "$f")"
    grep -E "^####|^CMP|^== |^base|^cand|^tB|^null|error|Segm|refused|^RESULT" "$f" | grep -v KBJSON | sed -E 's/ +/ /g' | cut -c1-150
done
