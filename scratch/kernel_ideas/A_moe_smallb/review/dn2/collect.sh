#!/usr/bin/env bash
# collect.sh TICKET... : wait for the tickets (quietly), then print the timing rows of every result file.
set -u
cd "$(dirname "$0")"
for t in "$@"; do bash ../../../_infra/gpu_wait.sh "$t" > "results/job_$t.log" 2>&1; echo "ticket $t rc=$?"; done
for f in results/chain_*.txt; do
    echo "== $f"
    grep -E "^\[h\] (b=|distinct)|bitexact=no|nonfinite=[1-9]|^CMP .* over|^base chain|^cand |^roofline" "$f" | grep -v "twin(hetsplit"
done
for f in results/parts_*.txt; do
    [ -s "$f" ] || continue
    echo "== $f"
    grep -E "^\[h\] (b=|distinct)|^kwide2|^cand .* down|^roofline" "$f"
done
