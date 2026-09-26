#!/usr/bin/env bash
# prof_att.sh <kernel-regex> <outdir> <harness args...> — ATT (per-instruction stall) trace of the
# named kernel on the dGPU, iterations 5-6 (= the first timed, cold-flushed rounds), short harness.
set -u
cd "$(dirname "$0")"
RE=$1; OUT=$2; shift 2
export C1_DIR="$PWD" C1_ROUNDS=2
ATT_ITERS='[5-6]' ../_infra/gpu_run.sh --dev dgpu --mb 115 --label "C1_dense_decode/att_$RE" --timeout 600 -- \
    bash ../_infra/prof.sh att dgpu "$RE" "$OUT" -- ./harness "$@" > "$OUT.log" 2>&1
tail -3 "$OUT.log"
for f in "$OUT"/stats_ui_output_*.csv; do
    [ -e "$f" ] || continue
    echo "== $f"
    python3 ~/scripts/att_top.py "$f" --by stall --top 22
done
