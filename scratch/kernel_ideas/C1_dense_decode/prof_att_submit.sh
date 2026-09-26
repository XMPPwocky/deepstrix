#!/usr/bin/env bash
# prof_att_submit.sh <kernel-regex> <outdir> <harness args...> — ENQUEUE (non-blocking) an ATT
# (per-instruction stall) trace of the named kernel on the dGPU, dispatch iterations 5-6 (= the
# first timed, cold-flushed rounds), short harness (C1_ROUNDS=2). Collect with wait_to.sh.
set -u
cd "$(dirname "$0")"
RE=$1; OUT=$2; shift 2
export C1_DIR="$PWD" C1_ROUNDS=2 ATT_ITERS='[5-6]'
bash ../_infra/gpu_submit.sh --dev dgpu --mb 400 --label "C1_dense_decode/att_$RE" --timeout 300 -- \
    bash ../_infra/prof.sh att dgpu "$RE" "$OUT" -- ./harness "$@"
