#!/usr/bin/env bash
# run_f16.sh MODE B N_EXP SEED ROUNDS [CAND]  -- harness on the iGPU only, beside the live hub (small footprint).
# Not through gpu_sched (not running); caller keeps runs to seconds. NEVER profile with ATT on the iGPU.
set -u
cd "$(dirname "$0")"
source ../_infra/env.sh
export ROCR_VISIBLE_DEVICES=1
export KB_CAND=${6:-cand_f16}
exec timeout 300 ./harness "$1" "$2" "$3" "$4" "$5"
