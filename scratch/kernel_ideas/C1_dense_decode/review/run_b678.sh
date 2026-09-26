#!/usr/bin/env bash
# The production grids nobody measured: 1024*b for b=6,7,8 (small_b_dense_dp4a allows up to 8) and 256*b for b=6,7.
set -u
REV="$(cd "$(dirname "$0")" && pwd)"
INFRA="$(cd "$REV/../../_infra" && pwd)"
cd "$REV"
export C1_DIR="$REV"
bash "$INFRA/gpu_run.sh" --dev dgpu --mb 100 --timeout 240 --label review/C1_dense_decode -- ./gpad_check 32768 6 32768 7 32768 8 8192 6 8192 7 > results/check_b678_r1.txt 2>&1
echo rc=$?
grep -E "^SHAPE|^q8_0_quantize_f32 grid=blocks\+(0|1) |^RESULT" results/check_b678_r1.txt | sed -E 's/ +/ /g; s/\| base vs CPU ref: //; s/xs_unwritten.*//' | cut -c1-75 | paste - - -
