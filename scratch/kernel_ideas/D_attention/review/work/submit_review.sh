#!/usr/bin/env bash
# Reviewer extras: (1) combine-split timing at b=1/4/8, (2) extended correctness (extra shapes x input regimes).
set -u
cd "$(dirname "$0")"
bash ../../../_infra/gpu_submit.sh --dev dgpu --mb 96 --label "review/D_attention" --timeout 280 -- \
    ./review_harness_gfx1201 gfx1201 . combine 1 4 8
bash ../../../_infra/gpu_submit.sh --dev dgpu --mb 96 --label "review/D_attention" --timeout 280 -- \
    ./review_harness_gfx1201 gfx1201 . check2
