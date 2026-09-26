#!/usr/bin/env bash
# Reviewer repro: 3 separate decode runs of the engineer's candidate list, plus one correctness run.
set -u
cd "$(dirname "$0")"
CANDS=attention_dec_flash3_part1_k128,attention_dec_flash3_part1_k64,attention_dec_flash2_part1_k128
for i in 1 2 3; do
    bash ../../../_infra/gpu_submit.sh --dev dgpu --mb 96 --label "review/D_attention" --timeout 280 -- \
        env CANDS="$CANDS" ./attn_harness2_gfx1201 gfx1201 . decode 1 2 3 4 5 8
done
bash ../../../_infra/gpu_submit.sh --dev dgpu --mb 96 --label "review/D_attention" --timeout 280 -- \
    env CANDS="$CANDS" ./attn_harness2_gfx1201 gfx1201 . check
