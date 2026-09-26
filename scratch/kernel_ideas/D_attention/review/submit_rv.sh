#!/usr/bin/env bash
# bash submit_rv.sh  -> 3 separate decode runs of the engineer's exact repro (harness2, CANDS=pf2,
# b = 1 2 3 4 5 8) + 1 extended check run (reviewer harness: extra shapes) + 1 extended decode run
# (b = 1 4 16). Prints one ticket per line.
set -u
cd "$(dirname "$0")"
for i in 1 2 3; do
    bash ../../_infra/gpu_submit.sh --dev dgpu --mb 96 --label "review/D_attention" --timeout 280 -- \
        env CANDS=attention_dec_smwsum_pf2 ./attn_harness2_gfx1201 gfx1201 . decode 1 2 3 4 5 8
done
bash ../../_infra/gpu_submit.sh --dev dgpu --mb 160 --label "review/D_attention" --timeout 280 -- \
    env CANDS=attention_dec_smwsum_pf2 ./attn_harness_rv_gfx1201 gfx1201 . check
bash ../../_infra/gpu_submit.sh --dev dgpu --mb 160 --label "review/D_attention" --timeout 280 -- \
    env CANDS=attention_dec_smwsum_pf2 ./attn_harness_rv_gfx1201 gfx1201 . decode 1 4 16
