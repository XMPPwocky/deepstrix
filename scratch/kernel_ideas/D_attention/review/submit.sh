#!/usr/bin/env bash
# Reviewer: submit N separate decode runs of the engineer's exact repro (their harness source,
# reviewer-built binaries in review/) + one correctness run; prints tickets.
#   bash review/submit.sh final 3            -> tickets for 3 decode runs
#   bash review/submit.sh check 1            -> ticket for the engineer's check mode
#   bash review/submit.sh extra 1            -> ticket for the reviewer's extra-shape check
set -u
cd "$(dirname "$0")"
what=${1:-final}; n=${2:-3}
CANDS=attention_dec_score_blk256,attention_dec_score_blk128
for i in $(seq 1 "$n"); do
    case "$what" in
        final)
            bash ../../_infra/gpu_submit.sh --dev dgpu --mb 96 --label "review/D_attention" --timeout 280 -- \
                env CANDS="$CANDS" ./attn_harness2_gfx1201 gfx1201 . decode 1 2 3 4 5 8 ;;
        check)
            bash ../../_infra/gpu_submit.sh --dev dgpu --mb 96 --label "review/D_attention" --timeout 200 -- \
                env CANDS="$CANDS" ./attn_harness2_gfx1201 gfx1201 . check ;;
        extra)
            bash ../../_infra/gpu_submit.sh --dev dgpu --mb 160 --label "review/D_attention" --timeout 200 -- \
                ./check_extra_gfx1201 gfx1201 . ;;
    esac
done
