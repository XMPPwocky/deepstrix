#!/usr/bin/env bash
# bash final.sh <tag> <N> <CANDS> [b...]  -> submit N separate decode runs (separate processes) of the
# given candidate list; tickets printed one per line; collect with `bash w.sh TICKET results/<tag>_run<i>.txt`
set -u
cd "$(dirname "$0")"
tag=$1; n=$2; cands=$3; shift 3
bs="$*"; [ -z "$bs" ] && bs="1 2 3 4 5 8"
for i in $(seq 1 "$n"); do
    bash ../_infra/gpu_submit.sh --dev dgpu --mb 96 --label "D_attention/${tag}_run$i" --timeout 280 -- \
        env CANDS="$cands" ./attn_harness2_gfx1201 gfx1201 . decode $bs
done
