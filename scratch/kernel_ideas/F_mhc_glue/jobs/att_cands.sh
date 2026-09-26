#!/usr/bin/env bash
# One scheduler job: ATT traces (dGPU) of the winning candidates and their baselines, sequentially
# (one rocprofv3 session at a time).  usage: bash jobs/att_cands.sh
set -u
cd "$(dirname "$0")/.."
export HARNESS_DIR=$PWD
rm -rf results/att_gemmc results/att_mvc results/att_mv results/att_topkc results/att_topk
bash ../_infra/prof.sh att dgpu f16_gemm_narrow results/att_gemmc -- ./harness_gfx1201 prof gemmc 512 2>&1 | tail -3
bash ../_infra/prof.sh att dgpu f16_matvec_batched_h20 results/att_mvc -- ./harness_gfx1201 prof mvc 1 2>&1 | tail -3
bash ../_infra/prof.sh att dgpu f16_matvec_batched results/att_mv -- ./harness_gfx1201 prof mv 1 2>&1 | tail -3
bash ../_infra/prof.sh att dgpu router_topk_wfred results/att_topkc -- ./harness_gfx1201 prof topkc 1 2>&1 | tail -3
bash ../_infra/prof.sh att dgpu router_topk_par results/att_topk -- ./harness_gfx1201 prof topk 1 2>&1 | tail -3
for d in att_gemmc att_mvc att_mv att_topkc att_topk; do
    echo "=== $d"
    for f in results/$d/stats_ui_output_*.csv; do
        [ -e "$f" ] || continue
        python3 ~/scripts/att_top.py "$f" --by stall --top 8 --total 2>&1 | tail -12
        break
    done
done
