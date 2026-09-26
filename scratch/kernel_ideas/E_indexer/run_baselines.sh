#!/usr/bin/env bash
# All remaining baselines + ATT profiles, serialized on the dGPU lock. Output under results/.
set -u
cd "$(dirname "$0")"
bash run.sh select base1 rounds=30
bash run.sh gather base1 rounds=30
bash run.sh cand base1 rounds=30
bash run.sh small base1 rounds=30
# ATT: per-instruction stalls on the production score kernel (n=235K, b=4) and the select kernel.
../_infra/gpu_run.sh --dev dgpu --mb 156 --label E_indexer/att_score --timeout 900 -- \
    bash ../_infra/prof.sh att dgpu indexer_score_wmma_batched_mw_e2m1 results/att_score -- ./harness_gfx1201 score . short n=235000 b=4 \
    > results/att_score_log.txt 2>&1
python3 ~/scripts/att_top.py results/att_score/stats_ui_output_*.csv --by stall --top 30 > results/att_score_top.txt 2>&1
python3 ~/scripts/att_top.py results/att_score/stats_ui_output_*.csv --by stall --group --top 20 >> results/att_score_top.txt 2>&1
../_infra/gpu_run.sh --dev dgpu --mb 40 --label E_indexer/att_select --timeout 900 -- \
    bash ../_infra/prof.sh att dgpu indexer_topk_select_batched_ilp results/att_select -- ./harness_gfx1201 select . short n=235000 b=4 \
    > results/att_select_log.txt 2>&1
python3 ~/scripts/att_top.py results/att_select/stats_ui_output_*.csv --by stall --top 30 > results/att_select_top.txt 2>&1
python3 ~/scripts/att_top.py results/att_select/stats_ui_output_*.csv --by stall --group --top 20 >> results/att_select_top.txt 2>&1
echo BASELINES_DONE
