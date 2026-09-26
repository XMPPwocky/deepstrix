#!/usr/bin/env bash
# ATT (per-instruction stall) traces of the production kernels, smaller footprint (32 experts).
# Usage: bash prof_att.sh [B] [N_EXP] [kernels...]
set -u
cd "$(dirname "$0")"
B=${1:-1024}; NE=${2:-32}; shift 2 || true
KS=${*:-kwide kwide2}
MB=900
R=../_infra/gpu_run.sh
for k in $KS; do
    case $k in
        kwide)  re='mxfp4_pair_matvec_fused_swiglu_kwide' ;;
        kwide2) re='mxfp4_matvec_par_by_expert_kwide2' ;;
        gu)     re='mxfp4_pair_matvec_fused_swiglu_wmma' ;;
        down)   re='mxfp4_matvec_par_by_expert_wmma' ;;
        *)      re=$k ;;
    esac
    $R --dev igpu --mb $MB --label B_moe_prefill/att_$k --timeout 900 -- bash ../_infra/prof.sh att igpu "$re" prof_att_$k -- ./harness prof_$k $B $NE > results/att_${k}_B${B}.log 2>&1
    for f in prof_att_$k/stats_ui_output_*.csv; do
        [ -f "$f" ] && python3 ~/scripts/att_top.py "$f" --by stall --top 30 > results/att_${k}_B${B}_top.txt 2>&1
    done
done
echo prof_att done
