#!/usr/bin/env bash
# Baseline profiling of the two production MoE kernels on the iGPU:
#   grid-bound vs exact grid timing, PMC passes, ATT on kwide and kwide2.
# Every GPU step goes through gpu_run.sh. Usage: bash prof_base.sh [B] [N_EXP]
set -u
cd "$(dirname "$0")"
B=${1:-1024}; NE=${2:-64}
MB=1600
R=../_infra/gpu_run.sh
$R --dev igpu --mb $MB --label B_moe_prefill/grid --timeout 600 -- ./harness grid $B $NE 1 25 > results/grid_B${B}_run1.txt 2>&1
for k in kwide kwide2; do
    $R --dev igpu --mb $MB --label B_moe_prefill/pmc_$k --timeout 600 -- bash ../_infra/prof.sh pmc "SQ_WAVES SQ_INSTS_VALU SQ_INSTS_LDS GRBM_GUI_ACTIVE" prof_pmc1_$k -- ./harness prof_$k $B $NE > results/pmc1_${k}_B${B}.log 2>&1
    $R --dev igpu --mb $MB --label B_moe_prefill/pmc_$k --timeout 600 -- bash ../_infra/prof.sh pmc "SQ_BUSY_CYCLES SQ_INSTS_TEX_LOAD SQ_INSTS_SALU MemUnitBusy" prof_pmc2_$k -- ./harness prof_$k $B $NE > results/pmc2_${k}_B${B}.log 2>&1
    $R --dev igpu --mb $MB --label B_moe_prefill/pmc_$k --timeout 600 -- bash ../_infra/prof.sh pmc "SQ_WAVE_CYCLES SQ_WAIT_INST_ANY SQ_INST_CYCLES_VALU SQ_INST_CYCLES_VMEM" prof_pmc3_$k -- ./harness prof_$k $B $NE > results/pmc3_${k}_B${B}.log 2>&1
    $R --dev igpu --mb $MB --label B_moe_prefill/pmc_$k --timeout 600 -- bash ../_infra/prof.sh pmc "VALUBusy MemUnitStalled" prof_pmc4_$k -- ./harness prof_$k $B $NE > results/pmc4_${k}_B${B}.log 2>&1
    $R --dev igpu --mb $MB --label B_moe_prefill/pmc_$k --timeout 600 -- bash ../_infra/prof.sh pmc "L2CacheHit LDSBankConflict" prof_pmc5_$k -- ./harness prof_$k $B $NE > results/pmc5_${k}_B${B}.log 2>&1
done
for k in kwide kwide2; do
    $R --dev igpu --mb $MB --label B_moe_prefill/att_$k --timeout 900 -- bash ../_infra/prof.sh att igpu "mxfp4.*$k\$" prof_att_$k -- ./harness prof_$k $B $NE > results/att_${k}_B${B}.log 2>&1
done
echo prof_base done
