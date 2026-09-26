#!/usr/bin/env bash
# Reproduce every F_mhc_glue number in NOTES.md / results/ (dGPU gfx1201, via the scheduler).
# Run from anywhere: bash scratch/kernel_ideas/F_mhc_glue/repro.sh
set -eu
cd "$(dirname "$0")"
S=../_infra/gpu_submit.sh
W=../_infra/gpu_wait.sh
bash build.sh                       # base_*_gfx1201.hsaco (unmodified in-tree, $KFLAGS_V41 --genco), cand_*.hsaco, harness_gfx1201
t=$(bash $S --dev dgpu --mb 400 --label F_mhc_glue/baseline --timeout 300 -- bash jobs/base_all.sh r2);   bash $W $t   # baselines (decode b=1/4 graph+direct, prefill chain warm+flush)
t=$(bash $S --dev dgpu --mb 600 --label F_mhc_glue/cands_r1 --timeout 300 -- bash jobs/cands.sh r1 all);  bash $W $t   # GEMM (incl. tails 65/100/128/1024), topk_wave, rms: correctness + A/B
t=$(bash $S --dev dgpu --mb 600 --label F_mhc_glue/winners_r2 --timeout 300 -- bash jobs/winners.sh r2); bash $W $t   # repeat runs
t=$(bash $S --dev dgpu --mb 600 --label F_mhc_glue/winners_r3 --timeout 300 -- bash jobs/winners.sh r3); bash $W $t
t=$(bash $S --dev dgpu --mb 500 --label F_mhc_glue/mv_topk_r1 --timeout 300 -- bash jobs/mv_topk.sh r1); bash $W $t   # router matvec (warm + cold-W rotation) + wfred top-k
t=$(bash $S --dev dgpu --mb 500 --label F_mhc_glue/mv_topk_r2 --timeout 300 -- bash jobs/mv_topk.sh r2); bash $W $t
t=$(bash $S --dev dgpu --mb 500 --label F_mhc_glue/mv_topk_r3 --timeout 300 -- bash jobs/mv_topk.sh r3); bash $W $t
t=$(bash $S --dev dgpu --mb 300 --label F_mhc_glue/rms_b512 --timeout 300 -- env HARNESS_DIR=$PWD ./harness_gfx1201 rms 512 direct); bash $W $t   # prefill-shape rms check
t=$(bash $S --dev dgpu --mb 500 --label F_mhc_glue/att_cands --timeout 600 -- bash jobs/att_cands.sh); bash $W $t   # ATT traces (dGPU only) of winners + baselines
bash isa_all.sh cand_gemm_gfx1201.hsaco cand_rms_gfx1201.hsaco cand_router_mv_gfx1201.hsaco cand_topk_gfx1201.hsaco cand_topk2_gfx1201.hsaco
