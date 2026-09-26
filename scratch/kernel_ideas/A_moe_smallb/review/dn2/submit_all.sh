#!/usr/bin/env bash
# Enqueue every review job on the iGPU scheduler; tickets to tickets.txt (collect with gpu_wait.sh).
set -u
cd "$(dirname "$0")"
S=../../../_infra/gpu_submit.sh
: > tickets.txt
sub() { bash $S "$@" | tee -a tickets.txt; }
# 2. the engineer's exact repro (chain A/B at b=2,4,8), three separate processes
sub --dev igpu --mb 450 --label review/A_moe_smallb -- bash run_chain_dn2.sh rep1 dn2_r2 "2 4 8"
sub --dev igpu --mb 450 --label review/A_moe_smallb -- bash run_chain_dn2.sh rep2 dn2_r2 "2 4 8"
sub --dev igpu --mb 450 --label review/A_moe_smallb -- bash run_chain_dn2.sh rep3 dn2_r2 "2 4 8"
# per-kernel numbers (claim: kwide2 184 -> 149 us at b=4 alone, bound grid)
sub --dev igpu --mb 450 --label review/A_moe_smallb -- bash run_parts_dn2.sh parts1 dn2_r2 "4 8"
# 3. shapes the engineer's claim did not list: b=1, odd b, b=7
sub --dev igpu --mb 450 --label review/A_moe_smallb -- bash run_chain_dn2.sh tails dn2_r2 "1 3 5 7"
# production-like regimes: many distinct experts, ~1 member per group (box 2 / hub), and 1 expert x 8 members
sub --dev igpu --mb 450 --label review/A_moe_smallb -- bash run_chain_dn2.sh E12 dn2_r2 "4 8" E=12 ppr=3
sub --dev igpu --mb 800 --label review/A_moe_smallb -- bash run_chain_dn2.sh E24 dn2_r2 "4 8" E=24 ppr=6 P=32
sub --dev igpu --mb 450 --label review/A_moe_smallb -- bash run_chain_dn2.sh E1 dn2_r2 "5 8" E=1 ppr=1
sub --dev igpu --mb 800 --label review/A_moe_smallb -- bash run_parts_dn2.sh E24parts dn2_r2 "4 8" E=24 ppr=6 P=32
echo "submitted: $(wc -l < tickets.txt) tickets"
