#!/usr/bin/env bash
# usage: npfsweep.sh <B list> <variants> <npf values...> -- in-kernel trace per V41_DGPU_G1_NPF
S=/home/claude-code/deepstrix/.claude/worktrees/wf_c1d97dd2-693-2/scratch/g1
B=$1; shift
V=$1; shift
for n in "$@"; do
  echo "##### NPF $n"
  "$S/run.sh" bench_g1_router_block /tmp/g1/npf_$n.log BENCH_ROUNDS=40 BENCH_B=$B V41_DGPU_G1_TS=1 V41_DGPU_G1_NPF=$n \
    "BENCH_G1_VARIANTS=$V" ${NPF_ENV} > /dev/null
  grep -v '^:3:' /tmp/g1/npf_$n.log | grep -E "${TS_PAT:-TRACE|flag seen max|logits max|tail end|collapse0 flag}" | grep -v ", 160, "
done
