#!/usr/bin/env bash
# usage: ablate.sh <dbg values...> -- g1 kernel durations (rocprof, dGPU only) per V41_DGPU_G1_DBG value
S=/home/claude-code/deepstrix/.claude/worktrees/wf_c1d97dd2-693-2/scratch/g1
for d in "$@"; do
  "$S/prof.sh" bench_g1_router_block /tmp/g1/abl_$d BENCH_ROUNDS=20 BENCH_B=${ABL_B:-1,4} "BENCH_G1_VARIANTS=${ABL_V:-1,1}" V41_DGPU_G1_DBG=$d ${ABL_ENV} > /dev/null
  echo "== dbg $d"
  python3 "$S/kdur.py" /tmp/g1/abl_$d | grep g1_router
done
