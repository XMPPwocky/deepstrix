#!/usr/bin/env bash
# usage: tsweep.sh <hsaco> <B list> <dbg values...> -- in-kernel phase trace per V41_DGPU_G1_DBG value
S=/home/claude-code/deepstrix/.claude/worktrees/wf_c1d97dd2-693-2/scratch/g1
H=$1; shift
B=$1; shift
for d in "$@"; do
  echo "=== dbg $d"
  "$S/run.sh" bench_g1_router_block /tmp/g1/tsweep_$d.log BENCH_ROUNDS=40 BENCH_B=$B V41_DGPU_G1_TS=1 V41_DGPU_G1_DBG=$d \
    V41_DGPU_G1_HSACO_GFX1201=$H ${TS_ENV} > /dev/null
  grep -v '^:3:' /tmp/g1/tsweep_$d.log | grep -E "${TS_PAT:-TRACE|flag seen max|logits max|tail end|tail row0|q8k}"
done
