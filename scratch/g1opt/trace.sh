#!/usr/bin/env bash
# trace.sh <tag> <B list> <variants> [ENV=val ...]: in-kernel phase trace of the hand-built /tmp/g1o/g1.hsaco
W=/home/claude-code/deepstrix/.claude/worktrees/wf_be8e2331-368-2
tag=$1; shift
bl=$1; shift
vs=$1; shift
"$W/scratch/g1opt/run.sh" bench_g1_router_block "/tmp/g1o/tr_$tag.log" V41_DGPU_G1_HSACO_GFX1201=/tmp/g1o/g1.hsaco V41_DGPU_G1_TS=1 \
  BENCH_ROUNDS="${ROUNDS:-30}" BENCH_B="$bl" "BENCH_G1_VARIANTS=$vs" "$@" > /dev/null
grep -v '^:3:' "/tmp/g1o/tr_$tag.log" | grep -E "${PAT:-TRACE|collapse flag|router flag seen max|router logits max|tail start|tail end|last finish|Error|no variant}"
