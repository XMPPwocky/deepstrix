#!/usr/bin/env bash
# hsaco.sh: compile kernels/g1_router_block.hip for gfx1201 (+ the .s) into /tmp/g1o/asm
W=/home/claude-code/deepstrix/.claude/worktrees/wf_be8e2331-368-2
mkdir -p /tmp/g1o/asm
cd /tmp/g1o/asm || exit 1
nix develop "$W" -c hipcc -O3 --genco --offload-arch=gfx1201 -DDEEPSTRIX_V41=1 -DMHC_N_EMBD=5120 -DMHC_HC_DIM=20480 -DROUTER_MAX_EXPERTS=512 \
  -save-temps "$W/crates/v4flash-kernels/kernels/g1_router_block.hip" -o /tmp/g1o/g1.hsaco 2>&1 | grep -E "error|warning" | head -20
ls -la /tmp/g1o/g1.hsaco
nix develop "$W" -c hipcc -O3 --genco --offload-arch=gfx1151 -DDEEPSTRIX_V41=1 -DMHC_N_EMBD=5120 -DMHC_HC_DIM=20480 -DROUTER_MAX_EXPERTS=512 \
  "$W/crates/v4flash-kernels/kernels/g1_router_block.hip" -o /tmp/g1o/g1_1151.hsaco 2>&1 | grep -E "error" | head -5
