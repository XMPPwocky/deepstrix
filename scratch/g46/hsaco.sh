#!/usr/bin/env bash
# hsaco.sh <kernel basename> : compile kernels/<name>.hip for gfx1201 with -save-temps into /tmp/g46/asm
W=/home/claude-code/deepstrix/.claude/worktrees/wf_be8e2331-368-6
mkdir -p /tmp/g46/asm
cd /tmp/g46/asm || exit 1
nix develop "$W" -c hipcc -O3 --genco --offload-arch=gfx1201 -DDEEPSTRIX_V41=1 -DMHC_N_EMBD=5120 -DMHC_HC_DIM=20480 -DROUTER_MAX_EXPERTS=512 \
  -save-temps "$W/crates/v4flash-kernels/kernels/$1.hip" -o /tmp/g46/$1.hsaco 2>&1 | grep -E "error|warning" | head -20
ls -la /tmp/g46/$1.hsaco
