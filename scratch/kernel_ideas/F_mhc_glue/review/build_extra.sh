#!/usr/bin/env bash
set -eu
D=/home/claude-code/deepstrix/.claude/worktrees/kernel-ideas-2026-09-26/scratch/kernel_ideas/F_mhc_glue/review/repro
cd "$D"
source ../../../_infra/env.sh
../../../_infra/kcc.sh -O2 --offload-arch=gfx1201 extra.cpp -o extra_gfx1201
echo "extra build ok"
