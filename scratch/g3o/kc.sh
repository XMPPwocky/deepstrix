#!/usr/bin/env bash
# usage: kc.sh <src.hip> <out.hsaco> [extra hipcc flags...] -- production kernel flags + resource usage
W=/home/claude-code/deepstrix/.claude/worktrees/wf_be8e2331-368-4
cd "$W" || exit 1
src=$1; shift
out=$1; shift
nix develop -c nice -n 19 hipcc -O3 --genco --offload-arch=gfx1201 -DDEEPSTRIX_V41=1 -DMHC_N_EMBD=5120 -DMHC_HC_DIM=20480 \
  -DROUTER_MAX_EXPERTS=512 "$@" -Rpass-analysis=kernel-resource-usage "$src" -o "$out" 2>&1 \
  | grep -E "error|Function Name|VGPRs|LDS Size|Occupancy|Spill|ScratchSize" | sed 's/.*remark: *//'
