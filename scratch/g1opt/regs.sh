#!/usr/bin/env bash
# regs.sh [prefix]: per-kernel VGPRs / scratch / LDS / calls of the last hsaco.sh build
S=/tmp/g1o/asm/g1_router_block-hip-amdgcn-amd-amdhsa-gfx1201.s
P=${1:-g1_router_block_r}
echo "calls: $(grep -c s_swappc $S)"
awk -v P="$P" '/\.amdhsa_kernel /{k=$2} k ~ "^"P && /amdhsa_group_segment_fixed_size|amdhsa_private_segment_fixed_size|amdhsa_next_free_vgpr/{printf "%s %s %s\n", k, $1, $2}' $S
