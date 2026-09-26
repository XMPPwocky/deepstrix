#!/usr/bin/env bash
# CPU-only: which wave-reduce / DPP builtins do the toolchain headers expose?
set -u
cd "$(dirname "$0")/.."
source ../_infra/env.sh
d=$(dirname "$(dirname "$(command -v hipcc)")")
echo "hip root: $d"
grep -rl "__reduce_max_sync\|__reduce_add_sync" "$d/include" 2>/dev/null | head -3
grep -rn "__builtin_amdgcn_permlanex16\|__builtin_amdgcn_update_dpp\|__builtin_amdgcn_mov_dpp" "$d/include/hip/amd_detail/"*.h 2>/dev/null | head -5
grep -rn "__shfl_xor(" "$d/include/hip/amd_detail/amd_warp_functions.h" 2>/dev/null | head -3
