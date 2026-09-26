#!/usr/bin/env bash
# flow_full.sh <kernel_full.s> : untruncated condensed flow with line numbers of the loop back-edges
set -u
cd "$(dirname "$0")"
grep -n -o "s_wait_loadcnt[_dscnt]* 0x[0-9a-f]*\|s_wait_dscnt 0x[0-9a-f]*\|s_barrier_signal\|v_wmma\|ds_load_b128\|ds_store_b128\|global_load_b128\|global_load_b32\|global_load_u16\|s_cbranch_[a-z]*\|v_cvt_f32_i32\|s_endpgm" "$1" | python3 -c '
import sys
prev=None; n=0; out=[]
for l in sys.stdin:
    ln, ins = l.rstrip().split(":",1)
    if ins == prev: n += 1
    else:
        if prev is not None: out.append(f"{n}x{prev}" if n>1 else prev)
        prev = ins; n = 1
out.append(f"{n}x{prev}" if n>1 else prev)
print(" | ".join(out))
'
