#!/usr/bin/env bash
# isa_flow.sh <hsaco> <kernel>  -> condensed instruction flow (memory/wait/barrier/wmma/branch only)
set -u
cd "$(dirname "$0")"
bash ../_infra/isa.sh "$1" gfx1201 --dis "$2" > "$2.$(basename "$1" .hsaco).s" 2>&1
grep -o "s_wait_loadcnt[_dscnt]* 0x[0-9a-f]*\|s_wait_dscnt 0x[0-9a-f]*\|s_barrier_signal\|v_wmma\|ds_load_b128\|ds_store_b128\|global_load_b128\|global_load_b32\|global_load_u16\|s_cbranch_[a-z]*\|v_cvt_f32_i32\|v_cvt_f16_f32\|v_cndmask\|s_endpgm" "$2.$(basename "$1" .hsaco).s" | uniq -c | while read n i; do printf "%s %s | " "$n" "$i"; done
echo
