#!/usr/bin/env bash
# Per-kernel resources (VGPR/SGPR/LDS/scratch) of the production code objects vs the wl candidate, and a
# disassembly diff of my cand_wl build vs the engineer's (the .hsaco bytes differ; the code should not).
set -u
cd "$(dirname "$0")"
I=../../../_infra
echo "=== base_pair (production kwide)"; bash $I/isa.sh base_pair_gfx1151.hsaco gfx1151 | grep -i 'kwide\|vgpr\|name' | grep -A3 -i 'fused_swiglu_kwide' | head -8
echo "=== base_down (production kwide2)"; bash $I/isa.sh base_down_gfx1151.hsaco gfx1151 | grep -A3 -i 'by_expert_kwide2' | head -8
echo "=== cand_wl (mine)"; bash $I/isa.sh cand_wl_gfx1151.hsaco gfx1151 --dis > /dev/null; bash $I/isa.sh cand_wl_gfx1151.hsaco gfx1151
echo "=== cand_wl (engineer's)"; cp ../../cand_wl_gfx1151.hsaco eng_cand_wl_gfx1151.hsaco; bash $I/isa.sh eng_cand_wl_gfx1151.hsaco gfx1151 --dis > /dev/null
if diff -q cand_wl_gfx1151.s eng_cand_wl_gfx1151.s > /dev/null; then echo "DISASSEMBLY IDENTICAL: my cand_wl build == engineer's"; else echo "disassembly differs:"; diff cand_wl_gfx1151.s eng_cand_wl_gfx1151.s | head -20; fi
