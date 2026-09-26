#!/usr/bin/env bash
# Reviewer: is the gateup rebuild of cand_wmma the same machine code as the engineer's hsaco?
# (md5 differs; the hsaco is an offload bundle, so go through isa.sh's disassembly.)
set -u
cd "$(dirname "$0")"
bash ../../../_infra/isa.sh cand_wmma_gfx1151.hsaco gfx1151 --dis wmma > dis_rv.s 2>&1
bash ../../../_infra/isa.sh ../../cand_wmma_gfx1151.hsaco gfx1151 --dis wmma > dis_eng.s 2>&1
wc -l dis_rv.s dis_eng.s
grep -v "\.hsaco\|cand_wmma" dis_rv.s | sed 's/^ *[0-9a-f]*://' > dis_rv.norm
grep -v "\.hsaco\|cand_wmma" dis_eng.s | sed 's/^ *[0-9a-f]*://' > dis_eng.norm
if diff -q dis_rv.norm dis_eng.norm > /dev/null; then echo "DISASSEMBLY IDENTICAL ($(wc -l < dis_rv.norm) lines)"; else echo "DISASSEMBLY DIFFERS"; diff dis_rv.norm dis_eng.norm | head -20; fi
head -12 dis_rv.s
