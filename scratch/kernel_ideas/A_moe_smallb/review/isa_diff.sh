#!/usr/bin/env bash
# Compare the reviewer's rebuilt candidate code object with the engineer's (md5 differs; is the ISA the same?)
set -u
cd "$(dirname "$0")"
mkdir -p eng_copy
cp ../cand_wl_c8dn2_gfx1151.hsaco eng_copy/
bash ../../_infra/isa.sh cand_wl_c8dn2_gfx1151.hsaco gfx1151 --dis cand_ > isa_rev.txt 2>&1
bash ../../_infra/isa.sh eng_copy/cand_wl_c8dn2_gfx1151.hsaco gfx1151 --dis cand_ > isa_eng.txt 2>&1
grep -v 'hsaco\|\.file\|\.ident\|eng_copy' isa_rev.txt > isa_rev.clean
grep -v 'hsaco\|\.file\|\.ident\|eng_copy' isa_eng.txt > isa_eng.clean
if diff -q isa_rev.clean isa_eng.clean > /dev/null; then echo "ISA IDENTICAL (cand_gate_up + cand_down)"; else echo "ISA DIFFERS"; diff isa_rev.clean isa_eng.clean | head -20; fi
wc -l isa_rev.clean isa_eng.clean
