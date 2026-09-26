#!/usr/bin/env bash
# Confirm the reviewer's cand7 code object has the same machine code as the engineer's
# (the .hsaco bytes differ only in ELF metadata).
set -u
cd "$(dirname "$0")"
source ../../../_infra/env.sh
llvm-objdump -d ../cand7_attn_gfx1201.hsaco | grep -v "file format" | sed 's/^ *[0-9a-f]*:[[:space:]]*//' > cand7_review.dis
llvm-objdump -d ../../cand7_attn_gfx1201.hsaco | grep -v "file format" | sed 's/^ *[0-9a-f]*:[[:space:]]*//' > cand7_eng.dis
wc -l cand7_review.dis cand7_eng.dis
if diff -q cand7_review.dis cand7_eng.dis >/dev/null; then echo "DISASM IDENTICAL"; else echo "DISASM DIFFERS"; diff cand7_review.dis cand7_eng.dis | head -20; fi
cmp -l ../cand7_attn_gfx1201.hsaco ../../cand7_attn_gfx1201.hsaco | wc -l
llvm-readelf --notes ../cand7_attn_gfx1201.hsaco 2>/dev/null | grep -E "vgpr_count|sgpr_count|lds_size|spill|\.name:" | head -20
