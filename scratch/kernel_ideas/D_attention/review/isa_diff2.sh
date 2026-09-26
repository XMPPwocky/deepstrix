#!/usr/bin/env bash
# Compare the pf2 kernel's ISA in the reviewer-built candidate object with the engineer's object
# (sha256 differs: check that the code is the same and only bundle metadata/paths differ).
set -u
cd "$(dirname "$0")"
bash ../../_infra/isa.sh cand_attn_gfx1201.hsaco gfx1201 --dis attention_dec_smwsum_pf2 > results/isa_pf2_rv.txt 2>&1
bash ../../_infra/isa.sh ../cand_attn_gfx1201.hsaco gfx1201 --dis attention_dec_smwsum_pf2 > results/isa_pf2_eng.txt 2>&1
wc -l results/isa_pf2_rv.txt results/isa_pf2_eng.txt
grep -v 'hsaco\|file format' results/isa_pf2_rv.txt | sed 's/^ *[0-9a-f]*:[[:space:]]*//' | sed 's#//.*##' > results/isa_pf2_rv.norm
grep -v 'hsaco\|file format' results/isa_pf2_eng.txt | sed 's/^ *[0-9a-f]*:[[:space:]]*//' | sed 's#//.*##' > results/isa_pf2_eng.norm
if diff -q results/isa_pf2_rv.norm results/isa_pf2_eng.norm > /dev/null; then
    echo "pf2 ISA IDENTICAL (engineer's object == rebuilt from cand_attn.hip)"
else
    echo "pf2 ISA DIFFERS:"
    diff results/isa_pf2_rv.norm results/isa_pf2_eng.norm | head -30
fi
echo "--- resource summary (rv build) ---"
grep -i -E "smwsum_pf2|vgpr|sgpr|lds|scratch|spill" results/isa_pf2_rv.txt | head -12
