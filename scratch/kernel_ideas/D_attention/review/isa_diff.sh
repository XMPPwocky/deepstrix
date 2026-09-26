#!/usr/bin/env bash
# Compare the ISA of the reviewer-built candidate object with the engineer's (hash differs: check
# that only metadata/paths differ, not code).
set -u
cd "$(dirname "$0")"
bash ../../_infra/in_env.sh llvm-objdump -d --offloading cand_attn_gfx1201.hsaco > results/isa_cand_rv.s 2>&1
bash ../../_infra/in_env.sh llvm-objdump -d --offloading ../cand_attn_gfx1201.hsaco > results/isa_cand_eng.s 2>&1
wc -l results/isa_cand_rv.s results/isa_cand_eng.s
grep -v 'file format' results/isa_cand_rv.s | grep -v '^/' | sed 's/^ *[0-9a-f]*:[[:space:]]*//' > results/isa_cand_rv.norm
grep -v 'file format' results/isa_cand_eng.s | grep -v '^/' | sed 's/^ *[0-9a-f]*:[[:space:]]*//' > results/isa_cand_eng.norm
if diff -q results/isa_cand_rv.norm results/isa_cand_eng.norm > /dev/null; then
    echo "ISA IDENTICAL (engineer's cand_attn hsaco == rebuilt from cand_attn.hip)"
else
    echo "ISA DIFFERS:"
    diff results/isa_cand_rv.norm results/isa_cand_eng.norm | head -30
fi
bash ../../_infra/in_env.sh llvm-readelf -n cand_attn_gfx1201.hsaco 2>/dev/null | grep -E "\.name:|\.vgpr_count|\.sgpr_count|\.group_segment_fixed_size|\.private_segment_fixed_size|\.max_flat_workgroup_size" | grep -B1 -A4 "attention_dec_smwsum_pf2" | head -12
