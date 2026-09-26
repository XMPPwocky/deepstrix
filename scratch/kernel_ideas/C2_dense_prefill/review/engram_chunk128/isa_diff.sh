#!/usr/bin/env bash
# Is the engineer's cand_gfx1201.hsaco the same i8x_db kernel as the reviewer's rebuild from cand.hip?
set -u
cd "$(dirname "$0")"
cp ../../cand_gfx1201.hsaco eng_cand_gfx1201.hsaco
bash ../../../_infra/isa.sh cand_gfx1201.hsaco gfx1201 --dis q8_0_gemm_wmma_i8x_db > isa_rev.txt 2>&1
bash ../../../_infra/isa.sh eng_cand_gfx1201.hsaco gfx1201 --dis q8_0_gemm_wmma_i8x_db > isa_eng.txt 2>&1
awk '/^[0-9a-f]+ <q8_0_gemm_wmma_i8x_db>:$/ {p=1} /^[0-9a-f]+ <q8_0_gemm_wmma_i8x_db_/ {p=0} p' cand_gfx1201.s > k_rev.s
awk '/^[0-9a-f]+ <q8_0_gemm_wmma_i8x_db>:$/ {p=1} /^[0-9a-f]+ <q8_0_gemm_wmma_i8x_db_/ {p=0} p' eng_cand_gfx1201.s > k_eng.s
wc -l k_rev.s k_eng.s
cut -c 20- k_rev.s > k_rev.body; cut -c 20- k_eng.s > k_eng.body
if diff -q k_rev.body k_eng.body > /dev/null; then echo "ISA_IDENTICAL i8x_db"; else echo "ISA_DIFFERS"; diff k_rev.body k_eng.body | head -20; fi
echo "--- resources (review build)"; grep -B1 -A6 'q8_0_gemm_wmma_i8x_db$' isa_rev.txt | head -20
grep -c 's_barrier' k_rev.s
