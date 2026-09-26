#!/usr/bin/env bash
# Reviewer: compare the ISA of the engineer's cand2 code object with the reviewer's rebuild (via
# isa.sh, which unbundles), and diff the blk128 body against the production kernel body.
set -u
cd "$(dirname "$0")"
cp ../cand2_attn_gfx1201.hsaco cand2_eng.hsaco
bash ../../_infra/isa.sh cand2_eng.hsaco gfx1201 --dis > /dev/null
bash ../../_infra/isa.sh cand2_attn_gfx1201.hsaco gfx1201 --dis > /dev/null
bash ../../_infra/isa.sh base_attention_dec_gfx1201.hsaco gfx1201 --dis > /dev/null
norm() { sed -e '/file format/d' -e '/^$/d' -e 's/^ *[0-9a-f]*:[[:space:]]*//' -e 's/^[0-9a-f]* </</' "$1"; }
norm cand2_eng.s > cand2_eng.norm
norm cand2_attn_gfx1201.s > cand2_mine.norm
if diff -q cand2_eng.norm cand2_mine.norm > /dev/null; then echo "ISA IDENTICAL: engineer's cand2 hsaco == reviewer rebuild"; else echo "ISA DIFFERS (engineer vs reviewer cand2)"; diff cand2_eng.norm cand2_mine.norm | head -20; fi
ext() { awk -v pat="$2" '/^[0-9a-f]+ <.*>:$/ {p = index($0, pat) > 0} p' "$1" | sed -e 's/^ *[0-9a-f]*:[[:space:]]*//' -e 's/^[0-9a-f]* </</' -e '/^$/d' -e 's/s_endpgm.*/s_endpgm/' ; }
ext base_attention_dec_gfx1201.s attention_dec_score_htiled_wmma_f16s > base_body.s
ext cand2_attn_gfx1201.s attention_dec_score_blk128 > blk128_body.s
ext cand2_attn_gfx1201.s attention_dec_score_blk256 > blk256_body.s
echo "base body lines: $(wc -l < base_body.s)  blk128 body lines: $(wc -l < blk128_body.s)  blk256: $(wc -l < blk256_body.s)"
echo "--- diff base vs blk128 (label/branch-offset lines excluded) ---"
diff <(grep -v "^<\|s_cbranch\|s_branch" base_body.s) <(grep -v "^<\|s_cbranch\|s_branch" blk128_body.s) | head -40
echo "--- instruction histogram delta (base -> blk128) ---"
diff <(awk '{print $1}' base_body.s | sort | uniq -c) <(awk '{print $1}' blk128_body.s | sort | uniq -c) | head -30
