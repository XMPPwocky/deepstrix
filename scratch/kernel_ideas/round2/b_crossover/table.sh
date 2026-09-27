#!/usr/bin/env bash
# table.sh : cross-run (r1..r5) crossover summary of every (b) measurement -> stdout
set -u
cd "$(dirname "$0")/results"
P="python3 ../../pairs.py"
short() { sed -e 's/ *| */ | /' -e 's/  */ /g' -e 's/ warm graph//' -e 's/ graph warm//' -e 's/COLD W (21 copies) graph/cold/' | cut -c1-190; }
echo "## any correctness failure:"; grep -h 'MISMATCH\|bitexact=no\|bit_diff=[1-9]\|diff=[1-9]' *_r[1-5].txt | grep -v -i twin | sort | uniq -c | head
echo "## F rms_fast";  $P --pairs 'old n=5120$=>fast n=5120$;;old n=1280$=>fast n=1280$;;old n=512$=>fast n=512$;;old head=>fast head' F_rms_r*.txt | short
echo "## F router h20"; $P F_router_r*.txt | short
echo "## F topk_wfred prior"; $P F_topk_r*.txt | short
echo "## F topk_wfred plain"; $P F_topkp_r*.txt | short
echo "## D fused / blk128"; $P --pairs 'base pair=>fused_vt_qreg_sp_d4;;base dec_score alone=>blk128 alone' D_decode_r*.txt | short
echo "## E score qreg"; $P --pairs 'base=>s2_w8n8_pf_hw' E_score_r*.txt E_score16_r*.txt | short
echo "## E select hybrid"; $P --pairs 'base select_ilp alone=>topk_select_v3_u8 alone' E_select_r*.txt E_select512_r*.txt | short
echo "## E threshold ilp"; $P --pairs 'base candidate_threshold$=>candidate_threshold_ilp$' E_cand_r*.txt E_cand64_r*.txt | short
echo "## G gather u4 vs old (+ pads)"; $P --tag COLD G_gather_r*.txt | short
echo "## C1 gemv tB"; $P --tag gemv --pairs 'base q8_0_gemv_bpack_warp8=>cand q8_0_gemv_bpack_tB[0-9]*$' C1_r*.txt | short
echo "## C1 grouped tB"; $P --tag grouped --pairs 'base q8_0_grouped_gemv_bpack=>cand q8_0_grouped_gemv_bpack_tB' C1_r*.txt | short
echo "## C1 shared fused (B r1 + tB down) vs tB chain"; $P --tag shared --pairs 'cand tB chain=>cand B r1 \+ tB down' C1_r*.txt | short
echo "## C1 quant wave (both padded)"; $P C1q_r*.txt | grep -v null | short
echo "## C2"; $P C2_r*.txt | short
echo "## A dn2 chain"; $P A_chain_b*_r*.txt | short
