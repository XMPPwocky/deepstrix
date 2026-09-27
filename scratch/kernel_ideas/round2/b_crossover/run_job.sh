#!/usr/bin/env bash
# run_job.sh <family> <tag> : ONE scheduler job's worth of crossover measurements (run from
# round2/b_crossover; outputs to results/<family>_<part>_<tag>.txt). Submitted by submit.sh.
set -u
cd "$(dirname "$0")"
fam=$1; tag=$2
R=results
case "$fam" in
F)
    ./xover_f rms 1 2 3 4 5 6 8 12 16 64 512 1024 > $R/F_rms_$tag.txt 2>&1
    ./xover_f router 1 2 3 4 5 6 8 12 16 32 64 > $R/F_router_$tag.txt 2>&1
    ./xover_f topk 1 2 3 4 5 6 8 12 16 24 32 64 128 512 > $R/F_topk_$tag.txt 2>&1
    ./xover_f topkp 1 4 16 24 32 48 64 96 128 512 > $R/F_topkp_$tag.txt 2>&1 ;;
D)
    CANDS=attention_dec_fused_vt_qreg_sp_d4,attention_dec_score_blk128 ./xattn gfx1201 ../../D_attention decode 1 2 3 4 5 6 8 12 16 > $R/D_decode_$tag.txt 2>&1 ;;
E)
    D=../../E_indexer
    SCORE_CANDS=s2_w8n8_pf_hw ./xe score $D n=32768 b=1,2,3,4,5,6,8 rounds=30 > $R/E_score_$tag.txt 2>&1
    SCORE_CANDS=s2_w8n8_pf_hw ./xe score $D n=12288 b=12,16 rounds=30 > $R/E_score16_$tag.txt 2>&1
    ./xe select $D n=131072 b=1,2,3,4,5,6,8,12,16 rounds=30 > $R/E_select_$tag.txt 2>&1
    E_STRIDE=32896 ./xe select $D n=32768 b=64,512 rounds=20 > $R/E_select512_$tag.txt 2>&1
    ./xe cand $D n=131072 b=1,2,3,4,5,6,8,12,16 rounds=30 > $R/E_cand_$tag.txt 2>&1
    E_STRIDE=32896 ./xe cand $D n=32768 b=8,16,24,32,48,64,128 rounds=30 > $R/E_cand64_$tag.txt 2>&1 ;;
G)
    GP_DIR=../a_gridpad ../a_gridpad/gpad_real gather 1 2 3 4 5 6 8 12 16 > $R/G_gather_$tag.txt 2>&1 ;;
C1)
    cd ../../C1_dense_decode
    C1_DIR=$PWD/intree ./harness gemv qa 1 1 : gemv qa 2 1 : gemv qa 3 1 : gemv qa 4 1 : gemv qa 5 1 : gemv qa 6 1 : gemv qa 8 1 : gemv qa 9 1 : gemv qa 10 1 \
        : gemv kv 1 : gemv kv 3 : gemv kv 6 : gemv kv 9 : gemv kv 10 \
        : gemv gate 1 : gemv gate 6 : gemv gate 9 : gemv gate 10 \
        : gemv down 1 : gemv down 6 : gemv down 9 : gemv down 10 \
        : grouped 1 : grouped 2 : grouped 3 : grouped 4 : grouped 5 : grouped 6 : grouped 7 : grouped 8 \
        : shared 1 : shared 2 : shared 3 : shared 4 : shared 5 : shared 6 : shared 7 : shared 8 \
        > ../round2/b_crossover/$R/C1_$tag.txt 2>&1
    C1_DIR=$PWD/intree C1_WARM=1 C1_GPAD=1 ./harness quant 5120 1 : quant 5120 4 : quant 5120 8 : quant 5120 16 : quant 1280 4 : quant 2304 4 \
        : quant 6144 16 : quant 6144 64 : quant 6144 128 : quant 5120 64 : quant 5120 512 : quant 32768 4 : quant 8192 16 \
        > ../round2/b_crossover/$R/C1q_$tag.txt 2>&1 ;;
C2)
    D=../../C2_dense_prefill; H="./xc2 gfx1201 $D"
    o=$R/C2_$tag.txt; : > $o
    for b in 65 128 256 512 1024; do $H kv --b $b --cand q8_0_gemm_wmma_f16x_db_bn64 --rounds 20 --flushmb 64 >> $o 2>&1; done
    for b in 65 128 256 512 768 1024; do $H qa --b $b --cand q8_0_gemm_wmma_f16x_db_bn64 --rounds 20 --flushmb 64 >> $o 2>&1; done
    for b in 65 128 192; do $H qb --b $b --cand q8_0_gemm_wmma_f16x_256x128 --rounds 20 --flushmb 48 >> $o 2>&1; done
    for b in 65 128 192 256 384; do $H woa --b $b --cand q8_0_gemm_wmma_f16x_256x128 --rounds 20 --flushmb 48 >> $o 2>&1; done
    for b in 17 24 32 48 64; do
        $H qb64 --b $b --cand q8_0_gemv_bpack_z16 --rounds 20 --flushmb 64 >> $o 2>&1
        $H woa64 --b $b --cand q8_0_grouped_gemv_bpack_z16 --rounds 20 --flushmb 64 >> $o 2>&1
    done
    for b in 17 24 32 40 48 56 64; do
        $H kv64 --b $b --cand q8_0_gemv_bpack_z16 --rounds 20 --flushmb 64 >> $o 2>&1
        $H qa64 --b $b --cand q8_0_gemv_bpack_z16 --rounds 20 --flushmb 64 >> $o 2>&1
        $H shg64 --b $b --cand q8_0_gemv_bpack_z16 --rounds 20 --flushmb 64 >> $o 2>&1
        $H shd64 --b $b --cand q8_0_gemv_bpack_z16 --rounds 20 --flushmb 64 >> $o 2>&1
    done
    for b in 9 16 32 64 128; do $H engram --m 6400 --b $b --cand q8_0_gemm_wmma_i8x_db --rounds 20 --flushmb 64 >> $o 2>&1; done ;;
A)
    D=../../A_moe_smallb
    for b in 1 2 3 4 5 6 8 12 16 24 32 64; do
        $D/harness_gfx1151 gfx1151 $D chain b=$b E=16 ppr=6 rounds=30 inner=8 cand=dn2_r2 > $R/A_chain_b${b}_$tag.txt 2>&1
    done ;;
esac
echo "done $fam $tag"
