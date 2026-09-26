#!/usr/bin/env bash
# Engram wkv (25600x6144 Q8_0, int8 activations) per 64-row chunk: production lds_tiled vs i8x twins.
# Full M now fits (hub down). run_engram.sh <run-tag> [--corr] [--b N]
set -u
cd "$(dirname "$0")"
TAG=${1:-r1}; shift || true
R=results/engram_$TAG; mkdir -p $R
./harness_gfx1201 gfx1201 . engram --m 25600 --cand i8x_r1,r1c_i8x,r1c_i8x_unc,r1c_i8x_aonc,r1c_i8x_both,q8_0_gemm_wmma_i8x,q8_0_gemm_wmma_i8x_db "$@" > $R/engram_m25600.txt 2>$R/engram_m25600.err
./harness_gfx1201 gfx1201 . engram --m 25600 --b 128 --cand i8x_r1,r1c_i8x,r1c_i8x_unc,r1c_i8x_aonc,r1c_i8x_both,q8_0_gemm_wmma_i8x,q8_0_gemm_wmma_i8x_db "$@" > $R/engram_m25600_b128.txt 2>$R/engram_m25600_b128.err
grep -h "^==\|^base\|^cand\|^CMP\|rror" $R/*.txt
