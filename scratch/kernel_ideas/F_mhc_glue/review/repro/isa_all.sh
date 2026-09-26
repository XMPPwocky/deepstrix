#!/usr/bin/env bash
# isa_all.sh [hsaco...]  -> per-kernel VGPR/SGPR/LDS/scratch for the given (or all base_) code objects
set -u
cd "$(dirname "$0")"
files=("$@")
[ ${#files[@]} -eq 0 ] && files=(base_mhc_fast_gfx1201.hsaco base_f16_gemm_wmma_gfx1201.hsaco base_rms_norm_gfx1201.hsaco base_router_topk_par_gfx1201.hsaco base_f16_matvec_gfx1201.hsaco base_hc_post_gfx1201.hsaco base_hc_sinkhorn_par_gfx1201.hsaco base_rms_norm_no_weight_gfx1201.hsaco)
for f in "${files[@]}"; do
    echo "## $f"
    bash ../_infra/isa.sh "$f" gfx1201 2>&1 | grep -v '^$'
done
