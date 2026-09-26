#!/usr/bin/env bash
# gemm_narrow_m24 reviewer build (private subdir; review/ itself is shared with other reviewers).
#   bash build_gemm_review.sh [all|base|cand|harness]
set -eu
cd "$(dirname "$0")"
source ../../../_infra/env.sh
ARCH=gfx1201
what=${1:-all}
if [ "$what" = all ] || [ "$what" = base ]; then
    for k in mhc_fast f16_gemm_wmma rms_norm_no_weight hc_sinkhorn_par hc_weighted_sum rms_norm hc_post f16_matvec router_topk_par vec_add mhc_arena f16_matvec_narrow; do
        ../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o base_${k}_$ARCH.hsaco
    done
fi
if [ "$what" = all ] || [ "$what" = cand ]; then
    ../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand_gemm.hip -o cand_gemm_$ARCH.hsaco
fi
if [ "$what" = all ] || [ "$what" = harness ]; then
    ../../../_infra/kcc.sh -O2 --offload-arch=$ARCH harness.cpp -o harness_$ARCH
fi
echo "gemm review build ok: $what"
