#!/usr/bin/env bash
# Build the production baseline code objects (unmodified in-tree sources, production flags),
# the candidate code objects, and the host harness — gfx1201 (dGPU) only.
#   bash build.sh            -> everything
#   bash build.sh base       -> only the production hsacos
#   bash build.sh cand       -> only cand_*.hip
#   bash build.sh harness    -> only harness.cpp
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
ARCH=gfx1201
what=${1:-all}
if [ "$what" = all ] || [ "$what" = base ]; then
    for k in mhc_fast f16_gemm_wmma rms_norm_no_weight hc_sinkhorn_par hc_weighted_sum rms_norm \
             hc_post f16_matvec router_topk_par vec_add mhc_arena f16_matvec_narrow; do
        ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o base_${k}_$ARCH.hsaco
    done
fi
if [ "$what" = all ] || [ "$what" = cand ]; then
    for c in cand_*.hip; do
        [ -e "$c" ] || continue
        ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$c" -o "${c%.hip}_$ARCH.hsaco"
    done
fi
if [ "$what" = all ] || [ "$what" = harness ]; then
    ../_infra/kcc.sh -O2 --offload-arch=$ARCH harness.cpp -o harness_$ARCH
fi
echo "build ok: $what"
