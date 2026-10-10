#!/usr/bin/env bash
# Build only what the f16-arm A/B needs: production baseline objects, cand_f16 (NTMAX from $1, default 2), harness.
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
ARCH=gfx1151
NT=${1:-2}
for k in mxfp4_pair_matvec mxfp4_matvec q8_k_quantize q2_k_accumulate_matvec_par moe_group_builder moe_work_items_builder; do
    [ -f base_${k}_$ARCH.hsaco ] || ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o base_${k}_$ARCH.hsaco
done
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH -I "$KERNELS_DIR" -DWM_NTMAX=$NT cand_f16.hip -o cand_f16_$ARCH.hsaco
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH -I "$KERNELS_DIR" cand_wmma.hip -o cand_wmma_$ARCH.hsaco
[ -x harness ] || ../_infra/kcc.sh -O2 --offload-arch=$ARCH harness.cpp -o harness
echo build ok
