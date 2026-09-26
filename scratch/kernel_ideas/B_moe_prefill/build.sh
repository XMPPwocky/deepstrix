#!/usr/bin/env bash
# B_moe_prefill: production baseline code objects (UNMODIFIED in-tree sources, production
# flags) + candidate code objects + the host harness. gfx1151 only (iGPU family).
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
ARCH=gfx1151
for k in mxfp4_pair_matvec mxfp4_matvec q8_k_quantize q2_k_accumulate_matvec_par moe_group_builder moe_work_items_builder; do
    ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o base_${k}_$ARCH.hsaco
done
for c in cand_wmma; do
    if [ -f $c.hip ]; then
        ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH $c.hip -o ${c}_$ARCH.hsaco
    fi
done
../_infra/kcc.sh -O2 --offload-arch=$ARCH harness.cpp -o harness
echo build ok
