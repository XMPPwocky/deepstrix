#!/usr/bin/env bash
# Family E (dGPU indexer): production baseline code objects from the UNMODIFIED in-tree
# sources, candidate code objects, and the harness. gfx1201 only (this family is dGPU).
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
ARCH=gfx1201
what=${1:-all}
if [ "$what" = all ] || [ "$what" = base ]; then
    for k in indexer_score_wmma indexer_topk_bitonic indexer_gather candidate_blocks indexer_qat f16_matvec vec_scale_inplace; do
        ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o base_${k}_$ARCH.hsaco
    done
fi
if [ "$what" = all ] || [ "$what" = cand ]; then
    for c in cand_*.hip; do
        [ -e "$c" ] || continue
        ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH -I"$KERNELS_DIR" "$c" -o "${c%.hip}_$ARCH.hsaco"
    done
fi
if [ "$what" = all ] || [ "$what" = harness ]; then
    ../_infra/kcc.sh -O2 --offload-arch=$ARCH harness.cpp -o harness_$ARCH.tmp && mv harness_$ARCH.tmp harness_$ARCH
fi
