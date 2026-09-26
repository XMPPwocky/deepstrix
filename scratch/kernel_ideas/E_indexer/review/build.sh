#!/usr/bin/env bash
# Reviewer's independent build into review/: baseline code objects from the UNMODIFIED in-tree
# sources ($KFLAGS_V41 --genco), the engineer's cand_topk.hip unchanged, the engineer's
# harness.cpp unchanged, plus the reviewer's extra-shape harness. Nothing in the engineer's
# directory is written.
set -eu
cd "$(dirname "$0")"
source ../../_infra/env.sh
ARCH=gfx1201
for k in indexer_topk_bitonic indexer_score_wmma; do
    ../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o base_${k}_$ARCH.hsaco
done
../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH -I"$KERNELS_DIR" ../cand_topk.hip -o cand_topk_$ARCH.hsaco
../../_infra/kcc.sh -O2 --offload-arch=$ARCH ../harness.cpp -o harness_$ARCH
../../_infra/kcc.sh -O2 --offload-arch=$ARCH extra.cpp -o extra_$ARCH
echo BUILD_OK
