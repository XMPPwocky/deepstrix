#!/usr/bin/env bash
# Build the production baseline code objects (UNMODIFIED in-tree sources, exact build.rs flags), every
# candidate cand_*.hip in this dir, and the harness. Usage: build.sh [gfx1151] [cand-name ...]
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
ARCH=${1:-gfx1151}
shift || true
if [ ! -s base_pair_$ARCH.hsaco ] || [ "${REBUILD_BASE:-0}" = 1 ]; then
    ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/mxfp4_pair_matvec.hip" -o base_pair_$ARCH.hsaco
    ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/mxfp4_matvec.hip" -o base_down_$ARCH.hsaco
    ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q2_k_accumulate_matvec_par.hip" -o base_q2k_$ARCH.hsaco
    ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q8_k_quantize.hip" -o base_q8k_$ARCH.hsaco
    ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/moe_group_builder.hip" -o base_grp_$ARCH.hsaco
    ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/moe_work_items_builder.hip" -o base_wi_$ARCH.hsaco
fi
for c in "$@"; do
    ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH -I"$KERNELS_DIR" cand_$c.hip -o cand_${c}_$ARCH.hsaco
done
if [ ! -x harness_$ARCH ] || [ harness.cpp -nt harness_$ARCH ]; then
    ../_infra/kcc.sh -O2 --offload-arch=$ARCH harness.cpp -o harness_$ARCH
fi
echo "build ok: $ARCH $*"
