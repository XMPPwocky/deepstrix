#!/usr/bin/env bash
# Reviewer build: baseline from the UNMODIFIED in-tree source with $KFLAGS_V41 (compared by hash to the
# engineer's), the engineer's candidate + tB7/tB9 (cand_bpack_rev.hip), and the reviewer harness.
set -eu
cd "$(dirname "$0")"
source ../../../_infra/env.sh
ARCH=gfx1201
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q8_0_matvec.hip" -o base_q8_$ARCH.hsaco &
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand_bpack_rev.hip -o cand_bpack_$ARCH.hsaco &
../../../_infra/kcc.sh -O2 --offload-arch=$ARCH -I../../../_infra harness_lanes.cpp -o harness_lanes &
wait
echo built
sha256sum base_q8_$ARCH.hsaco ../../base_q8_$ARCH.hsaco
bash ../../../_infra/isa.sh cand_bpack_$ARCH.hsaco $ARCH | grep -E "tB(4|5|6|7|8|9|10)\b|kernel|vgpr" | head -30
