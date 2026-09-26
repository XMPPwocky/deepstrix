#!/usr/bin/env bash
# Reviewer build: baseline from the UNMODIFIED in-tree q8_0_matvec_wmma.hip with $KFLAGS_V41,
# candidate from the engineer's cand.hip with the same flags, harness from harness.cpp. All into review/.
set -eu
cd "$(dirname "$0")"
mkdir -p results
source ../../../_infra/env.sh
ARCH=gfx1201
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q8_0_matvec_wmma.hip" -o base_q8_0_matvec_wmma_$ARCH.hsaco
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH ../../cand.hip -o cand_$ARCH.hsaco
../../../_infra/kcc.sh -O2 --offload-arch=$ARCH ../../harness.cpp -o harness_$ARCH
echo built
