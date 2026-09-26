#!/usr/bin/env bash
# Reviewer (f16x_256x128 claim) build: baseline from the UNMODIFIED in-tree source with $KFLAGS_V41,
# candidate from a copy of ../../cand.hip, harness from a copy of ../../harness.cpp.
set -eu
cd "$(dirname "$0")"
source ../../../_infra/env.sh
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=gfx1201 "$KERNELS_DIR/q8_0_matvec_wmma.hip" -o base_q8_0_matvec_wmma_gfx1201.hsaco
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=gfx1201 cand.hip -o cand_gfx1201.hsaco
../../../_infra/kcc.sh -O2 --offload-arch=gfx1201 harness.cpp -o harness_gfx1201
md5sum base_q8_0_matvec_wmma_gfx1201.hsaco ../../base_q8_0_matvec_wmma_gfx1201.hsaco cand_gfx1201.hsaco ../../cand_gfx1201.hsaco
echo built
