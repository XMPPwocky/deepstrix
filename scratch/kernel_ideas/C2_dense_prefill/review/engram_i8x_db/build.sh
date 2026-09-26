#!/usr/bin/env bash
# Reviewer (engram_i8x_db) rebuild of the engineer's exact sources (copied from ../../cand.hip,
# ../../harness.cpp) into this dir. Baseline = UNMODIFIED in-tree q8_0_matvec_wmma.hip with
# $KFLAGS_V41 --genco (same as ../../build_base.sh). harness_dn.cpp = harness.cpp with wide-range
# (denormal-reaching, zero-containing) scales and extra tail batches for correctness.
set -eu
cd "$(dirname "$0")"
source ../../../_infra/env.sh
ARCH=gfx1201
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q8_0_matvec_wmma.hip" -o base_q8_0_matvec_wmma_$ARCH.hsaco
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand.hip -o cand_$ARCH.hsaco
../../../_infra/kcc.sh -O2 --offload-arch=$ARCH harness.cpp -o harness_$ARCH
../../../_infra/kcc.sh -O2 --offload-arch=$ARCH harness_dn.cpp -o harness_dn_$ARCH
echo built
