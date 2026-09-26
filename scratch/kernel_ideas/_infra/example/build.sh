#!/usr/bin/env bash
# Build the production baseline code object (unmodified in-tree source, production flags),
# the candidate code object (same flags), and the host harness — for one arch.
set -eu
cd "$(dirname "$0")"
source ../env.sh
ARCH=${1:-gfx1151}
../kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/vec_add.hip" -o base_$ARCH.hsaco
../kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand.hip -o cand_$ARCH.hsaco
../kcc.sh -O2 --offload-arch=$ARCH harness.cpp -o harness_$ARCH
