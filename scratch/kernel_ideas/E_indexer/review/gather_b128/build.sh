#!/usr/bin/env bash
# Reviewer build for the E_indexer/gather_b128 claim: baseline hsaco from the UNMODIFIED in-tree
# indexer_gather.hip ($KFLAGS_V41 --genco), the engineer's cand_gather.hip unchanged, the engineer's
# harness.cpp unchanged, and the reviewer's correctness checker gcheck.cpp -- all into this
# directory. Nothing in the engineer's directory is written. CPU only (kcc.sh compile semaphore).
set -eu
cd "$(dirname "$0")"
source ../../../_infra/env.sh
ARCH=gfx1201
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/indexer_gather.hip" -o base_indexer_gather_$ARCH.hsaco
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH -I"$KERNELS_DIR" ../../cand_gather.hip -o cand_gather_$ARCH.hsaco
../../../_infra/kcc.sh -O2 --offload-arch=$ARCH ../../harness.cpp -o harness_$ARCH.tmp && mv harness_$ARCH.tmp harness_$ARCH
../../../_infra/kcc.sh -O2 --offload-arch=$ARCH gcheck.cpp -o gcheck_$ARCH.tmp && mv gcheck_$ARCH.tmp gcheck_$ARCH
b=NO; cmp -s base_indexer_gather_$ARCH.hsaco ../../base_indexer_gather_$ARCH.hsaco && b=yes
c=NO; cmp -s cand_gather_$ARCH.hsaco ../../cand_gather_$ARCH.hsaco && c=yes
echo "hsaco byte-identical to the engineer's?  base=$b  cand=$c"
