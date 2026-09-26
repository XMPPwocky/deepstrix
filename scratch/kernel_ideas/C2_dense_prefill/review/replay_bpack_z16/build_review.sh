#!/usr/bin/env bash
# Reviewer build for C2_dense_prefill/replay_bpack_z16. Baseline hsacos are compiled here from the
# UNMODIFIED in-tree sources with the exact production flags ($KFLAGS_V41 --genco --offload-arch=gfx1201);
# the candidate and the harness are compiled from read-only copies of the engineer's cand.hip / harness.cpp.
# Everything lands in this directory; the engineer's files are not touched.
set -eu
cd "$(dirname "$0")"
source ../../../_infra/env.sh
ARCH=gfx1201
cp ../../cand.hip cand.hip
cp ../../harness.cpp harness.cpp
for f in q8_0_matvec q8_0_grouped_matvec q8_0_matvec_wmma q8_k_quantize; do
  ../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$f.hip" -o base_${f}_$ARCH.hsaco
done
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand.hip -o cand_$ARCH.hsaco
../../../_infra/kcc.sh -O2 --offload-arch=$ARCH -I../../../_infra harness.cpp -o harness_$ARCH
sha256sum "$KERNELS_DIR/q8_0_matvec.hip" "$KERNELS_DIR/q8_0_grouped_matvec.hip" cand.hip harness.cpp > sources.sha256
cat sources.sha256
echo "KFLAGS_V41=$KFLAGS_V41"
echo built
