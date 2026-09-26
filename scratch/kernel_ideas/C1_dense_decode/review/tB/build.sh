#!/usr/bin/env bash
# Reviewer build (tB claim): baseline code objects from the UNMODIFIED in-tree sources with
# $KFLAGS_V41, candidate from a verbatim copy of the engineer's cand_bpack.hip, the engineer's
# harness.cpp verbatim, and the reviewer's odd-shape checker. Byte-compares baselines to the engineer's.
set -eu
cd "$(dirname "$0")"
source ../../../_infra/env.sh
ARCH=gfx1201
K=../../../_infra/kcc.sh
$K $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q8_0_matvec.hip" -o base_q8_$ARCH.hsaco
$K $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q8_0_grouped_matvec.hip" -o base_grp_$ARCH.hsaco
$K $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/swiglu.hip" -o base_swiglu_$ARCH.hsaco
$K $KFLAGS_V41 --genco --offload-arch=$ARCH src/cand_bpack.hip -o cand_bpack_$ARCH.hsaco
$K $KFLAGS_V41 --genco --offload-arch=$ARCH src/cand_shared.hip -o cand_shared_$ARCH.hsaco
$K -O2 --offload-arch=$ARCH -I../../../_infra src/harness.cpp -o harness
$K -O2 --offload-arch=$ARCH -I../../../_infra src/odd.cpp -o odd
echo built
for f in base_q8 base_grp base_swiglu; do
  if cmp -s "${f}_$ARCH.hsaco" "../../${f}_$ARCH.hsaco"; then echo "SAME $f hsaco as engineer's"; else echo "DIFF $f hsaco vs engineer's"; fi
done
cmp -s src/cand_bpack.hip ../../cand_bpack.hip && echo "SAME cand_bpack.hip source" || echo "DIFF cand_bpack.hip source"
cmp -s src/harness.cpp ../../harness.cpp && echo "SAME harness.cpp source" || echo "DIFF harness.cpp source"
