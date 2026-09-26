#!/usr/bin/env bash
# Review build: baselines from the UNMODIFIED in-tree sources + the engineer's candidates (copied,
# with the review additions appended) + harness copy, all with $KFLAGS_V41, into review/.
# Nothing in the engineer's dir is touched.
set -eu
cd "$(dirname "$0")"
source ../../../_infra/env.sh
ARCH=gfx1201
K=../../../_infra/kcc.sh
$K $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q8_0_matvec.hip" -o base_q8_$ARCH.hsaco &
$K $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q8_0_grouped_matvec.hip" -o base_grp_$ARCH.hsaco &
wait
$K $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/swiglu.hip" -o base_swiglu_$ARCH.hsaco &
$K $KFLAGS_V41 --genco --offload-arch=$ARCH cand_bpack.hip -o cand_bpack_$ARCH.hsaco &
wait
$K $KFLAGS_V41 --genco --offload-arch=$ARCH cand_shared.hip -o cand_shared_$ARCH.hsaco &
$K -O2 --offload-arch=$ARCH -I../../../_infra harness.cpp -o harness &
wait
echo built
bash ../../../_infra/isa.sh cand_shared_$ARCH.hsaco $ARCH > isa_cand_shared.txt || true
bash ../../../_infra/isa.sh base_q8_$ARCH.hsaco $ARCH > isa_base_q8.txt || true
grep -E "tB[1-8]_r1|swiglu_tB[1-8]\b" isa_cand_shared.txt || true
grep -E "bpack_warp8|quantize" isa_base_q8.txt || true
