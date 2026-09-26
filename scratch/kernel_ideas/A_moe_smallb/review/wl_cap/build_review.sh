#!/usr/bin/env bash
# Reviewer's independent build (claim A_moe_smallb/wl_cap): baseline code objects from the UNMODIFIED in-tree
# sources with the exact build.rs flags ($KFLAGS_V41), the engineer's cand_wl.hip (unchanged, compiled from
# its own dir), and the harness (unchanged harness.cpp). Everything lands in this dir; the engineer's files
# are not touched.
set -eu
cd "$(dirname "$0")"
source ../../../_infra/env.sh
ARCH=gfx1151
K=../../../_infra/kcc.sh
echo "KFLAGS_V41=$KFLAGS_V41"
$K $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/mxfp4_pair_matvec.hip" -o base_pair_$ARCH.hsaco &
$K $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/mxfp4_matvec.hip" -o base_down_$ARCH.hsaco &
wait
$K $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q2_k_accumulate_matvec_par.hip" -o base_q2k_$ARCH.hsaco &
$K $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q8_k_quantize.hip" -o base_q8k_$ARCH.hsaco &
wait
$K $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/moe_group_builder.hip" -o base_grp_$ARCH.hsaco &
$K $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/moe_work_items_builder.hip" -o base_wi_$ARCH.hsaco &
wait
$K $KFLAGS_V41 --genco --offload-arch=$ARCH -I"$KERNELS_DIR" ../../cand_wl.hip -o cand_wl_$ARCH.hsaco &
$K -O2 --offload-arch=$ARCH ../../harness.cpp -o harness_$ARCH &
wait
ls -la *.hsaco harness_$ARCH
cmp base_pair_$ARCH.hsaco ../../base_pair_$ARCH.hsaco && echo "base_pair identical to engineer's" || echo "base_pair DIFFERS from engineer's"
cmp base_down_$ARCH.hsaco ../../base_down_$ARCH.hsaco && echo "base_down identical to engineer's" || echo "base_down DIFFERS from engineer's"
cmp cand_wl_$ARCH.hsaco ../../cand_wl_$ARCH.hsaco && echo "cand_wl identical to engineer's" || echo "cand_wl DIFFERS from engineer's (bytes; compare the disassembly)"
echo "build ok"
