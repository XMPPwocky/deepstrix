#!/usr/bin/env bash
# Reviewer build for the dn2_member_outer claim: baseline down code object from the UNMODIFIED in-tree
# mxfp4_matvec.hip with $KFLAGS_V41, the engineer's cand_dn2_r2.hip (unchanged, from its own dir) and a
# rebuilt harness; all into review/dn2. cmp against the engineer's artifacts.
set -eu
cd "$(dirname "$0")"
source ../../../_infra/env.sh
ARCH=gfx1151
K=../../../_infra/kcc.sh
echo "KFLAGS_V41=$KFLAGS_V41"
$K $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/mxfp4_matvec.hip" -o chk_base_down_$ARCH.hsaco
$K $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/mxfp4_pair_matvec.hip" -o chk_base_pair_$ARCH.hsaco
$K $KFLAGS_V41 --genco --offload-arch=$ARCH -I"$KERNELS_DIR" ../../cand_dn2_r2.hip -o cand_dn2_r2_$ARCH.hsaco
$K -O2 --offload-arch=$ARCH ../../harness.cpp -o chk_harness_$ARCH
cmp chk_base_down_$ARCH.hsaco ../../base_down_$ARCH.hsaco && echo "base_down identical to engineer's" || echo "base_down DIFFERS from engineer's"
cmp chk_base_pair_$ARCH.hsaco ../../base_pair_$ARCH.hsaco && echo "base_pair identical to engineer's" || echo "base_pair DIFFERS from engineer's"
cmp cand_dn2_r2_$ARCH.hsaco ../../cand_dn2_r2_$ARCH.hsaco && echo "cand_dn2_r2 identical to engineer's" || echo "cand_dn2_r2 DIFFERS from engineer's"
echo "== ISA resources"
bash ../../../_infra/isa.sh cand_dn2_r2_$ARCH.hsaco $ARCH
bash ../../../_infra/isa.sh chk_base_down_$ARCH.hsaco $ARCH
echo "build ok"
