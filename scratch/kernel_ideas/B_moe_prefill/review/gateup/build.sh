#!/usr/bin/env bash
# review/gateup (reviewer of B_moe_prefill/wmma_i8_gateup): baseline hsacos from the UNMODIFIED
# in-tree sources with $KFLAGS_V41, the candidate from the engineer's ../../cand_wmma.hip
# (read-only), and harness_gu = the engineer's harness.cpp + an env override of the E8M0
# scale range (KB_SC_LO/KB_SC_SPAN). Everything lands in review/gateup/.
set -eu
cd "$(dirname "$0")"
source ../../../_infra/env.sh
ARCH=gfx1151
for k in mxfp4_pair_matvec mxfp4_matvec q8_k_quantize q2_k_accumulate_matvec_par moe_group_builder moe_work_items_builder; do
    ../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o base_${k}_$ARCH.hsaco
done
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH -I "$KERNELS_DIR" ../../cand_wmma.hip -o cand_wmma_$ARCH.hsaco
../../../_infra/kcc.sh -O2 --offload-arch=$ARCH harness_gu.cpp -o harness_gu
md5sum base_*.hsaco cand_wmma_$ARCH.hsaco ../../base_*.hsaco ../../cand_wmma_$ARCH.hsaco
echo build ok
