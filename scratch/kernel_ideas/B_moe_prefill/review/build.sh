#!/usr/bin/env bash
# Reviewer build (review/B_moe_prefill): baseline hsacos from the UNMODIFIED in-tree sources
# with $KFLAGS_V41, the candidate from ../cand_wmma.hip (engineer's file, read-only), the
# engineer's harness from ../harness.cpp, plus a reviewer harness with a tail/odd-shape mode.
# Everything lands in review/ so the engineer's files are untouched.
set -eu
cd "$(dirname "$0")"
source ../../_infra/env.sh
ARCH=gfx1151
for k in mxfp4_pair_matvec mxfp4_matvec q8_k_quantize q2_k_accumulate_matvec_par moe_group_builder moe_work_items_builder; do
    ../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o base_${k}_$ARCH.hsaco
done
../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH -I "$KERNELS_DIR" ../cand_wmma.hip -o cand_wmma_$ARCH.hsaco
../../_infra/kcc.sh -O2 --offload-arch=$ARCH -I ../../_infra ../harness.cpp -o harness
# reviewer harness = engineer's harness with main() renamed + an extra "tail" mode
sed 's/^int main(int argc, char\*\* argv) {/int orig_main(int argc, char** argv) {/' ../harness.cpp > harness_base.inc
../../_infra/kcc.sh -O2 --offload-arch=$ARCH -I ../../_infra -I .. harness_rev.cpp -o harness_rev
md5sum base_*.hsaco cand_wmma_$ARCH.hsaco ../base_*.hsaco ../cand_wmma_$ARCH.hsaco
echo build ok
