#!/usr/bin/env bash
# Reviewer's independent rebuild of the F_mhc_glue rms_fast claim.
# Copies the engineer's sources (unmodified) into review/repro/ and rebuilds the
# baseline hsacos from the in-tree kernels with $KFLAGS_V41 plus the candidate and harness.
set -eu
R=/home/claude-code/deepstrix/.claude/worktrees/kernel-ideas-2026-09-26/scratch/kernel_ideas/F_mhc_glue
D=$R/review/repro
mkdir -p "$D/results" "$D/jobs"
cp "$R/harness.cpp" "$R/cand_rms.hip" "$R/isa_all.sh" "$D/"
cp "$R/jobs/winners.sh" "$D/jobs/winners_orig.sh"
cd "$D"
source ../../../_infra/env.sh
ARCH=gfx1201
echo "KFLAGS_V41=$KFLAGS_V41"
echo "KERNELS_DIR=$KERNELS_DIR"
for k in mhc_fast f16_gemm_wmma rms_norm_no_weight hc_sinkhorn_par hc_weighted_sum rms_norm \
         hc_post f16_matvec router_topk_par vec_add mhc_arena f16_matvec_narrow; do
    ../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o base_${k}_$ARCH.hsaco
done
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand_rms.hip -o cand_rms_$ARCH.hsaco
../../../_infra/kcc.sh -O2 --offload-arch=$ARCH harness.cpp -o harness_$ARCH
echo "build ok"
echo "--- compare my baseline/candidate code objects with the engineer's ---"
cmp base_rms_norm_$ARCH.hsaco "$R/base_rms_norm_$ARCH.hsaco" && echo "base_rms_norm hsaco IDENTICAL to engineer's"
cmp cand_rms_$ARCH.hsaco "$R/cand_rms_$ARCH.hsaco" && echo "cand_rms hsaco IDENTICAL to engineer's"
cmp cand_rms.hip "$R/cand_rms.hip" && echo "cand_rms.hip unchanged"
