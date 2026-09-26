#!/usr/bin/env bash
# Reviewer build for E_indexer/candidate_threshold_ilp, isolated in review/threshold/ (review/ is
# shared with other reviewers). Baseline = UNMODIFIED in-tree candidate_blocks.hip with
# $KFLAGS_V41 --genco; candidate = the engineer's ../../cand_cand.hip as-is; the engineer's
# harness.cpp compiled unchanged; plus the reviewer's harness and BLOCK=256 ILP probe.
set -eu
cd "$(dirname "$0")"
source ../../../_infra/env.sh
ARCH=gfx1201
for k in candidate_blocks indexer_score_wmma; do
    ../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o base_${k}_$ARCH.hsaco
done
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH -I"$KERNELS_DIR" ../../cand_cand.hip -o cand_cand_$ARCH.hsaco
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH -I"$KERNELS_DIR" cand_review.hip -o cand_review_$ARCH.hsaco
../../../_infra/kcc.sh -O2 --offload-arch=$ARCH ../../harness.cpp -o eng_harness_$ARCH
../../../_infra/kcc.sh -O2 --offload-arch=$ARCH review_harness.cpp -o review_harness_$ARCH
echo "build ok"
ls -la *.hsaco eng_harness_$ARCH review_harness_$ARCH
