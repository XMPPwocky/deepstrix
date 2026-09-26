#!/usr/bin/env bash
# Reviewer build: production code objects from the UNMODIFIED in-tree sources with $KFLAGS_V41,
# byte-compared against the engineer's hsacos, plus the review harness. CPU only.
set -eu
cd "$(dirname "$0")"
source ../../_infra/env.sh
ARCH=gfx1201
for k in mhc_fast hc_weighted_sum rms_norm rms_norm_no_weight f16_gemm_wmma; do
    ../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o base_${k}_$ARCH.hsaco
    if cmp -s base_${k}_$ARCH.hsaco ../base_${k}_$ARCH.hsaco; then
        echo "hsaco $k: IDENTICAL to engineer's"
    else
        echo "hsaco $k: DIFFERS from engineer's (bytes: $(stat -c %s base_${k}_$ARCH.hsaco) vs $(stat -c %s ../base_${k}_$ARCH.hsaco))"
    fi
done
../../_infra/kcc.sh -O2 --offload-arch=$ARCH harness_review.cpp -o harness_review_$ARCH
echo "review build ok"
