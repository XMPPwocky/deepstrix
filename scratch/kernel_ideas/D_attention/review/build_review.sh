#!/usr/bin/env bash
# Reviewer build: baselines rebuilt from the UNMODIFIED in-tree sources with $KFLAGS_V41 into THIS dir,
# the engineer's harness compiled from his source (unmodified) into this dir, plus the reviewer's own
# correctness harness. Nothing in the engineer's dir is touched.
set -eu
cd "$(dirname "$0")"
source ../../_infra/env.sh
ARCH=gfx1201
for k in attention_mixed attention_dec fp4_kv_quant f16_roundtrip kv_cache_append f16_matvec; do
    ../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o base_${k}_$ARCH.hsaco
done
../../_infra/kcc.sh -O2 --offload-arch=$ARCH ../attn_harness2.cpp -o attn_harness2_$ARCH
../../_infra/kcc.sh -O2 --offload-arch=$ARCH review_f16rt.cpp -o review_f16rt_$ARCH
sha256sum base_*_$ARCH.hsaco ../base_*_$ARCH.hsaco
echo build_review ok
