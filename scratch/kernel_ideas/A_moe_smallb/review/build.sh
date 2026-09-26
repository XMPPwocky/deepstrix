#!/usr/bin/env bash
# Reviewer build: baseline code objects from the UNMODIFIED in-tree sources (exact build.rs flags),
# the engineer's candidate .hip files compiled from THEIR directory (read-only), harness from their
# harness.cpp -- all outputs under review/. Usage: bash review/build.sh [cand ...]
set -eu
cd "$(dirname "$0")"
ENG=..
source ../../_infra/env.sh
ARCH=gfx1151
for k in mxfp4_pair_matvec:pair mxfp4_matvec:down q2_k_accumulate_matvec_par:q2k q8_k_quantize:q8k moe_group_builder:grp moe_work_items_builder:wi; do
    src=${k%%:*}; tag=${k##*:}
    ../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$src.hip" -o base_${tag}_$ARCH.hsaco
done
for c in "$@"; do
    ../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH -I"$KERNELS_DIR" -I"$ENG" "$ENG/cand_$c.hip" -o cand_${c}_$ARCH.hsaco
done
../../_infra/kcc.sh -O2 --offload-arch=$ARCH "$ENG/harness.cpp" -o harness_$ARCH
echo "review build ok: $*"
