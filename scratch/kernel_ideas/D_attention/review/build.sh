#!/usr/bin/env bash
# Reviewer build: independent copies of the baselines (unmodified in-tree .hip, production flags),
# the engineer's cand2_attn.hip, the engineer's harness (unmodified source) and the reviewer's
# extra-shape check harness, all into review/.
set -eu
cd "$(dirname "$0")"
source ../../_infra/env.sh
ARCH=gfx1201
for k in attention_mixed attention_dec; do
    ../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o base_${k}_$ARCH.hsaco
done
../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH ../cand2_attn.hip -o cand2_attn_$ARCH.hsaco
../../_infra/kcc.sh -O2 --offload-arch=$ARCH ../attn_harness2.cpp -o attn_harness2_$ARCH
[ -f check_extra.cpp ] && ../../_infra/kcc.sh -O2 --offload-arch=$ARCH check_extra.cpp -o check_extra_$ARCH
echo review build ok
md5sum base_attention_dec_$ARCH.hsaco base_attention_mixed_$ARCH.hsaco cand2_attn_$ARCH.hsaco
