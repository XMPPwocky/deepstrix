#!/usr/bin/env bash
# Reviewer build: same compile lines as the engineer's build.sh/build2.sh, outputs ONLY under review/.
# Baselines from the UNMODIFIED in-tree sources with $KFLAGS_V41; candidate from the engineer's
# cand_attn.hip (untouched); harness2 from the engineer's attn_harness2.cpp (untouched); plus the
# reviewer's extended harness (bmax=16, extra shapes).
set -eu
cd "$(dirname "$0")"
source ../../_infra/env.sh
ARCH=gfx1201
../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/attention_mixed.hip" -o base_attention_mixed_$ARCH.hsaco
../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/attention_dec.hip" -o base_attention_dec_$ARCH.hsaco
../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH ../cand_attn.hip -o cand_attn_$ARCH.hsaco
../../_infra/kcc.sh -O2 --offload-arch=$ARCH ../attn_harness2.cpp -o attn_harness2_$ARCH
../../_infra/kcc.sh -O2 --offload-arch=$ARCH attn_harness_rv.cpp -o attn_harness_rv_$ARCH
sha256sum base_attention_mixed_$ARCH.hsaco ../base_attention_mixed_$ARCH.hsaco base_attention_dec_$ARCH.hsaco ../base_attention_dec_$ARCH.hsaco cand_attn_$ARCH.hsaco ../cand_attn_$ARCH.hsaco
echo build_rv ok
