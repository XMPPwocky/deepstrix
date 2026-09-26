#!/usr/bin/env bash
# Build: production baselines (UNMODIFIED in-tree sources, production flags), candidates, harnesses.
#   bash build.sh            -> everything for gfx1201
#   bash build.sh cand       -> only the candidate code object + attn harness
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
ARCH=gfx1201
what=${1:-all}
if [ "$what" = all ]; then
    ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/attention_mixed.hip" -o base_attention_mixed_$ARCH.hsaco
    ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/attention_dec.hip" -o base_attention_dec_$ARCH.hsaco
    ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/fp4_kv_quant.hip" -o base_fp4_kv_quant_$ARCH.hsaco
    ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/f16_roundtrip.hip" -o base_f16_roundtrip_$ARCH.hsaco
    ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/kv_cache_append.hip" -o base_kv_cache_append_$ARCH.hsaco
    ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/f16_matvec.hip" -o base_f16_matvec_$ARCH.hsaco
fi
if [ -f cand_attn.hip ]; then
    ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand_attn.hip -o cand_attn_$ARCH.hsaco
fi
if [ -f cand_misc.hip ]; then
    ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand_misc.hip -o cand_misc_$ARCH.hsaco
fi
../_infra/kcc.sh -O2 --offload-arch=$ARCH attn_harness.cpp -o attn_harness_$ARCH
if [ -f misc_harness.cpp ]; then
    ../_infra/kcc.sh -O2 --offload-arch=$ARCH misc_harness.cpp -o misc_harness_$ARCH
fi
echo build ok
