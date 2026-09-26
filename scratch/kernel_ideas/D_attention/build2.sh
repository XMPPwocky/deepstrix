#!/usr/bin/env bash
# Attempt-2 build: candidate code object cand2_attn + harness v2 (+ baselines if missing).
#   bash build2.sh          -> cand2 + harness2 (baselines rebuilt only if absent)
#   bash build2.sh all      -> everything (baselines from the UNMODIFIED in-tree sources, production flags)
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
ARCH=gfx1201
what=${1:-cand}
need_base=0
for k in attention_mixed attention_dec fp4_kv_quant f16_roundtrip kv_cache_append f16_matvec; do
    [ -f base_${k}_$ARCH.hsaco ] || need_base=1
done
if [ "$what" = all ] || [ $need_base = 1 ]; then
    for k in attention_mixed attention_dec fp4_kv_quant f16_roundtrip kv_cache_append f16_matvec; do
        ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o base_${k}_$ARCH.hsaco
    done
fi
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand2_attn.hip -o cand2_attn_$ARCH.hsaco
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand3_attn.hip -o cand3_attn_$ARCH.hsaco
[ -f cand4_attn.hip ] && ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand4_attn.hip -o cand4_attn_$ARCH.hsaco
[ -f cand5_attn.hip ] && ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand5_attn.hip -o cand5_attn_$ARCH.hsaco
[ -f cand6_attn.hip ] && ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand6_attn.hip -o cand6_attn_$ARCH.hsaco
[ -f cand7_attn.hip ] && ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand7_attn.hip -o cand7_attn_$ARCH.hsaco
../_infra/kcc.sh -O2 --offload-arch=$ARCH attn_harness2.cpp -o attn_harness2_$ARCH
echo build2 ok
