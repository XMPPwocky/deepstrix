#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
source ../../_infra/env.sh
ARCH=gfx1201
for k in rms_norm rope_tail fp4_kv_quant q8_0_matvec q8_k_quantize hc_post vec_add; do
  ../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o ${k}_$ARCH.hsaco
done
../../_infra/kcc.sh $KFLAGS_V41 -I"$KERNELS_DIR" --genco --offload-arch=$ARCH cand_fuse.hip -o cand_fuse_$ARCH.hsaco
../../_infra/kcc.sh -O2 --offload-arch=$ARCH chain_harness.cpp -o chain_harness
bash ../../_infra/isa.sh cand_fuse_$ARCH.hsaco $ARCH 2>/dev/null | head -8 || true
