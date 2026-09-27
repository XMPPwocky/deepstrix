#!/usr/bin/env bash
# gfx1151 copy of the gpad_real harness + the code objects the iGPU check needs.
set -eu
cd "$(dirname "$0")"
source ../../_infra/env.sh
ARCH=gfx1151
for k in q8_0_matvec fp4_kv_quant indexer_qat rope_tail q8_k_quantize indexer_gather; do
  ../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o base_${k}_$ARCH.hsaco
done
../../_infra/kcc.sh -O2 --offload-arch=$ARCH gpad_real.cpp -o gpad_real_gfx1151
