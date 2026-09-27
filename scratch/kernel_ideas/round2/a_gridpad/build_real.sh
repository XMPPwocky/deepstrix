#!/usr/bin/env bash
# Baseline code objects from the UNMODIFIED in-tree sources (production flags) + the gpad_real harness.
set -eu
cd "$(dirname "$0")"
source ../../_infra/env.sh
ARCH=gfx1201
for k in indexer_gather f16_matvec fp4_kv_quant rope_tail indexer_qat q8_k_quantize rms_norm kv_cache_append ${EXTRA_K:-}; do
  ../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o base_${k}_$ARCH.hsaco
done
../../_infra/kcc.sh -O2 --offload-arch=$ARCH gpad_real.cpp -o gpad_real
