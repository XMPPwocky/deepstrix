#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
source ../../_infra/env.sh
ARCH=gfx1201
for k in q8_0_matvec_wmma q8_0_matvec q8_k_quantize; do
  ../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o ${k}_$ARCH.hsaco
done
../../_infra/kcc.sh $KFLAGS_V41 -DKERNELS_WMMA_HIP="\"$KERNELS_DIR/q8_0_matvec_wmma.hip\"" --genco --offload-arch=$ARCH cand_i8x.hip -o cand_i8x_$ARCH.hsaco
../../_infra/kcc.sh -O2 --offload-arch=$ARCH harness.cpp -o harness
bash ../../_infra/isa.sh cand_i8x_$ARCH.hsaco $ARCH 2>/dev/null | grep -E 'kernel|i8x_' | head -10 || true
