#!/usr/bin/env bash
# (b) crossover builds: in-tree code objects for the F kernels + xover_f; the D harness copy with
# bmax 16 (xattn); the E harness copy with a configurable scores stride (xe).
set -eu
cd "$(dirname "$0")"
source ../../_infra/env.sh
ARCH=gfx1201
for k in rms_norm f16_matvec router_topk_par; do
  ../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o ${k}_$ARCH.hsaco
done
../../_infra/kcc.sh -O2 --offload-arch=$ARCH xover_f.cpp -o xover_f
../../_infra/kcc.sh -O2 --offload-arch=$ARCH xattn.cpp -o xattn
../../_infra/kcc.sh -O2 --offload-arch=$ARCH xe.cpp -o xe
../../_infra/kcc.sh -O2 --offload-arch=$ARCH xc2.cpp -o xc2
echo built
