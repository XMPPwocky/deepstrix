#!/usr/bin/env bash
# Baseline code objects: UNMODIFIED in-tree sources, production flags ($KFLAGS_V41 --genco), gfx1201.
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
ARCH=${1:-gfx1201}
for f in q8_0_matvec_wmma q8_0_matvec q8_0_grouped_matvec q8_k_quantize; do
  ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$f.hip" -o base_${f}_$ARCH.hsaco
done
echo built
