#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
source ../../_infra/env.sh
ARCH=gfx1201
../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/f16_matvec.hip" -o f16_matvec_$ARCH.hsaco
../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand_z16.hip -o cand_z16_$ARCH.hsaco
../../_infra/kcc.sh -O2 --offload-arch=$ARCH harness.cpp -o harness
bash ../../_infra/isa.sh cand_z16_$ARCH.hsaco $ARCH 2>/dev/null | grep -E "^kernel|VGPR|spill" | head -60 || true
