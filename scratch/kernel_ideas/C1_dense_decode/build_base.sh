#!/usr/bin/env bash
# Baseline code objects: UNMODIFIED in-tree sources, production flags ($KFLAGS_V41), gfx1201.
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
ARCH=gfx1201
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q8_0_matvec.hip" -o base_q8_$ARCH.hsaco
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q8_0_grouped_matvec.hip" -o base_grp_$ARCH.hsaco
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/swiglu.hip" -o base_swiglu_$ARCH.hsaco
echo built
bash ../_infra/isa.sh base_q8_$ARCH.hsaco $ARCH
bash ../_infra/isa.sh base_grp_$ARCH.hsaco $ARCH
bash ../_infra/isa.sh base_swiglu_$ARCH.hsaco $ARCH
