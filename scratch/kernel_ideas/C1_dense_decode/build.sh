#!/usr/bin/env bash
# Candidate code objects (same production flags) + the host harness. Baselines: build_base.sh.
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
ARCH=gfx1201
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand_bpack.hip -o cand_bpack_$ARCH.hsaco &
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand_shared.hip -o cand_shared_$ARCH.hsaco &
../_infra/kcc.sh -O2 --offload-arch=$ARCH -I../_infra harness.cpp -o harness &
wait
echo built
bash ../_infra/isa.sh cand_bpack_$ARCH.hsaco $ARCH
bash ../_infra/isa.sh cand_shared_$ARCH.hsaco $ARCH
