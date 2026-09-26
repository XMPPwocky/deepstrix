#!/usr/bin/env bash
# Build the harness (+ candidate code object if cand.hip exists). Baselines: build_base.sh.
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
ARCH=${1:-gfx1201}
if [ -f cand.hip ]; then ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand.hip -o cand_$ARCH.hsaco; fi
../_infra/kcc.sh -O2 --offload-arch=$ARCH harness.cpp -o harness_$ARCH
echo built
