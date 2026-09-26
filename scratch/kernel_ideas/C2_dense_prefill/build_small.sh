#!/usr/bin/env bash
# Build cand_small.hip (few kernels) as cand_small_gfx1201.hsaco — module-composition test.
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=gfx1201 cand_small.hip -o cand_small_gfx1201.hsaco 2>&1 | grep -i "error" || true
echo built_small
