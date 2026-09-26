#!/usr/bin/env bash
# Build cand_r1plus.hip (r1 source + dummy instantiations) as cand_r1plus_gfx1201.hsaco.
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=gfx1201 cand_r1plus.hip -o cand_r1plus_gfx1201.hsaco 2>&1 | grep -i " error" || true
echo built_r1plus
