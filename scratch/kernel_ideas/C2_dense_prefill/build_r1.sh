#!/usr/bin/env bash
# Build the first attempt's cand.hip (recovered from commit aab5a67) as cand_r1_gfx1201.hsaco.
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=gfx1201 cand_r1.hip -o cand_r1_gfx1201.hsaco 2>&1 | grep -v "warning\|^ *[0-9]* |\|^ *|\|^ *\^" || true
echo built_r1
