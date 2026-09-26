#!/usr/bin/env bash
# rebuild gpad_check only (GC_MIX / GC_ROUNDS added)
set -eu
cd "$(dirname "$0")"
source ../../_infra/env.sh
../../_infra/kcc.sh -O2 --offload-arch=gfx1201 -I../../_infra gpad_check.cpp -o gpad_check
echo built gpad_check
