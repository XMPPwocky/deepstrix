#!/usr/bin/env bash
# build_f16nt.sh NT : cand_f16nt<NT>_gfx1151.hsaco (WM_NTMAX=NT, so chunk up to 16*NT members) + rebuild the harness
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
ARCH=gfx1151
NT=$1
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH -I "$KERNELS_DIR" -DWM_NTMAX=$NT cand_f16.hip -o cand_f16nt${NT}_$ARCH.hsaco
rm -f harness
../_infra/kcc.sh -O2 --offload-arch=$ARCH harness.cpp -o harness
echo built nt=$NT
