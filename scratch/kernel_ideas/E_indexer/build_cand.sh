#!/usr/bin/env bash
# build_cand.sh <cand_name>...   — compile scratch candidate(s) cand_<name>.hip -> cand_<name>_gfx1201.hsaco
# (does not touch the baseline code objects or the harness).
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
for c in "$@"; do
    ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=gfx1201 -I"$KERNELS_DIR" "cand_$c.hip" -o "cand_${c}_gfx1201.hsaco.tmp"
    mv "cand_${c}_gfx1201.hsaco.tmp" "cand_${c}_gfx1201.hsaco"
    echo "built cand_${c}_gfx1201.hsaco"
done
