#!/usr/bin/env bash
# Build the double-buffered f16 candidates (NTMAX 2 and 3) with resource usage. CPU only.
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
ARCH=gfx1151
python3 mkf16.py cand_wide.hip mid_f16db.inc cand_f16db.hip
for NT in 2 3 4; do
  ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH -I "$KERNELS_DIR" -DWM_NTMAX=$NT \
    -Rpass-analysis=kernel-resource-usage cand_f16db.hip -o cand_f16db_nt${NT}_$ARCH.hsaco 2>&1 \
    | grep -E 'Function Name|VGPRs:|LDS Size|ScratchSize|Occupancy' | grep -B1 -A3 'wmma' | grep -vE '^--' | sed "s/^/NT$NT /"
done
echo build ok
