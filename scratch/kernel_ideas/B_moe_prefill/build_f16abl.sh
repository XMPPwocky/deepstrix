#!/usr/bin/env bash
# Regenerate cand_f16.hip from cand_wide.hip + mid_f16.inc and build the timing-only ablations.
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
ARCH=gfx1151
python3 -I "${MKF16:-./mkf16.py}" cand_wide.hip mid_f16.inc cand_f16.hip
B="$KFLAGS_V41 --genco --offload-arch=$ARCH -I $KERNELS_DIR"
../_infra/kcc.sh $B -DWM_NTMAX=2 cand_f16.hip -o cand_f16_$ARCH.hsaco
../_infra/kcc.sh $B -DWM_NTMAX=2 -DWMF_ABL_NOSTAGE cand_f16.hip -o cand_f16ns_$ARCH.hsaco
../_infra/kcc.sh $B -DWM_NTMAX=2 -DWMF_ABL_NOWLOAD cand_f16.hip -o cand_f16nw_$ARCH.hsaco
../_infra/kcc.sh $B -DWM_NTMAX=2 -DWMF_ABL_NOSTAGE -DWMF_ABL_NOWLOAD cand_f16.hip -o cand_f16nb_$ARCH.hsaco
echo ablations built
