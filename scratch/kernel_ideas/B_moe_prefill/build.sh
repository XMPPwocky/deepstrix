#!/usr/bin/env bash
# B_moe_prefill: production baseline code objects (UNMODIFIED in-tree sources, production
# flags) + candidate code objects + the host harness. gfx1151 only (iGPU family).
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
ARCH=gfx1151
for k in mxfp4_pair_matvec mxfp4_matvec q8_k_quantize q2_k_accumulate_matvec_par moe_group_builder moe_work_items_builder; do
    ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o base_${k}_$ARCH.hsaco
done
for c in cand_wmma cand_ablate cand_kw_pf cand_wmma2 cand_wmma3 cand_wmma4; do
    if [ -f $c.hip ]; then
        ../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH -I "$KERNELS_DIR" $c.hip -o ${c}_$ARCH.hsaco
    fi
done
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH -I "$KERNELS_DIR" -DWM_PROBE_UPPER=1 cand_wmma.hip -o cand_wmma_probe_$ARCH.hsaco
../_infra/kcc.sh -O2 --offload-arch=$ARCH harness.cpp -o harness
echo build ok
