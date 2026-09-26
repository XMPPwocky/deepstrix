#!/usr/bin/env bash
# build_intree.sh -- integration check (2026-09-26): rebuild the harness' code objects from the
# IN-TREE kernels after integration, under intree/, so `harness` (C1_DIR=intree) compares the
# in-tree tB / quantize_wave / fused-shared symbols against the in-tree runtime kernels:
#   base_q8     = crates/.../q8_0_matvec.hip          (runtime bpack + quantize + tB twins + wave)
#   base_grp    = crates/.../q8_0_grouped_matvec.hip  (runtime grouped bpack + grouped tB twins)
#   base_swiglu = crates/.../swiglu.hip
#   cand_bpack  = combo TU including BOTH in-tree GEMV files (the harness looks up gemv tB,
#                 grouped tB and q8_0_quantize_f32_wave in this one module)
#   cand_shared = crates/.../shared_expert_fused.hip
set -eu
cd "$(dirname "$0")"
source ../_infra/env.sh
ARCH=gfx1201
mkdir -p intree
cat > intree/cand_combo.hip <<EOF
#include "$KERNELS_DIR/q8_0_matvec.hip"
#include "$KERNELS_DIR/q8_0_grouped_matvec.hip"
EOF
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q8_0_matvec.hip" -o intree/base_q8_$ARCH.hsaco
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q8_0_grouped_matvec.hip" -o intree/base_grp_$ARCH.hsaco
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/swiglu.hip" -o intree/base_swiglu_$ARCH.hsaco
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH intree/cand_combo.hip -o intree/cand_bpack_$ARCH.hsaco
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/shared_expert_fused.hip" -o intree/cand_shared_$ARCH.hsaco
echo built
bash ../_infra/isa.sh intree/cand_bpack_$ARCH.hsaco $ARCH | grep -E "tB|wave|^kernel|VGPR|spill" | head -60 || true
bash ../_infra/isa.sh intree/cand_shared_$ARCH.hsaco $ARCH | head -40 || true
