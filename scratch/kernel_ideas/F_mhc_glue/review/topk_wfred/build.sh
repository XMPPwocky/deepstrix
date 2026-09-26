#!/usr/bin/env bash
# Reviewer build for F_mhc_glue/topk_wfred: baseline hsaco from the UNMODIFIED in-tree
# router_topk_par.hip with $KFLAGS_V41, the candidate from the engineer's cand_topk2.hip (read-only),
# and the reviewer harness. Everything lands in this directory.
set -eu
cd "$(dirname "$0")"
source ../../../_infra/env.sh
ARCH=gfx1201
echo "KFLAGS_V41=$KFLAGS_V41"
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/router_topk_par.hip" -o base_router_topk_par_$ARCH.hsaco
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH ../../cand_topk2.hip -o cand_topk2_$ARCH.hsaco
../../../_infra/kcc.sh -O2 --offload-arch=$ARCH rev_topk.cpp -o rev_topk
md5sum base_router_topk_par_$ARCH.hsaco ../../base_router_topk_par_$ARCH.hsaco cand_topk2_$ARCH.hsaco ../../cand_topk2_$ARCH.hsaco
echo "build ok"
