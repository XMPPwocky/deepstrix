#!/usr/bin/env bash
# Build the B harness candidate code object from the IN-TREE kernels/mxfp4_moe_wmma.hip (the
# integrated copy of cand_wmma.hip), so `KB_CAND=intree/intree_wmma ./harness cmp ...` checks the
# in-tree kernels against the production kwide / kwide2 baselines (base_*.hsaco in ..).
set -eu
cd "$(dirname "$0")/.."
source ../_infra/env.sh
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=gfx1151 -I "$KERNELS_DIR" "$KERNELS_DIR/mxfp4_moe_wmma.hip" -o intree/intree_wmma_gfx1151.hsaco
echo built intree/intree_wmma_gfx1151.hsaco
