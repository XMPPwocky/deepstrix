#!/usr/bin/env bash
# Reviewer rebuild: baselines from the UNMODIFIED in-tree sources with production flags, cand3/cand4, harness.
set -eu
cd "$(dirname "$0")"
source ../../../_infra/env.sh
ARCH=gfx1201
KCC=../../../_infra/kcc.sh
echo "KFLAGS_V41=$KFLAGS_V41"
echo "KERNELS_DIR=$KERNELS_DIR"
for k in attention_mixed attention_dec; do
    $KCC $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/$k.hip" -o base_${k}_$ARCH.hsaco
done
$KCC $KFLAGS_V41 --genco --offload-arch=$ARCH cand3_attn.hip -o cand3_attn_$ARCH.hsaco
$KCC $KFLAGS_V41 --genco --offload-arch=$ARCH cand4_attn.hip -o cand4_attn_$ARCH.hsaco
$KCC -O2 --offload-arch=$ARCH attn_harness2.cpp -o attn_harness2_$ARCH
echo build ok
