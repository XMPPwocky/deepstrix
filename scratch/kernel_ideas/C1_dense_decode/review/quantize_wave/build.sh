#!/usr/bin/env bash
# Reviewer build for claim C1_dense_decode/quantize_wave.
# Baselines: UNMODIFIED in-tree sources, $KFLAGS_V41, gfx1201 (byte-compared to the engineer's hsacos).
# Candidate: verbatim copy of the engineer's cand_bpack.hip (sha256 checked against the original).
# Harness: verbatim copy of the engineer's harness.cpp. odd.cpp: reviewer's correctness harness.
set -eu
cd "$(dirname "$0")"
source ../../../_infra/env.sh
ARCH=gfx1201
sha256sum -c <<EOF
39215b683e26955eb2d5a8a779b5855a09a5525c7099e6a52c8fa09302feefc0  cand_bpack.hip
ca800b48b8e85a21d7cdadb7489b0883ea0229333a70e4a846a49c7317f9fc2a  harness.cpp
EOF
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q8_0_matvec.hip" -o base_q8_$ARCH.hsaco &
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q8_0_grouped_matvec.hip" -o base_grp_$ARCH.hsaco &
wait
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/swiglu.hip" -o base_swiglu_$ARCH.hsaco &
../../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH cand_bpack.hip -o cand_bpack_$ARCH.hsaco &
wait
../../../_infra/kcc.sh -O2 --offload-arch=$ARCH -I../../../_infra harness.cpp -o harness &
../../../_infra/kcc.sh -O2 --offload-arch=$ARCH -I../../../_infra odd.cpp -o odd &
wait
echo built
for f in base_q8 base_grp base_swiglu cand_bpack; do
  if cmp -s "${f}_$ARCH.hsaco" "../../${f}_$ARCH.hsaco"; then echo "SAME ${f} hsaco as engineer's"; else echo "DIFF ${f} hsaco vs engineer's (bytes differ; may be build-id only)"; fi
done
bash ../../../_infra/isa.sh base_q8_$ARCH.hsaco $ARCH | grep -i -E "kernel|quantize" | head -20 || true
bash ../../../_infra/isa.sh cand_bpack_$ARCH.hsaco $ARCH | grep -i -E "kernel|quantize" | head -20 || true
