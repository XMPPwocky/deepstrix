#!/usr/bin/env bash
# review build: baseline hsacos from the UNMODIFIED in-tree sources with $KFLAGS_V41 (gfx1201),
# byte-compared against the engineer's; the engineer's harness.cpp rebuilt verbatim; gpad_check.
set -eu
cd "$(dirname "$0")"
source ../../_infra/env.sh
ARCH=gfx1201
../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q8_0_matvec.hip" -o base_q8_$ARCH.hsaco &
../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/q8_0_grouped_matvec.hip" -o base_grp_$ARCH.hsaco &
wait
../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH "$KERNELS_DIR/swiglu.hip" -o base_swiglu_$ARCH.hsaco &
cp ../harness.cpp harness_copy.cpp
../../_infra/kcc.sh -O2 --offload-arch=$ARCH -I../../_infra harness_copy.cpp -o harness &
wait
../../_infra/kcc.sh -O2 --offload-arch=$ARCH -I../../_infra gpad_check.cpp -o gpad_check
echo built
for f in base_q8 base_grp base_swiglu; do
  if cmp -s "${f}_$ARCH.hsaco" "../${f}_$ARCH.hsaco"; then echo "SAME $f hsaco as engineer's"; else echo "DIFF $f hsaco vs engineer's"; fi
done
bash ../../_infra/isa.sh base_q8_$ARCH.hsaco $ARCH | grep -i quantize || true
