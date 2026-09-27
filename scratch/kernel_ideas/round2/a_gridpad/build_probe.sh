#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
source ../../_infra/env.sh
ARCHS=${ARCHS:-gfx1201}
for ARCH in $ARCHS; do
  ../../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=$ARCH probe.hip -o probe_$ARCH.hsaco
  ../../_infra/kcc.sh -O2 --offload-arch=$ARCH probe_harness.cpp -o probe_harness_$ARCH
  ../../_infra/kcc.sh -O2 --offload-arch=$ARCH probe2.cpp -o probe2_$ARCH
done
