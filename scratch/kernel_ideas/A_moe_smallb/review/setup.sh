#!/usr/bin/env bash
# Review copy of the engineer's harness: identical sources, built in review/ so the engineer's files stay untouched.
set -eu
cd "$(dirname "$0")"
mkdir -p results
cp ../build.sh ../harness.cpp ../cand_kwide_c8.hip ../run_chain.sh ../run_parts.sh ../show.sh .
sed -i 's|\.\./_infra|../../_infra|g' build.sh run_chain.sh run_parts.sh show.sh
grep -n "_infra" build.sh run_chain.sh
sha256sum ../base_pair_gfx1151.hsaco ../cand_kwide_c8_gfx1151.hsaco
