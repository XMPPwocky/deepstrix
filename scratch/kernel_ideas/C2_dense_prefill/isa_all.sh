#!/usr/bin/env bash
# Print VGPR/SGPR/LDS/scratch per kernel for every baseline (and candidate) code object here.
set -eu
cd "$(dirname "$0")"
for f in "$@"; do
  echo "=== $f"
  bash ../_infra/isa.sh "$f" gfx1201
done
