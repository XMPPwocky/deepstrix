#!/usr/bin/env bash
# Print VGPR/SGPR/LDS/spills for every family-E code object (and candidates if present).
set -u
cd "$(dirname "$0")"
for f in base_*_gfx1201.hsaco cand_*_gfx1201.hsaco; do
    [ -e "$f" ] || continue
    echo "=== $f"
    bash ../_infra/isa.sh "$f" gfx1201 2>&1 | grep -v '^$'
done
