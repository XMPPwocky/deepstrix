#!/usr/bin/env bash
# Compile-only experiments: bash try_compile.sh SRC.hip OUTNAME "FLAGS"  -> prints ISA stats
set -u
cd "$(dirname "$0")"
source ../_infra/env.sh
SRC=$1; OUT=$2; FLAGS=${3:-}
../_infra/kcc.sh $KFLAGS_V41 --genco --offload-arch=gfx1151 -I "$KERNELS_DIR" $FLAGS "$SRC" -o "tmp_${OUT}.hsaco" 2>&1 | grep -i "error" | head -5
echo "== $OUT ($FLAGS)"
bash ../_infra/isa.sh "tmp_${OUT}.hsaco" gfx1151 2>&1 | grep -i "_wmma"
