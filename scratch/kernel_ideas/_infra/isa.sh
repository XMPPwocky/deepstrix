#!/usr/bin/env bash
# isa.sh <file.hsaco> <gfx1151|gfx1201> [--dis [symbol-substring]]
# Unbundles a `hipcc --genco` code object and prints per-kernel resources (VGPRs, SGPRs, LDS,
# scratch/spills, wavefront size). --dis also writes the disassembly next to the .hsaco
# (<file>.s) and prints the part for kernels matching the substring.
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
IN=$1; ARCH=$2; shift 2
ELF="${IN%.hsaco}.elf"
clang-offload-bundler --type=o --unbundle --input="$IN" --output="$ELF" \
    --targets=hipv4-amdgcn-amd-amdhsa--"$ARCH" 2>/dev/null || cp "$IN" "$ELF"
llvm-readelf --notes "$ELF" | python3 "$KI_ROOT/_infra/hsaco_notes.py"
if [ "${1:-}" = "--dis" ]; then
    llvm-objdump -d --no-show-raw-insn --mcpu="$ARCH" "$ELF" > "${IN%.hsaco}.s"
    echo "disassembly: ${IN%.hsaco}.s ($(wc -l < "${IN%.hsaco}.s") lines)"
    if [ -n "${2:-}" ]; then
        awk -v pat="$2" '/^[0-9a-f]+ <.*>:$/ {p = index($0, pat) > 0} p' "${IN%.hsaco}.s" | head -400
    fi
fi
