#!/usr/bin/env bash
# G0 (docs/v41/EMBED_PHASE_DESIGN.md §19.11): the qe_attn_g4 code-object gate.
# COMPILE-ONLY -- no GPU, nothing opens /dev/kfd. Run inside the dev shell:
#
#   CARGO_TARGET_DIR=target-v41 nice -n 19 nix develop -c scripts/qe_attn_g4_g0.sh [OUT]
#
# Compiles the three attention modules (qwen3_attn_g4 / g4n8 with -mcumode, g4wgp
# without) and qwen3_embed exactly as build.rs does for gfx1201 (-O3 --genco, the
# v41 model flags, $DEEPSTRIX_KERNEL_CFLAGS), unbundles each code object, and runs
# scripts/qe_attn_g4_g0.py:
#   G0a  per module: VGPRs <= 192, scratch 0, LDS 49152 / 32768 / 49152, occupancy
#        8 waves/SIMD, workgroup_processor_mode 0 / 0 / 1; qe_v_frag: scratch 0.
#   G0b  per attention loop: the steady-path instruction ledger (WMMA, ds_load_b128,
#        ds_store_b128, global_load_b128, v_exp_f32, barrier signal/wait, the signal's
#        placement, PV ds-wait pipelining, the non-trans VALU budget).
# Exit status non-zero on any violation. Outputs (.hsaco, .co, .s, .log, notes) in OUT
# (default $CARGO_TARGET_DIR/qe_attn_g4_g0, else target-v41/qe_attn_g4_g0).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
K="$ROOT/crates/v4flash-kernels/kernels"
OUT=${1:-${CARGO_TARGET_DIR:-$ROOT/target-v41}/qe_attn_g4_g0}
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
mkdir -p "$OUT"
ARCH=gfx1201
# shellcheck disable=SC2206
EXTRA=(${DEEPSTRIX_KERNEL_CFLAGS:-})
MODEL=(-DDEEPSTRIX_V41=1 -DMHC_N_EMBD=5120 -DMHC_HC_DIM=20480 -DROUTER_MAX_EXPERTS=512)

build() {   # stem, extra flags...
    local stem=$1; shift
    nice -n 19 hipcc -O3 --genco --offload-arch=$ARCH "${MODEL[@]}" "${EXTRA[@]}" "$@" \
        -Rpass-analysis=kernel-resource-usage "$K/$stem.hip" -o "$OUT/$stem.hsaco" > "$OUT/$stem.log" 2>&1 \
        || { cat "$OUT/$stem.log"; echo "G0: hipcc failed for $stem" >&2; exit 1; }
    clang-offload-bundler --type=o --unbundle --targets=hipv4-amdgcn-amd-amdhsa--$ARCH \
        --input="$OUT/$stem.hsaco" --output="$OUT/$stem.co"
    llvm-objdump -d --mcpu=$ARCH "$OUT/$stem.co" > "$OUT/$stem.s"
    llvm-readelf --notes "$OUT/$stem.co" > "$OUT/$stem.notes"
}

build qwen3_attn_g4 -mcumode
build qwen3_attn_g4n8 -mcumode
build qwen3_attn_g4wgp
build qwen3_embed

# The gate is about the code objects cargo embeds: when a cargo build exists, its
# instruction streams must equal these (same flags, same source).
TD=${CARGO_TARGET_DIR:-$ROOT/target-v41}
case "$TD" in /*) ;; *) TD="$ROOT/$TD" ;; esac
CARGO_FAIL=0
for stem in qwen3_attn_g4 qwen3_attn_g4n8 qwen3_attn_g4wgp qwen3_embed; do
    # shellcheck disable=SC2012
    src=$(ls -t "$TD"/release/build/v4flash-kernels-*/out/"${stem}_$ARCH.hsaco" 2>/dev/null | head -1 || true)
    if [ -z "$src" ]; then
        echo "G0: no cargo-built $stem (build first to compare)"
        continue
    fi
    clang-offload-bundler --type=o --unbundle --targets=hipv4-amdgcn-amd-amdhsa--$ARCH --input="$src" --output="$OUT/$stem.cargo.co"
    llvm-objdump -d --no-show-raw-insn --mcpu=$ARCH "$OUT/$stem.cargo.co" | sed 's@ *//.*@@' | grep -E '^\s' > "$OUT/$stem.cargo.ins"
    llvm-objdump -d --no-show-raw-insn --mcpu=$ARCH "$OUT/$stem.co" | sed 's@ *//.*@@' | grep -E '^\s' > "$OUT/$stem.g0.ins"
    if cmp -s "$OUT/$stem.cargo.ins" "$OUT/$stem.g0.ins"; then
        echo "G0: cargo-built $stem: instruction stream identical ($src)"
    else
        echo "G0: FAIL cargo-built $stem differs from the G0 build ($src): rebuild, or check DEEPSTRIX_KERNEL_CFLAGS"
        CARGO_FAIL=1
    fi
done
python3 "$ROOT/scripts/qe_attn_g4_g0.py" "$OUT" || exit 1
exit $CARGO_FAIL
