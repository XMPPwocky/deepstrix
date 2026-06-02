#!/usr/bin/env bash
# Bidirectional bisection over per-layer residuals to find the layer
# where deepstrix first diverges from ds4-CPU on the long_memory_archive
# case.
#
# Naming convention (consistent across ds4 + deepstrix dumps):
#   layer_KK_residual.bin = INPUT to layer K (= output of layer K-1)
#   K=0 is the embedded token vector (identical on both sides).
#   ds4 dumps K in 00..42 (no K=43 since no layer 43).
#   deepstrix also dumps K=43 (input to lm_head); ignored for cross-diff.
#
# DEEPSTRIX_SUBSTITUTE_RESIDUAL=K:PATH semantics:
#   "after our layer K, overwrite the residual with PATH"
#   == "feed PATH as the input to our layer K+1"
#   So to substitute ds4's input to layer L into our pass, use K=L-1
#   and PATH=/tmp/ds4-dump/layer_LL_residual.bin.
#
# DS4_INJECT_LAYER_INPUT_RESIDUAL=L:PATH semantics:
#   "when ds4 enters layer L, overwrite its input residual with PATH"
#   So to feed our input to ds4's layer L, use L=L and PATH=our file L.
#
# Prereqs:
#   - external/ds4/dump_residual run → /tmp/ds4-dump/layer_NN_residual.bin
#   - deepstrix-vector-test run with DEEPSTRIX_DUMP_RESIDUAL_DIR=/tmp/deepstrix-dump
#     for the SAME case → /tmp/deepstrix-dump/layer_NN_residual.bin
#   - both binaries rebuilt with the hooks
#
# Usage:
#   scripts/bisect_layer_residual.sh                # default sweep
#   scripts/bisect_layer_residual.sh 0 10 21 32 42  # custom layer set
#
# Each deepstrix run takes ~1 min, each ds4 run takes ~20 min. So the
# us-side sweep is fast; the ds4-side sweep is slow — we only run
# ds4-side for the LAYER OF INTEREST identified by the us-side sweep.

set -euo pipefail

cd "$(dirname "$0")/.."

GGUF=${GGUF:-/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}
VEC=${VEC:-external/ds4/tests/test-vectors/official.vec}
DS4_DUMP_DIR=${DS4_DUMP_DIR:-/tmp/ds4-dump}
DEEPSTRIX_DUMP_DIR=${DEEPSTRIX_DUMP_DIR:-/tmp/deepstrix-dump}
CASE=${CASE:-long_memory_archive}
BIN=${BIN:-./target/release/deepstrix-vector-test}

if [ ! -x "$BIN" ]; then
    echo "build $BIN first" >&2; exit 1
fi
[ -s "$DS4_DUMP_DIR/layer_00_residual.bin" ] || { echo "missing ds4 dumps in $DS4_DUMP_DIR" >&2; exit 1; }

# Layers L in [1..42] to feed as INPUT to our layer L.
# We use after_layer = L-1 in DEEPSTRIX_SUBSTITUTE_RESIDUAL.
# (Substituting "input to layer 0" is a no-op — that's the embedded
# token, identical by definition.)
if [ "$#" -gt 0 ]; then
    LAYERS=("$@")
else
    LAYERS=(1 5 10 15 21 28 35 42)
fi

# Baseline #1: no substitution, --substitute-eval path. Should match
# the original argmax ("Based") if our forward_token + prefill paths
# agree. If they don't agree at step 0, we have a separate bug.
echo "===== baseline (substitute_eval, NO substitution) ====="
unset DEEPSTRIX_SUBSTITUTE_RESIDUAL
nix develop --command "$BIN" \
    --gguf "$GGUF" --vec "$VEC" --case "$CASE" --substitute-eval 2>&1 \
    | grep -E "step  0|TOTAL" | head -3
echo

for L in "${LAYERS[@]}"; do
    LL=$(printf "%02d" "$L")
    AFTER=$((L - 1))
    FILE="$DS4_DUMP_DIR/layer_${LL}_residual.bin"
    if [ ! -s "$FILE" ]; then
        echo "===== layer $LL: SKIP (missing $FILE) ====="
        continue
    fi
    echo "===== substitute ds4's input-to-layer-$LL (== after our layer $AFTER) ====="
    DEEPSTRIX_SUBSTITUTE_RESIDUAL="${AFTER}:${FILE}" nix develop --command "$BIN" \
        --gguf "$GGUF" --vec "$VEC" --case "$CASE" --substitute-eval 2>&1 \
        | grep -E "step  0|substituted residual" | head -3
    echo
done

echo "===== ds4 → us sweep complete ====="
cat <<'EOF'
Interpretation:
  - If substituting at LAYER L still gives "Based" → our layers L..42 + head
    are CORRUPTING what should be a correct trajectory. Bug is downstream.
  - If substituting at LAYER L gives "Component" → our layers L..42 + head
    are CORRECT given a good input. Bug is upstream (in our 0..L-1).
  - The boundary (last L that gives "Based" → first L that gives "Component")
    pinpoints the divergence.

Then for the symmetric us → ds4 direction:
  DS4_INJECT_LAYER_INPUT_RESIDUAL=L:/tmp/deepstrix-dump/layer_LL_residual.bin \
    ./external/ds4/dump_residual <model> <prompt> /tmp/ds4-inject-out 16384
  Argmax at the end says what ds4 produces when it sees our intermediate.
  ~20 min per run. Only worth running for the suspect layer L.
EOF
