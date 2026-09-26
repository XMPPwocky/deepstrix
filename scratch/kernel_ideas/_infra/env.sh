# Source me: puts hipcc / rocprofv3 / llvm tools on PATH without paying `nix develop` per call.
# Caches `nix print-dev-env` (minus the shellHook, which runs rocm-smi) under $KI_STATE.
KI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KI_WT="$(cd "$KI_ROOT/../.." && pwd)"
export KI_ROOT KI_WT
export KI_STATE="${KI_STATE:-/home/claude-code/.claude/jobs/6b27263e/tmp/ki_state}"
mkdir -p "$KI_STATE/locks" "$KI_STATE/nixtop"
if [ ! -s "$KI_STATE/devenv.sh" ]; then
    (cd "$KI_WT" && nix print-dev-env 2>/dev/null) \
        | sed -e '/^eval "\${shellHook:-}"$/d' \
              -e "s|^export NIX_BUILD_TOP=.*|export NIX_BUILD_TOP=$KI_STATE/nixtop|" \
        > "$KI_STATE/devenv.sh.tmp" && mv "$KI_STATE/devenv.sh.tmp" "$KI_STATE/devenv.sh"
fi
# shellcheck disable=SC1091
source "$KI_STATE/devenv.sh"
export ROCPROFV3=/nix/store/nz5q0f805d5cx9nzf3pv3qpqarrrx4ag-rocprofiler-sdk-7.2.3/bin/rocprofv3
export ATT_DECODER_DIR=/nix/store/qzy5bk596ljy2nlj9ig4pynf8qj0mprm-rocprof-trace-decoder-0.1.7/lib
# Exactly the flags crates/v4flash-kernels/build.rs uses for the v41 feature.
export KFLAGS_V41="-O3 -DDEEPSTRIX_V41=1 -DMHC_N_EMBD=5120 -DMHC_HC_DIM=20480 -DROUTER_MAX_EXPERTS=512"
export KERNELS_DIR="$KI_WT/crates/v4flash-kernels/kernels"
