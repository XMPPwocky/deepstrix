#!/usr/bin/env bash
# in_env.sh <cmd args...> — run one command inside the cached ROCm dev env (hipcc, llvm-objdump,
# llvm-readelf, clang-offload-bundler, $ROCPROFV3, $KFLAGS_V41, $KERNELS_DIR ...).
# NOT for GPU work: anything that touches a GPU must go through gpu_run.sh.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
exec "$@"
