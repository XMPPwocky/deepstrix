#!/usr/bin/env bash
# build.sh [cargo args...] — build the V4.1 binaries from THIS worktree into the sweep's own target
# dir (never the production target-v41). Default: server + expertd, release, --features v41.
# Examples:  bash _infra/build.sh                       # both binaries
#            bash _infra/build.sh -p v4flash-kernels    # kernels crate only (fast compile check)
#            bash _infra/build.sh --tests -p v4flash-kernels
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export CARGO_TARGET_DIR=/home/claude-code/deepstrix/target-v41-ki
cd "$WT"
if [ $# -eq 0 ]; then set -- -p deepstrix-server -p deepstrix-expertd; fi
exec nix develop -c cargo build --release --features v41 "$@"
