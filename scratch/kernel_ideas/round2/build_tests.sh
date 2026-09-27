#!/usr/bin/env bash
# build_tests.sh <test-name>... : build (not run) the named v4flash-kernels integration tests into the
# sweep target dir, 2 jobs (the hub owns ~87 GB of RAM).
set -u
WT=/home/claude-code/deepstrix/.claude/worktrees/kernel-ideas-2026-09-26
export CARGO_TARGET_DIR=/home/claude-code/deepstrix/target-v41-ki
cd "$WT"
args=()
for t in "$@"; do args+=(--test "$t"); done
exec nix develop -c cargo test --release --features v41 -p v4flash-kernels -j 2 --no-run "${args[@]}"
