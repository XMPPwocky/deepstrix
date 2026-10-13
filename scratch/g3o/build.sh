#!/usr/bin/env bash
# usage: build.sh [test-name] -- builds the test binary with the worktree's OWN target dir (target-r2), nice 19
W=/home/claude-code/deepstrix/.claude/worktrees/wf_be8e2331-368-4
cd "$W" || exit 1
export CARGO_TARGET_DIR=$W/target-r2
TEST=${1:-attn_dec_g3_bitexact}
nix develop -c nice -n 19 cargo test --release --features v41 -p v4flash-kernels --test "$TEST" --no-run 2>&1 \
  | grep -E -A12 '^(error|warning: unused)' | head -80
ls -t target-r2/release/deps/"$TEST"-* 2>/dev/null | grep -v '\.d$' | head -1 > scratch/g3o/bin_"$TEST"
echo "built: $(cat scratch/g3o/bin_"$TEST")"
