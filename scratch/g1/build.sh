#!/usr/bin/env bash
# usage: build.sh <test-name>  -- builds the test binary into target-fuse and records its path
cd /home/claude-code/deepstrix/.claude/worktrees/wf_c1d97dd2-693-2 || exit 1
export CARGO_TARGET_DIR=/home/claude-code/deepstrix/.claude/worktrees/wf_c1d97dd2-693-2/target-fuse
TEST=$1
mkdir -p /tmp/g1
nix develop -c nice -n 19 cargo test --release --features v41 -p v4flash-kernels --test "$TEST" --no-run 2>&1 | grep -E -A12 '^error' | head -80
ls -t target-fuse/release/deps/"$TEST"-* | grep -v '\.d$' | head -1 > /tmp/g1/bin_"$TEST"
echo "built: $(cat /tmp/g1/bin_"$TEST")"
