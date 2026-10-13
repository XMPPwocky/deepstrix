#!/usr/bin/env bash
# usage: build.sh <test-name>  -- builds the test binary into target-r2 (nice 19) and records its path
W=/home/claude-code/deepstrix/.claude/worktrees/wf_be8e2331-368-2
cd $W || exit 1
export CARGO_TARGET_DIR=$W/target-r2
TEST=$1
mkdir -p /tmp/g1o
nix develop -c nice -n 19 cargo test --release --features v41 -p v4flash-kernels --test "$TEST" --no-run 2>&1 | grep -E -A12 '^error' | head -80
ls -t target-r2/release/deps/"$TEST"-* | grep -v '\.d$' | head -1 > /tmp/g1o/bin_"$TEST"
echo "built: $(cat /tmp/g1o/bin_"$TEST")"
