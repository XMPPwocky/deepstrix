#!/usr/bin/env bash
# usage: build.sh [test-name] -- builds the test binary with the worktree's OWN target dir, nice 19
cd /home/claude-code/deepstrix/.claude/worktrees/wf_c1d97dd2-693-6 || exit 1
export CARGO_TARGET_DIR=/home/claude-code/deepstrix/.claude/worktrees/wf_c1d97dd2-693-6/target-fuse
TEST=${1:-attn_dec_g3_bitexact}
nix develop -c nice -n 19 cargo test --release --features v41 -p v4flash-kernels --test "$TEST" --no-run 2>&1 \
  | grep -E -A12 '^(error|warning: unused)' | head -80
ls -t target-fuse/release/deps/"$TEST"-* 2>/dev/null | grep -v '\.d$' | head -1 > scratch/g3/bin_"$TEST"
echo "built: $(cat scratch/g3/bin_"$TEST")"
