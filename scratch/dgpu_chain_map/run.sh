#!/usr/bin/env bash
# usage: run.sh <test> <log> [ENV=val ...] -- runs the bench on the dGPU ONLY (HIP_VISIBLE_DEVICES=0 = gfx1201)
# with production's kernarg mode (VRAM kernargs, HDP flush workaround).
cd /home/claude-code/deepstrix/.claude/worktrees/wf_c1d97dd2-693-1 || exit 1
TEST=$1; shift
log=$1; shift
BIN=$(cat /tmp/fusemap/bin_"$TEST")
env HIP_VISIBLE_DEVICES=0 HIP_FORCE_DEV_KERNARG=1 DEBUG_CLR_KERNARG_HDP_FLUSH_WA=1 DEBUG_CLR_KERNARG_DEV_ARCHS=gfx1201 "$@" \
  timeout 2400 nix develop -c "$BIN" --ignored --nocapture > "$log" 2>&1
echo "rc $?"
grep -v '^:3:' "$log" | tail -n 400
