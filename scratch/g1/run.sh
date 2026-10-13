#!/usr/bin/env bash
# usage: run.sh <test> <log> [ENV=val ...] -- dGPU ONLY (HIP_VISIBLE_DEVICES=0 = gfx1201), production
# kernarg mode (VRAM kernargs + HDP flush workaround).
cd /home/claude-code/deepstrix/.claude/worktrees/wf_c1d97dd2-693-2 || exit 1
TEST=$1; shift
log=$1; shift
BIN=$(cat /tmp/g1/bin_"$TEST")
env HIP_VISIBLE_DEVICES=0 HIP_FORCE_DEV_KERNARG=1 DEBUG_CLR_KERNARG_HDP_FLUSH_WA=1 DEBUG_CLR_KERNARG_DEV_ARCHS=gfx1201 "$@" \
  timeout 1800 nix develop -c "$BIN" --ignored --nocapture --test-threads=1 > "$log" 2>&1
echo "rc $?"
grep -v '^:3:' "$log" | tail -n 60
