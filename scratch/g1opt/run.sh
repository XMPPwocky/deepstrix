#!/usr/bin/env bash
# usage: run.sh <test> <log> [ENV=val ...] [-- test filter args] -- dGPU ONLY (HIP_VISIBLE_DEVICES=0 =
# gfx1201), production kernarg mode (VRAM kernargs + HDP flush workaround). Never the hub, never the iGPU.
W=/home/claude-code/deepstrix/.claude/worktrees/wf_be8e2331-368-2
cd $W || exit 1
TEST=$1; shift
log=$1; shift
BIN=$(cat /tmp/g1o/bin_"$TEST")
envs=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
[ "$1" = "--" ] && shift
env HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 HIP_FORCE_DEV_KERNARG=1 DEBUG_CLR_KERNARG_HDP_FLUSH_WA=1 \
  DEBUG_CLR_KERNARG_DEV_ARCHS=gfx1201 "${envs[@]}" \
  timeout 1800 nix develop -c "$BIN" --ignored --nocapture --test-threads=1 "$@" > "$log" 2>&1
echo "rc $?"
grep -v '^:3:' "$log" | tail -n 80
