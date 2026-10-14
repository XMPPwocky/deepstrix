#!/usr/bin/env bash
# usage: run.sh <test-bin-name> <test-fn-filter> <log> [ENV=val ...] -- runs the built test on the dGPU ONLY
# (HIP_VISIBLE_DEVICES=0 = gfx1201 on box 1) with production's kernarg mode.
W=/home/claude-code/deepstrix/.claude/worktrees/wf_be8e2331-368-6
cd "$W" || exit 1
TB=$1; shift
FILTER=$1; shift
log=$1; shift
BIN=$(cat scratch/g46/bin_"$TB")
env HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 HIP_FORCE_DEV_KERNARG=1 DEBUG_CLR_KERNARG_HDP_FLUSH_WA=1 \
  DEBUG_CLR_KERNARG_DEV_ARCHS=gfx1201 "$@" \
  timeout 1800 nix develop -c "$BIN" "$FILTER" --ignored --nocapture --test-threads=1 > "$log" 2>&1
echo "rc $?"
grep -v '^:3:' "$log" | grep -v "^deepstrix dev\|Name:  \|^Devices" | tail -n 150
