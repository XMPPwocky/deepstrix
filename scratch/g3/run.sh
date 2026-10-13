#!/usr/bin/env bash
# usage: run.sh <test-fn-filter> <log> [ENV=val ...] -- runs the built test on the dGPU ONLY
# (HIP_VISIBLE_DEVICES=0 = gfx1201 on box 1) with production's kernarg mode.
cd /home/claude-code/deepstrix/.claude/worktrees/wf_c1d97dd2-693-6 || exit 1
FILTER=$1; shift
log=$1; shift
BIN=$(cat scratch/g3/bin_attn_dec_g3_bitexact)
env HIP_VISIBLE_DEVICES=0 HIP_FORCE_DEV_KERNARG=1 DEBUG_CLR_KERNARG_HDP_FLUSH_WA=1 DEBUG_CLR_KERNARG_DEV_ARCHS=gfx1201 "$@" \
  timeout 1800 nix develop -c "$BIN" "$FILTER" --ignored --nocapture --test-threads=1 > "$log" 2>&1
echo "rc $?"
grep -v '^:3:' "$log" | grep -v "^deepstrix dev\|Name:  \|^Devices" | tail -n 150
