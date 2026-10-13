#!/usr/bin/env bash
# usage: prof.sh <test> <outdir> [ENV=val ...] -- rocprofv3 kernel + memcpy trace of OUR bench process, dGPU ONLY
# (ROCR_VISIBLE_DEVICES=0 = gfx1201; never the hub, never ATT / thread trace, never the iGPU).
cd /home/claude-code/deepstrix/.claude/worktrees/wf_c1d97dd2-693-2 || exit 1
TEST=$1; shift
out=$1; shift
rm -rf "$out"; mkdir -p "$out"
BIN=$(cat /tmp/g1/bin_"$TEST")
ROCP=/nix/store/nz5q0f805d5cx9nzf3pv3qpqarrrx4ag-rocprofiler-sdk-7.2.3/bin/rocprofv3
env HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 HIP_FORCE_DEV_KERNARG=1 DEBUG_CLR_KERNARG_HDP_FLUSH_WA=1 \
  DEBUG_CLR_KERNARG_DEV_ARCHS=gfx1201 "$@" \
  timeout 1200 nix develop -c "$ROCP" --kernel-trace --memory-copy-trace -d "$out" -o kt --output-format csv \
  -- "$BIN" --ignored --nocapture --test-threads=1 > "$out/run.log" 2>&1
echo "rc $?"
find "$out" -name '*.csv' | head
