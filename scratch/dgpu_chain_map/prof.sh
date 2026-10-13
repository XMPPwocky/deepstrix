#!/usr/bin/env bash
# usage: prof.sh <outdir> [ENV=val ...] -- rocprofv3 kernel + memcpy trace of the bench, dGPU ONLY.
cd /home/claude-code/deepstrix/.claude/worktrees/wf_c1d97dd2-693-1 || exit 1
out=$1; shift
rm -rf "$out"; mkdir -p "$out"
BIN=$(cat /tmp/fusemap/bin_bench_dgpu_chain_map)
ROCP=/nix/store/nz5q0f805d5cx9nzf3pv3qpqarrrx4ag-rocprofiler-sdk-7.2.3/bin/rocprofv3
env HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 HIP_FORCE_DEV_KERNARG=1 DEBUG_CLR_KERNARG_HDP_FLUSH_WA=1 \
  DEBUG_CLR_KERNARG_DEV_ARCHS=gfx1201 BENCH_LABELS="$out/labels.txt" "$@" \
  timeout 2400 nix develop -c "$ROCP" --kernel-trace --memory-copy-trace -d "$out" -o kt --output-format csv \
  -- "$BIN" --ignored --nocapture > "$out/run.log" 2>&1
echo "rc $?"
find "$out" -name '*.csv' | head
grep -v '^:3:' "$out/run.log" | tail -n 120
