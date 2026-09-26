#!/usr/bin/env bash
cd /home/claude-code/deepstrix/.claude/worktrees/kernel-ideas-2026-09-26
F=crates/v4flash-kernels/src/het/forward_prefill.rs
for r in 3844,3858 3993,4004 4628,4640 4676,4686 5001,5012; do echo "== $r"; sed -n "${r}p" "$F"; done
echo ----CAP
grep -n "stage_cap\|cap.begin\|cap.end\|GraphCapture\|begin_capture" "$F" | sed -n 1,25p
echo ----STAGE
grep -n "fn stage\b\|fn stage(" crates/v4flash-kernels/src/het/*.rs | head
