#!/usr/bin/env bash
# Diagnose the 1024-row-lane 'restored' null failure (10-11 02:24 UTC): run lm_prefill_bitexact LM_CASES=restored at
# V41_PREFILL_LANE_ROWS=1024 twice -- (1) PRODUCTION pager config (unified LRU pool, no dense windows: V41_PREFILL_UNIFIED_POOL=1
# V41_PAGER_NO_DENSE=1) and (2) the dense-window config Gate A used -- with the row-span diff report (2a8afc94). Hub down for
# both (~8-10 min), back up on the SAME binary afterwards. Box 2 stays up. Redirect stdout to a FILE.
set -u
WT=/home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe
W=/home/claude-code/scratch-ms
MARK=/home/claude-code/logs/evtrace/deploy_markers.log
HUBPAT='^/home/claude-code/deepstrix[^ ]*/release/deepstrix-server '
SUPPAT='^bash /home/claude-code/run_v41_server.sh'
say() { echo "$(date -u +%T) $*"; }
if [ -p /dev/stdout ]; then echo "stdout is a pipe: redirect to a file"; exit 2; fi
BIN=$(ls -t "$WT"/target-gate/release/deps/lm_prefill_bitexact-* | grep -v '\.d$' | head -1)
stop_all() {
  bash "$W/stop_prod.sh" >/dev/null 2>&1
  if pgrep -f "$HUBPAT" >/dev/null || pgrep -f "$SUPPAT" >/dev/null; then pkill -KILL -f "$SUPPAT"; pkill -KILL -f "$HUBPAT"; sleep 5; fi
  ! pgrep -f "$HUBPAT" >/dev/null && ! pgrep -f "$SUPPAT" >/dev/null
}
bash "$WT/scratch/prefill_trace/wait_idle.sh" 20 1800 || say "WARNING: still busy; proceeding"
echo "$(date -u +%FT%T) diag1024: hub down" >> "$MARK"
say "hub down"; stop_all || { say "hub survived SIGKILL"; exit 1; }
export KI_STATE=/home/claude-code/.claude/jobs/b8eef48e/tmp/ki_state
# shellcheck disable=SC1091
source "$WT/scratch/kernel_ideas/_infra/env.sh"
common="HIP_VISIBLE_DEVICES=0,1 V41_PAGED_EXPERTS=1 V41_INDEX_K=1 V41_CANDIDATE_POOL=1 V41_LM_PREFILL=1 V41_LM_LAYER_OUTER=1 LM_CASES=restored V41_REMOTE_ADDR=10.99.0.2:7431 V41_T2_CATCHALL=2 V41_PREFILL_LANE_ROWS=1024"
for arm in prod_pager dense_windows; do
  extra=""
  [ $arm = prod_pager ] && extra="V41_PREFILL_UNIFIED_POOL=1 V41_PAGER_NO_DENSE=1"
  L=/home/claude-code/diag1024_${arm}_$(date -u +%Y%m%dT%H%M%SZ).log
  say "run $arm -> $L"
  ( cd "$WT/crates/v4flash-kernels" && env -u V41_SUB -u V41_PREFILL_SPLIT $common $extra timeout -k 30 1200 "$BIN" --ignored --nocapture ) > "$L" 2>&1
  say "$arm rc=$?"
  sed 's/\x1b\[[0-9;]*m//g' "$L" | grep -aE 'expert pager: .*windows|BIT-IDENTICAL|NULL|differ|test result|panicked' | head -14 | cut -c1-230
done
echo "$(date -u +%FT%T) diag1024: runs done, relaunching" >> "$MARK"
bash "$WT/scratch/prefill_trace/launch_hub.sh" && say "hub up"
