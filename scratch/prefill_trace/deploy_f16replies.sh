#!/usr/bin/env bash
# Box-2 prefill replies back to f16 (owner 10-11): drop V41_PREFILL_F16_REPLIES=0 from the hub env file (default = f16),
# hub-only restart after idle; on an unhealthy start restore the env file and relaunch. Redirect stdout to a FILE.
set -u
WT=/home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe
W=/home/claude-code/scratch-ms
ENVF=$W/hub_env_c_20261006.txt
ENVBAK=$ENVF.pre-f16replies-20261011
MARK=/home/claude-code/logs/evtrace/deploy_markers.log
HUBPAT='^/home/claude-code/deepstrix[^ ]*/release/deepstrix-server '
SUPPAT='^bash /home/claude-code/run_v41_server.sh'
say() { echo "$(date -u +%T) $*"; }
if [ -p /dev/stdout ]; then echo "stdout is a pipe: redirect to a file"; exit 2; fi
grep -q '^V41_PREFILL_F16_REPLIES=0$' "$ENVF" || { echo "env file has no V41_PREFILL_F16_REPLIES=0"; exit 1; }
stop_all() {
  bash "$W/stop_prod.sh" >/dev/null 2>&1
  if pgrep -f "$HUBPAT" >/dev/null || pgrep -f "$SUPPAT" >/dev/null; then pkill -KILL -f "$SUPPAT"; pkill -KILL -f "$HUBPAT"; sleep 5; fi
  ! pgrep -f "$HUBPAT" >/dev/null && ! pgrep -f "$SUPPAT" >/dev/null
}
bash "$WT/scratch/prefill_trace/wait_idle.sh" 20 1800 || say "WARNING: still busy; proceeding"
cp -p "$ENVF" "$ENVBAK" || exit 1
grep -v '^V41_PREFILL_F16_REPLIES=0$' "$ENVBAK" > "$ENVF.tmp" && mv "$ENVF.tmp" "$ENVF"
echo "$(date -u +%FT%T) deploy_f16replies: start" >> "$MARK"
say "hub down"
stop_all || { say "hub survived SIGKILL"; cp -p "$ENVBAK" "$ENVF"; exit 1; }
sleep 3
if bash "$WT/scratch/prefill_trace/launch_hub.sh" \
   && sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | tail -800 | grep -aq 'prefill VRAM loan: prefill lane set built'; then
  sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | tail -800 | grep -aE 'UNTESTED FIDELITY|ERROR|panicked' | tail -3 | cut -c1-160
  say "OK: hub up, f16 prefill replies (loan still on)"
  echo "$(date -u +%FT%T) deploy_f16replies: done" >> "$MARK"
  exit 0
fi
say "unhealthy: restoring the env file and relaunching"
stop_all
cp -p "$ENVBAK" "$ENVF"
bash "$WT/scratch/prefill_trace/launch_hub.sh" && say "rolled back"
echo "$(date -u +%FT%T) deploy_f16replies: ROLLED BACK" >> "$MARK"
exit 1
