#!/usr/bin/env bash
# Long-job threshold 16384 -> 4096 (owner 10-11 "kk"): every layer-major job gets the P1 prefill split + box 2's long-job
# handling (jobs < 16K used the decode partition: later windows 8.0 s p50 vs 3.6 s). V41_LONG_PREFILL_TOKENS is STATIC:
# env-file line + hub-only restart (box 2 stays warm). Unhealthy start -> env file restored, relaunch. Redirect to a FILE.
set -u
WT=/home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe
W=/home/claude-code/scratch-ms
ENVF=$W/hub_env_c_20261006.txt
ENVBAK=$ENVF.pre-long4096-20261011
MARK=/home/claude-code/logs/evtrace/deploy_markers.log
HUBPAT='^/home/claude-code/deepstrix[^ ]*/release/deepstrix-server '
SUPPAT='^bash /home/claude-code/run_v41_server.sh'
say() { echo "$(date -u +%T) $*"; }
if [ -p /dev/stdout ]; then echo "stdout is a pipe: redirect to a file"; exit 2; fi
grep -q '^V41_LONG_PREFILL_TOKENS=' "$ENVF" && { echo "env file already sets V41_LONG_PREFILL_TOKENS"; exit 1; }
stop_all() {
  bash "$W/stop_prod.sh" >/dev/null 2>&1
  if pgrep -f "$HUBPAT" >/dev/null || pgrep -f "$SUPPAT" >/dev/null; then pkill -KILL -f "$SUPPAT"; pkill -KILL -f "$HUBPAT"; sleep 5; fi
  ! pgrep -f "$HUBPAT" >/dev/null && ! pgrep -f "$SUPPAT" >/dev/null
}
hub_ok() {
  bash "$WT/scratch/prefill_trace/launch_hub.sh" \
    && sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | tail -800 | grep -aq 'prefill VRAM loan: prefill lane set built'
}
bash "$WT/scratch/prefill_trace/wait_idle.sh" 20 300 || say "WARNING: still busy after 5 min; proceeding"
cp -p "$ENVF" "$ENVBAK" || exit 1
{ cat "$ENVBAK"; echo "V41_LONG_PREFILL_TOKENS=4096"; } > "$ENVF.tmp" && mv "$ENVF.tmp" "$ENVF" || exit 1
echo "$(date -u +%FT%T) deploy_long4096: start" >> "$MARK"
say "hub down"
stop_all || { say "hub survived SIGKILL"; cp -p "$ENVBAK" "$ENVF"; exit 1; }
sleep 3
if hub_ok; then
  hp=$(pgrep -f "$HUBPAT")
  say "hub env: $(tr '\0' '\n' < /proc/$hp/environ | grep -E '^V41_LONG_PREFILL_TOKENS=')"
  say "OK: hub up with V41_LONG_PREFILL_TOKENS=4096"
  echo "$(date -u +%FT%T) deploy_long4096: done" >> "$MARK"
  exit 0
fi
say "unhealthy: restoring the env file and relaunching"
stop_all
cp -p "$ENVBAK" "$ENVF"
hub_ok && say "rolled back"
echo "$(date -u +%FT%T) deploy_long4096: ROLLED BACK" >> "$MARK"
exit 1
