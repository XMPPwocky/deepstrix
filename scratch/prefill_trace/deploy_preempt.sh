#!/usr/bin/env bash
# Hub-only deploy of the decode-preemption scheduler (docs/v41/MS_PREFILL_PREEMPT_DESIGN.md rev 2; code review APPROVE r3,
# fixes 0f29c56c): hub 3c2a6062 <- 6d8ffc87; env + knob files unchanged (V41_MS_PREEMPT defaults OFF -> the A/B flips it
# live). Box 2 untouched. Unhealthy -> back on the backup binary. Redirect stdout to a FILE.
set -u
WT=/home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe
W=/home/claude-code/scratch-ms
PROD=/home/claude-code/deepstrix/target-v41/release/deepstrix-server
NEW=$WT/target-gate/release/deepstrix-server
OLD_MD5=6d8ffc87
NEW_MD5=3c2a6062
BAK=$W/deepstrix-server.$OLD_MD5.bak
MARK=/home/claude-code/logs/evtrace/deploy_markers.log
HUBPAT='^/home/claude-code/deepstrix[^ ]*/release/deepstrix-server '
SUPPAT='^bash /home/claude-code/run_v41_server.sh'
say() { echo "$(date -u +%T) $*"; }
md5of() { md5sum < "$1" | cut -c1-8; }
if [ -p /dev/stdout ]; then echo "stdout is a pipe: redirect to a file"; exit 2; fi
[ "$(md5of "$PROD")" = "$OLD_MD5" ] || { echo "production is not $OLD_MD5"; exit 1; }
[ "$(md5of "$NEW")" = "$NEW_MD5" ] || { echo "new binary is not $NEW_MD5"; exit 1; }
[ -f "$BAK" ] || cp -p "$PROD" "$BAK" || exit 1
stop_all() {
  bash "$W/stop_prod.sh" >/dev/null 2>&1
  if pgrep -f "$HUBPAT" >/dev/null || pgrep -f "$SUPPAT" >/dev/null; then pkill -KILL -f "$SUPPAT"; pkill -KILL -f "$HUBPAT"; sleep 5; fi
  ! pgrep -f "$HUBPAT" >/dev/null && ! pgrep -f "$SUPPAT" >/dev/null
}
install_bin() { cp -p "$1" "$PROD.new" && mv -f "$PROD.new" "$PROD" && [ "$(md5of "$PROD")" = "$2" ]; }
hub_ok() {
  bash "$WT/scratch/prefill_trace/launch_hub.sh" \
    && sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | tail -800 | grep -aq 'prefill VRAM loan: prefill lane set built'
}
bash "$WT/scratch/prefill_trace/wait_idle.sh" 20 300 || say "WARNING: still busy after 5 min; proceeding"
echo "$(date -u +%FT%T) deploy_preempt: start $OLD_MD5 -> $NEW_MD5" >> "$MARK"
say "hub down"
stop_all || { say "hub survived SIGKILL"; exit 1; }
sleep 3
if install_bin "$NEW" "$NEW_MD5" && hub_ok; then
  say "OK: hub up on $NEW_MD5 (preemption built in, V41_MS_PREEMPT off)"
  echo "$(date -u +%FT%T) deploy_preempt: done" >> "$MARK"
  exit 0
fi
say "unhealthy: back on $OLD_MD5"
stop_all
install_bin "$BAK" "$OLD_MD5" || say "RESTORE FAILED"
hub_ok && say "rolled back"
echo "$(date -u +%FT%T) deploy_preempt: ROLLED BACK" >> "$MARK"
exit 1
