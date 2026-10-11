#!/usr/bin/env bash
# Hub-only deploy of the Engram RAW row cache + FADV_RANDOM (owner 10-11 "yes"; f6724e41, review r6 fixes): hub 0c957d7a <- e8669c98;
# env + knob files unchanged (ENGRAM_CACHE_MB default 1024 per table). Box 2 untouched. Unhealthy -> back on the
# backup binary. SKIP_IDLE=1 skips the idle wait. Redirect stdout to a FILE.
set -u
WT=/home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe
W=/home/claude-code/scratch-ms
PROD=/home/claude-code/deepstrix/target-v41/release/deepstrix-server
NEW=$WT/target-gate/release/deepstrix-server
OLD_MD5=e8669c98
NEW_MD5=0c957d7a
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
[ "${SKIP_IDLE:-0}" = 1 ] || bash "$WT/scratch/prefill_trace/wait_idle.sh" 20 300 || say "WARNING: still busy after 5 min; proceeding"
echo "$(date -u +%FT%T) deploy_engram: start $OLD_MD5 -> $NEW_MD5" >> "$MARK"
say "hub down"
stop_all || { say "hub survived SIGKILL"; exit 1; }
sleep 3
if install_bin "$NEW" "$NEW_MD5" && hub_ok; then
  say "OK: hub up on $NEW_MD5 (Engram raw row cache + FADV_RANDOM)"
  echo "$(date -u +%FT%T) deploy_engram: done" >> "$MARK"
  exit 0
fi
say "unhealthy: back on $OLD_MD5"
stop_all
install_bin "$BAK" "$OLD_MD5" || say "RESTORE FAILED"
hub_ok && say "rolled back"
echo "$(date -u +%FT%T) deploy_engram: ROLLED BACK" >> "$MARK"
exit 1
