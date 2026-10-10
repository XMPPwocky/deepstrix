#!/usr/bin/env bash
# Hub-only binary deploy (box 2 stays up and warm): stop supervisor + hub, install NEW (md5-checked), relaunch through
# launch_hub.sh (env file NAME=value lines only), health-check; on failure roll back to the previous binary.
#   deploy_hub.sh NEW_BIN NEW_MD5 OLD_MD5 "note"     (redirect stdout to a FILE)
set -u
NEW=$1; NEW_MD5=$2; OLD_MD5=$3; NOTE=${4:-deploy}
WT=/home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe
W=/home/claude-code/scratch-ms
PROD=/home/claude-code/deepstrix/target-v41/release/deepstrix-server
BAK=$W/deepstrix-server.$OLD_MD5.bak
MARK=/home/claude-code/logs/evtrace/deploy_markers.log
HUBPAT='^/home/claude-code/deepstrix[^ ]*/release/deepstrix-server '
SUPPAT='^bash /home/claude-code/run_v41_server.sh'
DRVPAT='^python3? [^ ]*/(ab_[a-z0-9_]+|sweep_[a-z0-9_]+|e2e_[a-z0-9_]+|arms_[a-z0-9_]+)\.py'
B2=10.43.0.95
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
say() { echo "$(date -u +%T) $*"; }
mark() { echo "$(date -u +%FT%T) deploy_hub: $*" >> "$MARK"; }
md5of() { md5sum < "$1" | cut -c1-8; }
if [ -p /dev/stdout ]; then echo "stdout is a pipe: redirect to a file"; exit 2; fi
pgrep -f -- "$DRVPAT" >/dev/null && { echo "an A/B driver is running; stop it first"; exit 1; }
[ "$(md5of "$PROD")" = "$OLD_MD5" ] || { echo "production binary is not $OLD_MD5"; exit 1; }
[ "$(md5of "$NEW")" = "$NEW_MD5" ] || { echo "new binary is not $NEW_MD5"; exit 1; }
[ -f "$BAK" ] || cp -p "$PROD" "$BAK" || exit 1
[ "$(md5of "$BAK")" = "$OLD_MD5" ] || { echo "backup $BAK is not $OLD_MD5"; exit 1; }
stop_all() {
  bash "$W/stop_prod.sh" >/dev/null 2>&1
  if pgrep -f "$HUBPAT" >/dev/null || pgrep -f "$SUPPAT" >/dev/null; then
    pkill -KILL -f "$SUPPAT"; pkill -KILL -f "$HUBPAT"; sleep 5
  fi
  ! pgrep -f "$HUBPAT" >/dev/null && ! pgrep -f "$SUPPAT" >/dev/null
}
b2_conn_free() {
  local i n
  for i in $(seq 1 120); do
    n=$($SSH $B2 "ss -Htn state established '( sport = :7431 )' | wc -l") || n=x
    [ "$n" = 0 ] && return 0
    sleep 1
  done
  say "box 2 still holds $n connection(s)"; return 1
}
install_bin() { cp -p "$1" "$PROD.new" && mv -f "$PROD.new" "$PROD" && [ "$(md5of "$PROD")" = "$2" ]; }
mark "start: $OLD_MD5 -> $NEW_MD5 ($NOTE)"
say "hub down"
stop_all || { say "a hub survived SIGKILL: NOT continuing"; mark "ABORT"; exit 1; }
b2_conn_free || say "WARNING: box 2 still shows a connection; launching anyway"
if install_bin "$NEW" "$NEW_MD5" && bash "$WT/scratch/prefill_trace/launch_hub.sh"; then
  say "OK: hub up on $NEW_MD5"; mark "done: hub $NEW_MD5 up ($NOTE)"; exit 0
fi
say "FAILED on $NEW_MD5: rolling back to $OLD_MD5"
stop_all
install_bin "$BAK" "$OLD_MD5" || say "RESTORE OF $BAK FAILED"
bash "$WT/scratch/prefill_trace/launch_hub.sh" && say "rolled back: hub up on $OLD_MD5"
mark "ROLLED BACK to $OLD_MD5"
exit 1
