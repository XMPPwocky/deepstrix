#!/usr/bin/env bash
# Restart the hub server with deploy/run-hub.sh, optionally installing a new binary.
#
#   deploy/restart-hub.sh [EXTERNAL_HOST]     # restart the current binary
#   INSTALL=path/to/deepstrix-server deploy/restart-hub.sh [EXTERNAL_HOST]
#
# EXTERNAL_HOST is passed to run-hub.sh: an IP or hostname to serve on besides
# 127.0.0.1 (omit for loopback only).
#
# Box 2's expertd must already be LISTENING: the hub connects to V41_REMOTE_ADDR
# during load and panics on refusal. To restart both: deploy/restart-expertd-b2.sh
# first, then this.
#
# Stop is SIGKILL: an idle server's worker blocks in `blocking_recv` and ignores
# SIGINT/SIGTERM. In-flight requests fail; clients (the agents) retry.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
LIVE=${DEEPSTRIX_BIN:-/home/claude-code/deepstrix/target-v41/release/deepstrix-server}
PAT="^$LIVE"
LOG=${LOG:-$HOME/logs/v41-server.log}

P=$(pgrep -f "$PAT" | head -1)
echo "hub pid: ${P:-none}  $(date -u +%T)"
[ -n "$P" ] && kill -9 "$P"
for _ in $(seq 1 30); do pgrep -f "$PAT" >/dev/null || break; sleep 1; done
pgrep -f "$PAT" >/dev/null && { echo "hub still alive; not relaunching"; exit 1; }

if [ -n "${INSTALL:-}" ]; then
  # Keep the previous binary for rollback (INSTALL=$LIVE.prev to roll back).
  cp -p "$LIVE" "$LIVE.prev"
  cp -p "$INSTALL" "$LIVE.new" && mv -f "$LIVE.new" "$LIVE"
  echo "installed $INSTALL (previous kept at $LIVE.prev)"
fi

# Never inherit V41_LOG_ATTACHED: run-hub.sh would then skip its log redirect
# and the server would log to /dev/null (looks hung while serving fine).
cd ~ && env -u V41_LOG_ATTACHED setsid nohup "$HERE/run-hub.sh" "$@" >/dev/null 2>&1 </dev/null &
sleep 3
for i in $(seq 1 300); do
  sed 's/\x1b\[[0-9;]*m//g' "$LOG" 2>/dev/null | grep -q 'listening on' && break
  sleep 1
done
echo "hub up check after ${i}s $(date -u +%T):"
sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -E 'listening on|multistream scheduler ON|ERROR|panicked' | tail -4 | cut -c1-200
