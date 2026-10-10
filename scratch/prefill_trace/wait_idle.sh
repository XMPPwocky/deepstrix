#!/usr/bin/env bash
# Block until the hub is idle: no ms.step / prefill window / chat request line in the server log for QUIET seconds.
# Exit 0 when idle, 1 on timeout.   wait_idle.sh [QUIET_S=20] [TIMEOUT_S=900]
QUIET=${1:-20}; TIMEOUT=${2:-900}
LOG=/home/claude-code/logs/v41-server.log
start=$(date +%s)
while :; do
  last=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -aE 'ms\.step|prefill_lm_window|prefill_job_progress|chat request|ms\.phase' | tail -1 | cut -c1-26)
  if [ -n "$last" ]; then
    lt=$(date -u -d "${last%Z}" +%s 2>/dev/null || echo 0)
    now=$(date +%s)
    if [ $((now - lt)) -ge "$QUIET" ]; then echo "$(date -u +%T) idle (last activity $last)"; exit 0; fi
  fi
  [ $(( $(date +%s) - start )) -ge "$TIMEOUT" ] && { echo "$(date -u +%T) timeout, still busy (last $last)"; exit 1; }
  sleep 2
done
