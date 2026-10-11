#!/usr/bin/env bash
# Turn the prefill VRAM loan ON in production (env-only; the binary already carries it): wait idle, add
# V41_PREFILL_VRAM_LOAN=1 V41_PREFILL_LANE_ROWS=1024 V41_MS_CHUNK_ROWS=2048 V41_MS_CHUNK_ROWS_IDLE=2048 to the env file,
# hub-only restart, require the "prefill lane set built" startup line; else revert the env file and relaunch.
# Redirect stdout to a FILE.
set -u
WT=/home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe
W=/home/claude-code/scratch-ms
ENVF=$W/hub_env_c_20261006.txt
ENVBAK=$ENVF.pre-loan-20261011b
MARK=/home/claude-code/logs/evtrace/deploy_markers.log
HUBPAT='^/home/claude-code/deepstrix[^ ]*/release/deepstrix-server '
SUPPAT='^bash /home/claude-code/run_v41_server.sh'
B2=10.43.0.95
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
say() { echo "$(date -u +%T) $*"; }
if [ -p /dev/stdout ]; then echo "stdout is a pipe: redirect to a file"; exit 2; fi
grep -q '^V41_PREFILL_VRAM_LOAN=' "$ENVF" && { echo "env file already has the loan"; exit 1; }
stop_all() {
  bash "$W/stop_prod.sh" >/dev/null 2>&1
  if pgrep -f "$HUBPAT" >/dev/null || pgrep -f "$SUPPAT" >/dev/null; then pkill -KILL -f "$SUPPAT"; pkill -KILL -f "$HUBPAT"; sleep 5; fi
  ! pgrep -f "$HUBPAT" >/dev/null && ! pgrep -f "$SUPPAT" >/dev/null
}
b2_conn_free() {
  local i n
  for i in $(seq 1 120); do
    n=$($SSH $B2 "ss -Htn state established '( sport = :7431 )' | wc -l") || n=x
    [ "$n" = 0 ] && return 0
    sleep 1
  done
  return 1
}
bash "$WT/scratch/prefill_trace/wait_idle.sh" 20 1800 || say "WARNING: still busy after 30 min; proceeding"
cp -p "$ENVF" "$ENVBAK" || exit 1
printf '%s\n' "V41_PREFILL_VRAM_LOAN=1" "V41_PREFILL_LANE_ROWS=1024" "V41_MS_CHUNK_ROWS=2048" "V41_MS_CHUNK_ROWS_IDLE=2048" >> "$ENVF"
echo "$(date -u +%FT%T) deploy_loan: start (env += loan, same binary)" >> "$MARK"
say "hub down"
stop_all || { say "hub survived SIGKILL"; cp -p "$ENVBAK" "$ENVF"; exit 1; }
b2_conn_free || say "WARNING: box 2 still shows a connection"
if bash "$WT/scratch/prefill_trace/launch_hub.sh" \
   && sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | tail -800 | grep -aq 'prefill VRAM loan: prefill lane set built'; then
  sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | grep -a 'prefill VRAM loan: prefill lane set built' | tail -1 | cut -c1-500
  sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | grep -aE 'ERROR|panicked' | tail -3 | cut -c1-200
  say "OK: hub up WITH the prefill VRAM loan"
  echo "$(date -u +%FT%T) deploy_loan: done, loan ON" >> "$MARK"
  exit 0
fi
say "loan start unhealthy: reverting the env file and relaunching"
sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | grep -aE 'prefill VRAM loan|ERROR|panicked' | tail -5 | cut -c1-300
stop_all
cp -p "$ENVBAK" "$ENVF"
bash "$WT/scratch/prefill_trace/launch_hub.sh" && say "rolled back: hub up without the loan"
echo "$(date -u +%FT%T) deploy_loan: ROLLED BACK" >> "$MARK"
exit 1
