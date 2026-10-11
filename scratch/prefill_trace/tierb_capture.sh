#!/usr/bin/env bash
# Tier B device trace around ONE long prefill, minimal exposure (Tier B regresses decode): wait for the 2nd window of a
# job > MIN_TOK tokens, set V41_EVTRACE_DEV=1 in the live knob file, let N_WIN windows complete, dump the ring, remove
# the line (trap: always removed). Prints the dump path + the UTC span to analyse with devscan.py.
#   tierb_capture.sh [MIN_TOK=30000] [N_WIN=3] [MAX_WAIT_S=7200]
set -u
MIN_TOK=${1:-30000}; N_WIN=${2:-3}; MAX=${3:-7200}
KF=/dev/shm/deepstrix-knobs.txt
RING=/home/claude-code/logs/evtrace-ring
LOG=/home/claude-code/logs/v41-server.log
off() { grep -v '^V41_EVTRACE_DEV=' "$KF" > "$KF.tmp.$$" && cat "$KF.tmp.$$" > "$KF"; rm -f "$KF.tmp.$$"; }
trap off EXIT
since=$(date -u +%FT%T)
log_done() { sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -a "prefill_lm_window done" | awk -v s="$1" '$1 > s'; }
job=""
for i in $(seq 1 $((MAX / 2))); do
  job=$(log_done "$since" | grep -oE 'tokens_done=[0-9]+ tokens_total=[0-9]+' \
        | awk -v m="$MIN_TOK" '{split($1,a,"=");split($2,b,"="); if (b[2]+0 > m && a[2]+0 < b[2]+0) print b[2]}' | tail -1)
  [ -n "$job" ] && break
  sleep 2
done
[ -z "$job" ] && { echo "TIMEOUT waiting for a > $MIN_TOK-token prefill"; exit 1; }
echo "V41_EVTRACE_DEV=1" >> "$KF"
t_on=$(date -u +%FT%T); echo "$t_on Tier B ON (job tokens_total=$job)"
n=0
for i in $(seq 1 120); do
  n=$(log_done "$t_on" | grep -c "tokens_total=$job ")
  [ "$n" -ge "$N_WIN" ] && break
  log_done "$t_on" | grep -q "tokens_done=$job tokens_total=$job" && break
  sleep 1
done
t_off=$(date -u +%FT%T)
before=$(ls -t "$RING"/*.evt 2>/dev/null | head -1)
echo all > "$RING/dump-request"
off
echo "$t_off Tier B OFF after $n windows; dump requested"
for i in $(seq 1 30); do
  new=$(ls -t "$RING"/*.evt 2>/dev/null | head -1)
  [ "$new" != "$before" ] && { sleep 3; echo "DUMP $new SPAN $t_on $t_off JOB $job"; exit 0; }
  sleep 1
done
echo "no new dump file"; exit 1
