#!/usr/bin/env bash
# TWO-BOX DEPLOY of the LAST-unit evict-behind (docs/v41/B2_LAST_WINDOW_EVICT_BEHIND_DESIGN.md rev 3; code review APPROVE):
#   box 2: target-b2lu1011 expertd 0f2b0ac0 (worktree-igpu-wmma-moe 7e2e0407, built on box 1) <- target-b2lo1011 37aaed7c;
#          same --experts / env / CPU pinning; knob file unchanged in content (V41_MOE_WMMA_F16=1 kept; new knob
#          prefill_budget_last NOT set = unlimited = OFF).
#   hub  : d7357c2b (target-gate, same rev) <- 96a9c669, env file unchanged.
#   gates (hub down, box 2 OLD): remote_experts_pin_loopback + remote_experts_nursery_loopback on box 1's iGPU (~15 s).
# ORDER: box 1 down, gates, box 2 down, box 2 up, box 1 up. Any failure -> both back on 37aaed7c + 96a9c669.
#   window_lastunit.sh [deploy|rollback] > ~/window_lastunit.log 2>&1   (never a pipe)
set -u
MODE=${1:-deploy}
WT=/home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe
W=/home/claude-code/scratch-ms
PROD=/home/claude-code/deepstrix/target-v41/release/deepstrix-server
HUB_NEW=$WT/target-gate/release/deepstrix-server
HUB_NEW_MD5=d7357c2b
HUB_OLD_MD5=96a9c669
HUB_BAK=$W/deepstrix-server.$HUB_OLD_MD5.bak
MARK=/home/claude-code/logs/evtrace/deploy_markers.log
HUBPAT='^/home/claude-code/deepstrix[^ ]*/release/deepstrix-server '
SUPPAT='^bash /home/claude-code/run_v41_server.sh'
B2=10.43.0.95
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
B2_NEW_BIN=/home/claude-code/deepstrix/target-b2lu1011/release/deepstrix-expertd
B2_NEW_MD5=0f2b0ac0
B2_OLD_BIN=/home/claude-code/deepstrix/target-b2lo1011/release/deepstrix-expertd
B2_OLD_MD5=37aaed7c
B2_EXPERTS=L0-L39:265-383
B2LOG=/home/claude-code/logs/expertd-lastunit-20261011.log
B2KN=/home/claude-code/expertd-knobs.txt
B2KN_BAK=/home/claude-code/expertd-knobs.txt.pre-lastunit-20261011
say() { echo "$(date -u +%T) $*"; }
mark() { echo "$(date -u +%FT%T) lastunit: $*" >> "$MARK"; }
md5of() { md5sum < "$1" | cut -c1-8; }
if [ -p /dev/stdout ]; then echo "stdout is a pipe: redirect to a file"; exit 2; fi

stop_all() {
  bash "$W/stop_prod.sh" >/dev/null 2>&1
  if pgrep -f "$HUBPAT" >/dev/null || pgrep -f "$SUPPAT" >/dev/null; then pkill -KILL -f "$SUPPAT"; pkill -KILL -f "$HUBPAT"; sleep 5; fi
  ! pgrep -f "$HUBPAT" >/dev/null && ! pgrep -f "$SUPPAT" >/dev/null
}
b2_conn_free() {
  local i n
  for i in $(seq 1 60); do
    n=$($SSH $B2 "ss -Htn state established '( sport = :7431 )' | wc -l") || n=x
    [ "$n" = 0 ] && return 0
    sleep 1
  done
  say "box 2 still holds $n connection(s)"; return 1
}
install_hub() { cp -p "$1" "$PROD.new" && mv -f "$PROD.new" "$PROD" && [ "$(md5of "$PROD")" = "$2" ]; }
hub_up() {  # 0 = listening (launch_hub.sh: one supervisor + one hub + a chat) and the loan built
  bash "$WT/scratch/prefill_trace/launch_hub.sh" \
    && sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | tail -800 | grep -aq 'prefill VRAM loan: prefill lane set built'
}

# b2_restart BIN KMODE(new|old): stop the daemon, write/restore the knob file, launch BIN with today's args + env.
b2_restart() {
  local bin=$1 km=$2
  say "box 2: restart -> $bin (knobs $km)"
  $SSH $B2 "B2BIN=$bin KMODE=$km B2_EXPERTS=$B2_EXPERTS B2LOG=$B2LOG B2KN=$B2KN B2KN_BAK=$B2KN_BAK bash -s" <<'EOS'
set -u
PAT="^/home/claude-code/deepstrix/target-[a-z0-9]*/release/deepstrix-expertd"
[ -x "$B2BIN" ] || { echo "MISSING $B2BIN"; exit 1; }
findmnt /weights >/dev/null && findmnt /weights2 >/dev/null || { echo "/weights or /weights2 not mounted"; exit 1; }
OLD=$(pgrep -f "$PAT" | head -1)
echo "old daemon pid: ${OLD:-none} ($(readlink /proc/${OLD:-0}/exe 2>/dev/null))"
if [ -n "$OLD" ]; then
  kill -TERM $OLD
  for i in $(seq 1 20); do kill -0 $OLD 2>/dev/null || break; sleep 1; done
  kill -0 $OLD 2>/dev/null && kill -KILL $OLD
  sleep 2
fi
pgrep -f "$PAT" >/dev/null && { echo "a daemon survived: not launching a second"; exit 1; }
for i in $(seq 1 120); do g=$(cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | sort -n | tail -1); [ "${g:-0}" -lt 2000000000 ] && break; sleep 1; done
echo "gtt before launch: $(awk -v g=${g:-0} 'BEGIN{printf "%.2f", g/1e9}') GB after ${i}s"
[ "${g:-0}" -lt 30000000000 ] || { echo "GTT still high: not launching on top of it"; exit 1; }
if [ "$KMODE" = new ]; then
  [ -f "$B2KN_BAK" ] || cp -p "$B2KN" "$B2KN_BAK" || { echo "knob backup failed"; exit 1; }
  { grep -v -E '^V41_MOE_WMMA_F16=' "$B2KN_BAK"; echo "V41_MOE_WMMA_F16=1"; } > "$B2KN.tmp.$$" && mv "$B2KN.tmp.$$" "$B2KN" || { echo "knob write failed"; exit 1; }
elif [ -f "$B2KN_BAK" ]; then
  cp -p "$B2KN_BAK" "$B2KN.tmp.$$" && mv "$B2KN.tmp.$$" "$B2KN"
fi
echo "knob file: $(grep -v '^#' "$B2KN" | tr '\n' ' ')"
N0=$(wc -l < "$B2LOG" 2>/dev/null || echo 0)
echo "=== $(date -u) launch $B2BIN --experts $B2_EXPERTS knobs [$KMODE] (window_lastunit) ===" >> "$B2LOG"
cd /home/claude-code/deepstrix || exit 1
nohup setsid env V41_B2_MODE_EVICT=1 V41_B2_PREFILL_STAGE=0 V41_EVTRACE_DIR=/home/claude-code/logs/evtrace V41_B2_KNOBS=$B2KN V41_B2_HITS_FIRST=1 V41_B2_MISS_PAR=4 V41_B2_PREFETCH_SETS=16 V41_EXPERT_MIRROR_DIR=/weights2/dsv4.1f V41_EXPERT_MIRROR_FRAC=0.70 taskset -c 8-15,24-31 /home/claude-code/b2-dev.sh "$B2BIN" --model /weights/dsv4.1f --experts "$B2_EXPERTS" --listen 0.0.0.0:7431 --decode-max-b 1 --load-threads 8 --log-every 0 --paged >> "$B2LOG" 2>&1 < /dev/null &
for i in $(seq 1 600); do ss -ltn | grep -q ":7431 " && break; pgrep -f "$PAT" >/dev/null || { echo "daemon exited"; tail -n +$((N0+1)) "$B2LOG" | tail -8; exit 1; }; sleep 1; done
NEWP=$(pgrep -f "$PAT" | head -1)
if ss -ltn | grep -q ":7431 "; then echo "daemon listening after ${i}s (pid $NEWP $(readlink /proc/${NEWP:-0}/exe) md5 $(md5sum < /proc/${NEWP:-0}/exe | cut -c1-8))"; else echo "NOT LISTENING after 600s"; tail -n +$((N0+1)) "$B2LOG" | tail -8; exit 1; fi
tail -n +$((N0+1)) "$B2LOG" | grep -a -E "expert shard: [0-9]+ experts|nursery ON|knobs: resolved|IGNORED|unknown" | cut -c1-400
echo "after ready: MemAvailable $(awk '/^MemAvailable/ {printf "%.1f", $2*1024/1e9}' /proc/meminfo) GB"
if [ "$KMODE" = new ] && ! tail -n +$((N0+1)) "$B2LOG" | grep -a "knobs: resolved" | grep -q "V41_MOE_WMMA_F16=1(File)"; then echo "KNOB: V41_MOE_WMMA_F16=1 not resolved from the file"; exit 5; fi
exit 0
EOS
}

rollback() {
  say "ROLLBACK: box 1 down, box 2 -> $B2_OLD_MD5, box 1 up on $HUB_OLD_MD5"
  stop_all || { say "a hub survived SIGKILL: production DOWN; rerun '$0 rollback'"; mark "ROLLBACK: hub survived"; return 1; }
  b2_conn_free || true
  b2_restart "$B2_OLD_BIN" old || say "!! box 2 did not come back on $B2_OLD_MD5"
  install_hub "$HUB_BAK" "$HUB_OLD_MD5" || say "!! restore of $HUB_BAK failed"
  hub_up && say "rolled back: hub $HUB_OLD_MD5 + box 2 $B2_OLD_MD5"
  mark "ROLLBACK done"
}

# ---- preflight (hub up) ----
[ "$($SSH $B2 "md5sum < $B2_OLD_BIN" | cut -c1-8)" = "$B2_OLD_MD5" ] || { echo "box-2 rollback binary is not $B2_OLD_MD5"; exit 1; }
if [ "$MODE" = rollback ]; then [ -f "$HUB_BAK" ] || { echo "no $HUB_BAK"; exit 1; }; rollback; exit $?; fi
[ "$MODE" = deploy ] || { echo "usage: $0 [deploy|rollback]"; exit 2; }
[ "$(md5of "$PROD")" = "$HUB_OLD_MD5" ] || { echo "production hub is not $HUB_OLD_MD5"; exit 1; }
[ "$(md5of "$HUB_NEW")" = "$HUB_NEW_MD5" ] || { echo "new hub is not $HUB_NEW_MD5"; exit 1; }
[ "$($SSH $B2 "md5sum < $B2_NEW_BIN" | cut -c1-8)" = "$B2_NEW_MD5" ] || { echo "box-2 new binary is not $B2_NEW_MD5"; exit 1; }
$SSH $B2 'P=$(pgrep -f "^/home/claude-code/deepstrix/target-[a-z0-9]*/release/deepstrix-expertd" | head -1); [ -n "$P" ] && md5sum < /proc/$P/exe | cut -c1-8' | grep -qx "$B2_OLD_MD5" \
  || { echo "box 2 is not running $B2_OLD_MD5"; exit 1; }
[ -f "$HUB_BAK" ] || cp -p "$PROD" "$HUB_BAK" || exit 1
[ "$(md5of "$HUB_BAK")" = "$HUB_OLD_MD5" ] || { echo "hub backup is not $HUB_OLD_MD5"; exit 1; }
say "preflight OK: hub $HUB_OLD_MD5 -> $HUB_NEW_MD5, box 2 $B2_OLD_MD5 -> $B2_NEW_MD5"

bash "$WT/scratch/prefill_trace/wait_idle.sh" 20 600 || say "WARNING: still busy after 10 min; proceeding"
mark "start: hub $HUB_OLD_MD5 -> $HUB_NEW_MD5, box 2 $B2_OLD_MD5 -> $B2_NEW_MD5"
say "box 1 down"
stop_all || { say "a hub survived SIGKILL: NOT continuing"; mark "ABORT: hub survived"; exit 1; }
b2_conn_free || say "WARNING: box 2 still shows a connection"
export KI_STATE=/home/claude-code/.claude/jobs/b8eef48e/tmp/ki_state
# shellcheck disable=SC1091
source "$WT/scratch/kernel_ideas/_infra/env.sh"
for t in remote_experts_pin_loopback remote_experts_nursery_loopback; do
  BIN=$(ls -t "$WT"/target-gate/release/deps/$t-* | grep -v '\.d$' | head -1)
  L=/home/claude-code/gate_${t}_$(date -u +%Y%m%dT%H%M%SZ).log
  ( cd "$WT/crates/v4flash-kernels" && env -u V41_SUB -u V41_KNOBS_FILE timeout -k 20 600 "$BIN" --ignored --nocapture ) > "$L" 2>&1
  R=$?
  say "gate $t rc=$R: $(grep -aE 'test result' "$L" | tail -1)"
  if [ $R -ne 0 ] || ! grep -aqE 'test result: ok\. [1-9]' "$L"; then
    mark "gate $t FAILED ($L)"; say "gate failed: hub back on $HUB_OLD_MD5, box 2 untouched"
    bash "$WT/scratch/prefill_trace/launch_hub.sh" && say "hub up on $HUB_OLD_MD5"; exit 1
  fi
done
mark "gates pass"
say "box 2 down -> up on $B2_NEW_MD5"
b2_restart "$B2_NEW_BIN" new
R=$?
if [ $R -ne 0 ]; then say "box 2 new daemon failed (rc=$R)"; mark "box 2 new rc=$R"; rollback; exit 1; fi
say "box 1 up on $HUB_NEW_MD5"
if install_hub "$HUB_NEW" "$HUB_NEW_MD5" && hub_up; then
  sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | tail -800 | grep -aE 'remote experts:|prefill VRAM loan: prefill lane set built|UNTESTED|ERROR|panicked' | tail -5 | cut -c1-240
  $SSH $B2 "N=\$(grep -a -n '^=== ' $B2LOG | tail -1 | cut -d: -f1); tail -n +\${N:-1} $B2LOG | grep -a -E 'connection from|expert shard|nursery ON|ERROR|panicked' | tail -5 | cut -c1-240"
  say "OK: hub $HUB_NEW_MD5 + box 2 $B2_NEW_MD5 (last-unit evict-behind, knob OFF)"
  mark "done: hub $HUB_NEW_MD5 + box 2 $B2_NEW_MD5 (last-unit, prefill_budget_last unset)"
  exit 0
fi
say "new hub unhealthy"
mark "new hub unhealthy -> rollback"
rollback
exit 1
