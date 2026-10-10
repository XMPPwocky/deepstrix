#!/usr/bin/env bash
# P1/P2a GPU window (docs/v41/P1_P2A_GPU_RUNBOOK.md): hub down -> Gate A -> ONE hub restart onto the gate binary (PASS) or
# back onto production unchanged (FAIL / timeout). Box 2 stays up and warm throughout. Redirect stdout to a FILE (never pipe).
set -u
WT=/home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe
W=/home/claude-code/scratch-ms
PROD=/home/claude-code/deepstrix/target-v41/release/deepstrix-server
GATE=$WT/target-gate/release/deepstrix-server
PROD_MD5=c18af233
GATE_MD5=fc67a363
BAK=$W/deepstrix-server.$PROD_MD5.bak
ENVF=$W/hub_env_c_20261006.txt
MARK=/home/claude-code/logs/evtrace/deploy_markers.log
DUMP=/home/claude-code/logs/prefill_logits_20261010.bin
HUBPAT='^/home/claude-code/deepstrix[^ ]*/release/deepstrix-server '
SUPPAT='^bash /home/claude-code/run_v41_server.sh'
DRVPAT='^python3? [^ ]*/(ab_[a-z0-9_]+|sweep_[a-z0-9_]+|e2e_[a-z0-9_]+)\.py'
URL=http://127.0.0.1:18080
B2=10.43.0.95
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
GATE_TIMEOUT=${GATE_TIMEOUT:-2700}
say() { echo "$(date -u +%T) $*"; }
mark() { echo "$(date -u +%FT%T) p1p2a window: $*" >> "$MARK"; }
md5of() { md5sum < "$1" | cut -c1-8; }

if [ -p /dev/stdout ]; then echo "stdout is a pipe: redirect to a file"; exit 2; fi
pgrep -f -- "$DRVPAT" >/dev/null && { echo "an A/B driver is running; stop it first"; exit 1; }
[ "$(md5of "$PROD")" = $PROD_MD5 ] || { echo "production binary is not $PROD_MD5"; exit 1; }
[ "$(md5of "$GATE")" = $GATE_MD5 ] || { echo "gate binary is not $GATE_MD5"; exit 1; }
[ -f "$BAK" ] || cp -p "$PROD" "$BAK" || exit 1
[ "$(md5of "$BAK")" = $PROD_MD5 ] || { echo "backup $BAK is not $PROD_MD5"; exit 1; }

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
launch() {  # $1 = md5 label, $2 = note, rest = extra env
  local md5=$1 note=$2; shift 2
  mapfile -t ENVV < <(grep -v -E '^(V41_LOG_ATTACHED|V41_PICK_TRACE|V41_PREFILL_LOGITS_DUMP)=' "$ENVF")
  ENVV+=("V41_PICK_TRACE=/home/claude-code/logs/picks-sub-$(date -u +%Y%m%d-%H%M).trace" "$@")
  local RAW; RAW=$(python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))')
  echo "$(date -u +%FT%T) $RAW DEPLOY src=$PROD env=$ENVF md5=$md5 $note" >> "$MARK"
  (cd ~ && env -i "${ENVV[@]}" setsid nohup ~/run_v41_server.sh >/dev/null 2>&1 </dev/null &)
  sleep 3
  local i ok=0
  for i in $(seq 1 400); do
    if sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | tail -400 | grep -q 'listening on'; then ok=1; break; fi
    sleep 1
  done
  local ns nh; ns=$(pgrep -fc "$SUPPAT"); nh=$(pgrep -fc "$HUBPAT")
  say "hub listening=$ok after ${i}s; supervisors=$ns hubs=$nh"
  [ "$ok" = 1 ] && [ "$ns" = 1 ] && [ "$nh" = 1 ] || return 1
  curl -s -m 10 $URL/healthz; echo
  python3 - "$URL" <<'EOF'
import json, sys, urllib.request
url = sys.argv[1]
models = json.loads(urllib.request.urlopen(url + "/v1/models", timeout=30).read())
chat = [m["id"] for m in models["data"] if "embedding" not in m["id"]][0]
req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps({"model": chat, "max_tokens": 16, "temperature": 0,
      "messages": [{"role": "user", "content": "Reply with the single word: ready"}]}).encode(), headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=600).read())
print(f"chat: {(r['choices'][0]['message'].get('content') or '')[:60]!r}, usage {r.get('usage')}")
sys.exit(0 if r["usage"]["completion_tokens"] > 0 else 1)
EOF
}
install_bin() {  # $1 = source, $2 = expected md5
  cp -p "$1" "$PROD.new" && mv -f "$PROD.new" "$PROD" && [ "$(md5of "$PROD")" = "$2" ]
}

mark "start: hub $PROD_MD5 down for Gate A (box 2 stays up)"
say "hub down"
stop_all || { say "a hub survived SIGKILL: NOT continuing; relaunch by hand"; mark "ABORT: hub survived SIGKILL"; exit 1; }
b2_conn_free || say "WARNING: box 2 still shows a connection; running the gate anyway"

say "Gate A start"
GLOG=/home/claude-code/gate_a_window_$(date -u +%Y%m%dT%H%M%SZ).log
V41_REMOTE_ADDR=10.99.0.2:7431 timeout -k 30 "$GATE_TIMEOUT" bash "$WT/scratch/prefill_trace/run_gate_a.sh" lazy,restored > "$GLOG" 2>&1
GRC=$?
# the gate binary must be gone before the hub relaunches
pkill -KILL -f '^/home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe/target-gate/release/deps/lm_prefill_bitexact' 2>/dev/null
sleep 2
PASS=0
if [ $GRC = 0 ] && grep -aqE 'test result: ok\. [1-9][0-9]* passed; 0 failed' "$GLOG"; then PASS=1; fi
say "Gate A rc=$GRC pass=$PASS log=$GLOG"
grep -aE 'test result|PASS|FAIL|mismatch|panicked|diff' "$GLOG" | tail -20 | cut -c1-200
mark "Gate A rc=$GRC pass=$PASS log=$GLOG"
b2_conn_free || say "WARNING: box 2 still shows a connection; launching anyway (it drops stale ones)"

if [ $PASS = 1 ]; then
  install_bin "$GATE" $GATE_MD5 || { say "install of the gate binary failed"; install_bin "$BAK" $PROD_MD5; PASS=0; }
fi
if [ $PASS = 1 ]; then
  if launch $GATE_MD5 "p1p2a: Gate A PASS, hub -> $GATE_MD5 (b8323821), knobs off, +V41_PREFILL_LOGITS_DUMP" "V41_PREFILL_LOGITS_DUMP=$DUMP"; then
    say "OK: hub up on $GATE_MD5"
    sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | grep -aE 'multistream scheduler ON|remote experts:|ERROR|panicked' | tail -4 | cut -c1-200
    mark "done: hub $GATE_MD5 up"
    exit 0
  fi
  say "UNHEALTHY on $GATE_MD5: rolling back to $PROD_MD5"
  stop_all
  install_bin "$BAK" $PROD_MD5 || say "RESTORE OF $BAK FAILED"
  launch $PROD_MD5 "p1p2a ROLLBACK: $GATE_MD5 unhealthy" && say "rolled back: hub up on $PROD_MD5"
  mark "ROLLED BACK to $PROD_MD5"
  exit 1
fi
launch $PROD_MD5 "p1p2a: Gate A not passed, hub back on $PROD_MD5 unchanged" && say "hub back up on $PROD_MD5 (gate not passed)"
mark "gate not passed: hub back on $PROD_MD5"
exit 1
