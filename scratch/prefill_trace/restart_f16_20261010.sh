#!/usr/bin/env bash
# Hub restart 10-10 (owner OK): binary f9fa7f87 (main 316182e3: f16-WMMA arm wired, V41_MOE_WMMA_F16 default off) +
# env V41_PAGER_NO_DENSE=1 (idle 384-slot dense window -> LRU; prod config never reaches the dense path: unified pool +
# union + 5425 slots < 65536 group ids) + V41_PREFILL_F16_REPLIES=0 (box-2 prefill partials as f32: the live null test).
# V41_PREFILL_LOGITS_DUMP rides this launch only (not in the env file). Box 2 stays up and warm.
# Rollback on an unhealthy start: previous binary fc67a363 + the previous env file. Redirect stdout to a FILE.
set -u
W=/home/claude-code/scratch-ms
WT=/home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe
PROD=/home/claude-code/deepstrix/target-v41/release/deepstrix-server
NEW=$WT/target-gate/release/deepstrix-server
OLD_MD5=fc67a363
NEW_MD5=f9fa7f87
BAK=$W/deepstrix-server.$OLD_MD5.bak
ENVF=$W/hub_env_c_20261006.txt
ENVBAK=$ENVF.pre-f16-20261010
MARK=/home/claude-code/logs/evtrace/deploy_markers.log
DUMP=/home/claude-code/logs/prefill_logits_20261010.bin
HUBPAT='^/home/claude-code/deepstrix[^ ]*/release/deepstrix-server '
SUPPAT='^bash /home/claude-code/run_v41_server.sh'
DRVPAT='^python3? [^ ]*/(ab_[a-z0-9_]+|sweep_[a-z0-9_]+|e2e_[a-z0-9_]+|arms_[a-z0-9_]+)\.py'
URL=http://127.0.0.1:18080
B2=10.43.0.95
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
say() { echo "$(date -u +%T) $*"; }
mark() { echo "$(date -u +%FT%T) restart f16: $*" >> "$MARK"; }
md5of() { md5sum < "$1" | cut -c1-8; }

if [ -p /dev/stdout ]; then echo "stdout is a pipe: redirect to a file"; exit 2; fi
pgrep -f -- "$DRVPAT" >/dev/null && { echo "an A/B driver is running; stop it first"; exit 1; }
[ "$(md5of "$PROD")" = $OLD_MD5 ] || { echo "production binary is not $OLD_MD5"; exit 1; }
[ "$(md5of "$NEW")" = $NEW_MD5 ] || { echo "new binary is not $NEW_MD5"; exit 1; }
grep -qE '^V41_(PAGER_NO_DENSE|PREFILL_F16_REPLIES)=' "$ENVF" && { echo "env file already has NO_DENSE / F16_REPLIES"; exit 1; }
grep -q '^V41_PREFILL_UNIFIED_POOL=1$' "$ENVF" || { echo "NO_DENSE needs V41_PREFILL_UNIFIED_POOL=1"; exit 1; }
[ -f "$BAK" ] || cp -p "$PROD" "$BAK" || exit 1
[ "$(md5of "$BAK")" = $OLD_MD5 ] || { echo "backup $BAK is not $OLD_MD5"; exit 1; }
cp -p "$ENVF" "$ENVBAK" || exit 1

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

printf '%s\n' "# restart f16 10-10 (owner OK): dense window -> LRU; box-2 prefill partials as f32" \
  "V41_PAGER_NO_DENSE=1" "V41_PREFILL_F16_REPLIES=0" >> "$ENVF"
mark "start: hub $OLD_MD5 -> $NEW_MD5, env +NO_DENSE +F16_REPLIES=0"
say "hub down"
stop_all || { say "a hub survived SIGKILL: NOT continuing"; cp -p "$ENVBAK" "$ENVF"; mark "ABORT: hub survived SIGKILL"; exit 1; }
b2_conn_free || say "WARNING: box 2 still shows a connection; launching anyway (it drops stale ones)"
install_bin "$NEW" $NEW_MD5 || { say "install failed; restoring"; install_bin "$BAK" $OLD_MD5; cp -p "$ENVBAK" "$ENVF"; launch $OLD_MD5 "restart f16: install failed, back on $OLD_MD5" "V41_PREFILL_LOGITS_DUMP=$DUMP"; exit 1; }
if launch $NEW_MD5 "restart f16: $NEW_MD5 (main 316182e3) +NO_DENSE +F16_REPLIES=0" "V41_PREFILL_LOGITS_DUMP=$DUMP"; then
  say "OK: hub up on $NEW_MD5"
  sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | grep -aE 'expert pager:|multistream scheduler ON|remote experts:|UNTESTED FIDELITY|ERROR|panicked' | tail -6 | cut -c1-220
  mark "done: hub $NEW_MD5 up"
  exit 0
fi
say "UNHEALTHY on $NEW_MD5: rolling back to $OLD_MD5 + previous env"
stop_all
install_bin "$BAK" $OLD_MD5 || say "RESTORE OF $BAK FAILED"
cp -p "$ENVBAK" "$ENVF"
launch $OLD_MD5 "restart f16 ROLLBACK" "V41_PREFILL_LOGITS_DUMP=$DUMP" && say "rolled back: hub up on $OLD_MD5"
mark "ROLLED BACK to $OLD_MD5"
exit 1
