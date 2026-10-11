#!/usr/bin/env bash
# Prefill VRAM loan GPU window (docs/v41/PREFILL_VRAM_LOAN_DESIGN.md; Fable code review r5 APPROVE):
#   hub down (after idle) -> 1. tests/prefill_loan_gpu (lend/overwrite/end bit-identity, ~1 min)
#                          -> 2. Gate A at 1024-row lanes (lm_prefill_bitexact lazy,restored, ~17-20 min)
#   both PASS -> hub up on the loan binary + env (V41_PREFILL_VRAM_LOAN=1, V41_PREFILL_LANE_ROWS=1024, V41_MS_CHUNK_ROWS[_IDLE]=2048)
#   else      -> hub back up on production unchanged.  Box 2 stays up. Redirect stdout to a FILE.
set -u
WT=/home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe
W=/home/claude-code/scratch-ms
PROD=/home/claude-code/deepstrix/target-v41/release/deepstrix-server
NEW=$WT/target-gate/release/deepstrix-server
OLD_MD5=${OLD_MD5:-99003020}
NEW_MD5=${NEW_MD5:-32e99f5f}
BAK=$W/deepstrix-server.$OLD_MD5.bak
ENVF=$W/hub_env_c_20261006.txt
ENVBAK=$ENVF.pre-loan-20261011
MARK=/home/claude-code/logs/evtrace/deploy_markers.log
HUBPAT='^/home/claude-code/deepstrix[^ ]*/release/deepstrix-server '
SUPPAT='^bash /home/claude-code/run_v41_server.sh'
B2=10.43.0.95
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
say() { echo "$(date -u +%T) $*"; }
mark() { echo "$(date -u +%FT%T) loan window: $*" >> "$MARK"; }
md5of() { md5sum < "$1" | cut -c1-8; }
if [ -p /dev/stdout ]; then echo "stdout is a pipe: redirect to a file"; exit 2; fi
[ "$(md5of "$PROD")" = "$OLD_MD5" ] || { echo "production binary is not $OLD_MD5"; exit 1; }
[ "$(md5of "$NEW")" = "$NEW_MD5" ] || { echo "new binary is not $NEW_MD5"; exit 1; }
grep -q '^V41_PREFILL_VRAM_LOAN=' "$ENVF" && { echo "env file already has the loan"; exit 1; }
LOANBIN=$(ls -t "$WT"/target-gate/release/deps/prefill_loan_gpu-* | grep -v '\.d$' | head -1)
[ -x "$LOANBIN" ] || { echo "no prefill_loan_gpu test binary"; exit 1; }
[ -f "$BAK" ] || cp -p "$PROD" "$BAK" || exit 1
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

say "waiting for the hub to go idle"
bash "$WT/scratch/prefill_trace/wait_idle.sh" 20 1800 || say "WARNING: still busy after 30 min; proceeding"
mark "start: hub $OLD_MD5 down for the loan GPU gates"
say "hub down"
stop_all || { say "a hub survived SIGKILL: NOT continuing"; mark "ABORT"; exit 1; }
b2_conn_free || say "WARNING: box 2 still shows a connection"

# 1. loan round trip
export KI_STATE=/home/claude-code/.claude/jobs/b8eef48e/tmp/ki_state
# shellcheck disable=SC1091
source "$WT/scratch/kernel_ideas/_infra/env.sh"
L1=/home/claude-code/loan_gpu_$(date -u +%Y%m%dT%H%M%SZ).log
say "1. prefill_loan_gpu -> $L1"
( cd "$WT/crates/v4flash-kernels" && HIP_VISIBLE_DEVICES=0,1 V41_PAGED_EXPERTS=1 V41_INDEX_K=1 V41_CANDIDATE_POOL=1 \
  V41_REMOTE_ADDR=10.99.0.2:7431 V41_PREFILL_LANE_ROWS=1024 timeout -k 30 900 "$LOANBIN" --ignored --nocapture ) > "$L1" 2>&1
R1=$?
P1=0; [ $R1 = 0 ] && grep -aq 'PASS: live decode KV bit-identical' "$L1" && P1=1
say "1. rc=$R1 pass=$P1"; grep -aE 'set |lend:|PASS|panicked|Error|error' "$L1" | head -8 | cut -c1-220
mark "1. prefill_loan_gpu rc=$R1 pass=$P1 log=$L1"

# 2. Gate A at 1024-row lanes
P2=0
if [ $P1 = 1 ]; then
  say "2. Gate A at 1024-row lanes (lazy,restored)"
  G=/home/claude-code/gate_a_loan_$(date -u +%Y%m%dT%H%M%SZ).log
  V41_REMOTE_ADDR=10.99.0.2:7431 V41_PREFILL_LANE_ROWS=1024 timeout -k 30 2700 bash "$WT/scratch/prefill_trace/run_gate_a.sh" lazy,restored > "$G" 2>&1
  R2=$?
  pkill -KILL -f '^/home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe/target-gate/release/deps/lm_prefill_bitexact' 2>/dev/null
  [ $R2 = 0 ] && grep -aqE 'test result: ok\. [1-9][0-9]* passed; 0 failed' "$G" && P2=1
  say "2. rc=$R2 pass=$P2"; grep -aE 'BIT-IDENTICAL|NULL|differ|test result|panicked' "$G" | head -8 | cut -c1-220
  mark "2. Gate A @1024 rc=$R2 pass=$P2 log=$G"
fi
b2_conn_free || say "WARNING: box 2 still shows a connection"

if [ $P1 = 1 ] && [ $P2 = 1 ]; then
  cp -p "$ENVF" "$ENVBAK" || exit 1
  printf '%s\n' "V41_PREFILL_VRAM_LOAN=1" "V41_PREFILL_LANE_ROWS=1024" "V41_MS_CHUNK_ROWS=2048" "V41_MS_CHUNK_ROWS_IDLE=2048" >> "$ENVF"
  if install_bin "$NEW" "$NEW_MD5" && bash "$WT/scratch/prefill_trace/launch_hub.sh" \
     && sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | tail -600 | grep -aq 'prefill VRAM loan: prefill lane set built'; then
    sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | grep -a 'prefill VRAM loan: prefill lane set built' | tail -1 | cut -c1-400
    say "OK: hub up on $NEW_MD5 with the prefill VRAM loan"
    mark "done: hub $NEW_MD5 up WITH the loan"
    exit 0
  fi
  say "loan hub unhealthy: rolling back to $OLD_MD5 + previous env"
  stop_all
  cp -p "$ENVBAK" "$ENVF"
  install_bin "$BAK" "$OLD_MD5" || say "RESTORE OF $BAK FAILED"
  bash "$WT/scratch/prefill_trace/launch_hub.sh" && say "rolled back: hub up on $OLD_MD5"
  mark "ROLLED BACK to $OLD_MD5"
  exit 1
fi
say "gates not passed: hub back on $OLD_MD5 unchanged"
bash "$WT/scratch/prefill_trace/launch_hub.sh" && say "hub up on $OLD_MD5"
mark "gates not passed (1=$P1 2=$P2): hub back on $OLD_MD5"
exit 1
