#!/usr/bin/env bash
# Async layer-outer staging window (V41_LM_ENGRAM_ASYNC + V41_LM_SEED_ASYNC, both default on; Fable review r6/r7):
#   hub down (after idle) -> Gate A at 1024-row lanes in PRODUCTION's pager config (unified pool, no dense windows: the
#   dense-window config has KNOWN_BUGS #58), cases lazy (lazy seed/Engram rows) + restored (eager), ~15-20 min
#   PASS -> hub up on the new binary (env unchanged: the knobs default on)
#   else -> hub back up on production unchanged.  Box 2 stays up. Redirect stdout to a FILE.
set -u
WT=/home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe
W=/home/claude-code/scratch-ms
PROD=/home/claude-code/deepstrix/target-v41/release/deepstrix-server
NEW=$WT/target-gate/release/deepstrix-server
OLD_MD5=${OLD_MD5:-96f2c3f7}
NEW_MD5=${NEW_MD5:-c234a624}
BAK=$W/deepstrix-server.$OLD_MD5.bak
MARK=/home/claude-code/logs/evtrace/deploy_markers.log
HUBPAT='^/home/claude-code/deepstrix[^ ]*/release/deepstrix-server '
SUPPAT='^bash /home/claude-code/run_v41_server.sh'
say() { echo "$(date -u +%T) $*"; }
mark() { echo "$(date -u +%FT%T) stage-async window: $*" >> "$MARK"; }
md5of() { md5sum < "$1" | cut -c1-8; }
if [ -p /dev/stdout ]; then echo "stdout is a pipe: redirect to a file"; exit 2; fi
[ "$(md5of "$PROD")" = "$OLD_MD5" ] || { echo "production binary is not $OLD_MD5"; exit 1; }
[ "$(md5of "$NEW")" = "$NEW_MD5" ] || { echo "new binary is not $NEW_MD5"; exit 1; }
BIN=$(ls -t "$WT"/target-gate/release/deps/lm_prefill_bitexact-* | grep -v '\.d$' | head -1)
[ -x "$BIN" ] || { echo "no lm_prefill_bitexact binary"; exit 1; }
[ -f "$BAK" ] || cp -p "$PROD" "$BAK" || exit 1
stop_all() {
  bash "$W/stop_prod.sh" >/dev/null 2>&1
  if pgrep -f "$HUBPAT" >/dev/null || pgrep -f "$SUPPAT" >/dev/null; then pkill -KILL -f "$SUPPAT"; pkill -KILL -f "$HUBPAT"; sleep 5; fi
  ! pgrep -f "$HUBPAT" >/dev/null && ! pgrep -f "$SUPPAT" >/dev/null
}
install_bin() { cp -p "$1" "$PROD.new" && mv -f "$PROD.new" "$PROD" && [ "$(md5of "$PROD")" = "$2" ]; }

bash "$WT/scratch/prefill_trace/wait_idle.sh" 20 1800 || say "WARNING: still busy after 30 min; proceeding"
mark "start: hub $OLD_MD5 down for Gate A"
say "hub down"
stop_all || { say "a hub survived SIGKILL: NOT continuing"; mark "ABORT"; exit 1; }
export KI_STATE=/home/claude-code/.claude/jobs/b8eef48e/tmp/ki_state
# shellcheck disable=SC1091
source "$WT/scratch/kernel_ideas/_infra/env.sh"
G=/home/claude-code/gate_a_stageasync_$(date -u +%Y%m%dT%H%M%SZ).log
say "Gate A (lazy,restored @1024, prod pager) -> $G"
( cd "$WT/crates/v4flash-kernels" && env -u V41_SUB -u V41_PREFILL_SPLIT \
    HIP_VISIBLE_DEVICES=0,1 V41_PAGED_EXPERTS=1 V41_INDEX_K=1 V41_CANDIDATE_POOL=1 V41_LM_PREFILL=1 V41_LM_LAYER_OUTER=1 \
    LM_CASES=lazy,restored V41_REMOTE_ADDR=10.99.0.2:7431 V41_T2_CATCHALL=2 V41_PREFILL_LANE_ROWS=1024 \
    V41_PREFILL_UNIFIED_POOL=1 V41_PAGER_NO_DENSE=1 V41_LM_ENGRAM_ASYNC=1 V41_LM_SEED_ASYNC=1 \
    timeout -k 30 2700 "$BIN" --ignored --nocapture ) > "$G" 2>&1
R=$?
pkill -KILL -f '^/home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe/target-gate/release/deps/lm_prefill_bitexact' 2>/dev/null
P=0; [ $R = 0 ] && grep -aqE 'test result: ok\. [1-9][0-9]* passed; 0 failed' "$G" && P=1
say "Gate A rc=$R pass=$P"
sed 's/\x1b\[[0-9;]*m//g' "$G" | grep -aE 'BIT-IDENTICAL|NULL|differ|test result|panicked' | head -10 | cut -c1-220
mark "Gate A rc=$R pass=$P log=$G"

if [ $P = 1 ]; then
  if install_bin "$NEW" "$NEW_MD5" && bash "$WT/scratch/prefill_trace/launch_hub.sh" \
     && sed 's/\x1b\[[0-9;]*m//g' ~/logs/v41-server.log | tail -800 | grep -aq 'prefill VRAM loan: prefill lane set built'; then
    say "OK: hub up on $NEW_MD5 (async seed + Engram staging)"
    mark "done: hub $NEW_MD5 up"
    exit 0
  fi
  say "new hub unhealthy: rolling back to $OLD_MD5"
  stop_all
  install_bin "$BAK" "$OLD_MD5" || say "RESTORE OF $BAK FAILED"
  bash "$WT/scratch/prefill_trace/launch_hub.sh" && say "rolled back: hub up on $OLD_MD5"
  mark "ROLLED BACK to $OLD_MD5"
  exit 1
fi
say "Gate A not passed: hub back on $OLD_MD5 unchanged"
bash "$WT/scratch/prefill_trace/launch_hub.sh" && say "hub up on $OLD_MD5"
mark "Gate A not passed: hub back on $OLD_MD5"
exit 1
