#!/usr/bin/env bash
# Box-1 side of the 990 Pro IOMMU passthrough test (iommu_990_test.sh runs as ROOT on box 2): waits for the root
# script's request flag, then box 1 down (after a short idle wait), box 2's expertd down, "go"; waits for "done";
# box 2's expertd up (same binary / args / env / knob file), box 1 up. Redirect stdout to a FILE.
#   iommu_watch.sh [MAX_WAIT_H=12] > ~/iommu_watch.log 2>&1
set -u
MAXH=${1:-12}
WT=/home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe
W=/home/claude-code/scratch-ms
HUBPAT='^/home/claude-code/deepstrix[^ ]*/release/deepstrix-server '
SUPPAT='^bash /home/claude-code/run_v41_server.sh'
B2=10.43.0.95
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
D=/home/claude-code/iommu_test
B2BIN=/home/claude-code/deepstrix/target-b2lu1011/release/deepstrix-expertd
B2LOG=/home/claude-code/logs/expertd-lastunit-20261011.log
MARK=/home/claude-code/logs/evtrace/deploy_markers.log
say() { echo "$(date -u +%T) $*"; }
mark() { echo "$(date -u +%FT%T) iommu-test: $*" >> "$MARK"; }
if [ -p /dev/stdout ]; then echo "stdout is a pipe: redirect to a file"; exit 2; fi
stop_all() {
  bash "$W/stop_prod.sh" >/dev/null 2>&1
  if pgrep -f "$HUBPAT" >/dev/null || pgrep -f "$SUPPAT" >/dev/null; then pkill -KILL -f "$SUPPAT"; pkill -KILL -f "$HUBPAT"; sleep 5; fi
  ! pgrep -f "$HUBPAT" >/dev/null && ! pgrep -f "$SUPPAT" >/dev/null
}
b2_down() {
  $SSH $B2 'PAT="^/home/claude-code/deepstrix/target-[a-z0-9]*/release/deepstrix-expertd"; P=$(pgrep -f "$PAT" | head -1)
    [ -n "$P" ] || exit 0; kill -TERM $P; for i in $(seq 1 20); do kill -0 $P 2>/dev/null || break; sleep 1; done
    kill -0 $P 2>/dev/null && kill -KILL $P; sleep 2; ! pgrep -f "$PAT" >/dev/null'
}
b2_up() {
  $SSH $B2 "B2BIN=$B2BIN B2LOG=$B2LOG bash -s" <<'EOS'
set -u
PAT="^/home/claude-code/deepstrix/target-[a-z0-9]*/release/deepstrix-expertd"
mountpoint -q /weights2 && mountpoint -q /weights || { echo "/weights or /weights2 not mounted"; exit 1; }
pgrep -f "$PAT" >/dev/null && { echo "a daemon is already running"; exit 1; }
for i in $(seq 1 120); do g=$(cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | sort -n | tail -1); [ "${g:-0}" -lt 2000000000 ] && break; sleep 1; done
echo "=== $(date -u) launch $B2BIN (iommu test) ===" >> "$B2LOG"
cd /home/claude-code/deepstrix || exit 1
nohup setsid env V41_B2_MODE_EVICT=1 V41_B2_PREFILL_STAGE=0 V41_EVTRACE_DIR=/home/claude-code/logs/evtrace V41_B2_KNOBS=/home/claude-code/expertd-knobs.txt V41_B2_HITS_FIRST=1 V41_B2_MISS_PAR=4 V41_B2_PREFETCH_SETS=16 V41_EXPERT_MIRROR_DIR=/weights2/dsv4.1f V41_EXPERT_MIRROR_FRAC=0.70 taskset -c 8-15,24-31 /home/claude-code/b2-dev.sh "$B2BIN" --model /weights/dsv4.1f --experts L0-L39:265-383 --listen 0.0.0.0:7431 --decode-max-b 1 --load-threads 8 --log-every 0 --paged >> "$B2LOG" 2>&1 < /dev/null &
for i in $(seq 1 600); do ss -ltn | grep -q ":7431 " && break; pgrep -f "$PAT" >/dev/null || { echo "daemon exited"; tail -5 "$B2LOG"; exit 1; }; sleep 1; done
ss -ltn | grep -q ":7431 " && echo "expertd listening after ${i}s" || { echo "NOT LISTENING"; exit 1; }
EOS
}

say "waiting up to ${MAXH} h for the root script's request ($B2:$D/req)"
for i in $(seq 1 $((MAXH * 360))); do $SSH $B2 "test -f $D/req" && break; sleep 10; done
$SSH $B2 "test -f $D/req" || { say "no request"; exit 1; }
say "request seen: $($SSH $B2 "cat $D/req")"
bash "$WT/scratch/prefill_trace/wait_idle.sh" 20 120 || say "WARNING: still busy after 2 min; proceeding"
mark "start: hub + box 2 down for the 990 Pro IOMMU test"
say "box 1 down"
stop_all || { say "a hub survived SIGKILL: NOT continuing (no go)"; mark "ABORT"; exit 1; }
say "box 2 expertd down"
if ! b2_down; then say "expertd did not stop: no go; hub back up"; bash "$WT/scratch/prefill_trace/launch_hub.sh"; exit 1; fi
$SSH $B2 "touch $D/go"
say "go given; waiting for the root script (up to 20 min)"
for i in $(seq 1 1200); do $SSH $B2 "test -f $D/done" && break; sleep 1; done
say "root script: $($SSH $B2 "cat $D/done 2>/dev/null || echo NO-DONE")"
$SSH $B2 "cat $D/root.log 2>/dev/null | tail -12"
say "box 2 expertd up"
b2_up || say "!! expertd did not come up (check /weights2 on box 2)"
say "box 1 up"
bash "$WT/scratch/prefill_trace/launch_hub.sh" && say "hub up"
mark "done: $($SSH $B2 "cat $D/done 2>/dev/null")"
$SSH $B2 "rm -f $D/req $D/go"
