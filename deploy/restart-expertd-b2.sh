#!/usr/bin/env bash
# Restart box 2's production expertd (run from box 1; ssh as claude-code@10.99.0.2).
# Checked in 2026-09-24 from ~/scratch-ms/restart_expertd_b2.sh.
# COLD POOL after this: experts refill at ~4-6 ms/miss, so decode is slow for a
# while. The hub reconnects on its next request.
set -u
ssh -o BatchMode=yes 10.99.0.2 '
OLD=$(pgrep -f "^/home/claude-code/deepstrix(-ooo)?/target-(b2|ooo)/release/deepstrix-expertd" | head -1)
echo "old daemon pid: ${OLD:-none}"
[ -n "$OLD" ] && { kill -TERM $OLD; for i in $(seq 1 20); do kill -0 $OLD 2>/dev/null || break; sleep 1; done; kill -0 $OLD 2>/dev/null && kill -KILL $OLD; sleep 2; }
# Wait for the old GTT pool to drop: teardown of 116 GB is asynchronous, launching
# on top of it OOM-kills the new daemon mid-preload, silently.
for i in $(seq 1 90); do g=$(cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | sort -n | tail -1); [ "${g:-0}" -lt 30000000000 ] && break; sleep 1; done
echo "gtt before launch: $((${g:-0}/1000000000)) GB after ${i}s"
cd /home/claude-code/deepstrix || exit 1
# Runtime knobs, reloaded by SIGUSR2 from this file (merge, merge_wait_us,
# miss_par, coalesce, mirror_frac). miss_par=1 is the value the code documents as
# its measured default: no gain at 4, each expert read is already 8 preads wide,
# and 4 concurrent misses put ~32 reads on the E100 where it does 3.2 GB/s
# against 4.5 at QD1-2. This script had been overriding it to 4. V41_B2_MISS_PAR
# below only sizes the pinned staging sets.
printf "merge=1\nmerge_wait_us=400\nmiss_par=1\ncoalesce=0\nmirror_frac=0.70\npark=1\n" > /home/claude-code/expertd-knobs.txt
nohup setsid env V41_B2_KNOBS=/home/claude-code/expertd-knobs.txt V41_B2_HITS_FIRST=1 V41_B2_MISS_PAR=4 V41_B2_PREFETCH_SETS=16 V41_EXPERT_MIRROR_DIR=/weights2/dsv4.1f V41_EXPERT_MIRROR_FRAC=0.70 taskset -c 8-15,24-31 /home/claude-code/b2-dev.sh /home/claude-code/deepstrix/target-b2/release/deepstrix-expertd --model /weights/dsv4.1f --experts L0-L39:230-383 --listen 0.0.0.0:7431 --decode-max-b 1 --load-threads 8 --log-every 0 --paged >> /home/claude-code/logs/expertd-s2sizing.log 2>&1 < /dev/null &
for i in $(seq 1 600); do ss -ltn | grep -q ":7431 " && break; sleep 1; done
ss -ltn | grep -q ":7431 " && echo "new daemon listening after ${i}s (pid $(pgrep -f "^/home/claude-code/deepstrix/target-b2/release/deepstrix-expertd" | head -1))" || { echo "NOT LISTENING after 600s"; tail -5 /home/claude-code/logs/expertd-s2sizing.log; exit 1; }
tail -3 /home/claude-code/logs/expertd-s2sizing.log | cut -c1-160'
