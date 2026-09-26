#!/usr/bin/env bash
# gpu_sched.sh <igpu|dgpu> — the per-device GPU job scheduler (ONE instance per device, started by
# the orchestrator: `nohup bash _infra/gpu_sched.sh igpu &`). Agents never run GPU work directly;
# they call gpu_submit.sh (async, returns a ticket) or gpu_run.sh (= submit + wait).
#
# Why a scheduler instead of a lock: 7 agents share two GPUs. A bare lock makes every agent block a
# shell for an unpredictable time and lets one family monopolise a device with a long run. This runner
#   * executes exactly ONE job per device at a time (no timing contamination);
#   * picks the next job FAIRLY: among families with pending jobs, the one with the least device
#     time in the last $FAIR_WINDOW seconds goes first (FIFO within a family);
#   * enforces the safety policy centrally: memory guards, per-job timeout (<= $MAX_JOB s),
#     no rocprofv3 ATT on the iGPU (hung the whole box 2026-09-26 04:01 UTC), HOLD file;
#   * records every job in _infra/runs.tsv and keeps stdout/stderr per ticket so agents can
#     submit several jobs, keep working (write the next candidate, read ISA) and collect later.
set -u
DEV=$1
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
Q="$KI_STATE/queue/$DEV"
mkdir -p "$Q/pending" "$Q/running" "$Q/done"
FAIR_WINDOW=${FAIR_WINDOW:-600}
MAX_JOB=${MAX_JOB:-900}
case "$DEV" in
    dgpu) ROCR=0; CARD=/sys/class/drm/card1/device ;;
    igpu) ROCR=1; CARD=/sys/class/drm/card2/device ;;
    *) echo "gpu_sched: dev must be igpu|dgpu" >&2; exit 64 ;;
esac
exec {sl}>"$KI_STATE/locks/sched.$DEV"
flock -n "$sl" || { echo "gpu_sched: another $DEV scheduler is running" >&2; exit 1; }
echo $$ > "$Q/sched.pid"
echo "gpu_sched[$DEV] up pid $$ $(date -u +%FT%TZ)"

# device seconds per family within the fairness window, from runs.tsv
family_usage() {  # $1 family -> seconds
    local now; now=$(date +%s)
    TZ=UTC awk -F'\t' -v fam="$1" -v dev="$DEV" -v now="$now" -v win="$FAIR_WINDOW" '
        $3==dev && index($2, fam"/")==1 {
            ts=$1; gsub(/[-T:Z]/, " ", ts); t=mktime(ts)
            if (now - t < win) { s=$6; sub(/s$/,"",s); tot+=s }
        } END { printf "%d", tot+0 }' "$KI_ROOT/_infra/runs.tsv" 2>/dev/null
}

pick_job() {  # prints the pending job file to run next, or nothing
    local best="" bestu=999999 f fam u
    declare -A seen
    for f in $(ls -1 "$Q/pending" 2>/dev/null | sort); do  # FIFO order by ticket
        fam=$(cut -d. -f3 <<<"$f")                         # ticket = <epoch>.<seq>.<family>.<dev>
        [ -n "$fam" ] || { mv "$Q/pending/$f" "$Q/done/$f" 2>/dev/null; echo "malformed ticket" > "$Q/done/$f.out"; echo 65 > "$Q/done/$f.rc"; continue; }
        [ -n "${seen[$fam]:-}" ] && continue
        seen[$fam]=1
        u=$(family_usage "$fam")
        if [ "$u" -lt "$bestu" ]; then bestu=$u; best=$f; fi
    done
    [ -n "$best" ] && echo "$best"
}

while :; do
    if [ -e "$KI_STATE/STOP_SCHED" ]; then echo "gpu_sched[$DEV] stop requested"; exit 0; fi
    if pgrep -f '[c]lock_ab\.sh' >/dev/null || [ -e "$KI_STATE/HOLD" ]; then sleep 10; continue; fi
    job=$(pick_job)
    if [ -z "$job" ]; then sleep 1; continue; fi
    mv "$Q/pending/$job" "$Q/running/$job" 2>/dev/null || continue
    # job file: KEY=VALUE lines (CWD, MB, LABEL, TMO) then a line "CMD" followed by the command
    CWD=$(sed -n 's/^CWD=//p' "$Q/running/$job"); MB=$(sed -n 's/^MB=//p' "$Q/running/$job")
    LABEL=$(sed -n 's/^LABEL=//p' "$Q/running/$job"); TMO=$(sed -n 's/^TMO=//p' "$Q/running/$job")
    CMD=$(sed -n '/^CMD$/,$p' "$Q/running/$job" | tail -n +2)
    out="$Q/done/$job.out"; rcfile="$Q/done/$job.rc"
    [ "$TMO" -le "$MAX_JOB" ] || TMO=$MAX_JOB
    refuse=""
    if [ "$DEV" = igpu ] && grep -qE -- '--att( |$)|prof\.sh +att' <<<"$CMD"; then
        refuse="REFUSED: rocprofv3 ATT on the iGPU hangs the whole machine (2026-09-26). Use PMC/ISA/ablations."
    fi
    if [ -z "$refuse" ] && [ "$DEV" = dgpu ]; then
        tot=$(( $(cat $CARD/mem_info_vram_total) / 1048576 )); used=$(( $(cat $CARD/mem_info_vram_used) / 1048576 ))
        [ $(( tot - used - MB - 270 )) -lt 0 ] && refuse="REFUSED: dGPU VRAM free $((tot-used)) MB < $MB MB + 270 MB margin (production hub may own it)"
    elif [ -z "$refuse" ]; then
        avail=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 ))
        [ $(( avail - MB - 3072 )) -lt 0 ] && refuse="REFUSED: MemAvailable $avail MB < $MB MB + 3 GB margin"
    fi
    b0=$(cat $CARD/gpu_busy_percent 2>/dev/null || echo NA)
    t0=$(date +%s.%N)
    if [ -n "$refuse" ]; then
        echo "$refuse" > "$out"; rc=66
    else
        # rocprofv3 sessions (ATT/PMC) are serialised MACHINE-WIDE: two concurrent thread-trace
        # sessions (dGPU + iGPU) were in flight when the box hung on 2026-09-26 04:01 UTC.
        if grep -qE 'rocprofv3|prof\.sh' <<<"$CMD"; then
            exec {pl}>"$KI_STATE/locks/rocprof.global"; flock "$pl"
        fi
        ( cd "$CWD" && ROCR_VISIBLE_DEVICES=$ROCR timeout --kill-after=10 "$TMO" bash -c "$CMD" ) > "$out" 2>&1
        rc=$?
        [ -n "${pl:-}" ] && { exec {pl}>&-; unset pl; }
        [ $rc -eq 124 ] && echo "gpu_sched: TIMED OUT after ${TMO}s" >> "$out"
    fi
    t1=$(date +%s.%N)
    b1=$(cat $CARD/gpu_busy_percent 2>/dev/null || echo NA)
    printf '%s\t%s\t%s\t%sMB\trc=%s\t%.1fs\tbusy_before=%s\tbusy_after=%s\t%s\n' \
        "$(date -u +%FT%TZ)" "$LABEL" "$DEV" "$MB" "$rc" "$(awk -v a="$t0" -v b="$t1" 'BEGIN{print b-a}')" \
        "$b0" "$b1" "$(head -c 200 <<<"$CMD" | tr '\n' ' ')" >> "$KI_ROOT/_infra/runs.tsv"
    echo "$rc" > "$rcfile"
    mv "$Q/running/$job" "$Q/done/$job"
done
