#!/usr/bin/env bash
# gpu_run.sh --dev igpu|dgpu --mb <device MB you will allocate> --label <family/idea> [--timeout S] -- cmd args...
#
# EVERY GPU-touching command (benchmarks, rocprofv3, correctness runs) goes through this.
# Box 1 is the LIVE production hub. This wrapper:
#   * serialises runs per device (flock), so agents never contaminate each other's timings;
#   * holds all runs while another job's clock A/B (clock_ab.sh) is measuring production,
#     or while $KI_STATE/HOLD exists (the orchestrator's emergency pause);
#   * exposes ONLY the requested GPU to the process (ROCR_VISIBLE_DEVICES) — inside the
#     harness the device is always HIP device 0, and for ATT use --att-gpu-index 0;
#   * refuses dGPU runs that would eat into the hub's VRAM (the hub leaves ~440 MB; TTM
#     silently evicts the server to GTT = 100x slower production) and iGPU runs that would
#     push host RAM into swap;
#   * logs every run to _infra/runs.tsv with device busy% before/after (production load).
set -u
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
DEV="" MB="" LABEL="" TMO=300
while [ $# -gt 0 ]; do
    case "$1" in
        --dev) DEV=$2; shift 2 ;;
        --mb) MB=$2; shift 2 ;;
        --label) LABEL=$2; shift 2 ;;
        --timeout) TMO=$2; shift 2 ;;
        --) shift; break ;;
        *) echo "gpu_run: unknown arg $1" >&2; exit 64 ;;
    esac
done
[ -n "$DEV" ] && [ -n "$MB" ] && [ -n "$LABEL" ] && [ $# -gt 0 ] || {
    echo "usage: gpu_run.sh --dev igpu|dgpu --mb N --label L [--timeout S] -- cmd..." >&2; exit 64; }
case "$DEV" in
    dgpu) ROCR=0; CARD=/sys/class/drm/card1/device; MAXMB=256 ;;
    igpu) ROCR=1; CARD=/sys/class/drm/card2/device; MAXMB=3072 ;;
    *) echo "gpu_run: --dev must be igpu or dgpu" >&2; exit 64 ;;
esac
[ "$MB" -le "$MAXMB" ] || { echo "gpu_run: --mb $MB exceeds the $DEV cap of $MAXMB MB" >&2; exit 65; }
[ "$TMO" -le 900 ] || TMO=900

waited=0
while pgrep -f '[c]lock_ab\.sh' >/dev/null || [ -e "$KI_STATE/HOLD" ]; do
    [ $waited -eq 0 ] && echo "gpu_run: holding (another job's clock A/B is measuring production, or HOLD set) ..." | tee /dev/stderr
    waited=1; sleep 20
done

exec {lk}>"$KI_STATE/locks/gpu.$DEV"
flock -w 5400 "$lk" || { echo "gpu_run: could not get the $DEV lock in 90 min" >&2; exit 75; }

if [ "$DEV" = dgpu ]; then
    tot=$(( $(cat $CARD/mem_info_vram_total) / 1048576 ))
    used=$(( $(cat $CARD/mem_info_vram_used) / 1048576 ))
    free=$(( tot - used ))
    # ~120 MB for the HIP context/queues + a 150 MB margin for the hub.
    if [ $(( free - MB - 270 )) -lt 0 ]; then
        echo "gpu_run: REFUSED dgpu run: VRAM free ${free} MB < ${MB} MB + 270 MB margin (hub owns the rest)" >&2
        exit 66
    fi
else
    avail=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 ))
    if [ $(( avail - MB - 3072 )) -lt 0 ]; then
        echo "gpu_run: REFUSED igpu run: MemAvailable ${avail} MB < ${MB} MB + 3 GB margin" >&2
        exit 66
    fi
fi

b0=$(cat $CARD/gpu_busy_percent 2>/dev/null || echo NA)
t0=$(date +%s.%N)
ROCR_VISIBLE_DEVICES=$ROCR timeout --kill-after=10 "$TMO" "$@"
rc=$?
t1=$(date +%s.%N)
b1=$(cat $CARD/gpu_busy_percent 2>/dev/null || echo NA)
printf '%s\t%s\t%s\t%sMB\trc=%s\t%.1fs\tbusy_before=%s\tbusy_after=%s\t%s\n' \
    "$(date -u +%FT%TZ)" "$LABEL" "$DEV" "$MB" "$rc" "$(awk -v a="$t0" -v b="$t1" 'BEGIN{print b-a}')" "$b0" "$b1" "$*" \
    >> "$KI_ROOT/_infra/runs.tsv"
[ $rc -eq 124 ] && echo "gpu_run: TIMED OUT after ${TMO}s" >&2
exit $rc
