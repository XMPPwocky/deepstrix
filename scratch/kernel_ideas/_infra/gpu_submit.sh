#!/usr/bin/env bash
# gpu_submit.sh --dev igpu|dgpu --mb N --label <family>/<idea> [--timeout S] -- cmd args...
# Enqueues a GPU job for the per-device scheduler and prints a TICKET immediately. The command
# runs later from the CURRENT directory with only the requested GPU visible (HIP device 0).
# Collect with: gpu_wait.sh TICKET   (blocks, prints the job's output, exits with its rc)
# Status:       gpu_queue.sh         (who is running, who is waiting, per-family usage)
# Prefer ONE job that measures several shapes/variants over many tiny jobs (ROCm init is ~0.3 s
# and every job is a queue round-trip). Submit, keep working, collect.
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
        *) echo "gpu_submit: unknown arg $1" >&2; exit 64 ;;
    esac
done
[ -n "$DEV" ] && [ -n "$MB" ] && [ -n "$LABEL" ] && [ $# -gt 0 ] || {
    echo "usage: gpu_submit.sh --dev igpu|dgpu --mb N --label family/idea [--timeout S] -- cmd..." >&2; exit 64; }
case "$DEV" in igpu|dgpu) ;; *) echo "gpu_submit: --dev must be igpu or dgpu" >&2; exit 64 ;; esac
[[ "$LABEL" == */* ]] || { echo "gpu_submit: --label must be <family>/<idea>" >&2; exit 64; }
Q="$KI_STATE/queue/$DEV"
[ -s "$Q/sched.pid" ] && kill -0 "$(cat "$Q/sched.pid")" 2>/dev/null || {
    echo "gpu_submit: no scheduler running for $DEV — tell the orchestrator (do NOT run GPU work directly)" >&2; exit 69; }
fam=${LABEL%%/*}
seq=$(( $(cat "$KI_STATE/queue/seq" 2>/dev/null || echo 0) + 1 )); echo $seq > "$KI_STATE/queue/seq"
ticket="$(date +%s).$(printf '%06d' $seq).${fam}.${DEV}"
cmd=""; for a in "$@"; do cmd+=$(printf '%q ' "$a"); done
{
    echo "CWD=$PWD"; echo "MB=$MB"; echo "LABEL=$LABEL"; echo "TMO=$TMO"; echo "CMD"; echo "$cmd"
} > "$Q/pending/$ticket.tmp" && mv "$Q/pending/$ticket.tmp" "$Q/pending/$ticket"
echo "$ticket"
