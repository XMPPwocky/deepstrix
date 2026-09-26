#!/usr/bin/env bash
# gpu_queue.sh — scheduler status: running job, pending jobs per family, device seconds per family
# in the fairness window. Check this BEFORE deciding whether to wait or to do CPU work first.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
for dev in dgpu igpu; do
    Q="$KI_STATE/queue/$dev"
    up=no; [ -s "$Q/sched.pid" ] && kill -0 "$(cat "$Q/sched.pid")" 2>/dev/null && up=yes
    echo "== $dev  scheduler=$up  hold=$([ -e "$KI_STATE/HOLD" ] && echo YES || echo no)"
    for f in "$Q"/running/*; do [ -e "$f" ] && echo "   RUNNING $(basename "$f")  $(sed -n 's/^LABEL=//p' "$f")  since $(date -u -r "$f" +%T)"; done
    n=$(ls -1 "$Q/pending" 2>/dev/null | wc -l)
    echo "   pending: $n"
    ls -1 "$Q/pending" 2>/dev/null | awk -F. '{c[$3]++} END {for (k in c) printf "     %-18s %d\n", k, c[k]}'
    now=$(date +%s)
    TZ=UTC awk -F'\t' -v dev="$dev" -v now="$now" '
        $3==dev { ts=$1; gsub(/[-T:Z]/, " ", ts); t=mktime(ts)
                  if (now-t < 600) { split($2,a,"/"); s=$6; sub(/s$/,"",s); u[a[1]]+=s } }
        END { for (k in u) printf "     used last 10 min: %-18s %.0f s\n", k, u[k] }' "$KI_ROOT/_infra/runs.tsv" 2>/dev/null
done
