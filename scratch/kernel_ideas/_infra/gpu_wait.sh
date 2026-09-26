#!/usr/bin/env bash
# gpu_wait.sh TICKET [TICKET...] — block until each job is done; print its output; exit with the
# LAST job's rc (a non-zero rc of any job is echoed as a line "== TICKET rc=N").
set -u
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
last=0
for t in "$@"; do
    dev=${t##*.}
    Q="$KI_STATE/queue/$dev"
    while [ ! -s "$Q/done/$t.rc" ]; do
        [ -e "$Q/pending/$t" ] || [ -e "$Q/running/$t" ] || [ -s "$Q/done/$t.rc" ] || { echo "gpu_wait: unknown ticket $t" >&2; exit 65; }
        sleep 1
    done
    rc=$(cat "$Q/done/$t.rc")
    echo "== $t rc=$rc"
    cat "$Q/done/$t.out"
    last=$rc
done
exit $last
