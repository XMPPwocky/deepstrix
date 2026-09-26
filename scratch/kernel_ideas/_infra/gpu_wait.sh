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
    echo "[ORCHESTRATOR NOTICE] Report live: the moment you have a measured win (>=5%, correctness checked, >=2 runs), a notable loss, or a new high-value idea, append ONE line to $KI_ROOT/_infra/WINS.log (format in its header). It is relayed to the user in real time."
    cat "$Q/done/$t.out"
    last=$rc
done
exit $last
