#!/usr/bin/env bash
# kcc.sh <hipcc args...> — hipcc behind a 2-slot semaphore (box 1 has ~6 GB RAM free
# while the hub server runs; 8 agents compiling at once would swap it).
# Examples:
#   kcc.sh $KFLAGS_V41 --genco --offload-arch=gfx1151 $KERNELS_DIR/mxfp4_matvec.hip -o base_gfx1151.hsaco
#   kcc.sh -O3 --offload-arch=gfx1151 harness.cpp -o harness      (host+device harness)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SLOTS=${KCC_SLOTS:-2}
while :; do
    for i in $(seq 0 $((SLOTS - 1))); do
        exec {fd}>"$KI_STATE/locks/kcc.$i"
        if flock -n "$fd"; then
            timeout 900 nice -n 5 hipcc -I"$KI_ROOT/_infra" "$@"
            rc=$?
            exec {fd}>&-
            exit $rc
        fi
        exec {fd}>&-
    done
    sleep 1
done
