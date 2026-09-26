#!/usr/bin/env bash
# submit.sh <tag> — reviewer re-run of the engineer's exact gemv sweep (submit_r3.sh ticket 2) and the
# grouped part of ticket 3, plus the odd-shape correctness job. Tickets -> results/tickets_<tag>.txt
set -u
cd "$(dirname "$0")"
TAG=${1:-v1}
export C1_DIR="$PWD"
SUB="bash ../../../_infra/gpu_submit.sh"
: > "results/tickets_${TAG}.txt"
$SUB --dev dgpu --mb 1000 --label review/C1_dense_decode --timeout 300 -- \
    ./harness gemv qa 1 : gemv qa 2 : gemv qa 4 : gemv qa 5 : gemv qa 8 \
    : gemv qb 1 : gemv qb 4 : gemv qb 5 : gemv qb 8 \
    : gemv kv 1 : gemv kv 4 : gemv kv 5 \
    : gemv wob 1 : gemv wob 4 : gemv wob 5 : gemv wob 8 \
    : gemv gate 1 : gemv gate 4 : gemv gate 5 \
    : gemv down 1 : gemv down 4 : gemv down 5 : gemv down 8 \
    : gemv engram_full 1 : gemv engram_full 4 : gemv engram_full 5 \
    : gemv head_full 1 : gemv head_full 4 : gemv head_full 5 : gemv head_full 8 | tee -a "results/tickets_${TAG}.txt"
$SUB --dev dgpu --mb 250 --label review/C1_dense_decode --timeout 200 -- \
    ./harness grouped 1 : grouped 2 : grouped 4 : grouped 5 : grouped 8 | tee -a "results/tickets_${TAG}.txt"
