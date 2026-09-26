#!/usr/bin/env bash
# Review jobs for the C1 shared_expert_fused_chain claim. Three separate process runs of the
# engineer's exact repro modes (shared 1/2/4/5/8) plus the untested production b=3, then two short
# correctness-only runs with NON-saturating activations (the engineer's inputs drive |gate|,|up|
# far past the swiglu clamp of 10, so mid is mostly exactly 0 or +-100).
set -u
cd "$(dirname "$0")"
export C1_DIR="$PWD"
SUB="bash ../../../_infra/gpu_submit.sh"
: > results/tickets.txt
for r in 1 2 3; do
    $SUB --dev dgpu --mb 250 --label review/C1_dense_decode --timeout 240 -- \
        ./harness shared 1 : shared 2 : shared 3 : shared 4 : shared 5 : shared 8 | tee -a results/tickets.txt
done
$SUB --dev dgpu --mb 250 --label review/C1_dense_decode --timeout 240 -- \
    env C1_XS_SCALE=0.01 C1_ROUNDS=5 ./harness shared 1 : shared 3 : shared 4 : shared 5 : shared 7 | tee -a results/tickets.txt
$SUB --dev dgpu --mb 250 --label review/C1_dense_decode --timeout 240 -- \
    env C1_XS_SCALE=0.003 C1_ROUNDS=5 ./harness shared 1 : shared 2 : shared 3 : shared 4 : shared 5 | tee -a results/tickets.txt
