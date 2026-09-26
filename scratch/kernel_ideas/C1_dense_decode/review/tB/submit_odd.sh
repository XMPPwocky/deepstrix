#!/usr/bin/env bash
# submit_odd.sh — reviewer odd-shape correctness job (tails, odd b, adversarial inputs, sentinels).
set -u
cd "$(dirname "$0")"
bash ../../../_infra/gpu_submit.sh --dev dgpu --mb 400 --label review/C1_dense_decode --timeout 200 -- ./odd "$PWD" | tee -a results/tickets_odd.txt
