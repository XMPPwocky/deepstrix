#!/usr/bin/env bash
# submit.sh <job: repro|review> <tag...>  -> enqueues one scheduler job per tag, prints tickets to review/tickets_<job>.txt
set -u
cd "$(dirname "$0")/../.."
job=$1; shift
: > "review/tickets_${job}.txt"
for t in "$@"; do
    bash ../_infra/gpu_submit.sh --dev dgpu --mb 400 --label review/F_mhc_glue --timeout 300 -- bash "review/jobs/${job}.sh" "$t" | tee -a "review/tickets_${job}.txt"
done
