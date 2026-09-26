#!/usr/bin/env bash
# save_job.sh TICKET results/name.txt : copy a finished scheduler job's output into results/
set -u
cd "$(dirname "$0")"
source ../_infra/env.sh
cp "$KI_STATE/queue/dgpu/done/$1.out" "$2" && echo "saved $2 ($(wc -l < "$2") lines)"
