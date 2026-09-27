#!/usr/bin/env bash
# submit.sh <tag> [families...] : one scheduler job per family (default: all), tickets to tickets.txt
set -eu
cd "$(dirname "$0")"
tag=$1; shift
fams=${*:-"F D E G C1 A"}
for f in $fams; do
  dev=dgpu; mb=128
  [ "$f" = A ] && { dev=igpu; mb=450; }
  t=$(bash ../../_infra/gpu_submit.sh --dev $dev --mb $mb --label round2/b_${f}_$tag --timeout 600 -- bash run_job.sh "$f" "$tag")
  echo "$t $tag job_$f" | tee -a tickets.txt
done
