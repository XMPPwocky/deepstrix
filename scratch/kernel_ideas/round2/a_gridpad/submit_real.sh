#!/usr/bin/env bash
# submit_real.sh TAG "SECTION [b ...]" ["SECTION [b ...]" ...] : one gpad_real job per quoted
# section spec on the dGPU; appends "ticket TAG section" lines to tickets.txt and prints them.
set -eu
cd "$(dirname "$0")"
tag=$1; shift
for spec in "$@"; do
  sec=${spec%% *}
  # shellcheck disable=SC2086
  t=$(bash ../../_infra/gpu_submit.sh --dev dgpu --mb 125 --label round2/a_${sec}_$tag --timeout 300 -- ./gpad_real $spec)
  echo "$t $tag $sec" | tee -a tickets.txt
done
