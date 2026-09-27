#!/usr/bin/env bash
# submit_probe2.sh TAG : one probe2 run on the dGPU; prints the ticket
set -eu
cd "$(dirname "$0")"
bash ../../_infra/gpu_submit.sh --dev dgpu --mb 80 --label round2/a_probe2_$1 --timeout 300 -- ./probe2_gfx1201 probe_gfx1201.hsaco
