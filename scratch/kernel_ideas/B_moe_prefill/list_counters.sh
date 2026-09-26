#!/usr/bin/env bash
# List rocprofv3 PMC counters (run under gpu_run.sh --dev igpu).
cd "$(dirname "$0")"
source ../_infra/env.sh
"$ROCPROFV3" -L 2>/dev/null
