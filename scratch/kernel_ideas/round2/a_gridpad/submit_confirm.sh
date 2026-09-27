#!/usr/bin/env bash
# submit_confirm.sh TAG : confirmation runs for the four guarded pad candidates (production shapes)
set -eu
cd "$(dirname "$0")"
export GP_ALL=1
bash submit_real.sh "$1" "fp8 1 4 8 64 128 256 512" "fp4 1 4 8 16 32 64 128 512" "rope 1 4 8 16 32 64 512" "cast 1 4 8 16 64 128 256 512"
