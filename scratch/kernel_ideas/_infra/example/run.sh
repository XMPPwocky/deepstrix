#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
ARCH=${1:-gfx1151}
DEV=igpu; [ "$ARCH" = gfx1201 ] && DEV=dgpu
../gpu_run.sh --dev $DEV --mb 16 --label _infra/example --timeout 60 -- ./harness_$ARCH $ARCH .
