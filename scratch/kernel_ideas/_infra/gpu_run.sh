#!/usr/bin/env bash
# gpu_run.sh --dev igpu|dgpu --mb <device MB you will allocate> --label <family>/<idea> [--timeout S] -- cmd args...
#
# Synchronous convenience wrapper = gpu_submit.sh + gpu_wait.sh: enqueues the command on the
# per-device scheduler (gpu_sched.sh), waits, prints the job's output, exits with its rc.
# Inside the command the requested GPU is HIP device 0. See gpu_submit.sh for the async form
# (submit several jobs, keep working, collect with gpu_wait.sh) and gpu_queue.sh for status.
# Every GPU-touching command goes through the scheduler: it serialises the device, enforces the
# memory guards, per-job timeout, the ATT policy, and fairness across families.
set -u
HERE="$(dirname "${BASH_SOURCE[0]}")"
t=$(bash "$HERE/gpu_submit.sh" "$@") || exit $?
exec bash "$HERE/gpu_wait.sh" "$t"
