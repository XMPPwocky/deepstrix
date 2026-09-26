#!/usr/bin/env bash
# Round 2: (a) rocprofv3 kernel trace = hardware per-dispatch durations of the quantize at grid vs grid+pad
#          (b) GC_MIX: a producer kernel between consecutive quantizes (production-like), 2 runs
set -u
REV="$(cd "$(dirname "$0")" && pwd)"
INFRA="$(cd "$REV/../../_infra" && pwd)"
SUB="bash $INFRA/gpu_submit.sh --dev dgpu --mb 100 --timeout 240 --label review/C1_dense_decode"
: > "$REV/tickets2.txt"
sub() { local name=$1; shift; local t; t=$("$@" 2>&1 | tail -1); echo "$name $t" >> "$REV/tickets2.txt"; echo "$name -> $t"; }
cd "$REV"
export C1_DIR="$REV"
SH="32768 1 32768 2 32768 3 32768 4 65568 1"
sub "trace_r1" $SUB -- env GC_ROUNDS=5 bash "$INFRA/prof.sh" trace "$REV/trace_r1" -- ./gpad_check $SH
sub "mix_r1" $SUB -- env GC_MIX=1 ./gpad_check $SH
sub "mix_r2" $SUB -- env GC_MIX=1 ./gpad_check $SH
cat "$REV/tickets2.txt"
