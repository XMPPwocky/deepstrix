#!/usr/bin/env bash
# Reviewer re-run of the engineer's EXACT lanes repro (their binary, their hsacos, their dir, the
# 'lanes' commands of submit_r3.sh's 3rd ticket) x3, plus untested shapes with the same binary.
set -u
REV="$(cd "$(dirname "$0")" && pwd)"
C1="$(cd "$REV/../.." && pwd)"
INFRA="$(cd "$REV/../../../_infra" && pwd)"
SUB="bash $INFRA/gpu_submit.sh --dev dgpu --mb 250 --timeout 240 --label review/C1_dense_decode"
: > "$REV/tickets_repro.txt"
sub() { local name=$1; shift; local t; t=$("$@" 2>&1 | tail -1); echo "$name $t" >> "$REV/tickets_repro.txt"; echo "$name -> $t"; }
cd "$C1"
export C1_DIR="$C1"
LANES="lanes qa 4 4 : lanes qa 5 5 : lanes wob 4 4 : lanes gate 3 3 : lanes down 4 4 : lanes qb 4 4"
for r in 1 2 3; do
    sub "eng_lanes_r$r" $SUB -- ./harness $LANES
done
# untested shapes with the engineer's binary: kv, odd split 4+3 (tB7 absent in their hsaco -> base only), 1+1, gate 4+4, wob 3+3, wob 4+3
sub "eng_extra_r1" $SUB -- ./harness lanes kv 4 4 : lanes qa 4 3 : lanes qa 1 1 : lanes gate 4 4 : lanes wob 3 3 : lanes wob 4 3
cat "$REV/tickets_repro.txt"
