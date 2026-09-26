#!/usr/bin/env bash
# Review of C1_dense_decode/quantize_grid_pad. Submits (tickets -> review/tickets.txt):
#  (a) the engineer's EXACT repro (submit_quant_grid.sh shapes, their binary + hsacos from their dir):
#      gpad0 / gpad1 x 3 runs, gpad3 + nograph x 1
#  (b) the same shapes with the harness REBUILT verbatim in review/ against freshly built baseline hsacos
#  (c) gpad_check (same-process interleaved grid vs grid+1/+7/+64, correctness incl. odd/tail shapes) x 3 + nograph
set -u
REV="$(cd "$(dirname "$0")" && pwd)"
C1="$(cd "$REV/.." && pwd)"
INFRA="$(cd "$REV/../../_infra" && pwd)"
SHAPES="quant 32768 2 : quant 32768 4 : quant 32768 1 : quant 65536 1 : quant 32800 2 : quant 16384 4 : quant 5120 4 : quant 8192 4"
SUB="bash $INFRA/gpu_submit.sh --dev dgpu --mb 100 --timeout 200 --label review/C1_dense_decode"
: > "$REV/tickets.txt"
sub() { # name cmd...
    local name=$1; shift
    local t
    t=$("$@" 2>&1 | tail -1)
    echo "$name $t" >> "$REV/tickets.txt"
    echo "$name -> $t"
}
cd "$C1"
export C1_DIR="$C1"
for r in 1 2 3; do
    sub "eng_gpad0_r$r" $SUB -- ./harness $SHAPES
    sub "eng_gpad1_r$r" $SUB -- env C1_GPAD=1 ./harness $SHAPES
done
sub "eng_gpad3_r1" $SUB -- env C1_GPAD=3 ./harness $SHAPES
sub "eng_nograph_r1" $SUB -- env C1_NOGRAPH=1 ./harness $SHAPES
cd "$REV"
export C1_DIR="$REV"
sub "rebuilt_gpad0_r1" $SUB -- ./harness $SHAPES
sub "rebuilt_gpad1_r1" $SUB -- env C1_GPAD=1 ./harness $SHAPES
CHK="32768 1 32768 2 32768 3 32768 4 32768 5 8192 8 8192 5 5120 1 5120 5 2304 4 65504 1 65568 1 131040 1 131104 1 262144 1 32 1 1024 1"
for r in 1 2 3; do
    sub "check_r$r" $SUB -- ./gpad_check $CHK
done
sub "check_nograph_r1" $SUB -- env GC_NOGRAPH=1 ./gpad_check $CHK
cat "$REV/tickets.txt"
