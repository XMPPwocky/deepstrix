#!/usr/bin/env bash
# Reviewer jobs for claim C1_dense_decode/quantize_wave (all via the dGPU scheduler, label review/C1_dense_decode).
#   3x the engineer's EXACT 3rd ticket of submit_r3.sh (grouped/quant/swiglu/lanes, --mb 250, timeout 240)
#   1x extra shapes the engineer did not time (b=1 at every K, b=5 = max production b per lane, odd tails)
#   1x the odd/adversarial correctness harness
# Tickets go to results/tickets.txt (one per line, tagged).
set -u
cd "$(dirname "$0")"
export C1_DIR="$PWD"
SUB="bash ../../../_infra/gpu_submit.sh"
: > results/tickets.txt
for r in 1 2 3; do
  echo -n "repro_r$r " >> results/tickets.txt
  $SUB --dev dgpu --mb 250 --label review/C1_dense_decode --timeout 240 -- \
    ./harness grouped 1 : grouped 2 : grouped 4 : grouped 5 : grouped 8 \
    : quant 5120 1 : quant 5120 4 : quant 5120 8 : quant 1280 4 : quant 8192 4 : quant 32768 4 : quant 2304 4 \
    : swiglu 1 : swiglu 4 \
    : lanes qa 4 4 : lanes qa 5 5 : lanes wob 4 4 : lanes gate 3 3 : lanes down 4 4 : lanes qb 4 4 | tee -a results/tickets.txt
done
echo -n "extra " >> results/tickets.txt
$SUB --dev dgpu --mb 100 --label review/C1_dense_decode --timeout 200 -- \
  ./harness quant 5120 1 : quant 1280 1 : quant 2304 1 : quant 8192 1 : quant 32768 1 \
  : quant 5120 5 : quant 1280 5 : quant 2304 5 : quant 8192 5 : quant 32768 5 \
  : quant 5120 2 : quant 32768 2 : quant 1312 1 : quant 224 1 : quant 5120 3 | tee -a results/tickets.txt
echo -n "extra_gpad1 " >> results/tickets.txt
$SUB --dev dgpu --mb 100 --label review/C1_dense_decode --timeout 200 -- \
  env C1_GPAD=1 ./harness quant 32768 4 : quant 32768 2 : quant 5120 4 | tee -a results/tickets.txt
echo -n "odd " >> results/tickets.txt
$SUB --dev dgpu --mb 100 --label review/C1_dense_decode --timeout 120 -- ./odd | tee -a results/tickets.txt
cat results/tickets.txt
