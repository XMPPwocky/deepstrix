#!/usr/bin/env bash
# Reviewer: shapes the engineer did not test, one scheduler job. Output: review/results/extra_<name>.txt
#   b1_E6     : 1 row, 6 distinct experts (n_wi = bound = 6 < cap 8)
#   b3_E1     : 3 rows all on ONE expert (single group of 3)
#   b5_E2     : odd row count, 2 groups of 5
#   b8_E8     : max production b, 8 distinct experts, n_wi = cap exactly
#   b4_E24    : hub regime -- 4 rows x 6 picks over 24 experts (n_wi ~ 20 > cap: WGs loop 2-3 items), P=24
#   b8_E48    : hub/merged regime -- 8 rows x 6 picks over 48 experts (n_wi ~ 40, 5 items/WG), P=48
#   b12_E4    : above production max (groups of 12 > chunk 8 -> the c8 builder splits groups)
set -u
cd "$(dirname "$0")"
mkdir -p results
C=wl_c8dn2.wl.c8,dn2_r2,kwide_c8.c8,wl.wl
run() {
    name=$1; shift
    ./harness_gfx1151 gfx1151 . chain rounds=40 inner=8 wl=8 cand=$C "$@" > results/extra_${name}.txt 2>&1
    echo "== $name rc=$? ($*)"
    grep -E "^CMP .* over|bitexact=no|nonfinite=[1-9]|^variant|us  |error|Error|fault|^\[h\] b=|^\[h\] distinct" results/extra_${name}.txt | grep -v KBJSON | grep -v "twin(hetsplit"
}
run b1_E6   b=1  E=6  ppr=6 P=16
run b3_E1   b=3  E=1  ppr=1 P=16
run b5_E2   b=5  E=2  ppr=2 P=16
run b8_E8   b=8  E=8  ppr=6 P=16
run b4_E24  b=4  E=24 ppr=6 P=24
run b8_E48  b=8  E=48 ppr=6 P=48
run b12_E4  b=12 E=4  ppr=3 P=16
