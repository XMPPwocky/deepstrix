#!/usr/bin/env bash
# Submit every review GPU job (label review/A_moe_smallb) and print the tickets to tickets.txt.
#  rep1..rep3 : the engineer's exact repro (chain, wl.wl, b=2 4 8, wl=8) as 3 separate processes/jobs
#  parts      : per-kernel breakdown at b=1 2 4 8 (bound grid) with the wl.wl candidate
#  partsx     : same with exact=1 (exact grid, no empties) -> isolates the loop kernel's own overhead
#  tails      : shapes the engineer did not put in the claim: b=1 3 5 7 (and 6)
#  hub        : hub-like many-distinct regime: E=16 ppr=6 at b=5 (n_wi up to 16 > cap 8, WGs loop 2 items)
#  e12        : E=12 ppr=6 at b=8 (n_wi 12 > cap) and E=6 ppr=6 b=8 (all 6 picks real, n_wi=6)
set -u
cd "$(dirname "$0")"
S=../../../_infra/gpu_submit.sh
: > tickets.txt
sub() { local name=$1; shift; t=$(bash $S --dev igpu --mb 450 --label review/A_moe_smallb --timeout 300 -- "$@"); echo "$name $t" | tee -a tickets.txt; }
sub rep1  bash run_review.sh chain rep1 wl.wl "2 4 8" wl=8
sub rep2  bash run_review.sh chain rep2 wl.wl "2 4 8" wl=8
sub rep3  bash run_review.sh chain rep3 wl.wl "2 4 8" wl=8
sub parts bash run_review.sh parts parts1 wl.wl "1 2 4 8" wl=8
sub partsx bash run_review.sh parts partsx1 wl.wl "1 2 4 8" wl=8 exact=1
sub tails bash run_review.sh chain tails1 wl.wl "1 3 5 6 7" wl=8
sub hub   bash run_review.sh chain hubE16 wl.wl "5" wl=8 E=16 ppr=6
sub e12   bash run_review.sh chain E12 wl.wl "8" wl=8 E=12 ppr=6
sub e6    bash run_review.sh chain E6 wl.wl "8" wl=8 E=6 ppr=6
