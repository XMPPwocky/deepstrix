#!/usr/bin/env bash
# table.sh : collect (e) runs r2/r3 and print the 3-run summary
set -u
cd "$(dirname "$0")"
while read -r t; do bash ../../_infra/gpu_wait.sh "$t" > /dev/null 2>&1; done < tickets.txt
cd results
P='prod GEMM alone=>cand GEMM alone i8x_db_ld;;prod GEMM alone=>cand GEMM alone i8x_128_ld;;prod GEMM alone=>cand GEMM alone i8x_256x128_ld$;;prod GEMM alone=>cand GEMM alone i8x_bn64_ld;;prod: cast=>cand: quant \+ i8x_256x128_ld$;;prod: cast=>cand: quant \+ i8x_bn64_ld;;prod: cast=>cand: quant \+ i8x_128_ld'
for s in qa kv shg shd wob qb woa qa1024; do
  python3 ../../pairs.py --pairs "$P" ${s}_r1.txt ${s}_r2.txt ${s}_r3.txt | sed -e 's/  */ /g' -e 's/ cold(flush [0-9]*MB) direct//' | cut -c1-200
done
