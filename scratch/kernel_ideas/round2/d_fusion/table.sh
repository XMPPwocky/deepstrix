#!/usr/bin/env bash
# table.sh : collect the chain runs listed in tickets.txt (r2..r6) and print the cross-run table
set -u
cd "$(dirname "$0")"
i=2
while read -r t; do
  [ -s results/chain_r$i.txt ] || bash ../../_infra/gpu_wait.sh "$t" > results/chain_r$i.txt 2>&1
  i=$((i + 1))
done < tickets.txt
grep -h 'RESULT' results/chain_r*.txt | sort | uniq -c
grep -h 'MISMATCH' results/chain_r*.txt | head
python3 ../pairs.py --pairs 'kv: rms=>kv: fused;;qa: rms \+ cast=>qa: rms \+ quant;;qa: rms \+ cast=>qa: fused;;q: memcpy=>q: rope_copy;;oproj: rope_inv \+ cast=>oproj: rope_inv \+ quant;;oproj: rope_inv \+ cast=>oproj: fused;;dead: cast_input=>dead-free: quant \(;;dead: cast_low=>dead-free: quant low' results/chain_r[2-9].txt | sed -e 's/decode chains //' -e 's/ warm graph inner=10//' -e 's/  */ /g'
