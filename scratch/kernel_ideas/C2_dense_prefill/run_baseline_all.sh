#!/usr/bin/env bash
# Baseline sweep: every production shape, cold-weight (flush) and warm, b=512 (+256/1024 where it fits).
set -u
cd "$(dirname "$0")"
R=results/baseline_${1:-r1}
mkdir -p $R
for sh in qa qb kv woa wob shg shd; do
  ./run.sh $sh > $R/${sh}_b512_flush.txt 2>&1
  ./run.sh $sh --warm > $R/${sh}_b512_warm.txt 2>&1
done
for sh in qa kv shg shd wob; do
  ./run.sh $sh --b 256 > $R/${sh}_b256_flush.txt 2>&1
  ./run.sh $sh --b 1024 > $R/${sh}_b1024_flush.txt 2>&1
done
./run.sh engram --m 12800 > $R/engram_m12800_b64_flush.txt 2>&1
./run.sh engram --m 6400 > $R/engram_m6400_b64_flush.txt 2>&1
./run.sh engram --m 12800 --warm > $R/engram_m12800_b64_warm.txt 2>&1
for c in 5120 32768 8192; do ./run.sh cast_$c > $R/cast_${c}_b512.txt 2>&1; done
for sh in qb64 kv64 wob64 woa64; do
  ./run.sh $sh > $R/${sh}_b64_flush.txt 2>&1
  ./run.sh $sh --warm > $R/${sh}_b64_warm.txt 2>&1
done
grep -h "^==\|^base\|^cand\|^\[kb\] flush\|error\|Error" $R/*.txt
