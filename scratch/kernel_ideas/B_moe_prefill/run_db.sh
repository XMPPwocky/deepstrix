#!/usr/bin/env bash
# Double-buffered f16 arm vs the single-buffered f16 arm (production at V41_MOE_WMMA_F16=1), iGPU only, beside the live hub.
# Regimes: the 1024-row-lane MoE call (B=1024 over ~162 hub experts: ~38 members/expert), 2048, and the old bench point.
set -u
cd "$(dirname "$0")"
OUT=results/f16db_$(date -u +%Y%m%dT%H%M%SZ).txt
{
for cfg in "1024 162" "2048 162" "3072 48"; do
  set -- $cfg
  echo "=== cmp (correctness vs kwide) B=$1 N_EXP=$2: f16db nt2"
  KB_CHUNK=32 bash run_f16.sh cmp "$1" "$2" 7 1 cand_f16db_nt2 2>&1 | grep -E 'rel_rmse|FAIL|PASS|error' | head -6
  echo "=== abx B=$1 N_EXP=$2: A = f16 single-buffer (prod), B = f16 double-buffer, both chunk 32"
  KB_CAND2=cand_f16db_nt2 KB_CHUNK_A=32 KB_CHUNK_B=32 bash run_f16.sh abx "$1" "$2" 7 12 cand_f16 2>&1 | grep -vE '^\s*$' | tail -12
done
echo "=== abx B=1024 N_EXP=162: A = f16db nt2 (chunk 32), B = f16db nt3 (chunk 48)"
KB_CAND2=cand_f16db_nt3 KB_CHUNK_A=32 KB_CHUNK_B=48 bash run_f16.sh abx 1024 162 7 12 cand_f16db_nt2 2>&1 | grep -vE '^\s*$' | tail -12
echo "=== abx B=1024 N_EXP=162: A = f16db nt3 (chunk 48), B = f16 single-buffer nt4 (chunk 64)"
KB_CAND2=cand_f16nt4 KB_CHUNK_A=48 KB_CHUNK_B=64 bash run_f16.sh abx 1024 162 7 12 cand_f16db_nt3 2>&1 | grep -vE '^\s*$' | tail -12
} > "$OUT" 2>&1
cat "$OUT"
