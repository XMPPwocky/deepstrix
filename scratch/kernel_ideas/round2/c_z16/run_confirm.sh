#!/usr/bin/env bash
# run_confirm.sh <tag> [intree] : confirmation job (base, base+1, h8 tight-NB, h8 tight-NB +1) for the
# wired shapes; with "intree" the h8 variants are the in-tree f16_matvec_batched_z16_n<NB> symbols.
set -u
cd "$(dirname "$0")"
tag=$1
export Z_CONFIRM=1
[ "${2:-}" = intree ] && export Z_INTREE=1
./harness idxq 1 2 3 4 5 6 8 16 32 64 > results/cidxq_$tag.txt 2>&1
./harness comp1 1 2 3 4 5 6 8 16 32 64 > results/ccomp1_$tag.txt 2>&1
./harness proj 1 2 3 4 5 6 8 16 > results/cproj_$tag.txt 2>&1
[ "${2:-}" = intree ] && ./harness router 8 16 32 48 64 > results/crouter_$tag.txt 2>&1
grep -h 'MISMATCH\|RESULT' results/c*_$tag.txt
