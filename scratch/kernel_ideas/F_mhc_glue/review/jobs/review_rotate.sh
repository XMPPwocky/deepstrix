#!/usr/bin/env bash
# cold-X regime via 4 rotating X copies (168 MB > 64 MB MALL), inner=1; three separate processes
set -u
cd "$(dirname "$0")/.."
export HARNESS_DIR=$PWD
for t in r1 r2 r3; do ./harness_gfx1201 gemm 512 rotate > results/gemm_512_rotate_${t}.txt 2>&1; done
grep -h "^==\|^[a-zA-Z].*[0-9]\.[0-9][0-9] " results/gemm_512_rotate_r*.txt
echo "bitexact YES: $(grep -h '^CMP' results/gemm_512_rotate_r*.txt | grep -c 'bitexact=YES')  no: $(grep -h '^CMP' results/gemm_512_rotate_r*.txt | grep -c 'bitexact=no')"
