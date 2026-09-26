#!/usr/bin/env bash
# Compare the reviewer's rebuilt code objects with the engineer's at the instruction level
# (the .hsaco bundles differ in a header byte; the disassembly is what matters).
set -u
cd "$(dirname "$0")"
source ../../../_infra/env.sh
ARCH=gfx1151
od -A d -c -N 32 cand_dn2_r2_$ARCH.hsaco | head -3
od -A d -c -N 32 ../../cand_dn2_r2_$ARCH.hsaco | head -3
unb() { clang-offload-bundler --type=o --unbundle --input="$1" --output="$2" --targets=hipv4-amdgcn-amd-amdhsa--$ARCH 2>/dev/null || cp "$1" "$2"; }
unb ../../cand_dn2_r2_$ARCH.hsaco eng_cand.elf
unb ../../base_down_$ARCH.hsaco eng_base.elf
unb ../../base_pair_$ARCH.hsaco eng_pair.elf
unb cand_dn2_r2_$ARCH.hsaco my_cand.elf
unb chk_base_down_$ARCH.hsaco my_base.elf
unb chk_base_pair_$ARCH.hsaco my_pair.elf
for p in eng_cand my_cand eng_base my_base eng_pair my_pair; do
    llvm-objdump -d --no-show-raw-insn --mcpu=$ARCH $p.elf | sed 's/^ *[0-9a-f]*://' | sed 's#//.*##' > $p.s
done
wc -l *.s
for pair in "my_cand eng_cand" "my_base eng_base" "my_pair eng_pair"; do
    set -- $pair
    if diff -q $1.s $2.s >/dev/null; then echo "$1 == $2 (instruction-identical)"; else echo "$1 != $2"; diff $1.s $2.s | head -10; fi
done
for p in eng_cand eng_base eng_pair; do
    printf "%s: md5 elf text = " $p; llvm-objcopy -O binary --only-section=.text $p.elf /dev/stdout | md5sum
done
for p in my_cand my_base my_pair; do
    printf "%s: md5 elf text = " $p; llvm-objcopy -O binary --only-section=.text $p.elf /dev/stdout | md5sum
done
