#!/usr/bin/env bash
# kasm.sh <kernel symbol> : extract that kernel's ISA from the last hsaco.sh build into /tmp/g1o/asm/<sym>.s
S=/tmp/g1o/asm/g1_router_block-hip-amdgcn-amd-amdhsa-gfx1201.s
awk -v K="$1:" 'index($0, K) == 1 {f = 1; print; next} f && /^g1_router_block_[a-z0-9_]*:/ {exit} f {print}' "$S" > "/tmp/g1o/asm/$1.s"
wc -l "/tmp/g1o/asm/$1.s"
