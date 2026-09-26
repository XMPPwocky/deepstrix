#!/usr/bin/env bash
# isa_cand.sh <hsaco-base-name> [kernel symbols...]: resources + instruction mix of a candidate code object
set -u
cd "$(dirname "$0")"
f=$1; shift
bash ../_infra/isa.sh ${f}_gfx1151.hsaco gfx1151 --dis 2>&1 | head -20
for k in "$@"; do
    echo "== $k"
    awk -v pat="<$k>" '/^[0-9a-f]+ <.*>:$/ {p = index($0, pat) > 0} p' ${f}_gfx1151.s > ${f}_$k.s
    echo "lines $(wc -l < ${f}_$k.s)  global_load $(grep -c global_load ${f}_$k.s)  vmcnt-waits $(grep -c 's_waitcnt vmcnt' ${f}_$k.s)  barriers $(grep -c s_barrier ${f}_$k.s)  movrel $(grep -c movrel ${f}_$k.s)  scratch $(grep -c scratch_ ${f}_$k.s)  ds_load $(grep -c ds_load ${f}_$k.s)  dot4 $(grep -c v_dot4 ${f}_$k.s)"
    grep -o "global_load_[a-z0-9]*" ${f}_$k.s | sort | uniq -c
done
