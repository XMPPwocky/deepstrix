#!/usr/bin/env bash
# run_wlsens.sh: WL cap sensitivity (one job). E=4 regime at b=4,8 with wl=4/6/16, and a many-expert regime
# (E=12 ppr=6, n_wi=12 > cap) at b=8 with wl=8/16 and the uncapped loop.
set -u
cd "$(dirname "$0")"
run() { # tag b wl extra...
    local tag=$1 b=$2 wl=$3; shift 3
    ./harness_gfx1151 gfx1151 . chain b=$b E=4 ppr=3 rounds=40 inner=8 cand=wl_c8dn2.wl.c8,combo_c8dn2.c8 wl=$wl "$@" > results/chain_${tag}_b${b}.txt 2>&1
    echo "== $tag b=$b"; grep -A6 "^variant" results/chain_${tag}_b${b}.txt | grep -v KBJSON
}
run wl4 8 4
run wl6 8 6
run wl16 8 16
run wl4 4 4
run wl16 4 16
run E12wl8 8 8 E=12 ppr=6
run E12wl16 8 16 E=12 ppr=6
run E12wl48 8 48 E=12 ppr=6
