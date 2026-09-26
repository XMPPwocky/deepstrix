#!/usr/bin/env bash
# isa_all.sh: resources + instruction mix for every named candidate (gu = cand_gate_up, dn = cand_down)
set -u
cd "$(dirname "$0")"
for c in "$@"; do
    case $c in
        gu*|kwide*) k=cand_gate_up ;;
        dn*) k=cand_down ;;
        *) k="cand_gate_up cand_down" ;;
    esac
    echo "### $c"
    bash isa_cand.sh cand_$c $k 2>&1 | grep -v "^disassembly\|^kernel"
done
