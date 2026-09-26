#!/usr/bin/env bash
# show.sh <mode:chain|parts> <tag> [b-list]: print the timing tables of a run
set -u
cd "$(dirname "$0")"
mode=$1; tag=$2; blist=${3:-"1 2 4 8"}
for b in $blist; do
    f=results/${mode}_${tag}_b${b}.txt
    [ -s "$f" ] || continue
    echo "== $mode $tag b=$b"
    grep -E "bitexact=no|nonfinite=[1-9]|error|Error|fault" "$f" | grep -v "twin(hetsplit" | grep -v KBJSON
    grep -A40 "^variant" "$f" | grep -v KBJSON | grep -v "^roofline"
done
