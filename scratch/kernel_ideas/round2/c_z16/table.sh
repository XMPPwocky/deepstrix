#!/usr/bin/env bash
# table.sh : cross-run summary of the (c) confirmation runs (c1-c3 candidate file, c4-c6 in-tree)
set -u
cd "$(dirname "$0")/results"
P="python3 ../../pairs.py"
short() { sed -e 's/ [0-9]*x[0-9]* \(b=[0-9]*\) COLD W ([0-9]* copies) graph */ \1/' -e 's/vs base f16_matvec_batched *//' -e 's/  */ /g'; }
grep -h 'MISMATCH\|RESULT' c*_c[1-6].txt | sort | uniq -c
for s in cidxq ccomp1 cproj; do $P ${s}_c[1-6].txt | short; done
$P crouter_c[4-6].txt | short
