#!/usr/bin/env bash
# Wait on round-2 tickets, save outputs, summarise mix runs and the rocprof kernel trace per grid size.
set -u
REV="$(cd "$(dirname "$0")" && pwd)"
INFRA="$(cd "$REV/../../_infra" && pwd)"
while read -r name t; do
    [ -n "$name" ] || continue
    out="$REV/results/$name.txt"
    if [ ! -s "$out" ] || ! grep -q "^== .*rc=" "$out"; then
        bash "$INFRA/gpu_wait.sh" "$t" > "$out" 2>&1
        echo "rc=$? $name -> $out"
    fi
done < "$REV/tickets2.txt"
for f in "$REV"/results/mix_*.txt; do
    echo "--- $(basename "$f") $(grep -m1 '^== .*rc=' "$f" | cut -c1-60) $(grep ^RESULT "$f")"
    grep -E "^SHAPE|^q8_0_quantize_f32 grid=blocks\+(0|1) " "$f" | sed -E 's/ +/ /g; s/\| base vs CPU ref: //; s/xs_unwritten.*//' | cut -c1-75 | paste - - -
done
echo "--- trace_r1 $(grep -m1 '^== .*rc=' "$REV/results/trace_r1.txt" | cut -c1-60)"
csv=$(ls "$REV"/trace_r1/*kernel_trace.csv 2>/dev/null | head -1)
if [ -n "$csv" ]; then
python3 - "$csv" <<'EOF'
import csv, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
by = {}
for r in rows:
    if 'q8_0_quantize_f32' not in r['Kernel_Name']: continue
    g = int(r['Grid_Size_X']) // int(r['Workgroup_Size_X'])
    d = (int(r['End_Timestamp']) - int(r['Start_Timestamp'])) / 1000.0
    by.setdefault(g, []).append(d)
print("rocprofv3 kernel-trace: q8_0_quantize_f32 hardware duration per dispatch, by grid (WGs)")
for g in sorted(by):
    v = sorted(by[g]); n = len(v)
    print(f"  grid={g:6d} n={n:4d} med={st.median(v):7.2f} us  p10={v[int(n*0.1)]:7.2f}  p90={v[int(n*0.9)]:7.2f}  min={v[0]:7.2f}")
EOF
else
    echo "no kernel_trace.csv found"; ls "$REV"/trace_r1 2>/dev/null
fi
