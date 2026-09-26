#!/usr/bin/env bash
# kd_meta.sh <file.elf> [kernel-substr] : real per-kernel metadata (LDS, VGPR, SGPR, scratch) from the code object notes
set -u
cd "$(dirname "$0")"
bash ../_infra/in_env.sh llvm-readelf --notes "$1" 2>/dev/null | python3 -c '
import sys, re
txt = sys.stdin.read()
sub = sys.argv[1] if len(sys.argv) > 1 else ""
cur = None
for line in txt.split("\n"):
    m = re.search(r"\.name:\s+(\S+)", line)
    if m and not line.strip().startswith(".name:") is False and "kd" not in m.group(1):
        cur = m.group(1)
    for key in (".group_segment_fixed_size", ".vgpr_count", ".sgpr_count", ".private_segment_fixed_size", ".agpr_count", ".vgpr_spill_count"):
        if key in line and cur and sub in cur:
            print(f"{cur:40s} {key:30s} {line.strip().split()[-1]}")
' "${2:-}"
