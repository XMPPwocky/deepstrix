#!/usr/bin/env python3
"""Summarise `llvm-readelf --notes` of an AMDGPU code object: one row per kernel."""
import re
import sys

KEYS = {".name", ".vgpr_count", ".sgpr_count", ".group_segment_fixed_size",
        ".private_segment_fixed_size", ".vgpr_spill_count", ".sgpr_spill_count",
        ".wavefront_size"}

txt = sys.stdin.read()
rows, cur, in_args = [], {}, False
for line in txt.splitlines():
    m = re.match(r"(\s*)-?\s*(\.[a-z_]+):\s*(.*)", line)
    if not m:
        continue
    indent, k, v = m.groups()
    if k == ".args":
        in_args = True
        continue
    if k == ".name" and in_args and len(indent) > 6:
        continue  # an argument's .name
    if k in KEYS:
        if k == ".name":
            in_args = False
            if cur.get(".name") and ".vgpr_count" in cur:
                rows.append(cur)
                cur = {}
        cur[k] = v.strip()
if cur:
    rows.append(cur)
print(f"{'kernel':64s} {'vgpr':>5s} {'sgpr':>5s} {'lds_B':>7s} {'scratch':>7s} {'vspill':>6s} {'wave':>4s}")
for r in rows:
    if ".vgpr_count" not in r:
        continue
    print(f"{r.get('.name', '?')[:64]:64s} {r.get('.vgpr_count', ''):>5s} {r.get('.sgpr_count', ''):>5s} "
          f"{r.get('.group_segment_fixed_size', ''):>7s} {r.get('.private_segment_fixed_size', ''):>7s} "
          f"{r.get('.vgpr_spill_count', ''):>6s} {r.get('.wavefront_size', ''):>4s}")
