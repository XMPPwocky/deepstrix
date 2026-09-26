#!/usr/bin/env python3
"""isa_summary.py <file.s> <symbol> [--full]
Run-length summary of one kernel's instruction stream: loads / waits / barriers / wmma / branches.
"""
import re, sys
path, sym = sys.argv[1], sys.argv[2]
full = "--full" in sys.argv
lines = open(path).read().split("\n")
body, on = [], False
for l in lines:
    m = re.match(r"^[0-9a-f]+ <(.*)>:$", l)
    if m:
        on = (m.group(1) == sym)
        continue
    if on and l.strip():
        body.append(l)
print(f"{sym}: {len(body)} lines")
classes = [
    ("global_load", r"\bglobal_load"), ("ds_load", r"\bds_load"), ("ds_store", r"\bds_store"),
    ("global_store", r"\bglobal_store"), ("scratch", r"\bscratch_"),
    ("s_wait_loadcnt", r"\bs_wait_loadcnt\b"), ("s_wait_loadcnt_dscnt", r"\bs_wait_loadcnt_dscnt\b"),
    ("s_wait_dscnt", r"\bs_wait_dscnt\b"), ("s_wait_kmcnt", r"\bs_wait_kmcnt"), ("s_wait_storecnt", r"\bs_wait_storecnt"),
    ("s_barrier", r"\bs_barrier_wait"), ("v_wmma", r"\bv_wmma"), ("branch", r"\bs_cbranch"),
    ("s_load", r"\bs_load"), ("v_cvt_f16", r"\bv_cvt_f16_f32"), ("v_exp", r"\bv_exp_f32"),
]
counts = {c: 0 for c, _ in classes}
seq = []
for i, l in enumerate(body):
    ins = l.strip().split()[0] if l.strip() else ""
    cls = None
    for c, r in classes:
        if re.search(r, l):
            cls = c
            break
    if cls:
        counts[cls] += 1
        if full:
            wait = ""
            if cls.startswith("s_wait"):
                wait = " ".join(l.strip().split()[1:3])
            seq.append((i, cls, wait, l.strip()[:60]))
print("counts:", {k: v for k, v in counts.items() if v})
if full:
    prev, n, first, extra = None, 0, 0, ""
    for i, cls, w, txt in seq:
        key = cls + ((" " + w) if w else "")
        if key != prev:
            if prev is not None:
                print(f"  @{first:5d} {prev:28s} x{n}")
            prev, n, first = key, 1, i
        else:
            n += 1
    if prev is not None:
        print(f"  @{first:5d} {prev:28s} x{n}")
