"""Kernel durations by (name, grid WGs) from a rocprofv3 kernel trace: p10 / p50 / p90 in us."""
import csv
import collections
import glob
import sys

d = sys.argv[1]
f = glob.glob(f"{d}/**/kt_kernel_trace.csv", recursive=True) + glob.glob(f"{d}/kt_kernel_trace.csv")
rows = list(csv.DictReader(open(f[0])))
D = collections.defaultdict(list)
for r in rows:
    n = r["Kernel_Name"].split("(")[0][:44]
    wg = int(r["Workgroup_Size_X"]) * int(r["Workgroup_Size_Y"]) * int(r["Workgroup_Size_Z"])
    g = int(r["Grid_Size_X"]) * int(r["Grid_Size_Y"]) * int(r["Grid_Size_Z"]) // max(1, wg)
    D[(n, g, wg, r.get("VGPR_Count", ""), r.get("LDS_Block_Size", ""))].append((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3)


def p(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(round((len(s) - 1) * q)))]


for k, v in sorted(D.items()):
    if any(x in k[0] for x in ["fma_f32", "ceil_read"]):
        continue
    print(f"{k[0]:<44} wgs {k[1]:>5} x{k[2]:<4} vgpr {k[3]:>4} lds {k[4]:>6}  n {len(v):>5}  p10 {p(v,.1):7.2f}  p50 {p(v,.5):7.2f}  p90 {p(v,.9):7.2f}")
