#!/usr/bin/env python3
"""Merge measured device ceilings + per-kernel microbench rows into the
roofline table for docs/v41/KERNEL_ROOFLINE.md.

Inputs (JSON lines):
  ceilings.json  from bench_device_ceilings (CEIL_JSON=...)
  kernels.json   from bench_v41_kernel_roofline (BENCH_JSON=...)

For every kernel row: roofline time = max(bytes / BW_ceiling, flops / peak)
where the peak is chosen by compute class (inferred from the kernel name):
  wmma_f16  : *wmma* kernels (f16 matrix cores)
  dp4a      : q8_0_gemv*, grouped_gemv, mxfp4* (v_dot4 inner loops)
  fma_f32   : everything else (elementwise / reductions / softmax)
Per-token (decode) or per-lane-chunk (prefill) cost = p50 × launches/layer ×
layers; wasted = measured − roofline. Rows are sorted by wasted time.
"""
import json, sys, math, os, re
from collections import defaultdict

def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows

def ceilings(path):
    c = defaultdict(dict)
    for r in load(path):
        c[r["dev"]][r["metric"]] = (r["value"], r["pattern"])
    return c

def compute_class(name):
    n = name.lower()
    if "wmma" in n:
        return "wmma_f16"
    if any(k in n for k in ("q8_0_gemv", "grouped_gemv", "mxfp4", "f16_matvec", "q8_0_gemm")):
        return "dp4a" if "f16_matvec" not in n else "fma_f32"
    return "fma_f32"

def peak_for(cl, dev, C):
    if cl == "wmma_f16":
        return C[dev].get("wmma_f16", (0, ""))[0] * 1e12
    if cl == "dp4a":
        return C[dev].get("dp4a", (0, ""))[0] * 1e12
    return C[dev].get("fma_f32", (0, ""))[0] * 1e12

def bw_for(dev, C):
    # the streaming-read ceiling is the honest bound for weight/KV readers;
    # copy-heavy elementwise kernels are bounded by the r+w figure instead.
    return C[dev].get("bw_read", (0, ""))[0] * 1e9, C[dev].get("bw_copy", (0, ""))[0] * 1e9

def fmt_us(us):
    if us >= 1e6:
        return f"{us/1e6:.2f} s"
    if us >= 1e3:
        return f"{us/1e3:.2f} ms"
    return f"{us:.1f} µs"

def confidence(r, ratio):
    # launch-bound tiny kernels (< 8 µs) and rows whose byte model is a
    # guess get low confidence; large streaming kernels get high.
    if r["p50_us"] < 8:
        return "low (launch-bound)"
    if r["bytes"] > 32e6 or r["flops"] > 5e9:
        return "high"
    return "med"

def main():
    sp = os.path.dirname(os.path.abspath(__file__))
    C = ceilings(os.path.join(sp, "ceilings.json"))
    rows = load(os.path.join(sp, "kernels.json"))
    if not rows:
        print("no kernels.json rows", file=sys.stderr)
        sys.exit(1)
    out = []
    for r in rows:
        dev = r["dev"]
        cl = compute_class(r["name"])
        bw_read, bw_copy = bw_for(dev, C)
        peak = peak_for(cl, dev, C)
        # elementwise kernels write as much as they read: use the r+w ceiling
        bw = bw_copy if (cl == "fma_f32" and r["flops"] < 4 * r["bytes"]) else bw_read
        t_bw = r["bytes"] / bw * 1e6 if bw > 0 else float("nan")
        t_fl = r["flops"] / peak * 1e6 if peak > 0 else 0.0
        t_roof = max(t_bw, t_fl)
        bound = "BW" if t_bw >= t_fl else "compute"
        n_calls = r["launches_per_layer"] * r["layers"]
        meas = r["p50_us"] * n_calls
        roof = t_roof * n_calls
        out.append(dict(r, cls=cl, bound=bound, t_roof=t_roof, meas_total=meas, roof_total=roof,
                        wasted=meas - roof, ratio=(r["p50_us"] / t_roof if t_roof > 0 else float("inf")),
                        gbs=r["bytes"] / 1e9 / (r["p50_us"] / 1e6), tflops=r["flops"] / 1e12 / (r["p50_us"] / 1e6),
                        pct=(t_roof / r["p50_us"] * 100.0), n_calls=n_calls))
    # group by path
    paths = sorted(set(r["path"] for r in out), key=lambda p: (p != "decode", p))
    for p in paths:
        sub = sorted([r for r in out if r["path"] == p], key=lambda r: -r["wasted"])
        unit = "token" if p == "decode" else "lane-chunk"
        tot_meas = sum(r["meas_total"] for r in sub)
        tot_roof = sum(r["roof_total"] for r in sub)
        print(f"\n### {p}: sum of measured per-{unit} kernel time by device")
        for dev in ("dgpu", "igpu"):
            m = sum(r["meas_total"] for r in sub if r["dev"] == dev)
            ro = sum(r["roof_total"] for r in sub if r["dev"] == dev)
            if m > 0:
                print(f"- {dev}: measured {fmt_us(m)} vs roofline {fmt_us(ro)} → {m/ro:.2f}× (wasted {fmt_us(m-ro)})")
        print(f"\n| # | kernel | dev | shape | launches/layer × layers | bytes | FLOPs | bound | p50 | achieved | % of ceiling | meas/roof | wasted per {unit} | conf |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(sub, 1):
            ach = f"{r['gbs']:.0f} GB/s" if r["bound"] == "BW" else f"{r['tflops']:.1f} TF/s"
            print(f"| {i} | `{r['name']}` | {r['dev']} | {r['shape']} | {r['launches_per_layer']:g} × {r['layers']:g} | {r['bytes']/1e6:.2f} MB | {r['flops']/1e9:.2f} G | {r['bound']} ({r['cls']}) | {fmt_us(r['p50_us'])} | {ach} | {r['pct']:.0f}% | {r['ratio']:.1f}× | {fmt_us(r['wasted'])} | {confidence(r, r['ratio'])} |")
    # ceilings echo
    print("\n### ceilings used")
    for dev in C:
        for k, (v, pat) in sorted(C[dev].items()):
            print(f"- {dev} {k}: {v:.1f} ({pat})")

if __name__ == "__main__":
    main()
