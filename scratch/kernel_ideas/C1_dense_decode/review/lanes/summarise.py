#!/usr/bin/env python3
"""Per results file and section: production 2-lane cost (sum of the two separately-cold launches, the
engineer's arithmetic), the one-graph 2-lane cost (reviewer), tB+tB, merged tB, ratios."""
import glob, json, os, re, sys
root = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(root, "results", "*.txt")))
for f in files:
    print("===", os.path.basename(f))
    sec = None
    rows = {}
    def flush():
        if not sec or not rows:
            return
        l0 = next((v for k, v in rows.items() if k.startswith("base lane0")), None)
        l1 = next((v for k, v in rows.items() if k.startswith("base lane1")), None)
        one = next((v for k, v in rows.items() if "ONE graph" in k and k.startswith("base 2 lanes")), None)
        tt = next((v for k, v in rows.items() if "ONE graph" in k and k.startswith("tB")), None)
        m = next((v for k, v in rows.items() if k.startswith("cand merged tB") and "_w4" not in k), None)
        rt = next((v for k, v in rows.items() if k.startswith("base merged")), None)
        null = next((v for k, v in rows.items() if k.startswith("null")), None)
        chain = next((v for k, v in rows.items() if k.startswith("base chain")), None)
        tchain = next((v for k, v in rows.items() if k.startswith("tB") and "chain" in k), None)
        out = [sec]
        if l0 and l1:
            out.append(f"2xcold={l0+l1:.1f}")
        if one:
            out.append(f"1graph={one:.1f}")
        if chain:
            out.append(f"chain2lanes={chain:.1f}")
        if tchain:
            out.append(f"tBchain={tchain:.1f}")
        if tt:
            out.append(f"tB+tB={tt:.1f}")
        if m:
            out.append(f"merged={m:.1f}")
            if l0 and l1:
                out.append(f"m/2cold={m/(l0+l1):.3f}")
            if one:
                out.append(f"m/1graph={m/one:.3f}")
            if chain:
                out.append(f"m/chain={m/chain:.3f}")
            if tchain:
                out.append(f"m/tBchain={m/tchain:.3f}")
            if tt:
                out.append(f"m/tB+tB={m/tt:.3f}")
        if rt:
            out.append(f"runtime_merged={rt:.1f}")
        if null:
            out.append(f"null={null:.1f}")
        print("  " + "  ".join(out))
    for line in open(f, errors="replace"):
        if line.startswith("#### "):
            flush(); sec = line.strip()[5:]; rows = {}
        elif line.startswith("KBJSON {\"tag\""):
            try:
                j = json.loads(line[7:])
                rows[j["variant"]] = j["med_us"]
            except Exception:
                pass
        elif line.startswith("CMP") and ("bitexact=no" in line) and "cpu_ref" not in line:
            print("  !! " + line.strip()[:140])
        elif line.startswith("CMP") and "NO tB" in line:
            print("  !! " + line.strip()[:140])
    flush()
