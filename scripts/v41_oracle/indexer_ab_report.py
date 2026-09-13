"""Turn an `indexer_ab.py` run directory into the tables for
docs/v41/INDEXER_DENSE_VS_SPARSE.md.

  python3 indexer_ab_report.py ~/.cache/deepstrix/v41/dense_ab_4096
"""
import glob
import json
import math
import os
import sys

d = sys.argv[1]
cfg = json.load(open(os.path.expanduser("~/.cache/deepstrix/models/dsv4.1f/inference/config.json")))
TOPK, RATIOS = cfg["index_topk"], cfg["compress_ratios"][: cfg["n_layers"]]

rowdirs = sorted(glob.glob(os.path.join(d, "row*_*")))
teach = [json.load(open(os.path.join(r, "teacher.json"))) for r in rowdirs]
T = len(teach[0]["nll"])
modes = [t["mode"] for t in teach]
print(f"{d}\nT={T} positions, rows {modes}\n")

EDGES = [e for e in (0, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192) if e < T] + [T]


def band(v, lo, hi):
    s = v[lo:hi]
    return sum(s) / len(s)


# ---------------------------------------------------------------- 1. quality
print("A. Teacher-forced quality per row (NLL in nats of the true next token, top-1 accuracy).")
print("   The duplicate control row is the same math as row 0; its gap is pure arithmetic noise.")
hdr = f"{'positions':>14} {'ctx':>6} {'r1 store/512':>12} {'r2 store/512':>12} " + \
      "".join(f"{'NLL row'+str(i):>11}" for i in range(len(teach))) + \
      "".join(f"{'dNLL r'+str(i)+' +-se':>15}" for i in range(1, len(teach))) + \
      "".join(f"{'acc r'+str(i):>9}" for i in range(len(teach)))
print(hdr)
def paired(i, lo, hi):
    """mean and standard error of the per-position NLL difference row i - row 0"""
    dd = [teach[i]["nll"][p] - teach[0]["nll"][p] for p in range(lo, hi)]
    n = len(dd)
    mu = sum(dd) / n
    var = sum((x - mu) ** 2 for x in dd) / max(n - 1, 1)
    return mu, math.sqrt(var / n)


for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    mid = (lo + hi) // 2
    hh = min(hi, T - 1)
    nl = [band(t["nll"], lo, hh) for t in teach]
    ac = [band(t["hit"], lo, hh) for t in teach]
    pr = [paired(i, lo, hh) for i in range(1, len(teach))]
    print(f"{lo:>6}..{hi-1:<7} {mid+1:>6} {(mid+1)/TOPK:>12.2f} {((mid+1)//2)/TOPK:>12.2f} "
          + "".join(f"{x:>11.4f}" for x in nl)
          + "".join(f"{m:>+7.4f}+-{e:.4f}" for m, e in pr)
          + "".join(f"{x:>9.4f}" for x in ac))
nl = [sum(t["nll"][: T - 1]) / (T - 1) for t in teach]
ac = [sum(t["hit"][: T - 1]) / (T - 1) for t in teach]
pr = [paired(i, 0, T - 1) for i in range(1, len(teach))]
print("  whole prompt:".ljust(42) + "".join(f"{x:>11.4f}" for x in nl)
      + "".join(f"{m:>+7.4f}+-{e:.4f}" for m, e in pr)
      + "".join(f"{x:>9.4f}" for x in ac))

# ------------------------------------------------- 2. pairwise divergence
for f in sorted(glob.glob(os.path.join(d, "compare*.json"))):
    c = json.load(open(f))
    tag = "row0 vs " + ("row1 (%s)" % c["rows"][1] if "rows" in c else os.path.basename(f))
    agree, kl = c["top1_agree"], c["kl_sd"]
    first = next((i for i, v in enumerate(agree) if not v), None)
    print(f"\nB. {tag}: overall top-1 agreement {sum(agree)/T:.4f}; "
          f"first top-1 flip at position {first} (context {None if first is None else first+1});"
          f" mean KL {sum(kl)/T:.5f} nats; mean TV {sum(c['tv'])/T:.5f}")
    print(f"{'positions':>14} {'ctx':>6} {'agree':>8} {'meanKL':>9} {'medKL':>9} {'p90KL':>9} "
          f"{'maxKL':>9} {'TV':>8} {'top5/5':>8} {'H(row0)':>8}")
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        mid = (lo + hi) // 2
        k = sorted(kl[lo:hi])
        n = len(k)
        print(f"{lo:>6}..{hi-1:<7} {mid+1:>6} {band(agree,lo,hi):>8.4f} {sum(k)/n:>9.5f} "
              f"{k[n//2]:>9.5f} {k[int(0.9*n)]:>9.5f} {k[-1]:>9.5f} {band(c['tv'],lo,hi):>8.4f} "
              f"{band(c['top5_agree'],lo,hi):>8.3f} {band(c['ent_sparse'],lo,hi):>8.3f}")

# ------------------------------------------------- 3. attention-mass probe
mp = os.path.join(d, "attn_mass.pt")
if os.path.exists(mp):
    import torch
    mass = torch.load(mp)
    dense_rows = [i for i, m in enumerate(modes) if m == "dense"]
    print("\nC. Attention mass the DENSE row puts on compressed rows its own indexer would have dropped\n"
          "   (fraction of each query's total attention probability; chaos-free, read off the softmax).\n"
          "   'of compressed' = the same mass as a share of all compressed-row mass.")
    bands = [(lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    print(f"{'layer':>6} {'ratio':>5} " + " ".join(f"{str(lo)+'..'+str(hi-1):>14}" for lo, hi in bands))
    for L in sorted(mass):
        dr, cm = mass[L]
        for b in dense_rows:
            cells = []
            for lo, hi in bands:
                x = dr[b, lo:hi].float().mean().item()
                y = cm[b, lo:hi].float().mean().item()
                cells.append(f"{x*100:>6.2f}% /{(x/max(y,1e-9))*100:>5.1f}%")
            print(f"L{L:02d}".rjust(6) + f" {RATIOS[L]:>5} " + " ".join(f"{c:>14}" for c in cells))

# ---------------------------------------------------------------- 4. rows
r = json.load(open(os.path.join(d, "rows.json")))
print("\nC. Compressed rows scored by each index-source layer in this prefill "
      f"(prompt of {T} tokens; the store is the whole prompt, queries see a causal prefix of it):")
for layer, ratio, n, vis, sel in r:
    print(f"  L{layer:02d} ratio={ratio}  store {n:>6} rows  mean visible/query {vis:>9.1f}  "
          f"mean kept by the indexer {sel:>7.1f}  dense/sparse {vis/max(sel,1e-9):>6.2f}x")
