"""Compare two oracle dump dirs layer by layer (e.g. CPU oracle vs the GPU
port, or two oracle runs for determinism). Reports per-layer max|Δ|, its
position, and Δ scaled by the vector's max magnitude — the same scoring the
standing V4-Flash prefill oracles use (5e-2 of scale is their bar).

Usage: python3 compare.py DIR_A DIR_B [--tol 5e-2]
"""
import argparse
import glob
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=float, default=5e-2)
    args = ap.parse_args()
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.a, "*.pt")))
    worst = 0.0
    print(f"{'tensor':24s} {'max|Δ|':>10s} {'scale':>10s} {'Δ/scale':>10s}  {'argmax'}")
    for n in names:
        pb = os.path.join(args.b, n)
        if not os.path.exists(pb):
            print(f"{n:24s} (missing in B)")
            continue
        ta = torch.load(os.path.join(args.a, n)).float()
        tb = torch.load(pb).float()
        if ta.shape != tb.shape:
            print(f"{n:24s} SHAPE {tuple(ta.shape)} vs {tuple(tb.shape)}")
            continue
        d = (ta - tb).abs()
        mx = d.max().item()
        scale = ta.abs().max().item() or 1.0
        rel = mx / scale
        worst = max(worst, rel)
        note = ""
        if n.startswith("logits"):
            note = "MATCH" if ta.argmax().item() == tb.argmax().item() else "DIFFER"
        flag = "  <-- over tol" if rel > args.tol else ""
        print(f"{n:24s} {mx:10.3e} {scale:10.3f} {rel:10.3e}  {note}{flag}")
    print(f"\nworst Δ/scale = {worst:.3e}  ({'PASS' if worst <= args.tol else 'FAIL'} at tol {args.tol:.0e})")


if __name__ == "__main__":
    main()
