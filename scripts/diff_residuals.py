#!/usr/bin/env python3
"""Diff per-layer residual dumps between ds4-CPU and deepstrix.

Usage: diff_residuals.py [DS4_DIR] [DEEPSTRIX_DIR]
Default: /tmp/ds4-dump /tmp/deepstrix-dump

Both dirs must contain files layer_NN_residual.bin (HC_DIM=16384 f32 LE
each). ds4 has layers 00..42 (43 files); deepstrix has 00..43 (44).
We compare the intersection 00..42.

Per-layer stats:
  - max |diff| (Linf), L2 norm of diff
  - cosine similarity (dot/||x||/||y||)
  - relative L2 = ||x-y|| / ||x||
"""
import os
import struct
import sys
from pathlib import Path

DEFAULT_DS4 = "/tmp/ds4-dump"
DEFAULT_DS = "/tmp/deepstrix-dump"
HC_DIM = 16384


def load(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != HC_DIM * 4:
        raise SystemExit(f"{path}: expected {HC_DIM*4} bytes, got {len(data)}")
    return list(struct.unpack(f"<{HC_DIM}f", data))


def diff(a, b):
    diffs = [x - y for x, y in zip(a, b)]
    linf = max(abs(d) for d in diffs)
    l2_diff = sum(d * d for d in diffs) ** 0.5
    l2_a = sum(x * x for x in a) ** 0.5
    l2_b = sum(y * y for y in b) ** 0.5
    dot = sum(x * y for x, y in zip(a, b))
    cos = dot / (l2_a * l2_b) if l2_a and l2_b else float("nan")
    rel = l2_diff / l2_a if l2_a else float("nan")
    return linf, l2_diff, rel, cos, l2_a, l2_b


def main():
    ds4_dir = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DS4)
    ds_dir = Path(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DS)

    print(f"comparing: {ds4_dir} (ds4-CPU)  vs  {ds_dir} (deepstrix)")
    print(f"{'layer':>5}  {'Linf':>10}  {'L2_diff':>10}  {'rel_L2':>10}  "
          f"{'cosine':>10}  {'||ds4||':>10}  {'||us||':>10}")
    print("-" * 80)

    for n in range(43):  # ds4 has 0..42
        nn = f"{n:02}"
        ds4_path = ds4_dir / f"layer_{nn}_residual.bin"
        ds_path = ds_dir / f"layer_{nn}_residual.bin"
        if not ds4_path.exists() or not ds_path.exists():
            print(f"{nn:>5}  missing")
            continue
        a = load(ds4_path)
        b = load(ds_path)
        linf, l2_diff, rel, cos, la, lb = diff(a, b)
        print(f"{nn:>5}  {linf:>10.4f}  {l2_diff:>10.4f}  {rel:>10.4e}  "
              f"{cos:>10.6f}  {la:>10.4f}  {lb:>10.4f}")


if __name__ == "__main__":
    main()
