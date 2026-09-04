#!/usr/bin/env python3
"""Regenerate crates/v4flash-vision/tests/data/pillow_resize_cases.json.

Emits real-Pillow reference digests for `Image.resize` (default BICUBIC)
and `ImageOps.pad(..., color=(127,127,127))` — the two PIL operations
`v4flash_vision::resize` reimplements. Run:

    nix-shell -p 'python3.withPackages(ps: [ps.pillow])' \
      --run 'python3 scripts/gen_pillow_resize_vectors.py' \
      > crates/v4flash-vision/tests/data/pillow_resize_cases.json

Cases include letterbox deltas = 3 (mod 4), where Pillow's `round()`
paste offset differs from the `int()` truncation older versions used.
"""
import json
from PIL import Image, ImageOps
import PIL

def fnv1a64(b: bytes) -> str:
    h = 0xcbf29ce484222325
    for x in b:
        h ^= x
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h

def pattern(x, y, c):
    return ((x * 7 + y * 13 + c * 101) & 255)

def make(w, h, seed=0):
    im = Image.new("RGB", (w, h))
    px = im.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (pattern(x + seed, y, 0), pattern(x, y + seed, 1), pattern(x, y, 2))
    return im

specs = [
    (30, 20, 14, 14, "resize"), (20, 30, 42, 42, "resize"),
    (100, 61, 56, 28, "resize"), (7, 9, 70, 90, "resize"),
    (63, 63, 63, 28, "resize"), (63, 63, 28, 63, "resize"),
    (37, 41, 37, 41, "resize"), (256, 144, 112, 42, "resize"),
    (100, 50, 42, 42, "pad"), (50, 100, 42, 42, "pad"),
    (101, 33, 70, 28, "pad"), (33, 101, 28, 70, "pad"),
    (99, 100, 42, 42, "pad"), (100, 99, 42, 42, "pad"),
    (17, 5, 140, 42, "pad"), (5, 17, 42, 140, "pad"),
    (100, 40, 43, 14, "pad"), (40, 100, 14, 43, "pad"),
    (200, 91, 91, 42, "pad"), (640, 480, 126, 70, "pad"),
    (1920, 1080, 154, 84, "pad"),
]
cases = []
for (sw, sh, ow, oh, op) in specs:
    seed = sw % 5
    src = make(sw, sh, seed=seed)
    out = src.resize((ow, oh)) if op == "resize" else ImageOps.pad(src, (ow, oh), color=(127, 127, 127))
    data = out.tobytes()
    cases.append({
        "op": op, "src_w": sw, "src_h": sh, "out_w": ow, "out_h": oh, "seed": seed,
        "res_w": out.width, "res_h": out.height,
        "len": len(data), "fnv1a64": fnv1a64(data),
        "head": data[:24].hex(), "tail": data[-24:].hex(),
    })
print(json.dumps({"pillow": PIL.__version__, "pattern": "((x*7 + y*13 + c*101) & 255) with x+=seed for c=0, y+=seed for c=1", "cases": cases}, indent=1))
