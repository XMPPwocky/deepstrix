#!/usr/bin/env python3
"""Pure-Python (no torch / no PIL) re-implementation of the DeepSeek-V4-Flash
Vision-Exp reference layout functions — `grid_tokens`, `solve_resize_ratio`,
`safe_resize`, the integer prologue of `load_image`, and `build_image_block`
(all from `inference/image_processor.py`) — used to generate
`crates/v4flash-vision/tests/data/layout_cases.json`.

The torch ops in `build_image_block` are replaced by explicit index loops.
Everything else is verbatim math.

    python3 scripts/gen_vision_layout_vectors.py > crates/v4flash-vision/tests/data/layout_cases.json
"""
import json
import math
import sys

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4

PATCH = 14
DOWNSAMPLE = 3
MAX_N_TOKEN = 384
MAX_WH_RATIO = 8
MIN_PIXELS = 147456


def grid_tokens(best_height, best_width, patch_size, downsample_ratio):
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(max_w * patch_size * downsample_ratio / width,
                   max_h * patch_size * downsample_ratio / height)
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width, patch_size, downsample_ratio)
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width, patch_size, downsample_ratio)
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget)
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def plan(height, width):
    """Integer prologue of `load_image` (everything before the pixel ops)."""
    orig_w, orig_h = width, height
    if width > height * MAX_WH_RATIO:
        width = height * MAX_WH_RATIO
    if 0 < width * height < MIN_PIXELS:
        ratio = (MIN_PIXELS / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / PATCH) * PATCH
    best_height = math.ceil(height / PATCH) * PATCH
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height, width, best_height, best_width, PATCH, DOWNSAMPLE, MAX_N_TOKEN)
    n_vit_h, n_vit_w = best_height // PATCH, best_width // PATCH
    plain_resize = orig_w >= MAX_WH_RATIO * orig_h
    return dict(best_h=best_height, best_w=best_width, n_vit_h=n_vit_h, n_vit_w=n_vit_w,
                n_llm_h=n_llm_h, n_llm_w=n_llm_w, plain_resize=plain_resize)


def build_image_block(n_llm_h, n_llm_w, start_pos):
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_PAD] * (row_len * pad_h)
    # order = arange(rows*row_len).view(rows//2, 2, row_len).transpose(1, 2).reshape(-1)
    order = []
    for p in range(rows // 2):
        for c in range(row_len):
            for r in range(2):
                order.append((p * 2 + r) * row_len + c)
    # image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = arange(n_llm_h*n_llm_w)
    image_idx = [-1] * (rows * row_len)
    for r in range(n_llm_h):
        for c in range(n_llm_w):
            image_idx[r * row_len + c] = r * n_llm_w + c
    perm = [image_idx[o] for o in order if image_idx[o] >= 0]
    out = ([IMAGE_PAD] * compress_pad + [IMAGE_START] + [types[o] for o in order]
           + [IMAGE_PAD] * pad_last + [IMAGE_END])
    return out, perm


def main():
    sizes = []
    for s in [1, 13, 14, 15, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]:
        sizes.append((s, s))
    # (height, width) — aspect extremes and the reference examples
    sizes += [
        (1080, 1920), (1920, 1080), (768, 1024), (1024, 768), (384, 2208), (2208, 384),
        (64, 4096), (4096, 64), (1, 4096), (4096, 1), (100, 1000), (1000, 100),
        (100, 800), (100, 801), (100, 799), (7, 64), (64, 7), (300, 2400), (2400, 300),
        (720, 1280), (2160, 3840), (4096, 2048), (2048, 4096), (33, 999), (999, 33),
        (500, 4000), (4000, 500), (17, 4096), (4096, 17),
    ]
    cases = []
    for (h, w) in sizes:
        p = plan(h, w)
        blocks = []
        for sp in range(8):
            types, perm = build_image_block(p["n_llm_h"], p["n_llm_w"], sp)
            entry = {"start_pos": sp, "len": len(types),
                     "compress_pad": COMPRESS_PAD_TO - 1 - sp % COMPRESS_PAD_TO}
            if sp < 4:
                entry["types"] = types
            blocks.append(entry)
        _, perm = build_image_block(p["n_llm_h"], p["n_llm_w"], 0)
        cases.append({"height": h, "width": w, **p, "perm": perm, "blocks": blocks})
    json.dump({"patch": PATCH, "downsample": DOWNSAMPLE, "max_n_token": MAX_N_TOKEN,
               "max_wh_ratio": MAX_WH_RATIO, "min_pixels": MIN_PIXELS, "cases": cases},
              sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
