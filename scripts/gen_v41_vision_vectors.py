#!/usr/bin/env python3
"""Dump CANONICAL DeepSeek-V4.1 vision-tower vectors for
`crates/v4flash-vision/tests/canonical_v41.rs`, plus the V4.1 image-processor
layout table for `tests/layout_v41_vectors.rs`.

Loads `vision.*` / `aligner.*` / `image_{start,newline,end}` straight from the
HF safetensors (bf16, zero-copy mmap via scripts/v41_oracle/loader.py) into the
UNMODIFIED reference modules `inference/vision.py::{ViT, Aligner}` (f32 on the
CPU), runs `inference/image_processor.load_image` + `image_token_types`, and
writes, per case <tag>:

    <tag>.json          grid dims, block `types`, `perm` (identity for V4.1), timings
    <tag>.patches.f32   [n][588]      ViT input (bf16-rounded, as the reference makes it)
    <tag>.hidden.f32    [n][1024]     post-`vision.norm`
    <tag>.aligner.f32   [n_llm][5120] aligner rows, row-major over the LLM grid
    <tag>.block.f32     [n_block][5120] the merged span (`merge_image_embeddings`)

and `layout_cases_v41.json` — `plan_image_grid` over a size sweep (pure
integer/float math from the reference, no pixels).

Cases: `synth4x6` (the LCG grid `tower_encode.rs::synth_image(4, 6)` uses),
`png640x480` (a generated deterministic PNG), and every `--image` given
(default: the checkpoint's own `inference/examples/images/{corn,carrots}.jpeg`).

Memory: ~1.9 GB of f32 weights + activations (< 3.5 GB for the 3774-patch
carrots image). NEVER loads the language model. Run beside the server with:

  cd scripts && choom -n 1000 nix-shell -p python3Packages.torch python3Packages.numpy \
      python3Packages.pillow --run "python3 gen_v41_vision_vectors.py --out ~/.cache/deepstrix/v41/vision_canon"
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("V41_MODEL", os.path.expanduser("~/.cache/deepstrix/models/dsv4.1f"))
sys.path.insert(0, os.path.join(HERE, "v41_oracle"))  # loader.py (mmap safetensors reader)
sys.path.insert(1, os.path.join(MODEL, "inference"))  # vision.py, image_processor.py


class Args:
    """The `vision_*` / `dim` / `image_token_id` fields of `ModelArgs`, from inference/config.json."""

    def __init__(self, cfg):
        for k, v in cfg.items():
            if k.startswith("vision_") or k in ("dim", "image_token_id"):
                setattr(self, k, v)
        self.vision_max_wh_ratio = cfg.get("vision_max_wh_ratio", None)


def make_png(path):
    """Deterministic 640x480: sinusoidal gradients + circle, rect, triangle, diagonals
    (same picture as scripts/gen_canonical_vision_vectors.py)."""
    from PIL import Image, ImageDraw

    W, H = 640, 480
    img = Image.new("RGB", (W, H))
    px = img.load()
    for y in range(H):
        for x in range(W):
            fx, fy = x / W, y / H
            r = 0.5 + 0.35 * math.sin(6.0 * fx) * math.cos(4.0 * fy)
            g = 0.5 + 0.30 * math.hypot(fx - 0.4, fy - 0.6)
            b = 0.5 + 0.25 * math.sin(9.0 * (fx + fy))
            px[x, y] = tuple(int(max(0.0, min(1.0, v)) * 255) for v in (r, g, b))
    d = ImageDraw.Draw(img)
    d.ellipse([80, 60, 300, 260], fill=(230, 40, 40), outline=(0, 0, 0), width=5)
    d.rectangle([340, 120, 580, 300], fill=(30, 90, 220), outline=(255, 255, 0), width=7)
    d.polygon([(160, 440), (320, 300), (480, 440)], fill=(20, 200, 90))
    d.line([(0, 0), (W, H)], fill=(255, 255, 255), width=3)
    d.line([(0, H), (W, 0)], fill=(0, 0, 0), width=3)
    img.save(path, "PNG", compress_level=6)
    return img.size


def lcg_patches(n_h, n_w):
    """The exact grid `tower_encode.rs::synth_image` builds (same LCG, same constants)."""
    st = np.uint32(0x12345678)
    v = np.empty(n_h * n_w * 3 * 14 * 14, dtype=np.float32)
    with np.errstate(over="ignore"):
        for i in range(v.size):
            st = np.uint32(np.uint32(st * np.uint32(1664525)) + np.uint32(1013904223))
            v[i] = (np.float32(st >> np.uint32(8)) / np.float32(1 << 24)) * 2.0 - 1.0
    return v.reshape(n_h * n_w, 3, 14, 14)


LAYOUT_SIZES = [
    # (width, height): tiny, min_pixels edges, typical, the two example photos, big, extreme aspect
    (1, 1), (7, 9), (28, 42), (100, 100), (543, 543), (544, 544), (545, 545), (300, 700),
    (640, 480), (480, 640), (450, 308), (1024, 701), (800, 600), (1024, 768), (1280, 720),
    (1920, 1080), (1080, 1920), (2000, 2000), (2560, 1440), (3840, 2160), (4000, 3000),
    (3000, 4000), (8000, 8000), (1000, 3000), (3000, 1000), (100, 8000), (8000, 100),
    (10, 5000), (5000, 10), (1, 1000), (1000, 1), (1, 100000), (100000, 1), (20000, 30),
    (30, 20000), (333, 777), (777, 333), (1023, 1), (1, 1023), (2048, 2048), (1366, 768),
    (600, 1200), (1200, 600), (5000, 5000), (12000, 300), (300, 12000), (999, 1001),
    (1001, 999), (2222, 3333), (14, 14), (15, 15), (41, 43), (42, 42), (43, 41),
]


def dump_layout_cases(IP, args, out):
    cases = []
    for w, h in LAYOUT_SIZES:
        n_llm_h, n_llm_w, best_h, best_w = IP.plan_image_grid(w, h, args)
        p = args.vision_patch_size
        types = IP.image_token_types(n_llm_h, n_llm_w).tolist()
        cases.append(dict(width=w, height=h, best_h=best_h, best_w=best_w,
                          n_vit_h=best_h // p, n_vit_w=best_w // p, n_llm_h=n_llm_h, n_llm_w=n_llm_w,
                          n_tokens=IP.num_image_tokens(n_llm_h, n_llm_w), types=types))
    json.dump(dict(patch=args.vision_patch_size, downsample=args.vision_downsample_ratio,
                   max_n_token=args.vision_max_n_token, min_pixels=args.vision_min_pixels,
                   max_wh_ratio=args.vision_max_wh_ratio, image_token_id=args.image_token_id,
                   types_enum=dict(IMAGE_START=IP.IMAGE_START, IMAGE=IP.IMAGE,
                                   IMAGE_NEW_LINE=IP.IMAGE_NEW_LINE, IMAGE_END=IP.IMAGE_END),
                   cases=cases), open(out, "w"), indent=1)
    print(f"layout: {len(cases)} cases -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", action="append", default=None,
                    help="image file(s) to run (default: the checkpoint's example photos)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-tower", action="store_true", help="only write layout_cases_v41.json")
    a = ap.parse_args()

    import torch
    import image_processor as IP
    import vision as V
    from loader import Checkpoint

    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(os.path.join(MODEL, "inference", "config.json")))
    args = Args(cfg)
    print(f"config: vision {args.vision_n_layers}L dim {args.vision_dim} heads {args.vision_n_heads} "
          f"inter {args.vision_inter_dim} patch {args.vision_patch_size} theta {args.vision_rope_theta} "
          f"ds {args.vision_downsample_ratio} max_n_token {args.vision_max_n_token} "
          f"min_pixels {args.vision_min_pixels} wh_ratio {args.vision_max_wh_ratio} text dim {args.dim} "
          f"image_token_id {args.image_token_id}", flush=True)

    dump_layout_cases(IP, args, os.path.join(a.out, "layout_cases_v41.json"))
    if a.skip_tower:
        return

    ck = Checkpoint(MODEL)
    def T(name):
        t = ck.get(name)
        assert t.dtype == torch.bfloat16, (name, t.dtype)
        return t.float()

    t0 = time.time()
    vit = V.ViT(args)
    vit.patch_embed.proj.weight.copy_(T("vision.patch_embed.proj.weight"))
    vit.patch_embed.proj.bias.copy_(T("vision.patch_embed.proj.bias"))
    for l, blk in enumerate(vit.blocks):
        p = f"vision.blocks.{l}."
        blk.norm1.weight.copy_(T(p + "norm1.weight"))
        blk.norm2.weight.copy_(T(p + "norm2.weight"))
        blk.attn.wqkv.weight.copy_(T(p + "attn.wqkv.weight"))
        blk.attn.wqkv.bias.copy_(T(p + "attn.wqkv.bias"))
        blk.attn.wo.weight.copy_(T(p + "attn.wo.weight"))
        blk.attn.wo.bias.copy_(T(p + "attn.wo.bias"))
        blk.mlp.w1.weight.copy_(T(p + "mlp.w1.weight"))
        blk.mlp.w2.weight.copy_(T(p + "mlp.w2.weight"))
    vit.norm.weight.copy_(T("vision.norm.weight"))
    aligner = V.Aligner(args)
    aligner.w1.weight.copy_(T("aligner.w1.weight"))
    aligner.w1.bias.copy_(T("aligner.w1.bias"))
    aligner.w2.weight.copy_(T("aligner.w2.weight"))
    aligner.w2.bias.copy_(T("aligner.w2.bias"))
    SENT = {IP.IMAGE_START: T("image_start"), IP.IMAGE_NEW_LINE: T("image_newline"), IP.IMAGE_END: T("image_end")}
    vit.eval()
    aligner.eval()
    n_par = sum(p.numel() for p in vit.parameters()) + sum(p.numel() for p in aligner.parameters())
    print(f"weights loaded: {n_par:,} params in {time.time() - t0:.1f} s", flush=True)

    def dump(tag, patches, nvh, nvw, nlh, nlw, src=None):
        x = patches.float().contiguous().reshape(patches.shape[0], -1)
        t0 = time.time(); hidden = vit(x, nvh, nvw)
        t1 = time.time(); rows = aligner(hidden, nvh, nvw); t2 = time.time()
        assert rows.shape == (nlh * nlw, args.dim), (rows.shape, nlh, nlw)
        types = IP.image_token_types(nlh, nlw)
        block = torch.empty(types.numel(), args.dim)
        for t, vec in SENT.items():
            block[types == t] = vec
        block[types == IP.IMAGE] = rows  # aligner rows in reading order (merge_image_embeddings)
        assert int((types == IP.IMAGE).sum()) == rows.shape[0]
        json.dump(dict(tag=tag, source=src, n_vit_h=int(nvh), n_vit_w=int(nvw), n_llm_h=int(nlh), n_llm_w=int(nlw),
                       n_patches=int(x.shape[0]), text_dim=args.dim, types=[int(t) for t in types],
                       perm=list(range(nlh * nlw)), n_block=int(types.numel()),
                       vit_ms=(t1 - t0) * 1e3, aligner_ms=(t2 - t1) * 1e3),
                  open(f"{a.out}/{tag}.json", "w"), indent=1)
        for name, arr in (("patches", x), ("hidden", hidden), ("aligner", rows), ("block", block)):
            arr.reshape(-1).numpy().astype("<f4").tofile(f"{a.out}/{tag}.{name}.f32")
        print(f"[{tag}] {tuple(x.shape)} grid {nvh}x{nvw} -> llm {nlh}x{nlw} = {rows.shape[0]} rows, "
              f"block {types.numel()} tokens | vit {(t1-t0)*1e3:.0f} ms aligner {(t2-t1)*1e3:.0f} ms | "
              f"aligner mean {rows.mean():+.6f} std {rows.std():.6f} | rss {rss_mb():.0f} MB", flush=True)

    dump("synth4x6", torch.from_numpy(lcg_patches(4, 6)), 4, 6, 2, 2)

    png = f"{a.out}/png640x480.png"
    print("png:", make_png(png), flush=True)
    images = [("png640x480", png)]
    for path in (a.image or [os.path.join(MODEL, "inference", "examples", "images", f) for f in ("corn.jpeg", "carrots.jpeg")]):
        tag = os.path.splitext(os.path.basename(path))[0].replace(".", "_")
        images.append((tag, path))
    for tag, path in images:
        patches, nvh, nvw, nlh, nlw = IP.load_image({"url": path}, args)
        dump(tag, patches, nvh, nvw, nlh, nlw, src=os.path.abspath(path))
    print("done", flush=True)


def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


if __name__ == "__main__":
    main()
