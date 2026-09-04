#!/usr/bin/env python3
"""Dump CANONICAL vision-tower vectors for `tests/canonical_vs_python.rs`.

Loads `mmproj-F16.gguf` straight into the reference `inference/vision.py`
`ViT` / `Aligner` modules (f32 on CPU), runs `image_processor.load_image`
and `build_image_block`, and writes, per case <tag>:

    <tag>.json          grid dims, block `types`, `perm`
    <tag>.patches.f32   [n][588]    ViT input (bf16-rounded, as the reference makes it)
    <tag>.hidden.f32    [n][1024]   post-`v.post_ln`
    <tag>.aligner.f32   [n_llm][4096]
    <tag>.block.f32     [n_block][4096]

Cases: `synth4x6` (the LCG grid `tower_encode.rs::synth_image(4, 6)` uses)
and `real640x480` (a generated PNG with gradients + shapes).

Needs `vision.py` and `image_processor.py` from the DeepSeek reference on
`--ref-dir`. Run:

  nix-shell -p python3Packages.torch python3Packages.numpy python3Packages.pillow \
    --run "python3 scripts/gen_canonical_vision_vectors.py --out /tmp/canon \
             --ref-dir <dir with vision.py + image_processor.py>"

Then:

  DEEPSTRIX_MMPROJ=/persist/lumi/models/dsv4f-exp-q2-k-xl/mmproj-F16.gguf \
  DEEPSTRIX_VISION_DEVICE=1 CANON_DIR=/tmp/canon CANON_PNG=/tmp/canon/test640x480.png \
  cargo test --release -p v4flash-vision --test canonical_vs_python \
      -- --ignored --test-threads=1 --nocapture
"""
import argparse, io, json, math, os, struct, mmap, sys, time
import numpy as np

# ------------------------------------------------------------- GGUF reader
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
GGML_NP = {0: np.float32, 1: np.float16}


class Gguf:
    """Header parse + mmap'd f16/f32 tensor access. Numpy shape = reversed GGUF dims."""

    def __init__(self, path):
        self.f = open(path, 'rb')
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.p = 0
        assert self._rd(4) == b'GGUF'
        ver, = struct.unpack('<I', self._rd(4))
        assert ver == 3, ver
        nt, nkv = struct.unpack('<QQ', self._rd(16))
        self.kv = {}
        for _ in range(nkv):
            k = self._rstr()
            t, = struct.unpack('<I', self._rd(4))
            self.kv[k] = self._rval(t)
        self.tensors = {}
        for _ in range(nt):
            name = self._rstr()
            nd, = struct.unpack('<I', self._rd(4))
            dims = struct.unpack('<' + 'Q' * nd, self._rd(8 * nd))
            ty, = struct.unpack('<I', self._rd(4))
            off, = struct.unpack('<Q', self._rd(8))
            self.tensors[name] = (dims, ty, off)
        align = self.kv.get('general.alignment', 32)
        self.data0 = (self.p + align - 1) // align * align

    def _rd(self, n):
        b = self.mm[self.p:self.p + n]
        assert len(b) == n
        self.p += n
        return b

    def _rstr(self):
        n, = struct.unpack('<Q', self._rd(8))
        return self._rd(n).decode('utf-8', 'replace')

    def _rval(self, t):
        if t == 8:
            return self._rstr()
        if t == 9:
            et, = struct.unpack('<I', self._rd(4))
            n, = struct.unpack('<Q', self._rd(8))
            return [self._rval(et) for _ in range(n)]
        return struct.unpack('<' + FMT[t], self._rd(SZ[t]))[0]

    def get(self, name):
        dims, ty, off = self.tensors[name]
        n = 1
        for d in dims:
            n *= d
        a = np.frombuffer(self.mm, dtype=GGML_NP[ty], count=n, offset=self.data0 + off)
        return a.reshape(tuple(reversed(dims)))


class Args:
    """`clip.vision.*` from the mmproj KV, plus the text `dim`."""
    vision_patch_size = 14
    vision_dim = 1024
    vision_n_heads = 16
    vision_inter_dim = 2816
    vision_n_layers = 32
    vision_rope_theta = 10000.0
    vision_downsample_ratio = 3
    dim = 4096
    vision_max_wh_ratio = 8
    vision_min_pixels = 147456
    vision_max_n_token = 384


def make_png(path):
    """Deterministic 640x480: sinusoidal gradients + circle, rect, triangle, diagonals."""
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
    with np.errstate(over='ignore'):
        for i in range(v.size):
            st = np.uint32(np.uint32(st * np.uint32(1664525)) + np.uint32(1013904223))
            v[i] = (np.float32(st >> np.uint32(8)) / np.float32(1 << 24)) * 2.0 - 1.0
    return v.reshape(n_h * n_w, 3, 14, 14)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mmproj", default="/persist/lumi/models/dsv4f-exp-q2-k-xl/mmproj-F16.gguf")
    ap.add_argument("--ref-dir", required=True, help="dir holding vision.py + image_processor.py")
    a = ap.parse_args()

    sys.path.insert(0, a.ref_dir)
    import torch
    import vision as V
    import image_processor as IP

    torch.set_grad_enabled(False)
    torch.set_num_threads(min(12, os.cpu_count() or 8))
    os.makedirs(a.out, exist_ok=True)
    args = Args()
    g = Gguf(a.mmproj)
    T = lambda n: torch.from_numpy(np.ascontiguousarray(g.get(n)).astype(np.float32))

    vit = V.ViT(args)
    # GGUF [14,14,3,1024] -> numpy (1024, 3, 14, 14) -> (1024, 588), input order (c, y, x).
    vit.patch_embed.proj.weight.copy_(T("v.patch_embd.weight").reshape(1024, 3 * 14 * 14))
    vit.patch_embed.proj.bias.copy_(T("v.patch_embd.bias"))
    for l, blk in enumerate(vit.blocks):
        p = f"v.blk.{l}."
        blk.norm1.weight.copy_(T(p + "ln1.weight"))
        blk.norm2.weight.copy_(T(p + "ln2.weight"))
        blk.attn.wqkv.weight.copy_(torch.cat([T(p + "attn_q.weight"), T(p + "attn_k.weight"), T(p + "attn_v.weight")], 0))
        blk.attn.wqkv.bias.copy_(torch.cat([T(p + "attn_q.bias"), T(p + "attn_k.bias"), T(p + "attn_v.bias")], 0))
        blk.attn.wo.weight.copy_(T(p + "attn_out.weight"))
        blk.attn.wo.bias.copy_(T(p + "attn_out.bias"))
        # `MLP.w1` is one Linear(dim, 2*inter) that `chunk(2)`s into gate|up.
        blk.mlp.w1.weight.copy_(torch.cat([T(p + "ffn_gate.weight"), T(p + "ffn_up.weight")], 0))
        blk.mlp.w2.weight.copy_(T(p + "ffn_down.weight"))
    vit.norm.weight.copy_(T("v.post_ln.weight"))

    aligner = V.Aligner(args)
    aligner.w1.weight.copy_(T("mm.1.weight"))
    aligner.w1.bias.copy_(T("mm.1.bias"))
    aligner.w2.weight.copy_(T("mm.2.weight"))
    aligner.w2.bias.copy_(T("mm.2.bias"))
    SENT = {0: T("v.token_embd.img_start"), 1: T("v.token_embd.img_pad"),
            3: T("v.image_newline"), 4: T("v.token_embd.img_end")}
    vit.eval()
    aligner.eval()
    n_par = sum(p.numel() for p in vit.parameters()) + sum(p.numel() for p in aligner.parameters())
    print(f"weights loaded: {n_par:,} params", flush=True)

    def dump(tag, patches, nvh, nvw, nlh, nlw, start_pos=0):
        x = patches.float().contiguous()
        t0 = time.time(); hidden = vit(x, nvh, nvw)
        t1 = time.time(); rows = aligner(hidden, nvh, nvw); t2 = time.time()
        assert rows.shape == (nlh * nlw, 4096), (rows.shape, nlh, nlw)
        types, perm = IP.build_image_block(nlh, nlw, start_pos)
        block = torch.empty(len(types), 4096)
        k = 0
        for i, t in enumerate(types):
            if int(t) == IP.IMAGE:
                block[i] = rows[int(perm[k])]; k += 1
            else:
                block[i] = SENT[int(t)]
        assert k == len(perm)
        json.dump(dict(tag=tag, n_vit_h=int(nvh), n_vit_w=int(nvw), n_llm_h=int(nlh),
                       n_llm_w=int(nlw), n_patches=int(x.shape[0]), start_pos=start_pos,
                       types=[int(t) for t in types], perm=[int(v) for v in perm],
                       n_block=len(types), vit_ms=(t1 - t0) * 1e3, aligner_ms=(t2 - t1) * 1e3),
                  open(f"{a.out}/{tag}.json", "w"), indent=1)
        for name, arr in (("patches", x), ("hidden", hidden), ("aligner", rows), ("block", block)):
            arr.reshape(-1).numpy().astype("<f4").tofile(f"{a.out}/{tag}.{name}.f32")
        print(f"[{tag}] {tuple(x.shape)} grid {nvh}x{nvw} -> llm {nlh}x{nlw} = {rows.shape[0]} rows, "
              f"block {len(types)} tokens | vit {(t1-t0)*1e3:.0f} ms aligner {(t2-t1)*1e3:.0f} ms | "
              f"aligner mean {rows.mean():+.6f} std {rows.std():.6f}", flush=True)

    dump("synth4x6", torch.from_numpy(lcg_patches(4, 6)), 4, 6, 2, 2)

    png = f"{a.out}/test640x480.png"
    print("png:", make_png(png), flush=True)
    patches, nvh, nvw, nlh, nlw = IP.load_image({"url": png}, args)
    dump("real640x480", patches, nvh, nvw, nlh, nlw)
    print("done")


if __name__ == "__main__":
    main()
