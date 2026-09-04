#!/usr/bin/env python3
"""Fetch the text-side vision routing bias (`ffn.gate.bias_vl`) from the HF
safetensors of deepseek-ai/DeepSeek-V4-Flash-Vision-Exp and write the
deepstrix sidecar `bias_vl.bin` (43 × 256 f32 little-endian, layer-major).

The unsloth GGUF upload (2026-09) lacks `blk.N.exp_probs_b_vl.bias`, so the
engine reads this sidecar instead:
    ~/.cache/deepstrix/models/<gguf-basename>/bias_vl.bin

Tensor names found in model.safetensors.index.json (2026-09-03):
    layers.{0..42}.ffn.gate.bias_vl   dtype F32  shape [256]
    (sibling keys: layers.N.ffn.gate.bias [256] F32 = exp_probs_b,
     layers.N.ffn.gate.tid2eid [129280, 6] I64 on layers 0-2,
     layers.N.ffn.gate.weight [256, 4096] BF16)
The 43 tensors are spread over 46 shards (model-000NN-of-00048.safetensors);
each is fetched with a single ~1 KiB HTTP range request after reading the
shard's JSON header. Only stdlib is used.

Usage:
    python3 scripts/fetch_bias_vl.py --gguf /persist/.../DeepSeek-V4-Flash-Vision-Exp-UD-Q2_K_XL-00001-of-00003.gguf
    python3 scripts/fetch_bias_vl.py --out /path/to/bias_vl.bin
"""
import argparse
import json
import os
import re
import struct
import sys
import urllib.request

REPO = "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
N_LAYERS = 43
N_EXPERTS = 256
NAME_RE = re.compile(r"^layers\.(\d+)\.ffn\.gate\.bias_vl$")


def http_get(url, rng=None, timeout=60):
    headers = {"User-Agent": "deepstrix-fetch-bias-vl/1"}
    tok = os.environ.get("HF_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    if rng is not None:
        headers["Range"] = "bytes=%d-%d" % rng
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def shard_header(shard, cache):
    if shard in cache:
        return cache[shard]
    url = BASE + shard
    n = struct.unpack("<Q", http_get(url, (0, 7)))[0]
    hdr = json.loads(http_get(url, (8, 8 + n - 1)))
    cache[shard] = (n, hdr)
    return cache[shard]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", help="text GGUF path; sidecar goes to ~/.cache/deepstrix/models/<stem>/bias_vl.bin")
    ap.add_argument("--out", help="explicit output path (overrides --gguf)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.out:
        out = args.out
    elif args.gguf:
        stem = os.path.basename(args.gguf)
        stem = stem[:-5] if stem.endswith(".gguf") else stem
        out = os.path.expanduser(f"~/.cache/deepstrix/models/{stem}/bias_vl.bin")
    else:
        ap.error("need --gguf or --out")
    if os.path.exists(out) and not args.force:
        print(f"exists: {out} (use --force)")
        return 0

    index = json.loads(http_get(BASE + "model.safetensors.index.json"))["weight_map"]
    found = {}
    for name, shard in index.items():
        m = NAME_RE.match(name)
        if m:
            found[int(m.group(1))] = (name, shard)
    missing = [l for l in range(N_LAYERS) if l not in found]
    if missing:
        sys.exit(f"missing bias_vl for layers {missing}; found {sorted(found)}")
    extra = sorted(l for l in found if l >= N_LAYERS)
    if extra:
        print(f"note: ignoring bias_vl for layers {extra} (beyond {N_LAYERS})")

    hdr_cache = {}
    buf = bytearray()
    for l in range(N_LAYERS):
        name, shard = found[l]
        n, hdr = shard_header(shard, hdr_cache)
        ent = hdr[name]
        assert ent["dtype"] == "F32", (name, ent)
        assert ent["shape"] == [N_EXPERTS], (name, ent)
        a, b = ent["data_offsets"]
        assert b - a == N_EXPERTS * 4
        data = http_get(BASE + shard, (8 + n + a, 8 + n + b - 1))
        assert len(data) == N_EXPERTS * 4, (name, len(data))
        vals = struct.unpack("<%df" % N_EXPERTS, data)
        assert all(v == v and abs(v) < 1e6 for v in vals), name
        buf += data
        print(f"layer {l:2d}  {shard}  {name}  min={min(vals):+.4f} max={max(vals):+.4f}")
    assert len(buf) == N_LAYERS * N_EXPERTS * 4
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf)
    os.replace(tmp, out)
    print(f"wrote {out} ({len(buf)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
