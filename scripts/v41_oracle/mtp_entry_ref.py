#!/usr/bin/env python3
"""Reference for the DSpark drafter ENTRY stage:

    main_x = mtp.0.main_norm(mtp.0.main_proj(main_hidden))

`main_proj.weight` is F8_E4M3 with a SEPARATE `main_proj.scale` [160, 480] of
F8_E8M0 32x32 block scales.  `Checkpoint.get` is a zero-copy view in the
on-disk dtype and does NOT apply those scales -- multiplying by the raw e4m3
bytes gives a plausible-looking but wrong projection (cos 0.946 vs the engine).
This script does the dequantisation, so it needs numpy only, no torch: the one
part that needs the real model is `main_hidden`, which dump_mtp_ref.py writes.

    nix-shell -p python3Packages.numpy --run \
      'python3 scripts/v41_oracle/mtp_entry_ref.py'
"""
import argparse, json, os, struct, sys
import numpy as np

BLK = 32


def _e4m3_lut():
    """torch float8_e4m3fn: bias 7, finite-only (S.1111.111 is NaN, no inf)."""
    out = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if (b >> 7) else 1.0
        e, m = (b >> 3) & 0xF, b & 0x7
        if e == 0:
            v = m * 2.0 ** -9          # subnormal: 2^(1-7) * m/8
        elif e == 0xF and m == 0x7:
            v = np.nan
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        out[b] = s * v
    return out


def _e8m0(b):
    """torch float8_e8m0fnu: 2^(e-127); 0xFF is NaN."""
    v = np.exp2(b.astype(np.float32) - 127.0)
    return np.where(b == 0xFF, np.nan, v)


class Raw:
    """Minimal safetensors reader: returns the raw bytes of a tensor."""

    def __init__(self, root):
        self.root = root
        self.index = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
        self._hdr = {}

    def _shard(self, fname):
        if fname not in self._hdr:
            path = os.path.join(self.root, fname)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._hdr[fname] = (json.loads(f.read(n)), 8 + n, path)
        return self._hdr[fname]

    def get(self, name, dtype):
        hdr, base, path = self._shard(self.index[name])
        meta = hdr[name]
        a, b = meta["data_offsets"]
        with open(path, "rb") as f:
            f.seek(base + a)
            buf = f.read(b - a)
        return np.frombuffer(buf, dtype=dtype).reshape(meta["shape"])


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/.cache/deepstrix/models/dsv4.1f"))
    ap.add_argument("--dump", default=os.path.expanduser(
        "~/.cache/deepstrix/v41/agentic/main/mtp_ref"))
    a = ap.parse_args()

    mh = np.fromfile(f"{a.dump}/main_hidden.bin", dtype=np.float32)
    ck = Raw(a.model)
    wb = ck.get("mtp.0.main_proj.weight", np.uint8)       # [5120, 15360] e4m3
    sb = ck.get("mtp.0.main_proj.scale", np.uint8)        # [160, 480]    e8m0
    wn = bf16_to_f32(ck.get("mtp.0.main_norm.weight", np.uint16))
    out_f, in_f = wb.shape
    assert mh.shape == (in_f,), f"main_hidden {mh.shape} vs weight in {in_f}"
    assert sb.shape == (out_f // BLK, in_f // BLK), f"scale {sb.shape}"

    # Per-32-block dots in f32 (exact enough over 32 terms), then scale and
    # accumulate across the 480 blocks in f64 so the reference is not itself
    # limited by f32 accumulation over 15360 terms.
    w = _e4m3_lut()[wb].reshape(out_f // BLK, BLK, in_f // BLK, BLK)
    x = mh.reshape(in_f // BLK, BLK)
    part = np.einsum("iajb,jb->iaj", w, x, optimize=True).astype(np.float64)
    proj = (part * _e8m0(sb).astype(np.float64)[:, None, :]).sum(-1).reshape(out_f)

    eps = 1e-20
    main_x = proj * (1.0 / np.sqrt((proj * proj).mean() + eps)) * wn.astype(np.float64)
    proj32, mx32 = proj.astype(np.float32), main_x.astype(np.float32)

    old = np.fromfile(f"{a.dump}/main_x.bin", dtype=np.float32)
    cos = float(old @ mx32 / (np.linalg.norm(old) * np.linalg.norm(mx32)))
    print(f"proj    mean {proj32.mean():12.6f} std {proj32.std():12.6f} absmax {np.abs(proj32).max():12.6f}")
    print(f"main_x  mean {mx32.mean():12.6f} std {mx32.std():12.6f} absmax {np.abs(mx32).max():12.6f}")
    print(f"cos(new, previous unscaled main_x.bin) = {cos:.6f}")

    proj32.tofile(f"{a.dump}/proj.bin")
    mx32.tofile(f"{a.dump}/main_x.bin")
    print(f"rewrote {a.dump}/proj.bin and main_x.bin (block scales APPLIED)")


if __name__ == "__main__":
    sys.exit(main())
