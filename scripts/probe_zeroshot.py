#!/usr/bin/env python3
"""Zero-shot router probe: how well does layer L's OWN gate do on layer 19/20's
activation? Reads a `V41_PROBE_DUMP` file; no training, no second model run.

This is the measurement that prices the whole probe idea. Under a prior, the
data a probe needs scales with the SQUARE of the correction's norm, so a good
zero-shot is worth ~10x less data than a bad one -- and a good ENOUGH zero-shot
needs no probe at all (break-even is ~20-35% precision; see
docs/v41/PREFETCH_STUDY_2026-09-18.md).

    gate:  scores = sqrt(softplus(W_L @ x / gate_temp));  pick = topk(scores + b_L)
    x is ALREADY ffn_norm'd -- that is what the dump stores.

Usage: probe_zeroshot.py <dump> [--model DIR] [--limit N]
       probe_zeroshot.py --check-weights [--model DIR]
"""
import json, os, struct, sys
import numpy as np

DEF_MODEL = os.path.expanduser("~/.cache/deepstrix/models/dsv4.1f")
GATE_TEMP = 1.0          # reference default; absent from config.json

# ---------------- safetensors ----------------
_DT = {"F32": np.float32, "F16": np.float16, "BF16": None, "F8_E4M3": None, "U8": np.uint8, "I8": np.int8}

class St:
    def __init__(self, d):
        self.d = os.path.realpath(d)
        ip = os.path.join(self.d, "model.safetensors.index.json")
        self.map = json.load(open(ip))["weight_map"] if os.path.exists(ip) else None
        self._hdr = {}

    def _header(self, shard):
        if shard not in self._hdr:
            p = os.path.realpath(os.path.join(self.d, shard))
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._hdr[shard] = (json.loads(f.read(n)), 8 + n, p)
        return self._hdr[shard]

    def get(self, name):
        shard = self.map[name]
        hdr, base, path = self._header(shard)
        e = hdr[name]
        a, b = e["data_offsets"]
        with open(path, "rb") as f:
            f.seek(base + a)
            raw = f.read(b - a)
        dt, shape = e["dtype"], e["shape"]
        if dt == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(shape)
        if dt in ("F32", "F16"):
            return np.frombuffer(raw, dtype=_DT[dt]).astype(np.float32).reshape(shape)
        raise SystemExit(f"{name}: unhandled dtype {dt} (shape {shape}) -- needs dequant")

def gates(st, layers):
    """-> {layer: (W [n_expert, dim] f32, b [n_expert] f32)}"""
    out = {}
    for L in layers:
        out[L] = (st.get(f"layers.{L}.ffn.gate.weight"), st.get(f"layers.{L}.ffn.gate.bias"))
    return out

def score(W, b, X):
    """The reference gate, exactly: sqrtsoftplus then bias, bias steers only."""
    z = (X @ W.T) / GATE_TEMP
    s = np.sqrt(np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0))   # stable softplus
    return s + b

# ---------------- dump reader ----------------
def read_dump(path, limit=None):
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"DSPROBE1":
            raise SystemExit(f"{path}: bad magic {magic!r}")
        dim, n_used, dst0, n_dst, n_src = struct.unpack("<5I", f.read(20))
        src = list(struct.unpack(f"<{n_src}I", f.read(4 * n_src)))
        rec = 8 + n_src * dim * 2 + n_dst * n_used * 2
        blob = f.read()
    n = len(blob) // rec
    if limit:
        n = min(n, limit)
    acts = np.zeros((n, n_src, dim), dtype=np.float32)
    picks = np.zeros((n, n_dst, n_used), dtype=np.int16)
    phase = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        o = i * rec
        phase[i] = blob[o]
        a0 = o + 8
        acts[i] = np.frombuffer(blob, dtype=np.float16, count=n_src * dim, offset=a0).astype(np.float32).reshape(n_src, dim)
        p0 = a0 + n_src * dim * 2
        picks[i] = np.frombuffer(blob, dtype=np.int16, count=n_dst * n_used, offset=p0).reshape(n_dst, n_used)
    return dict(src=src, dst0=dst0, n_dst=n_dst, n_used=n_used, dim=dim,
                acts=acts, picks=picks, phase=phase)

def main():
    model = DEF_MODEL
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]
    st = St(model)

    if "--check-weights" in sys.argv:
        # Run this BEFORE collecting data: proves the gate tensors load and the
        # bias really is the load-balancing term it is claimed to be.
        g = gates(st, [20, 29, 39])
        for L, (W, b) in g.items():
            print(f"L{L}: W {W.shape} {W.dtype}  |W|_F {np.linalg.norm(W):8.2f}   "
                  f"b {b.shape}  mean {b.mean():+.4f}  std {b.std():.4f}  "
                  f"min {b.min():+.4f}  max {b.max():+.4f}")
        W20, W39 = g[20][0], g[39][0]
        cs = (W20 * W39).sum() / (np.linalg.norm(W20) * np.linalg.norm(W39))
        print(f"\ncos(W20, W39) = {cs:+.4f}   (how differently the two ends of the decoder route)")
        return

    dump = sys.argv[1]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    d = read_dump(dump, limit)
    n = len(d["acts"])
    print(f"samples {n:,}   src layers {d['src']}   targets L{d['dst0']}..{d['dst0']+d['n_dst']-1}")
    print(f"   decode {int((d['phase']==0).sum()):,}   prefill {int((d['phase']==1).sum()):,}\n")
    tgt = list(range(d["dst0"], d["dst0"] + d["n_dst"]))
    G = gates(st, tgt)
    print(f"{'src':>4} {'dst':>4} {'top6':>7} {'r@8':>7} {'r@32':>7} {'r@64':>7} {'bias-only r@32':>15}")
    for si, s in enumerate(d["src"]):
        for L in tgt:
            if L not in (d["dst0"], d["dst0"] + d["n_dst"] // 2, d["dst0"] + d["n_dst"] - 1):
                continue
            W, b = G[L]
            sc = score(W, b, d["acts"][:, si, :])
            truth = d["picks"][:, L - d["dst0"], :].astype(np.int64)
            hit = {}
            for k in (6, 8, 32, 64):
                top = np.argpartition(-sc, k, axis=1)[:, :k]
                hit[k] = np.mean([len(set(top[i]) & set(truth[i][truth[i] >= 0])) /
                                  max(1, (truth[i] >= 0).sum()) for i in range(n)])
            bo = np.argpartition(-b, 32)[:32]
            bhit = np.mean([len(set(bo) & set(truth[i][truth[i] >= 0])) /
                            max(1, (truth[i] >= 0).sum()) for i in range(n)])
            print(f"{s:>4} {L:>4} {hit[6]:>7.3f} {hit[8]:>7.3f} {hit[32]:>7.3f} "
                  f"{hit[64]:>7.3f} {bhit:>15.3f}")

main()
