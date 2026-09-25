"""Expert-weight round trips for the oracle (`oracle.py --expert-rt`).

A covered routed expert's MXFP4 weights are replaced by what a ggml quantization
gives back (MXFP4 -> f32 -> ggml quantize -> f32), so the reference prices a
low-bit expert format end to end. Two sources:

  q8_0        quantized on the fly (0.14 s/expert);
  cache:DIR   pre-quantized bytes written by build_rt_cache.py
              (DIR/meta.json names the type; DIR/Lnn/Eeee.bin = w1|w3|w2).

Weights are dequantized in a thread pool (ctypes releases the GIL) a few experts
ahead of the forward loop, bounded to `window` experts in flight: box 2 has
~12 GB free beside expertd and one f32 expert is 142 MB. Disk reads (cache files,
or the checkpoint for q8_0) are serialised behind one lock: box 2's drives serve
production expert misses, and a miss queues behind reads already in flight.
"""
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import ggml_rt as G
from kernel import dequant_fp4

WHICH = ("w1", "w3", "w2")  # gate [2304,5120], up [2304,5120], down [5120,2304]; rows = output dim


def expert_path(root: str, L: int, e: int) -> str:
    return os.path.join(root, f"L{L:02d}", f"E{e:03d}.bin")


def expert_f32(ckpt, L: int, e: int, io_lock=None) -> dict:
    out = {}
    for which in WHICH:
        p = f"layers.{L}.ffn.experts.{e}.{which}."
        if io_lock is None:
            w, sc = ckpt.get(p + "weight"), ckpt.get(p + "scale")
        else:
            with io_lock:  # fault the mmap'd bytes in here, one reader at a time
                w, sc = ckpt.get(p + "weight").clone(), ckpt.get(p + "scale").clone()
        out[which] = dequant_fp4(w, sc).numpy()
    return out


NO_ZERO = (G.IQ2_XXS, G.IQ2_XS, G.IQ2_S)  # grids with no zero value


def balance_zero_signs(w: np.ndarray, seed: int) -> np.ndarray:
    """IQ2 grids have no zero, and ggml gives +0.0 a positive sign, so MXFP4's exact zeros
    (~12% of V4.1's expert weights) would all become the same small POSITIVE value: a
    coherent per-column bias. An input aligned with it is amplified layer over layer. With
    IQ2_S every zero went positive and the model collapsed (the first V3b run, KL 15 nats);
    IQ2_XXS, whose sign parity flips some, went 67% positive. Give the zeros balanced
    random signs at 1e-7 (far below any grid step): the bias falls to the random-error level
    and the error RMS is unchanged."""
    z = w == 0
    if not z.any():
        return w
    w = w.copy()
    w[z] = np.where(np.random.default_rng(seed).random(int(z.sum())) < 0.5, -1e-7, 1e-7).astype(np.float32)
    return w


def quantize_expert(t: int, ws: dict, imx_in=None, imx_down=None, key=None) -> bytes:
    """w1/w3 share the FFN-input importance, w2 takes the down-input one; None = uniform.
    key = (layer, expert) seeds the zero-sign balancing for the IQ2 types."""
    out = []
    for i, w in enumerate(WHICH):
        src = ws[w]
        if t in NO_ZERO:
            L, e = key if key is not None else (0, 0)
            src = balance_zero_signs(src, (L * 1009 + e) * 3 + i)
        out.append(G.quantize(t, src, imx_down if w == "w2" else imx_in))
    return b"".join(out)


def dequantize_expert(t: int, blob: bytes, shapes: dict) -> dict:
    out, off = {}, 0
    for w in WHICH:
        rows, cols = shapes[w]
        n = G.lib().ggml_row_size(t, cols) * rows
        out[w] = torch.from_numpy(G.dequantize(t, blob[off:off + n], rows, cols))
        off += n
    assert off == len(blob), (off, len(blob))
    return out


def load_cover(spec: str | None, n_expert: int = 384):
    """None/'all' -> every expert; a hot-set JSON path (one id list per layer) -> every
    expert NOT in it, i.e. box 2's cold set."""
    if spec in (None, "", "all"):
        return None
    hot = json.load(open(spec))
    return {(L, e) for L in range(len(hot)) for e in range(n_expert) if e not in set(hot[L])}


class ExpertRT:
    def __init__(self, spec: str, ckpt, cover=None, threads: int = 8, window: int = 8):
        self.spec, self.ckpt, self.cover, self.window = spec, ckpt, cover, window
        if spec == "q8_0":
            self.type, self.dir = G.Q8_0, None
        elif spec.startswith("cache:"):
            self.dir = spec[len("cache:"):]
            self.meta = json.load(open(os.path.join(self.dir, "meta.json")))
            self.type = G.NAMES[self.meta["type"]]
        else:
            raise ValueError(f"--expert-rt {spec!r}: expected q8_0 or cache:DIR")
        self.pool = ThreadPoolExecutor(threads)
        self.io = threading.Lock()
        self.pending: dict = {}
        self.queue: list = []
        self.stats = {"calls": 0, "rt_calls": 0, "experts": 0}

    def covers(self, L: int, e: int) -> bool:
        return self.cover is None or (L, e) in self.cover

    def _load(self, L: int, e: int) -> dict:
        if self.dir is None:
            ws = expert_f32(self.ckpt, L, e, self.io)
            blob = quantize_expert(self.type, ws)
            shapes = {w: v.shape for w, v in ws.items()}
        else:
            with self.io, open(expert_path(self.dir, L, e), "rb") as f:
                blob = f.read()
            shapes = {"w1": (2304, 5120), "w3": (2304, 5120), "w2": (5120, 2304)}
        return dequantize_expert(self.type, blob, shapes)

    def _fill(self):
        while self.queue and len(self.pending) < self.window:
            key = self.queue.pop(0)
            self.pending[key] = self.pool.submit(self._load, *key)

    def begin_layer(self, L: int, experts: list):
        """Queue this layer's covered experts in the order the forward loop visits them."""
        self.queue = [(L, e) for e in experts if self.covers(L, e)]
        self._fill()

    def get(self, L: int, e: int) -> dict:
        w = self.pending.pop((L, e)).result()
        self._fill()
        self.stats["experts"] += 1
        return w
