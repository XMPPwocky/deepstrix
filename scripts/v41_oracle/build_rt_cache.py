"""Pre-quantize routed experts through ggml for `oracle.py --expert-rt cache:DIR`.

    build_rt_cache.py --type iq2_xxs --cover all|HOT.json --imatrix none|IMX.pt --out DIR --procs 12

Box 2's drives serve production expert misses, and a miss queues behind reads
already in flight, so ALL disk I/O happens on this process's main thread, one
expert at a time: it reads each expert's packed MXFP4 bytes sequentially, ships
them to a worker, and writes the worker's quantized bytes back (fsync'd per file,
so the kernel never flushes a large writeback burst). The workers only compute.
At most 2 x procs experts are in flight (~19 MB each).

Resumable: an expert whose DIR/Lnn/Eeee.bin exists is skipped; each file is written
to a temp name, fsync'd and renamed, so a killed build never leaves a torn file.
--imatrix takes an `oracle.py --imatrix-capture` file: w1/w3 get the mean x^2 of the
expert's routed FFN-input rows, w2 the mean h^2 of its down input. An expert the
capture never saw (n = 0), or --imatrix none, gets a uniform imatrix (ggml requires
one for IQ2_XXS/IQ2_XS; all-ones leaves its own sqrt(sigma2 + x^2) weighting in charge).
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import ggml_rt as G  # noqa: E402
from expert_rt import WHICH, expert_path, load_cover, quantize_expert  # noqa: E402

FREE_FLOOR = 50 << 30  # never fill the cache disk past this much free space


def _init():
    torch.set_num_threads(1)


def read_raw(ckpt, L: int, e: int) -> dict:
    """The expert's packed MXFP4 tensors as plain bytes (sequential reads, this thread)."""
    raw = {}
    for w in WHICH:
        for part in ("weight", "scale"):
            t = ckpt.get(f"layers.{L}.ffn.experts.{e}.{w}.{part}")
            raw[w, part] = (t.contiguous().view(torch.uint8).numpy().copy(), str(t.dtype).split(".")[-1], tuple(t.shape))
    return raw


def _work(job):
    from kernel import dequant_fp4
    t, L, e, raw, imx_in, imx_down = job
    ws = {}
    for w in WHICH:
        (wb, wdt, wsh), (sb, sdt, ssh) = raw[w, "weight"], raw[w, "scale"]
        wt = torch.from_numpy(wb).view(getattr(torch, wdt)).view(wsh)
        st = torch.from_numpy(sb).view(getattr(torch, sdt)).view(ssh)
        ws[w] = dequant_fp4(wt, st).numpy()
    return L, e, quantize_expert(t, ws, imx_in, imx_down)


def write_atomic(path: str, blob: bytes):
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, blob)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=sorted(G.NAMES))
    ap.add_argument("--cover", default="all", help="'all' or a hot-set JSON (quantize the experts NOT in it)")
    ap.add_argument("--imatrix", default="none", help="'none' or an oracle.py --imatrix-capture .pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many experts (throughput tests)")
    a = ap.parse_args()
    from loader import Checkpoint
    ckpt = Checkpoint(os.environ["V41_MODEL"])
    t = G.NAMES[a.type]
    cover = load_cover(a.cover)
    keys = [(L, e) for L in range(a.layers) for e in range(384) if cover is None or (L, e) in cover]
    os.makedirs(a.out, exist_ok=True)
    meta = {"type": a.type, "imatrix": a.imatrix, "ggml_lib": os.environ.get("GGML_LIB", G._DEFAULT)}
    mp = os.path.join(a.out, "meta.json")
    if os.path.exists(mp):
        old = json.load(open(mp))
        if (old["type"], old["imatrix"]) != (meta["type"], meta["imatrix"]):
            sys.exit(f"{mp} was built as {old['type']} / {old['imatrix']}; refusing to mix")
    json.dump(meta, open(mp, "w"), indent=1)
    for L in range(a.layers):
        os.makedirs(os.path.join(a.out, f"L{L:02d}"), exist_ok=True)
    todo = [k for k in keys if not os.path.exists(expert_path(a.out, *k))]
    if a.limit:
        todo = todo[:a.limit]
    per_expert = sum(G.lib().ggml_row_size(t, c) * r for r, c in ((2304, 5120), (2304, 5120), (5120, 2304)))
    need, free = len(todo) * per_expert, shutil.disk_usage(a.out).free
    print(f"{a.type} imatrix={a.imatrix} cover={a.cover}: {len(keys)} experts, {len(keys) - len(todo)} already built, "
          f"{len(todo)} to do on {a.procs} procs -> {a.out}  ({need / 2**30:.0f} GiB needed, {free / 2**30:.0f} GiB free)",
          flush=True)
    if free - need < FREE_FLOOR:
        sys.exit(f"not enough space: would leave {(free - need) / 2**30:.0f} GiB free (< {FREE_FLOOR >> 30})")
    imx = None if a.imatrix == "none" else torch.load(a.imatrix)

    def job(L, e):
        imx_in = imx_down = None
        if imx is not None and L in imx and int(imx[L]["n"][e]) > 0:
            n = float(imx[L]["n"][e])
            imx_in = (imx[L]["x2"][e] / n).numpy().astype(np.float32)
            imx_down = (imx[L]["h2"][e] / n).numpy().astype(np.float32)
        return t, L, e, read_raw(ckpt, L, e), imx_in, imx_down

    t0, done, it, inflight = time.time(), 0, iter(todo), set()
    with ProcessPoolExecutor(a.procs, initializer=_init) as pool:
        while True:
            while len(inflight) < 2 * a.procs:
                k = next(it, None)
                if k is None:
                    break
                inflight.add(pool.submit(_work, job(*k)))
            if not inflight:
                break
            fin, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for f in fin:
                L, e, blob = f.result()
                write_atomic(expert_path(a.out, L, e), blob)
                done += 1
                if done % 100 == 0 or done == len(todo):
                    dt = time.time() - t0
                    print(f"  {done}/{len(todo)}  {done / dt:.2f} experts/s  eta {(len(todo) - done) / (done / dt) / 60:.0f} min",
                          flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
