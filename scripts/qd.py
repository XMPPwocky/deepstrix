#!/usr/bin/env python3
"""O_DIRECT random-read scaling vs queue depth, at the expert-read size (19.25 MB).
Mimics box 2's pager miss path. Page cache bypassed; offsets random and 4K aligned."""
import os, sys, mmap, time, random, threading

SZ = 19.25 * 1024 * 1024
SZ = int(SZ) // 4096 * 4096           # 20,185,088 B, exactly 4K aligned
path = sys.argv[1]
fsz = os.path.getsize(path)
noff = (fsz - SZ) // 4096
rng = random.Random(1234)

def run(qd, iters):
    bufs = [mmap.mmap(-1, SZ) for _ in range(qd)]
    fds  = [os.open(path, os.O_RDONLY | os.O_DIRECT) for _ in range(qd)]
    offs = [[rng.randrange(noff) * 4096 for _ in range(iters)] for _ in range(qd)]
    errs = []
    def worker(i):
        try:
            for o in offs[i]:
                got = 0
                while got < SZ:
                    n = os.preadv(fds[i], [memoryview(bufs[i])[got:]], o + got)
                    if n == 0: break
                    got += n
        except Exception as e: errs.append(e)
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(qd)]
    t0 = time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join()
    dt = time.perf_counter() - t0
    for fd in fds: os.close(fd)
    for b in bufs: b.close()
    if errs: raise errs[0]
    total = qd * iters * SZ
    return total / dt / 1e9, dt / (qd * iters) * 1000   # GB/s, ms per read

print(f"file {path}  read size {SZ/1e6:.2f} MB  O_DIRECT random\n")
print(f"{'QD':>3} {'agg GB/s':>9} {'ms/read':>9} {'vs QD1':>7} {'miss ms':>8}")
base = None
for qd in [1, 2, 3, 4, 6, 8, 12, 16]:
    gbs, msr = run(qd, max(4, 32 // qd))
    if base is None: base = gbs
    print(f"{qd:3d} {gbs:9.2f} {msr:9.2f} {gbs/base:6.2f}x {SZ/1e6/gbs*qd/qd*1000/1e3*1e3/1e3:8.2f}")
