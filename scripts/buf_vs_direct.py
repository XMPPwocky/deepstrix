#!/usr/bin/env python3
"""Same 20.2 MB random reads, O_DIRECT vs buffered, 3 concurrent (the miss path's shape)."""
import os, sys, mmap, time, random, threading

SZ = int(19.25*1024*1024)//4096*4096
ROLE = SZ//3//4096*4096          # ~6.3 MB, one role
path = sys.argv[1]
fsz = os.path.getsize(path); noff = (fsz - SZ)//4096
def run(direct, threads, size, iters):
    rng = random.Random(7)
    flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
    bufs = [mmap.mmap(-1, size) for _ in range(threads)]
    fds  = [os.open(path, flags) for _ in range(threads)]
    offs = [[rng.randrange(noff)*4096 for _ in range(iters)] for _ in range(threads)]
    def w(i):
        for o in offs[i]:
            got = 0
            while got < size:
                n = os.preadv(fds[i], [memoryview(bufs[i])[got:]], o+got)
                if n == 0: break
                got += n
    ths=[threading.Thread(target=w,args=(i,)) for i in range(threads)]
    t0=time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join()
    dt=time.perf_counter()-t0
    for fd in fds: os.close(fd)
    for b in bufs: b.close()
    return dt/iters*1000, threads*iters*size/dt/1e9

print(f"{'arm':>34} {'ms/expert':>10} {'GB/s':>7}")
for label, direct, th, size in [
    ("1 thread x 20.2MB  O_DIRECT",  True, 1, SZ),
    ("1 thread x 20.2MB  buffered", False, 1, SZ),
    ("3 threads x 6.3MB  O_DIRECT",  True, 3, ROLE),
    ("3 threads x 6.3MB  buffered", False, 3, ROLE),
]:
    ms, gbs = run(direct, th, size, 8)
    print(f"{label:>34} {ms:10.2f} {gbs:7.2f}")
