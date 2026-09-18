import os, sys, time, mmap, threading, random
path, mode, threads, chunk, nreads = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
flags = os.O_RDONLY | (os.O_DIRECT if mode == 'direct' else 0)
fd = os.open(path, flags)
size = os.fstat(fd).st_size
A = 4096
def ds():
    for l in open('/proc/diskstats'):
        f = l.split()
        if f[2] == 'nvme0n1':
            return int(f[3]), int(f[5]) * 512, int(f[6]), int(f[12])  # reads, bytes, read_ticks_ms, io_ticks_ms
res = [None] * threads
def worker(i):
    buf = mmap.mmap(-1, chunk)
    rng = random.Random(i * 7919 + int(time.time() * 1e6))
    lat = []
    for _ in range(nreads):
        off = rng.randrange(0, (size - chunk) // A) * A
        t0 = time.perf_counter()
        n = os.preadv(fd, [buf], off)
        lat.append(time.perf_counter() - t0)
        assert n == chunk, n
    res[i] = lat
d0 = ds(); t0 = time.perf_counter()
ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
[t.start() for t in ts]; [t.join() for t in ts]
wall = time.perf_counter() - t0; d1 = ds()
lat = sorted(x for r in res for x in r)
tot = threads * nreads * chunk
p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] * 1e3
dreads = d1[0] - d0[0]; dbytes = d1[1] - d0[1]; dticks = d1[2] - d0[2]; dio = d1[3] - d0[3]
print(f"{mode:8s} T={threads:2d} chunk={chunk/1e6:6.2f}MB n={nreads:3d} | wall {wall*1e3:8.1f} ms  agg {tot/wall/1e9:5.2f} GB/s | per-read ms p50 {p(.5):6.2f} p90 {p(.9):6.2f} max {p(1):6.2f} | chunk-rate {chunk/(lat[len(lat)//2])/1e9:5.2f} GB/s"
      f" || device: {dreads} reqs avg {dbytes/max(dreads,1)/1024:5.0f} KB, {dbytes/1e9:5.2f} GB, busy {dio/wall/10:4.0f}%, avg-inflight {dticks/max(dio,1):4.1f}")
