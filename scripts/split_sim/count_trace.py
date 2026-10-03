#!/usr/bin/env python3
"""Stream a (large) pick trace and write per-segment router pick counts of the
DECODE batches (b <= 6), for placement-drift checks. Never loads the file:
reads line by line and drops the pages it has read (POSIX_FADV_DONTNEED) so the
page cache of the production box is not displaced.

  count_trace.py TRACE OUT.pkl [--seg-batches 200000]

OUT: {'segments': [ [layer][expert] counts ], 'seg_batches': N, 'bytes': ...}
"""
import os
import pickle
import sys

N_LAYER, N_EXPERT = 40, 384


def main():
    path, out = sys.argv[1], sys.argv[2]
    seg_n = int(sys.argv[sys.argv.index('--seg-batches') + 1]) if '--seg-batches' in sys.argv else 200000
    fd = os.open(path, os.O_RDONLY)
    size = os.fstat(fd).st_size
    segs = []
    cur = [[0] * N_EXPERT for _ in range(N_LAYER)]
    n_b = 0
    # batch state: P lines with (layer, b); O lines patch the batch just completed
    last_rows = None  # (layer, b, [rows]) of the last completed decode batch, counted lazily
    pend = None
    pos = 0
    buf = b''
    CH = 64 << 20

    def flush_batch(bt):
        nonlocal n_b, cur
        layer, b, rows = bt
        cl = cur[layer]
        for row in rows:
            for e in row:
                if 0 <= e < N_EXPERT:
                    cl[e] += 1
        n_b += 1
        if n_b % seg_n == 0:
            segs.append(cur)
            cur = [[0] * N_EXPERT for _ in range(N_LAYER)]

    while True:
        chunk = os.pread(fd, CH, pos)
        if not chunk:
            break
        os.posix_fadvise(fd, pos, len(chunk), os.POSIX_FADV_DONTNEED)
        pos += len(chunk)
        data = buf + chunk
        lines = data.split(b'\n')
        buf = lines.pop()
        for raw in lines:
            k = raw[:1]
            if k == b'P':
                p = raw.split()
                layer, b = int(p[1]), int(p[2])
                if b > 6:
                    if pend is not None:
                        pend = None
                    continue
                ids = [int(x) for x in p[3:]]
                if pend is not None and (pend[0] != layer or pend[1] != b or len(pend[2]) >= b):
                    pend = None
                if pend is None:
                    if last_rows is not None:
                        flush_batch(last_rows)
                        last_rows = None
                    pend = (layer, b, [])
                pend[2].append(ids)
                if len(pend[2]) == b:
                    last_rows = pend
                    pend = None
            elif k == b'O':
                p = raw.split()
                layer, b, r = int(p[1]), int(p[2]), int(p[3])
                if last_rows is not None and last_rows[0] == layer and last_rows[1] == b and r < b:
                    last_rows[2][r] = [int(x) for x in p[4:]]
        if pos % (1 << 30) < CH:
            print(f'{pos / 1e9:.1f} / {size / 1e9:.1f} GB, decode batches {n_b}', file=sys.stderr, flush=True)
    if last_rows is not None:
        flush_batch(last_rows)
    if any(any(r) for r in cur):
        segs.append(cur)
    os.close(fd)
    with open(out, 'wb') as fh:
        pickle.dump({'segments': segs, 'seg_batches': seg_n, 'bytes': pos, 'n_batches': n_b}, fh)
    print(f'segments {len(segs)}, decode batches {n_b}', file=sys.stderr)


if __name__ == '__main__':
    main()
