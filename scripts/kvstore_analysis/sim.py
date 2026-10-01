#!/usr/bin/env python3
"""Replay the on-disk prompt snapshots (creation order) as requests against a
simulated chunked store: C-position chunks keyed by token prefix, FULL tails at
every prompt end, ENCODER tails every K positions crossed by a prefill (usable
only with a suffix > 128). No eviction (the corpus is what survived the 100 GiB
LRU). Compares restored tokens with today's whole-snapshot rule and with the
token LCP upper bound, and sums the bytes each design writes."""
import json, os, sys
import numpy as np
ROOT = '/home/claude-code/.cache/deepstrix/snapshots-v41'
CK = {int(x) for x in open('/home/claude-code/.claude/jobs/16b63e08/tmp/ckpt_counts.txt').read().split()}
BPP = 2760          # bytes per position (4 stores: comp rows + E2M1 keys)
FULL_TAIL = 5.67e6  # enc+dec windows, accumulators, DSpark ring
ENC_TAIL = 2.65e6
W = 128
snaps = []
for d in os.listdir(ROOT):
    try:
        meta = json.load(open(os.path.join(ROOT, d, 'meta.json')))
        toks = np.fromfile(os.path.join(ROOT, d, 'tokens.bin'), dtype='<i4')
    except Exception:
        continue
    if len(toks) in CK:
        continue
    snaps.append((meta.get('created_at_unix', 0), toks))
snaps.sort(key=lambda s: s[0])

def lcp(a, b):
    m = min(len(a), len(b)); ne = np.nonzero(a[:m] != b[:m])[0]
    return int(ne[0]) if len(ne) else m

def run(C, K):
    full_tails = []   # (prompt index, length)
    enc_tails = []    # (prompt index, position)
    restored_new = restored_old = lcp_tot = total = 0
    chunk_keys = set()
    bytes_new = bytes_old = 0
    for i, (_, t) in enumerate(snaps):
        L = len(t); total += L
        # LCP against every older prompt
        l_any = 0; best_old = 0
        lcps = []
        for j in range(i):
            l = lcp(t, snaps[j][1]); lcps.append(l)
            l_any = max(l_any, l)
            if l == len(snaps[j][1]) and l < L:
                best_old = max(best_old, l)
        lcp_tot += l_any
        restored_old += best_old
        # new store: deepest usable tail whose prefix matches
        best = 0
        for (j, T) in full_tails:
            if T <= L - 1 and lcps[j] >= T:
                best = max(best, T)
        for (j, T) in enc_tails:
            if T <= L - (W + 1) and lcps[j] >= T:
                best = max(best, T)
        restored_new += best
        # writes: chunks for positions prefilled, new chunk keys only
        for k in range(best // C, L // C):
            key = (k, hash(t[:(k + 1) * C].tobytes()))
            if key not in chunk_keys:
                chunk_keys.add(key); bytes_new += C * BPP
        bytes_new += FULL_TAIL + (L % C) * BPP
        full_tails.append((i, L))
        for T in range((best // K + 1) * K, L, K):
            enc_tails.append((i, T)); bytes_new += ENC_TAIL
        bytes_old += L * BPP + 5.24e6
    return dict(C=C, K=K, prompts=len(snaps), total=total, lcp=lcp_tot / total,
                old=restored_old / total, new=restored_new / total,
                gb_new=bytes_new / 1e9, gb_old=bytes_old / 1e9, chunks=len(chunk_keys))

for C, K in [(1024, 4096), (1024, 8192), (1024, 16384), (1024, 32768), (4096, 8192)]:
    r = run(C, K)
    print(f"C={r['C']:5d} K={r['K']:6d}: prompts {r['prompts']}, tokens {r['total']:,}; restored: LCP bound {r['lcp']:.2%}, "
          f"today {r['old']:.2%}, chunked {r['new']:.2%}; written: today {r['gb_old']:.1f} GB, chunked {r['gb_new']:.1f} GB ({r['chunks']} chunks)")
