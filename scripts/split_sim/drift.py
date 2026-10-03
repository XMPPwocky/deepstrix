#!/usr/bin/env python3
"""Placement drift: how much decode pick mass a top-K placement learned on one
window still covers on a later window.

  drift.py TODAY_SEGS.pkl [PREMOVE_SEGS.pkl]

Coverage(K, learn, eval) = sum over layers of eval picks on learn's top-K / eval picks,
averaged over layers. Reports in-sample (learn = eval), learn on the first half of
today -> eval second half, learn on the previous segment, and (with the pre-move
counts) learn on the whole pre-move trace / its last segments -> eval today's
second half.
"""
import pickle
import statistics as st
import sys

NL, NE = 40, 384


def add(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def total(segs):
    out = [[0] * NE for _ in range(NL)]
    for s in segs:
        out = add(out, s)
    return out


def topk(counts, k):
    return [set(sorted(range(NE), key=lambda e: (-counts[l][e], e))[:k]) for l in range(NL)]


def coverage(sets, counts):
    v = []
    for l in range(NL):
        t = sum(counts[l])
        if t:
            v.append(sum(counts[l][e] for e in sets[l]) / t)
    return st.fmean(v)


def main():
    today = pickle.load(open(sys.argv[1], 'rb'))['segments']
    pre = pickle.load(open(sys.argv[2], 'rb'))['segments'] if len(sys.argv) > 2 else None
    h = len(today) // 2
    first, second = total(today[:h]), total(today[h:])
    ks = (10, 25, 50, 103, 133)
    print(f'today: {len(today)} segments; first half {h}, second half {len(today) - h}')
    rows = [('in-sample (eval = learn)', second), ('today first half', first)]
    if h >= 1:
        rows.append(('today: segment just before eval', today[h - 1]))
    if pre:
        rows.append(('pre-move trace, all', total(pre)))
        rows.append(('pre-move trace, last 3 segments', total(pre[-3:])))
        rows.append(('pre-move trace, first 3 segments', total(pre[:3])))
    print(f"{'learned on':40s} " + ' '.join(f'top-{k:<4d}' for k in ks) + '  (coverage of today second-half picks)')
    for name, c in rows:
        print(f'{name:40s} ' + ' '.join(f'{coverage(topk(c, k), second):8.3f}' for k in ks))
    if pre:
        # drift within the pre-move trace: learn on segment i, eval on segment i+d
        print('\nwithin the pre-move trace: coverage of segment i+d by top-K of segment i (mean over i)')
        for k in (25, 103):
            out = []
            for d in (1, 2, 4, 8, 16, 32):
                v = [coverage(topk(pre[i], k), pre[i + d]) for i in range(0, len(pre) - d, max(1, (len(pre) - d) // 12))]
                ins = [coverage(topk(pre[i + d], k), pre[i + d]) for i in range(0, len(pre) - d, max(1, (len(pre) - d) // 12))]
                out.append(f'd={d}: {st.fmean(v):.3f} (in-sample {st.fmean(ins):.3f})')
            print(f'  top-{k}: ' + '; '.join(out))


if __name__ == '__main__':
    main()
