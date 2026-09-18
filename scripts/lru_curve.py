#!/usr/bin/env python3
"""Exact LRU-residency curve from a DEEPSTRIX_EXPERT_TRACE (u16 LE, 6 picks x 40 layers per token).

Replays the pager's own policy: per (layer, token) the picks are de-duplicated
(`ExpertPager::ensure` does `!ids.contains`), each hit touches the LRU, each miss
evicts the least-recently-used slot. Reports misses per generated token for a
sweep of pool sizes, plus the static top-N-by-frequency placement for comparison.
"""
import sys, collections, array

N_LAYER, TOPK = 40, 6
path = sys.argv[1]
sizes = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else \
    [384,650,1024,1418,2186,3072,4096,6144,8192,11915,15360]
warm_frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5  # ignore the first half (cold)

buf = array.array('H'); buf.frombytes(open(path,'rb').read())
per_tok = N_LAYER*TOPK
ntok = len(buf)//per_tok
print(f"trace: {len(buf)} picks = {ntok} tokens x {N_LAYER} layers x {TOPK}")

# per-token, per-layer deduped key list
toks = []
for t in range(ntok):
    rows = []
    for l in range(N_LAYER):
        seen = []
        for k in range(TOPK):
            e = buf[t*per_tok + l*TOPK + k]
            if e < 384 and e not in seen: seen.append(e)
        rows.append([(l, e) for e in seen])
    toks.append(rows)

reqs_per_tok = sum(len(r) for r in toks[0])
print(f"deduped requests/token ~ {sum(sum(len(r) for r in tk) for tk in toks)/ntok:.1f}")

t0 = int(ntok*warm_frac)
print(f"\n{'slots':>7} {'GB':>6} {'%resident':>9} | {'miss/tok(all)':>13} {'hit(all)':>9} | {'miss/tok(warm)':>14} {'hit(warm)':>10}")
for S in sizes:
    lru = collections.OrderedDict()
    miss=req=0; wmiss=wreq=0
    for ti, tk in enumerate(toks):
        warm = ti >= t0
        for row in tk:
            for key in row:
                req += 1
                if warm: wreq += 1
                if key in lru:
                    lru.move_to_end(key)
                else:
                    miss += 1
                    if warm: wmiss += 1
                    lru[key] = 1
                    if len(lru) > S: lru.popitem(last=False)
    wt = ntok - t0
    print(f"{S:7d} {S*18.8/1000:6.1f} {100*S/15360:8.1f}% | {miss/ntok:13.1f} {1-miss/req:9.4f} | "
          f"{wmiss/wt:14.1f} {1-wmiss/max(wreq,1):10.4f}")

# static placement oracle: top-S (layer,expert) by pick frequency over the WARM half
freq = collections.Counter()
for tk in toks[t0:]:
    for row in tk:
        for key in row: freq[key]+=1
print(f"\ndistinct (layer,expert) touched, warm half: {len(freq)} of {N_LAYER*384}")
order=[k for k,_ in freq.most_common()]
tot=sum(freq.values())
print(f"{'slots':>7} | static-top-N hit (warm)  miss/tok")
cum=0; i=0
for S in sizes:
    while i < min(S, len(order)):
        cum += freq[order[i]]; i+=1
    hit = cum/tot
    print(f"{S:7d} | {hit:20.4f}  {(1-hit)*tot/(ntok-t0):8.1f}")
