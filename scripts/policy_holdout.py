#!/usr/bin/env python3
"""Split-sample comparison of pager policies. Fit on the first half of the
trace, EVALUATE ON THE SECOND HALF. The original lru_curve.py fit the static
frequency table on the warm half and scored it on that same half (in-sample),
which is why static appeared to dominate."""
import sys, collections, array, heapq

N_LAYER, TOPK = 40, 6
path = sys.argv[1]
sizes = [384,650,1024,1418,2186,3072,4096,6144,8192,11915]

buf = array.array('H'); buf.frombytes(open(path,'rb').read())
per_tok = N_LAYER*TOPK
ntok = len(buf)//per_tok

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

split = ntok//2
fit, ev = toks[:split], toks[split:]
flat = lambda tks: [k for tk in tks for row in tk for k in row]
ev_flat = flat(ev)
print(f"trace: {ntok} tokens, fit[0:{split}] eval[{split}:{ntok}], "
      f"{len(ev_flat)/len(ev):.1f} deduped requests/token")

# frequency table from the FIT half only
freq = collections.Counter(flat(fit))
order = [k for k,_ in freq.most_common()]
print(f"distinct (layer,expert) in fit half: {len(freq)}; in eval half: {len(set(ev_flat))}; "
      f"eval keys never seen in fit: {len(set(ev_flat)-set(freq))}")

def run_lru(S, pins=frozenset()):
    lru = collections.OrderedDict()
    for k in flat(fit):                       # warm
        if k in lru: lru.move_to_end(k)
        else:
            lru[k]=1
            if len(lru) > S-len(pins): lru.popitem(last=False)
    m=0
    for k in ev_flat:
        if k in pins: continue
        if k in lru: lru.move_to_end(k)
        else:
            m+=1; lru[k]=1
            if len(lru) > S-len(pins): lru.popitem(last=False)
    return m/len(ev)

def run_static(S):
    res = set(order[:S])
    return sum(1 for k in ev_flat if k not in res)/len(ev)

def run_belady(S):
    nxt = collections.defaultdict(list)
    for i,k in enumerate(ev_flat): nxt[k].append(i)
    for k in nxt: nxt[k].reverse()
    cache=set(order[:S])                       # same warm start as static
    for k in cache: nxt[k]  # touch
    m=0; heap=[]
    def push(k,i):
        j = nxt[k][-1] if nxt[k] else float('inf')
        heapq.heappush(heap,(-j,k))
    for k in cache: push(k,0)
    for i,k in enumerate(ev_flat):
        if nxt[k] and nxt[k][-1]==i: nxt[k].pop()
        if k in cache:
            push(k,i); continue
        m+=1
        if len(cache) >= S:
            while heap:
                negj,vk = heapq.heappop(heap)
                if vk in cache and (not nxt[vk] or -negj == (nxt[vk][-1] if nxt[vk] else float('inf'))):
                    cache.discard(vk); break
            else: cache.pop()
        cache.add(k); push(k,i)
    return m/len(ev)

print(f"\n{'slots':>7} {'GB':>6} {'%res':>6} | {'LRU':>8} {'static-freq':>12} {'pin50%+LRU':>11} {'Belady':>8}")
for S in sizes:
    pins = frozenset(order[:S//2])
    print(f"{S:7d} {S*18.8/1000:6.1f} {100*S/15360:5.1f}% | {run_lru(S):8.1f} {run_static(S):12.1f} "
          f"{run_lru(S,pins)+sum(1 for k in ev_flat if k in pins)*0/len(ev):11.1f} {run_belady(S):8.1f}")
