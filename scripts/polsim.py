import struct, random
from collections import OrderedDict, defaultdict
P="/home/claude-code/.cache/deepstrix/v41/expert_trace_v41_decode313.bin"
raw=open(P,'rb').read(); n=len(raw)//2; L=40; K=6; NE=384
vals=struct.unpack('<%dH'%n, raw); T=n//(L*K)
tok=[[vals[(t*L+l)*K:(t*L+l)*K+K] for l in range(L)] for t in range(T)]
print(f"T={T} L={L} K={K} maxid={max(vals)}")
WARM=range(0,128); EVAL=range(128,T)
def keys(t): return [(l,e) for l in range(L) for e in tok[t][l]]
seen=set(); comp=0
for t in WARM:
    seen.update(keys(t))
for t in EVAL:
    for k in keys(t):
        if k not in seen: comp+=1; seen.add(k)
print(f"compulsory (first-touch) misses in eval half: {comp/len(EVAL):.1f}/token  (floor for ANY policy/pool)")
# per-token distinct picks
print(f"distinct picks/token: {sum(len(set(keys(t))) for t in EVAL)/len(EVAL):.1f} of {L*K}")

def lru_run(cap, scan=None, prob_frac=None, eval_only_layers=None):
    """LRU with optional probationary segment (S3-FIFO-lite / 2Q): scan keys enter a small FIFO
    of prob_frac*cap; promoted to main LRU on a hit."""
    main=OrderedDict(); prob=OrderedDict()
    pcap=int(prob_frac*cap) if prob_frac else 0; mcap=cap-pcap
    def touch(k, is_scan=False):
        if k in main: main.move_to_end(k); return True
        if k in prob:
            if is_scan: return True
            del prob[k]; main[k]=1
            if len(main)>mcap: main.popitem(last=False)
            return True
        if is_scan and pcap>0:
            prob[k]=1
            if len(prob)>pcap: prob.popitem(last=False)
        else:
            main[k]=1
            if len(main)>mcap: main.popitem(last=False)
        return False
    for t in WARM:
        for k in keys(t): touch(k)
    if scan:
        for k in scan: touch(k, is_scan=True)
    miss=0; cnt=0
    for t in EVAL:
        for k in keys(t):
            if eval_only_layers is not None and k[0] not in eval_only_layers: continue
            cnt+=1
            if not touch(k): miss+=1
    return miss/len(EVAL)

def belady(cap):
    seq=[]; 
    for t in range(T): seq.extend(keys(t))
    nxt=defaultdict(list)
    for i,k in enumerate(seq): nxt[k].append(i)
    import bisect
    cache=set(); miss=0; start=len(WARM)*L*K
    for i,k in enumerate(seq):
        if k in cache: continue
        if i>=start: miss+=1
        if len(cache)>=cap:
            # evict the one whose next use is farthest
            far=None; fk=None
            for c in cache:
                lst=nxt[c]; j=bisect.bisect_right(lst,i)
                nu=lst[j] if j<len(lst) else 10**9
                if far is None or nu>far: far=nu; fk=c
            cache.remove(fk)
        cache.add(k)
    return miss/len(EVAL)

random.seed(0)
# a "short prefill" union stream: 20 encoder layers x 250 local experts (box 1 share), inserted between warm and eval
scan=[(l,e) for l in range(20) for e in range(0,250)]
for cap in (2186, 3274, 4096, 6144):
    base=lru_run(cap)
    flushed=lru_run(cap, scan=scan)
    prot=lru_run(cap, scan=scan, prob_frac=0.2)
    print(f"slots={cap:5d}: LRU {base:5.1f}  | after 5000-key prefill scan: plain LRU {flushed:5.1f}, 20% probationary {prot:5.1f}  | Belady {belady(cap):5.1f}")
