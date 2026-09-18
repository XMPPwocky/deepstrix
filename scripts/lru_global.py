import array, collections
N_LAYER, TOPK = 40, 6
buf = array.array('H'); buf.frombytes(open('/home/claude-code/.cache/deepstrix/v41/expert_trace_v41_decode313.bin','rb').read())
per = N_LAYER*TOPK; ntok = len(buf)//per
toks = [[[buf[t*per+l*TOPK+k] for k in range(TOPK)] for l in range(N_LAYER)] for t in range(ntok)]
# global LRU: misses per token over time, in 32-token bins, for several sizes; plus how many are first-touch (compulsory)
for S in (4096, 6160, 7395, 8192):
    lru=collections.OrderedDict(); seen=set(); bins=[]; comp=[]
    for t in range(ntok):
        m=c=0
        for l in range(N_LAYER):
            for e in toks[t][l]:
                k=(l,e)
                if k in lru: lru.move_to_end(k)
                else:
                    m+=1
                    if k not in seen: c+=1
                    seen.add(k); lru[k]=1
                    if len(lru)>S: lru.popitem(last=False)
        bins.append(m); comp.append(c)
    b=lambda a,x: sum(a[x:x+32])/32
    print(f"S={S:5d}: misses/tok by 32-tok bin: " + " ".join(f"{b(bins,x):5.1f}" for x in range(0,ntok,32)) + f"  | compulsory (first touch): " + " ".join(f"{b(comp,x):4.1f}" for x in range(0,ntok,32)))
# per-layer regions 268/40 and 154/154, same bins
for name,R in (("regions 268/40",[268]*20+[40]*20),("regions 154/154",[154]*40),("regions 268/154 (8440 slots, does not fit)",[268]*20+[154]*20)):
    lrus=[collections.OrderedDict() for _ in range(N_LAYER)]; bins=[]
    for t in range(ntok):
        m=0
        for l in range(N_LAYER):
            for e in toks[t][l]:
                d=lrus[l]
                if e in d: d.move_to_end(e)
                else:
                    m+=1; d[e]=1
                    if len(d)>R[l]: d.popitem(last=False)
        bins.append(m)
    print(f"{name:44s}: " + " ".join(f"{sum(bins[x:x+32])/32:5.1f}" for x in range(0,ntok,32)))
