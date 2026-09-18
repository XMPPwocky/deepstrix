import array, collections, math
N_LAYER, TOPK, N_EXP = 40, 6, 384
buf = array.array('H'); buf.frombytes(open('/home/claude-code/.cache/deepstrix/v41/expert_trace_v41_decode313.bin','rb').read())
per_tok = N_LAYER*TOPK; ntok = len(buf)//per_tok
print("tokens", ntok)
# picks[t][l] = list of expert ids (deduped)
picks = [[[] for _ in range(N_LAYER)] for _ in range(ntok)]
dups = 0; oob = 0
for t in range(ntok):
    for l in range(N_LAYER):
        s = []
        for k in range(TOPK):
            e = buf[t*per_tok + l*TOPK + k]
            if e >= N_EXP: oob += 1; continue
            if e in s: dups += 1; continue
            s.append(e)
        picks[t][l] = s
print("dups", dups, "oob", oob)
# per-layer frequency
freq = [collections.Counter() for _ in range(N_LAYER)]
for t in range(ntok):
    for l in range(N_LAYER):
        freq[l].update(picks[t][l])
tot = [sum(c.values()) for c in freq]
print("\nlayer distinct top6 top12 top24 top40 top64 top96 (share of picks)")
for l in range(N_LAYER):
    mc = [c for _, c in freq[l].most_common()]
    cum = lambda n: sum(mc[:n])/tot[l]
    print(f"{l:2d} {len(mc):4d}  {cum(6):.3f} {cum(12):.3f} {cum(24):.3f} {cum(40):.3f} {cum(64):.3f} {cum(96):.3f}")
# LRU hit rate per layer at S slots (sequential over tokens), report warm half
def lru_layer(l, S, warm_from):
    lru = collections.OrderedDict(); miss=0; req=0
    for t in range(ntok):
        for e in picks[t][l]:
            if t >= warm_from: req += 1
            if e in lru: lru.move_to_end(e)
            else:
                if t >= warm_from: miss += 1
                lru[e] = 1
                if len(lru) > S: lru.popitem(last=False)
    return miss/max(req,1)
print("\nLRU miss fraction (warm half) at slots/layer: 25, 40, 80, 154, 268")
for S in (25,40,80,154,268):
    ms = [lru_layer(l, S, ntok//2) for l in range(N_LAYER)]
    print(f"S={S:3d} enc(0-19) mean {sum(ms[:20])/20:.3f}  dec(20-39) mean {sum(ms[20:])/20:.3f}  all {sum(ms)/40:.3f}  misses/tok {sum(ms)*6:.1f}")
# how many picks per token per layer are duplicates? (n distinct)
nd = collections.Counter(len(picks[t][l]) for t in range(ntok) for l in range(N_LAYER))
print("\ndistinct picks per (tok,layer):", dict(nd))
# cross-half stability: top-N fit on first half, coverage on second half
print("\nsplit-sample coverage: top-N fit on tokens 0..127, scored on 128..255 (vs in-sample)")
h = ntok//2
for N in (6, 12, 24, 40, 64):
    ins = 0; oos = 0; tot2 = 0
    for l in range(N_LAYER):
        f1 = collections.Counter(); f2 = collections.Counter()
        for t in range(h): f1.update(picks[t][l])
        for t in range(h, ntok): f2.update(picks[t][l])
        top1 = set(e for e,_ in f1.most_common(N)); top2 = set(e for e,_ in f2.most_common(N))
        oos += sum(f2[e] for e in top1); ins += sum(f2[e] for e in top2); tot2 += sum(f2.values())
    print(f"N={N:2d}  in-sample {ins/tot2:.3f}  out-of-sample {oos/tot2:.3f}")
