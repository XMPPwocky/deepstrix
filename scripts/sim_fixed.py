#!/usr/bin/env python3
"""Decode expert-phase simulation on FIXED semantics (all 6 picks computed).
Trace: 256 tok x 40 layers x 6 picks. Warm half = tokens 128..255.
Cost model per layer: expert_phase = max(box1, box2, dgpu); token = 40*(chain+phase) + fixed.
"""
import array, collections, sys
N_LAYER, TOPK, N_EXPERT = 40, 6, 384
buf = array.array('H'); buf.frombytes(open('/home/claude-code/.cache/deepstrix/v41/expert_trace_v41_decode313.bin','rb').read())
per = N_LAYER*TOPK; ntok = len(buf)//per
toks = [[[buf[t*per+l*TOPK+k] for k in range(TOPK)] for l in range(N_LAYER)] for t in range(ntok)]
T0 = ntok//2

hot=[]
for line in open('/home/claude-code/.cache/deepstrix/hot_experts.txt'):
    hot.append([int(x.split(':')[0]) for x in line.strip().split(',') if x])

def freq_fit(t_lo, t_hi):
    f=[collections.Counter() for _ in range(N_LAYER)]
    for t in range(t_lo,t_hi):
        for l in range(N_LAYER):
            for e in toks[t][l]: f[l][e]+=1
    return [[e for e,_ in c.most_common()] for c in f]
FIT_ALL   = freq_fit(0, ntok)        # in-sample oracle
FIT_HALF  = freq_fit(0, T0)          # split-sample
FIT_HOT   = hot                      # accumulated file (64 entries/layer max)

class LRU:
    def __init__(self, cap): self.cap=cap; self.d=collections.OrderedDict()
    def touch(self, key):
        if key in self.d: self.d.move_to_end(key); return True
        self.d[key]=1
        if len(self.d)>self.cap: self.d.popitem(last=False)
        return False

def run(name, box1_static, box1_lru, dgpu_static, box2_mode, box2_size, claim_cap=None,
        c1=98.0, f1=120.0, cd=29.0, fd=40.0, I2=175.0, s2=95.0, F2=4000.0,
        chain_us=470.0, ensure_us=100.0, other_us=200.0, fixed_us=1300+250+400, engram_us=8000.0,
        drop_to=None, quiet=False):
    """box1_static: list per layer of sets (pinned); box1_lru: slots (global LRU over pairs);
       dgpu_static: per-layer sets; box2_mode: 'regions' (box2_size = list per layer) or 'global' (int).
       drop_to: if set, compute only the first k picks (rank order in trace = router order) -- deliberate drop."""
    lru1 = LRU(box1_lru) if box1_lru>0 else None
    if box2_mode=='regions':
        lru2 = [LRU(box2_size[l]) for l in range(N_LAYER)]
    else:
        lru2g = LRU(box2_size)
    acc = collections.defaultdict(float); n=0
    per_layer_phase=[0.0]*N_LAYER
    for t in range(ntok):
        warm = t>=T0
        tok_phase=0.0; tok_faults=0; n1s=n2s=nds=0
        for l in range(N_LAYER):
            picks = toks[t][l][:drop_to] if drop_to else toks[t][l]
            n1=nd=m=0; faults=0
            for e in picks:
                key=(l,e)
                if e in dgpu_static[l]:
                    nd+=1; continue
                local = (e in box1_static[l]) or (lru1 is not None and key in lru1.d)
                if local and (claim_cap is None or n1<claim_cap):
                    n1+=1
                    if lru1 is not None and key in lru1.d: lru1.touch(key)
                    continue
                # box 1 LRU admission: only into FREE slots (catch-all rule) -- approximated: never admit once full
                if lru1 is not None and len(lru1.d)<lru1.cap and (claim_cap is None or n1<claim_cap):
                    lru1.touch(key); n1+=1; continue
                m+=1
                if box2_mode=='regions':
                    if not lru2[l].touch(e): faults+=1
                else:
                    if not lru2g.touch(key): faults+=1
            b1 = f1 + c1*n1 if n1 else 0.0
            bd = fd + cd*nd if nd else 0.0
            b2 = I2 + s2*m + F2*faults if m else 0.0
            ph = max(b1,bd,b2)
            tok_phase+=ph; tok_faults+=faults; n1s+=n1; n2s+=m; nds+=nd
            if warm: per_layer_phase[l]+=ph
        if warm:
            n+=1
            acc['phase']+=tok_phase; acc['faults']+=tok_faults; acc['n1']+=n1s; acc['n2']+=n2s; acc['nd']+=nds
    ph = acc['phase']/n/1000; faults=acc['faults']/n
    tok_ms = (N_LAYER*(chain_us+ensure_us+other_us) + fixed_us + engram_us)/1000 + ph
    if not quiet:
        print(f"{name:62s} phase {ph:6.2f} ms  faults/tok {faults:5.1f}  n1 {acc['n1']/n/40:4.2f} n2 {acc['n2']/n/40:4.2f} nd {acc['nd']/n/40:4.2f} | token {tok_ms:6.1f} ms = {1000/tok_ms:5.1f} tok/s")
    return ph, faults, tok_ms

def sets_top(fit, N, layers=range(N_LAYER)):
    return [set(fit[l][:N]) if l in layers else set() for l in range(N_LAYER)]
EMPTY=[set() for _ in range(N_LAYER)]
ENC=range(0,20); DEC=range(20,40)
REG_TODAY=[268]*20+[40]*20
REG_BAL=[154]*40

print(f"trace {ntok} tokens; warm half = {T0}..{ntok-1}\n")
print("=== A. Where we are (post-fix), box2 per-layer regions 268/40, box1 = hot top-64 enc pinned + 25 LRU, no dGPU")
print("    chain 470 + ensure 100 + other 200 us/layer; engram 8 ms; F2 = 4 ms/fault; I2 = 175")
run("A0 today (hot top-64 enc pinned, 25 LRU, regions 268/40)", sets_top(FIT_HOT,64,ENC), 25, EMPTY, 'regions', REG_TODAY)
run("A1 same, claim cap 4", sets_top(FIT_HOT,64,ENC), 25, EMPTY, 'regions', REG_TODAY, claim_cap=4)
run("A2 nothing local (mode 2), regions 268/40", EMPTY, 0, EMPTY, 'regions', REG_TODAY)
print()
print("=== B. Box 2 pool shape (nothing local on box 1 = mode 2)")
run("B0 regions 268/40 (today)", EMPTY, 0, EMPTY, 'regions', REG_TODAY)
run("B1 regions balanced 154/154", EMPTY, 0, EMPTY, 'regions', REG_BAL)
run("B2 GLOBAL pool 6160", EMPTY, 0, EMPTY, 'global', 6160)
run("B3 GLOBAL pool 6160, F2=1.5 ms (3 roles parallel)", EMPTY, 0, EMPTY, 'global', 6160, F2=1500)
run("B4 GLOBAL pool 6160, ZERO faults (steady-state ideal)", EMPTY, 0, EMPTY, 'global', 6160, F2=0)
run("B5 GLOBAL pool 8192 (152 GB -- does not exist), F2=4ms", EMPTY, 0, EMPTY, 'global', 8192)
print()
print("=== C. Zero-fault expert phase: who computes what (box2 global 6160, F2=0)  -- in-sample oracle fits")
run("C0 all remote", EMPTY, 0, EMPTY, 'global', 6160, F2=0)
run("C1 + dGPU top-6/layer (oracle)", EMPTY, 0, sets_top(FIT_ALL,6), 'global', 6160, F2=0)
run("C2 + dGPU top-6/layer (hot file)", EMPTY, 0, sets_top(FIT_HOT,6), 'global', 6160, F2=0)
run("C3 + dGPU top-6 (split fit)", EMPTY, 0, sets_top(FIT_HALF,6), 'global', 6160, F2=0)
run("C4 box1 top-40/layer all layers (oracle), cap 3", sets_top(FIT_ALL,40), 0, EMPTY, 'global', 6160, F2=0, claim_cap=3)
run("C5 box1 top-40 (oracle) cap 3 + dGPU top-6 (oracle)", sets_top(FIT_ALL,40), 0, sets_top(FIT_ALL,6), 'global', 6160, F2=0, claim_cap=3)
run("C6 box1 top-40 (split) cap 3 + dGPU top-6 (split)", sets_top(FIT_HALF,40), 0, sets_top(FIT_HALF,6), 'global', 6160, F2=0, claim_cap=3)
run("C7 box1 top-64 (hot) cap 3 + dGPU top-6 (hot)", sets_top(FIT_HOT,64), 0, sets_top(FIT_HOT,6), 'global', 6160, F2=0, claim_cap=3)
run("C8 box1 top-116 all layers (oracle) cap 3 + dGPU top-8 (oracle)", sets_top(FIT_ALL,116), 0, sets_top(FIT_ALL,8), 'global', 6160, F2=0, claim_cap=3)
run("C9 box1 top-116 (split) cap 3 + dGPU top-8 (split)", sets_top(FIT_HALF,116), 0, sets_top(FIT_HALF,8), 'global', 6160, F2=0, claim_cap=3)
run("C10 EVERYTHING on box1 (all 6 local, no remote)", [set(range(384)) for _ in range(40)], 0, EMPTY, 'global', 6160, F2=0)
run("C11 box1 everything cap 3 + dGPU top-6 oracle, rest box2", [set(range(384)) for _ in range(40)], 0, sets_top(FIT_ALL,6), 'global', 6160, F2=0, claim_cap=3)
print()
print("=== D. Sensitivity to the box-2 intercept I2 (C6 config, split fit, F2=0)")
for I2 in (175, 300, 400, 600):
    run(f"D I2={I2}", sets_top(FIT_HALF,40), 0, sets_top(FIT_HALF,6), 'global', 6160, F2=0, claim_cap=3, I2=I2)
print()
print("=== E. Serial chain sensitivity (C6 config, F2=0, I2=175): chain+ensure+other per layer, engram")
for chain,ens,oth,eng,lbl in ((470,100,200,8000,"today"),(470,0,200,500,"engram fixed, ensure skipped"),(350,0,100,500,"+chain fused to 350, other 100"),(230,0,50,500,"chain at byte roofline 230")):
    run(f"E {lbl}", sets_top(FIT_HALF,40), 0, sets_top(FIT_HALF,6), 'global', 6160, F2=0, claim_cap=3, chain_us=chain, ensure_us=ens, other_us=oth, engram_us=eng)
print()
print("=== F. Deliberate drop (compute only first k of 6 picks; trace order = router rank order, UNVERIFIED), mode 2, global 6160, F2=0, chain today")
for k in (6,4,3,2,1):
    run(f"F k={k}", EMPTY, 0, EMPTY, 'global', 6160, F2=0, drop_to=k)
print("    ... with engram fixed + ensure skipped + chain 350:")
for k in (6,4,3,2):
    run(f"F' k={k}", EMPTY, 0, EMPTY, 'global', 6160, F2=0, drop_to=k, chain_us=350, ensure_us=0, other_us=100, engram_us=500)
