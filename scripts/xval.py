#!/usr/bin/env python3
"""Cross-DOMAIN validation of a frequency-ranked expert placement.

Fit the table on one trace, score coverage on traces from unrelated prompts.
usage: xval.py FIT.bin  SPLIT.bin  off:len:label [off:len:label ...]
       (offsets/lengths in TOKENS within SPLIT.bin)
"""
import sys, array, collections
N_LAYER, TOPK = 40, 6
def toks(path, lo=0, hi=None):
    buf=array.array('H'); buf.frombytes(open(path,'rb').read())
    per=N_LAYER*TOPK; n=len(buf)//per
    hi = n if hi is None else min(hi,n)
    out=[]
    for t in range(lo,hi):
        row=[]
        for l in range(N_LAYER):
            seen=[]
            for k in range(TOPK):
                e=buf[t*per+l*TOPK+k]
                if e<384 and e not in seen: seen.append(e)
            row+=[(l,e) for e in seen]
        out.append(row)
    return out
fit_path, split_path = sys.argv[1], sys.argv[2]
fit=collections.Counter()
for r in toks(fit_path):
    for k in r: fit[k]+=1
order=[k for k,_ in fit.most_common()]
print(f"fit: {fit_path.split('/')[-1]}  ({sum(fit.values())} picks, {len(fit)} distinct pairs)\n")
print(f"{'domain':<10} {'tokens':>7} {'k=77':>8} {'k=154':>8} {'k=200':>8} {'unseen':>8}")
for spec in sys.argv[3:]:
    off,ln,label = spec.split(':'); off,ln = int(off),int(ln)
    ev=collections.Counter()
    for r in toks(split_path, off, off+ln):
        for k in r: ev[k]+=1
    tot=sum(ev.values())
    if tot==0: print(f"{label:<10} {'(empty)':>7}"); continue
    cov=[]
    for k in (77,154,200):
        sel=set(order[:k*N_LAYER])
        cov.append(100*sum(v for kk,v in ev.items() if kk in sel)/tot)
    unseen=100*sum(v for kk,v in ev.items() if kk not in fit)/tot
    print(f"{label:<10} {ln:7d} {cov[0]:7.1f}% {cov[1]:7.1f}% {cov[2]:7.1f}% {unseen:7.1f}%")
