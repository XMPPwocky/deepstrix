#!/usr/bin/env python3
"""Held-out coverage: fit the placement on one half of the trace, score the other.

An in-sample fit is an ORACLE BOUND, not a deployable placement. The gap between
them is what a static hot tier actually loses to routing drift.
"""
import sys, array, collections
N_LAYER, TOPK, N_EXPERT = 40, 6, 384
buf=array.array('H'); buf.frombytes(open(sys.argv[1],'rb').read())
per_tok=N_LAYER*TOPK; ntok=len(buf)//per_tok
def counts(lo,hi):
    f=[collections.Counter() for _ in range(N_LAYER)]
    for t in range(lo,hi):
        for l in range(N_LAYER):
            seen=set()
            for k in range(TOPK):
                e=buf[t*per_tok+l*TOPK+k]
                if e<N_EXPERT and e not in seen:
                    seen.add(e); f[l][e]+=1
    return f
half=ntok//2
fit=counts(0,half); test=counts(half,ntok)
tot=sum(sum(f.values()) for f in test)
print(f"fit on tokens 0..{half}, scored on {half}..{ntok}  ({tot} picks scored)\n")
print(f"{'k/layer':>8} {'VRAM GB':>8} {'IN-SAMPLE':>10} {'HELD-OUT':>9} {'oracle(test)':>13} {'retained':>9}")
for k in [4,6,8,12,16,22,32,48]:
    ins  = sum(sum(c for _,c in fit[l].most_common(k)) for l in range(N_LAYER))
    inst = sum(fit[l].total() for l in range(N_LAYER))
    ho   = sum(sum(test[l][e] for e,_ in fit[l].most_common(k)) for l in range(N_LAYER))
    orc  = sum(sum(c for _,c in test[l].most_common(k)) for l in range(N_LAYER))
    print(f"{k:8d} {k*N_LAYER*18.8/1000:8.1f} {100*ins/inst:9.1f}% {100*ho/tot:8.1f}% {100*orc/tot:12.1f}% {100*ho/orc:8.1f}%")
