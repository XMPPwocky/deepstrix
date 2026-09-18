#!/usr/bin/env python3
"""Stationary miss rate implied by access skew alone (no discovery term):
share of ALL accesses captured by the top-S keys by frequency."""
import sys, array, collections
N_LAYER,TOPK=40,6
buf=array.array('H'); buf.frombytes(open(sys.argv[1],'rb').read())
pt=N_LAYER*TOPK; ntok=len(buf)//pt
freq=collections.Counter()
for t in range(ntok):
    for l in range(N_LAYER):
        seen=[]
        for k in range(TOPK):
            e=buf[t*pt+l*TOPK+k]
            if e<384 and e not in seen: seen.append(e); freq[(l,e)]+=1
tot=sum(freq.values()); order=freq.most_common()
print(f"{len(freq)} distinct keys, {tot} accesses, {tot/ntok:.0f}/token")
print(f"\n{'slots':>7} {'GB':>6} {'%of 15360':>10} {'access share':>13} {'implied miss/tok':>17}")
cum=0;i=0
for S in [1024,2186,3072,4096,6144,8192,11915,15360]:
    while i<min(S,len(order)): cum+=order[i][1]; i+=1
    sh=cum/tot
    print(f"{S:7d} {S*18.8/1000:6.1f} {100*S/15360:9.1f}% {sh:13.4f} {(1-sh)*tot/ntok:17.1f}")
