#!/usr/bin/env python3
"""V4.1 dGPU hot-expert placement from a post-submit-mask-fix routing trace.

Format (weights.rs::parse_hot_expert_file): one line per layer, comma-separated
`id:count`, descending. Counts present => the loader allocates the k_avg*N_LAYER
budget by GLOBAL GREEDY, so skewed layers get more slots than flat ones.
"""
import sys, array, collections
N_LAYER, TOPK, N_EXPERT = 40, 6, 384
TOP = 96                                   # enough for any k_avg up to 96/layer
buf=array.array('H'); buf.frombytes(open(sys.argv[1],'rb').read())
per_tok=N_LAYER*TOPK; ntok=len(buf)//per_tok
freq=[collections.Counter() for _ in range(N_LAYER)]
for t in range(ntok):
    for l in range(N_LAYER):
        seen=set()
        for k in range(TOPK):
            e=buf[t*per_tok+l*TOPK+k]
            if e<N_EXPERT and e not in seen:
                seen.add(e); freq[l][e]+=1
out=[]
for l in range(N_LAYER):
    out.append(",".join(f"{e}:{c}" for e,c in freq[l].most_common(TOP)))
open(sys.argv[2],"w").write("\n".join(out)+"\n")
tot=sum(sum(f.values()) for f in freq)
print(f"wrote {sys.argv[2]}: {N_LAYER} layers x top-{TOP}, from {ntok} tokens, {tot} picks")
print(f"distinct touched/layer: min {min(len(f) for f in freq)} mean {sum(len(f) for f in freq)/N_LAYER:.0f} max {max(len(f) for f in freq)}")
for k in (6,16,22):
    cov=sum(sum(c for _,c in f.most_common(k)) for f in freq)
    print(f"  uniform top-{k:2d}/layer covers {100*cov/tot:5.1f}% of picks ({k*N_LAYER} experts, {k*N_LAYER*18.8/1000:.1f} GB)")
