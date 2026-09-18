import pickle, collections, re, bisect
tracks, lv = pickle.load(open('leaves.pkl','rb'))
steps=pickle.load(open('steps.pkl','rb'))
RE=0x52454d5400000001
s=sorted(lv[RE])
kinds=collections.Counter()
bvals=collections.Counter()
for (b,e,n) in s:
    k=re.sub(r'L\d+','L#',n)
    k=re.sub(r'rtt=\d+us link=\d+us remote=\d+us','rtt/link/remote',k)
    k=re.sub(r'b=\d+','b=N',k)
    kinds[k]+=1
    m=re.search(r'b=(\d+)',n)
    if m: bvals[('decode' if n.startswith('decode') else 'batched', int(m.group(1)))]+=1
for k,c in kinds.most_common(): print(f"  {c:>6} {k}")
print()
print("b values:", sorted(bvals.items(), key=lambda kv:-kv[1])[:20])
# which steps contain non-'decode' submits?
st=[x[0] for x in s]
print()
print("step -> counts of 'submit L# b=N' (batched) vs 'decode submit'")
for idx,(c,b,e,n) in enumerate(steps):
    i=bisect.bisect_left(st,b); j=bisect.bisect_left(st,e)
    seg=s[i:j]
    bat=sum(1 for x in seg if x[2].startswith('submit'))
    dec=sum(1 for x in seg if x[2].startswith('decode submit'))
    if idx<6 or bat or idx in (46,47,48,49,50,113):
        print(f"  {idx:>4} {'VERIFYcls' if c else 'decodecls'} n={n} batched_submits={bat} decode_submits={dec}")
