import re, statistics as st, collections, sys
path=sys.argv[1]
txt=open(path,errors='replace').read()
def kv(l): return dict(re.findall(r'(\w+)=("?[-\w./]+"?)', l))
def clean(v): return v.strip('"')
stages=[kv(l) for l in txt.splitlines() if 'het.stage' in l]
ranges=[("SHORT ctx",20,120),("LONG ctx",2600,2900)]
if len(sys.argv)>2:
    ranges=[("R%d"%i, int(a), int(b)) for i,(a,b) in enumerate(zip(sys.argv[2::2], sys.argv[3::2]))]
for label, lo, hi in ranges:
    acc=collections.defaultdict(list); calls=collections.defaultdict(list); toks=set()
    for d in stages:
        p=int(clean(d['token_pos']))
        if not (lo<=p<=hi): continue
        toks.add(p)
        k=(clean(d['device']), clean(d['stage']))
        acc[k].append(int(clean(d['total_us']))); calls[k].append(int(clean(d['calls'])))
    if not toks: continue
    print(f"\n=== {label} pos {lo}-{hi} ({len(toks)} tokens) ===")
    rows=sorted(((st.mean(v), k, st.mean(calls[k])) for k,v in acc.items()), reverse=True)
    dg=ig=0
    for m,k,c in rows:
        print(f"  {k[0]:5s} {k[1]:36s} {m:9.1f} us   calls {c:6.1f}   {m/max(c,1):7.1f} us/call")
        if k[0]=='dgpu': dg+=m
        else: ig+=m
    print(f"  -- dgpu stage total {dg:.1f} us   igpu stage total {ig:.1f} us")
