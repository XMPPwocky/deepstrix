import pickle, bisect, re, collections
tracks, lv = pickle.load(open('leaves.pkl','rb'))
steps=pickle.load(open('steps.pkl','rb'))
RE=0x52454d5400000001
s=sorted(lv[RE]); st=[x[0] for x in s]
for idx,(c,b,e,n) in enumerate(steps):
    i=bisect.bisect_left(st,b); j=bisect.bisect_left(st,e)
    bs=collections.Counter()
    for (x,y,nm) in s[i:j]:
        m=re.match(r'submit L\d+ b=(\d+)',nm)
        if m: bs[int(m.group(1))]+=1
        elif nm.startswith('decode submit'): bs['dec']+=1
    print(f"{idx:>4} nlayers={n:>2} wall={(e-b)/1e6:8.2f}ms  b={dict(bs)}")
