import pickle, bisect, re, collections
tracks, lv = pickle.load(open('leaves.pkl','rb'))
RE=0x52454d5400000001; DG=0x4450475500000001
s=sorted(lv[RE])
fwd=[]  # (start_of_submitL0, kind, b)
marks=[]
for (b,e,n) in s:
    m=re.match(r'(decode )?submit L(\d+)(?: b=(\d+))?$', n)
    if m: marks.append((b,e,'decode' if m.group(1) else 'batch', int(m.group(2)), int(m.group(3)) if m.group(3) else 1))
print("marks", len(marks))
# forward = run of L0..L39
fw=[]
cur=None
for (b,e,k,L,bb) in marks:
    if L==0:
        if cur: fw.append(cur)
        cur={'start':b,'kind':k,'b':bb,'lastL':0,'end':e,'lays':[b]}
    elif cur is not None:
        cur['lastL']=L; cur['end']=e; cur['lays'].append(b)
if cur: fw.append(cur)
print("forwards", len(fw))
# extend each forward end to the start of the next forward
dgs=sorted(lv[DG]); dst=[x[0] for x in dgs]
for i,f in enumerate(fw):
    f['wend'] = fw[i+1]['start'] if i+1<len(fw) else f['end']
    # true start: back up to the dgpu.mhc_pre_attn preceding submit L0
    j=bisect.bisect_left(dst,f['start'])
    k=j
    while k>0 and dgs[k-1][2]!='dgpu.mhc_pre_attn': k-=1
    f['wstart']=dgs[k-1][0] if k>0 else f['start']
cnt=collections.Counter((f['kind'],f['b'],f['lastL']) for f in fw)
print(cnt)
pickle.dump(fw, open('fwd.pkl','wb'))
for i,f in enumerate(fw):
    w=(f['wend']-f['wstart'])/1e6
    print(f"{i:>4} {f['kind']:>6} b={f['b']:>2} lastL={f['lastL']:>2} wall={w:9.2f} ms")
