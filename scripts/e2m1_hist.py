import json,struct,os,glob,re,collections
import numpy as np
files=sorted(glob.glob('*.incomplete'), key=os.path.getsize, reverse=True)
f=files[0]; size=os.path.getsize(f)
with open(f,'rb') as fh:
    L=struct.unpack('<Q',fh.read(8))[0]; hdr=json.loads(fh.read(L))
hdr.pop('__metadata__',None); base=8+L
layers=sorted({int(m.group(1)) for k in hdr for m in [re.search(r'layers\.(\d+)\.',k)] if m})
print(f"file={f[:30]}… landed={size/1e9:.2f} GB tensors={len(hdr)} layers={layers}")
inv=collections.OrderedDict()
for k,v in hdr.items():
    g=re.sub(r'experts\.\d+\.','experts.E.',k)
    if g not in inv: inv[g]=(v['dtype'],v['shape'])
for g,(d,sh) in inv.items(): print(f"  {g:62s} {d:8s} {sh}")
ready=[(k,v) for k,v in hdr.items() if re.search(r'\.experts\.\d+\.',k) and k.endswith('.weight') and base+v['data_offsets'][1]<=size]
print(f"\nfully-landed routed expert weight tensors: {len(ready)}  dtype={ready[0][1]['dtype'] if ready else None}")
hist=np.zeros(16,dtype=np.int64); n=0
with open(f,'rb') as fh:
    for k,v in ready[:12]:
        a,b=v['data_offsets']; fh.seek(base+a); buf=np.frombuffer(fh.read(b-a),dtype=np.uint8)
        hist+=np.bincount(buf&0x0F,minlength=16); hist+=np.bincount(buf>>4,minlength=16); n+=2*buf.size
    k,v=ready[0]; sk=k.replace('.weight','.weight_scale_inv')
    if sk in hdr and base+hdr[sk]['data_offsets'][1]<=size:
        a2,b2=hdr[sk]['data_offsets']; fh.seek(base+a2); sc=np.frombuffer(fh.read(b2-a2),dtype=np.uint8)
        print(f"scale tensor {hdr[sk]['dtype']} {hdr[sk]['shape']} raw byte range {sc.min()}..{sc.max()} → 2^{int(sc.min())-127}..2^{int(sc.max())-127}; distinct={len(np.unique(sc))}")
vals=[0,.5,1,1.5,2,3,4,6]; p=hist/hist.sum()
print(f"\nE2M1 code distribution over {n/1e6:.1f}M weights ({len(ready[:12])} tensors):")
print("  mag   P(+)     P(-)     P(|.|)")
for i in range(8): print(f"  {vals[i]:<4}  {p[i]:.4f}   {p[i+8]:.4f}   {p[i]+p[i+8]:.4f}")
H=-(p[p>0]*np.log2(p[p>0])).sum()
pm=np.array([p[i]+p[i+8] for i in range(8)]); Hm=-(pm[pm>0]*np.log2(pm[pm>0])).sum()
print(f"\nentropy of 4-bit code: {H:.3f} bits/weight (magnitude {Hm:.3f} + sign {1-pm[0]:.3f})")
print(f"→ zero-order lossless ≈ {H:.3f} + 0.25 (scales) = {H+0.25:.3f} bpw; MXFP4 4.25, IQ3_S 3.44, IQ3_XXS 3.06")
with open(f,'rb') as fh:
    k,v=ready[0]; a,b=v['data_offsets']; fh.seek(base+a); buf=np.frombuffer(fh.read(b-a),dtype=np.uint8)
codes=np.empty(2*buf.size,dtype=np.uint8); codes[0::2]=buf&0x0F; codes[1::2]=buf>>4
mag=(codes&7).astype(np.int64)
t=mag[:(mag.size//4)*4].reshape(-1,4); key=t[:,0]*512+t[:,1]*64+t[:,2]*8+t[:,3]
cnt=np.bincount(key,minlength=4096); srt=np.sort(cnt)[::-1]; tot=cnt.sum()
print(f"\n4-tuple magnitude codebook coverage ({k}, {tot/1e6:.1f}M groups): top-256={srt[:256].sum()/tot:.2%} top-512={srt[:512].sum()/tot:.2%} top-1024={srt[:1024].sum()/tot:.2%} distinct={(cnt>0).sum()}")
# per-32-block: how many distinct magnitudes / is the block max always 4 or 6?
blk=mag[:(mag.size//32)*32].reshape(-1,32); bm=blk.max(1)
print("block-of-32 max magnitude code distribution:", {vals[i]: round(float((bm==i).mean()),4) for i in range(8) if (bm==i).any()})
