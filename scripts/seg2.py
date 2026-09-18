import pickle, collections
tracks, sl = pickle.load(open('slices.pkl','rb'))
DG=0x4450475500000001
names=collections.Counter(n for (b,e,n) in sl[DG])
print("distinct dgpu.compute stage names:", len(names))
for n,c in names.most_common(60): print(f"  {c:>6} {n}")
