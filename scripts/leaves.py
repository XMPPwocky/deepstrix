import pickle, collections
tracks, sl = pickle.load(open('slices.pkl','rb'))
out={}
for u, s in sl.items():
    ss=sorted(s, key=lambda x:(x[0],-x[1]))
    lv=[]
    for i,cur in enumerate(ss):
        if i+1<len(ss) and ss[i+1][0] < cur[1]:
            continue   # has a nested child
        lv.append(cur)
    out[u]=lv
    print(tracks[u], len(s), "->leaves", len(lv))
pickle.dump((tracks,out), open('leaves.pkl','wb'))
