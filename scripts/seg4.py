import pickle
layers,cls=pickle.load(open('layers.pkl','rb'))
# walk layers, build steps of 40 within a class run
steps=[]; i=0
while i < len(layers):
    c=cls[i]; j=i
    while j<len(layers) and cls[j]==c: j+=1
    run=list(range(i,j))
    # chunk into 40s
    for k in range(0,len(run),40):
        ch=run[k:k+40]
        steps.append((c,layers[ch[0]][0],layers[ch[-1]][1],len(ch)))
    i=j
print(f"{len(steps)} steps")
for idx,(c,b,e,n) in enumerate(steps):
    w=(e-b)/1e6
    tag='VERIFY' if c else 'decode'
    flag=' <<<' if w>200 else ''
    print(f"{idx:>4} {tag} n={n:>2} wall={w:9.2f} ms{flag}")
import pickle as p; p.dump(steps, open('steps.pkl','wb'))
