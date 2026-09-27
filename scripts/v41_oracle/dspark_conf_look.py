import json, math
B='/home/claude-code/.cache/deepstrix/v41/agentic'
sig=lambda x: 1/(1+math.exp(-x))
def load(run, name): return json.load(open(f'{B}/{run}/dspark_accept_{name}.json'))
def summ(tag, R, d=None):
    n=len(R)
    print(f"\n== {tag}: {n} steps, positions {R[0]['i']}..{R[-1]['i']}" + (f", conf_auc {d.get('conf_auc')}" if d else ""))
    print("depth  mean conf  mean sigmoid(conf)  hit rate  prefix-acc  mean RS-acc  mean main entropy")
    pre=[0]*5
    for r in R:
        ok=True
        for k in range(5):
            ok = ok and r['hit'][k]; pre[k]+=ok
    for k in range(5):
        c=[r['conf'][k] for r in R]; h=[r['hit'][k] for r in R]
        rs=[r['rs'][k] for r in R] if 'rs' in R[0] else [float('nan')]
        en=[r['ent'][k] for r in R] if 'ent' in R[0] else [float('nan')]
        print(f"  d{k+1}    {sum(c)/n:+6.2f}        {sum(map(sig,c))/n:.3f}           {sum(h)/n:.3f}     {pre[k]/n:.3f}       {sum(rs)/len(rs):.3f}        {sum(en)/len(en):.2f}")
    # conf-predicted E: product of sigmoids along the block (if conf is P(accept_k | accepted <k))
    pe=sum(1+sum(math.prod(sig(r['conf'][j]) for j in range(k+1)) for k in range(5)) for r in R)/n
    ae=1+sum(p/n for p in pre)
    print(f"  E actual {ae:.2f}   E predicted by conf (chain of sigmoids) {pe:.2f}")
def calib(tag, Rs):
    buckets=[(-99,-1),(-1,0),(0,1),(1,2),(2,4),(4,99)]
    print(f"\n-- calibration {tag}: conf bucket -> hit rate (all depths pooled; hits counted only where earlier drafts all hit)")
    for lo,hi in buckets:
        n=h=0
        for r in Rs:
            for k in range(5):
                if all(r['hit'][:k]) and lo<=r['conf'][k]<hi:
                    n+=1; h+=r['hit'][k]
        if n: print(f"   [{lo:>3},{hi:>3})  n={n:4d}  hit {h/n:.2f}   sigmoid-mid {sig((max(lo,-4)+min(hi,6))/2):.2f}")
p=load('prose','base'); g=load('gen2','base')
summ('prose base (74-row seed)', p['records'], p)
summ('gen2 base (128-row seed)', g['records'], g)
ns=load('gen2','noseed'); summ('gen2 noseed', ns['records'], ns)
P=p['records']
# prose window fill: ring rows at step i = min(128, i+1) (positions 0..i written); full from i=127
summ('prose, window NOT yet full (i < 127)', [r for r in P if r['i']<127])
summ('prose, window full (i >= 127)', [r for r in P if r['i']>=127])
calib('gen2 base', g['records']); calib('prose base', P)
