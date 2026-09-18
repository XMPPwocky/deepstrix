import pickle, collections, bisect
tracks, lv = pickle.load(open('leaves.pkl','rb'))
steps = pickle.load(open('steps.pkl','rb'))
DG=0x4450475500000001; IG=0x4950475500000001; RE=0x52454d5400000001
for u in lv: lv[u].sort()
V=[(b,e) for (c,b,e,n) in steps if c and n==40]
D=[(b,e) for (c,b,e,n) in steps if not c and n==40]
def inwin(u,wins):
    st=[x[0] for x in lv[u]]
    out=[]
    for (b,e) in wins:
        i=bisect.bisect_left(st,b); j=bisect.bisect_left(st,e)
        out.append(lv[u][i:j])
    return out
def report(u,wins,label,topn=20):
    print(f"===== {tracks[u]} — {label} ({len(wins)} steps) =====")
    agg=collections.Counter(); cnt=collections.Counter()
    tot_gap=0; tot_busy=0; tot_wall=sum(e-b for b,e in wins)
    biggest=[]
    for seg,(wb,we) in zip(inwin(u,wins),wins):
        seg=[s for s in seg]
        prev_end=None; prev_name='<step start>'
        for (b,e,n) in seg:
            if '.wait' in (n or ''): pass
            if prev_end is not None and b>prev_end:
                g=b-prev_end; agg[(prev_name,n)]+=g; cnt[(prev_name,n)]+=1; tot_gap+=g
                biggest.append((g,prev_name,n))
            tot_busy += (e-b) if '.wait' not in (n or '') else 0
            prev_end = e if prev_end is None else max(prev_end,e)
            prev_name=n
    print(f"  wall {tot_wall/1e6:.1f} ms | busy {tot_busy/1e6:.1f} ms ({100*tot_busy/tot_wall:.1f}%) | inter-leaf gap {tot_gap/1e6:.1f} ms ({100*tot_gap/tot_wall:.1f}%)")
    print(f"  per step: wall {tot_wall/len(wins)/1e6:.2f} busy {tot_busy/len(wins)/1e6:.2f} gap {tot_gap/len(wins)/1e6:.2f} ms")
    print(f"  top {topn} gap kinds by TOTAL (ms tot | n | avg us | ms/step):")
    for (p,c),t in agg.most_common()[:topn]:
        n=cnt[(p,c)]
        print(f"    {t/1e6:>8.1f} | {n:>5} | {t/n/1000:>9.1f} | {t/len(wins)/1e6:>7.2f} | {p}  ->  {c}")
    print()
report(DG,V,"VERIFY steps")
report(IG,V,"VERIFY steps")
report(DG,D,"decode steps",15)
report(IG,D,"decode steps",10)
