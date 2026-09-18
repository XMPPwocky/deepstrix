import re, statistics as st, sys
path=sys.argv[1]; tag=sys.argv[2]
txt=open(path,errors='replace').read()
def kv(l): return dict(re.findall(r'(\w+)=("?[-\w./]+"?)', l))
toks=[kv(l) for l in txt.splitlines() if 'het.token.summary' in l]
# split into runs: a new run starts when pos decreases or jumps by >200
runs=[]; cur=[]; last=None
for t in toks:
    p=int(t['token_pos'].strip('"'))
    if last is not None and (p < last or p-last > 200):
        runs.append(cur); cur=[]
    cur.append(t); last=p
runs.append(cur)
print(f"--- {tag} ---")
for i,r in enumerate(runs):
    sel=r[4:]
    if len(sel)<5: continue
    def m(k):
        v=[int(t[k].strip('"')) for t in sel if k in t]
        return st.mean(v) if v else 0
    print(f" run{i} pos {sel[0]['token_pos']:>5}..{sel[-1]['token_pos']:>5} n={len(sel):3d} | "
          f"ms/tok {m('total_us')/1000:7.1f} tok/s {1e6/m('total_us'):5.2f} | "
          f"misses {m('pager_misses'):6.1f} ms/miss {(m('pager_read_us')+m('pager_h2d_us'))/max(m('pager_misses'),1)/1000:5.2f} | "
          f"ensure {m('pager_ensure_us')/1000:7.1f} sel_sync {m('sel_sync_us')/1000:5.1f} "
          f"dgpu_stage {m('dgpu_busy_us')/1000:6.1f} other {(m('total_us')-m('pager_ensure_us')-m('sel_sync_us'))/1000:5.1f}")
