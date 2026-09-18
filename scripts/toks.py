import re, statistics as st, sys
path=sys.argv[1]
txt=open(path,errors='replace').read()
def kv(l): return dict(re.findall(r'(\w+)=("?[-\w./]+"?)', l))
toks=[kv(l) for l in txt.splitlines() if 'het.token.summary' in l]
groups=[]; cur=[]; last=10**9
for t in toks:
    p=int(t['token_pos'].strip('"'))
    if p < last: 
        if cur: groups.append(cur)
        cur=[]
    cur.append(t); last=p
if cur: groups.append(cur)
fields=('total_us','dgpu_busy_us','igpu_busy_us','sel_sync_us','pager_ensure_us','pager_read_us','pager_h2d_us','pager_misses','host_us','sync_us')
for gi,g in enumerate(groups):
    sel=g[4:]
    if len(sel)<5: continue
    def m(k):
        v=[int(t[k].strip('"')) for t in sel if k in t]
        return st.mean(v) if v else 0
    print(f"req{gi}: pos {sel[0]['token_pos']}..{sel[-1]['token_pos']} n={len(sel)}  "
          f"ms/tok {m('total_us')/1000:7.1f}  tok/s {1e6/m('total_us'):5.2f}  "
          f"misses {m('pager_misses'):6.1f}  ensure {m('pager_ensure_us')/1000:7.1f}  "
          f"read {m('pager_read_us')/1000:7.1f}  h2d {m('pager_h2d_us')/1000:6.1f}  "
          f"sel_sync {m('sel_sync_us')/1000:6.1f}  dgpu {m('dgpu_busy_us')/1000:6.1f}  "
          f"other {(m('total_us')-m('pager_ensure_us')-m('sel_sync_us'))/1000:6.1f}")
