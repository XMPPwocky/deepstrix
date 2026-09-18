import re,sys,datetime as dt
p=sys.argv[1]
raw=open(p,'rb').read().replace(b'\x00',b'').decode('utf-8','replace')
raw=re.sub(r'\x1b\[[0-9;]*m','',raw)
ts=lambda s: dt.datetime.fromisoformat(s.replace('Z','+00:00'))
reqs=[]; cur=None
for line in raw.splitlines():
    m=re.match(r'(\S+)\s+INFO\s+(\S+):\s+(.*)',line)
    if not m: continue
    t,mod,msg=m.groups(); t=ts(t)
    if 'engram rows_for_chunk' in msg:
        cur={'t_pf_start':t,'tok':[], 'pf_lines':[]}; reqs.append(cur)
    if cur is None: continue
    if 'ced_replay' in msg: cur['ced']=msg
    if msg.startswith('prefill req_len'): cur['t_pf_end']=t
    if 'expert pager (request)' in msg: cur['pager']=msg
    if 'het.token.summary' in msg:
        d=dict(re.findall(r'(\w+)=(\d+)',msg)); d['t']=t; cur['tok'].append(d)
    if 'decode heartbeat' in msg: cur.setdefault('hb',[]).append(re.search(r'tok_per_s="([\d.]+)"',msg).group(1))
    if 'saved live to disk' in msg: cur.setdefault('saves',[]).append(t)
for i,r in enumerate(reqs):
    tk=r['tok']; n=len(tk)
    if n==0: continue
    tot=sum(int(d['total_us']) for d in tk)/n
    sel=sum(int(d['sel_sync_us']) for d in tk)/n
    rtt=sum(int(d['remote_rtt_us']) for d in tk)/n
    ens=sum(int(d['pager_ensure_us']) for d in tk)/n
    loop_wall=(tk[-1]['t']-tk[0]['t']).total_seconds()
    pf=(r['t_pf_end']-r['t_pf_start']).total_seconds()
    pre_first=(tk[0]['t']-r['t_pf_end']).total_seconds()
    post=(r['saves'][-1]-tk[-1]['t']).total_seconds() if r.get('saves') else float('nan')
    print(f"req{i}: n_tok={n} prefill_wall={pf:.2f}s ({r.get('ced','')[:70]})")
    print(f"   pf_end->first_summary={pre_first*1e3:.0f}ms  loop_wall(first..last summary)={loop_wall:.2f}s = {loop_wall/(n-1)*1e3:.1f} ms/tok  last_summary->save={post*1e3:.0f}ms")
    print(f"   mean total_us={tot:.0f} sel_sync={sel:.0f} rtt={rtt:.0f} ensure={ens:.0f}  => in-loop gap={loop_wall/(n-1)*1e3-tot/1e3:.1f} ms/tok ; heartbeats={r.get('hb')}")
    if 'pager' in r: print('   ', r['pager'][:300])
    # distribution of total_us
    v=sorted(int(d['total_us']) for d in tk); print(f"   total_us min={v[0]} p50={v[n//2]} p90={v[int(n*.9)]} max={v[-1]}")
