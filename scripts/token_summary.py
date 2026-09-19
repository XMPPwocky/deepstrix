#!/usr/bin/env python3
# Summarise the last N `het.token.summary` lines of the live server log.
#   python3 scripts/token_summary.py [N=600] [LOG=/home/claude-code/logs/v41-server.log]
# link/srv/sel_sync are the knobs' scoreboard (see docs/v41/LINK_IDLE_LATENCY.md).
import re,sys,statistics as st
ansi=re.compile(r'\x1b\[[0-9;]*m')
rows=[]
LOG=sys.argv[2] if len(sys.argv)>2 else '/home/claude-code/logs/v41-server.log'
for line in open(LOG,errors='replace'):
    if 'het.token.summary' not in line: continue
    line=ansi.sub('',line)
    d=dict(re.findall(r'(\w+)=(-?\d+)',line))
    d={k:int(v) for k,v in d.items()}
    d['ts']=line[:26]
    rows.append(d)
N=int(sys.argv[1]) if len(sys.argv)>1 else 600
rows=rows[-N:]
print("tokens",len(rows),"from",rows[0]['ts'],"to",rows[-1]['ts'])
keys=['total_us','host_us','sync_us','remote_rtt_us','remote_srv_us','remote_link_us','sel_sync_us','pager_ensure_us','pager_read_us','pager_h2d_us','pager_misses','engram_us','gap_us','sample_us','pre_us','engram_stage_us','peer_bytes','token_pos']
def q(v,p):
    v=sorted(v); return v[min(len(v)-1,int(p*len(v)))]
print(f"{'field':18}{'p10':>10}{'p50':>10}{'p90':>10}{'mean':>10}")
for k in keys:
    v=[r.get(k,0) for r in rows]
    print(f"{k:18}{q(v,.1):>10}{q(v,.5):>10}{q(v,.9):>10}{st.mean(v):>10.0f}")
import collections
c=collections.Counter(r.get('pager_misses',0) for r in rows)
print("miss hist",sorted(c.items()))
for m in [0,1,2,3]:
    sub=[r for r in rows if r.get('pager_misses',0)==m] if m<3 else [r for r in rows if r.get('pager_misses',0)>=3]
    if sub: print(f"misses{'>=' if m==3 else '='}{m}: n={len(sub)} total p50={q([r['total_us'] for r in sub],.5)} rtt p50={q([r['remote_rtt_us'] for r in sub],.5)} srv p50={q([r['remote_srv_us'] for r in sub],.5)} ensure p50={q([r['pager_ensure_us'] for r in sub],.5)} selsync p50={q([r['sel_sync_us'] for r in sub],.5)}")
tot=sum(r['total_us'] for r in rows)/1e6
print(f"sum total {tot:.1f}s -> {len(rows)/tot:.2f} tok/s over summed token time")
