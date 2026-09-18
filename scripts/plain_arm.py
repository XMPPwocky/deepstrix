import json, subprocess, time, sys
ST='/home/claude-code/.cache/deepstrix/expert_stats.json'
def load(): 
    d=json.load(open(ST)); return d['decode']['tokens'], d['decode']['counts']
for i in range(1,5):
    req = 'agentic_req.json' if i<4 else 'req_wal.json'
    t0,c0=load(); w0=time.time()
    out=subprocess.run(['curl','-s','-m','3600','http://127.0.0.1:18141/v1/chat/completions','-H','Content-Type: application/json','--data','@'+req],capture_output=True,text=True).stdout
    wall=time.time()-w0
    time.sleep(1.5)  # last harvest+save
    t1,c1=load()
    try: u=json.loads(out)['usage']; ct=u['completion_tokens']
    except Exception: ct=-1
    dist=[]
    for l in range(40):
        row=[c1[l*384+e]-c0[l*384+e] for e in range(384)]
        n=sum(1 for v in row if v>0)
        if n: dist.append(n)
    dist.sort()
    print(f"PLAIN req{i} {req} completion_tokens={ct} decode_tokens_delta={t1-t0} wall={wall:.1f}s tok/s={ct/wall:.2f} decode_distinct_median={dist[len(dist)//2] if dist else 0} max={dist[-1] if dist else 0} min={dist[0] if dist else 0} layers={len(dist)}", flush=True)
