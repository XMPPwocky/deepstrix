import json,time,urllib.request,subprocess,re
para=("The expert pager keeps resident expert weights in a pool keyed by layer and id. "
      "On a routed miss it evicts the least recently used slot and pages from disk. ")
def ask(body, mt):
    P={"model":"deepseek-v4.1-flash","max_tokens":mt,"temperature":0,
       "messages":[{"role":"user","content":body+"\n\nWrite a detailed paragraph about caching."}]}
    r=urllib.request.Request("http://127.0.0.1:18080/v1/chat/completions",
      data=json.dumps(P).encode(),headers={"Content-Type":"application/json"})
    t=time.time(); d=json.load(urllib.request.urlopen(r,timeout=2400)); e=time.time()-t
    return d['usage']['prompt_tokens'], d['usage']['completion_tokens'], e
LOG="/home/claude-code/logs/v41-prod.log"
def tail_tokens(n=60):
    out=subprocess.run(["bash","-c",f"sed 's/\\x1b\\[[0-9;]*m//g' {LOG} | grep 'het.token.summary' | tail -{n}"],
                       capture_output=True,text=True).stdout
    return [{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',l)} for l in out.strip().split('\n') if l]
F=["total_us","remote_rtt_us","remote_srv_us","remote_link_us","sel_sync_us","pager_ensure_us","pager_read_us","pager_misses"]
hdr={"total_us":"total","remote_rtt_us":"rtt","remote_srv_us":"srv","remote_link_us":"link",
     "sel_sync_us":"sel_sync","pager_ensure_us":"pg_ens","pager_read_us":"pg_read","pager_misses":"miss"}
print(f"{'ctx':>7} {'tok/s':>6} " + " ".join(f"{hdr[f]:>9}" for f in F), flush=True)
for nsec in (25, 250, 1000):
    body="FIXED CORPUS. "+"".join(f"Sec {i}. {para}" for i in range(nsec))
    for _ in range(3):            # warm: same prompt -> prefix cache + experts resident
        ask(body, 30)
    p,c,e=ask(body, 60)           # measured
    rows=[r for r in tail_tokens(60) if "total_us" in r][-45:]
    med={f: sorted(r.get(f,0) for r in rows)[len(rows)//2] for f in F}
    print(f"{p:>7} {c/e:>6.2f} " + " ".join(f"{med[f]:>9}" for f in F), flush=True)
