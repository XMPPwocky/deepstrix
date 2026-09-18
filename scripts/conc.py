import json, sys, time, threading, urllib.request
PROMPTS = ["Explain how a B-tree index works.",
           "Describe how a hash join executes.",
           "Explain MVCC in a database engine.",
           "Describe how a write-ahead log recovers."]
def one(i, out, mt):
    body = json.dumps({"model":"deepseek-v4.1-flash",
        "messages":[{"role":"user","content":PROMPTS[i % len(PROMPTS)]}],
        "max_tokens":mt,"temperature":0}).encode()
    t0=time.perf_counter()
    try:
        r=urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:18141/v1/chat/completions", body,
            {"Content-Type":"application/json"}), timeout=1800)
        d=json.loads(r.read()); out[i]=(d["usage"]["completion_tokens"], time.perf_counter()-t0)
    except Exception as e:
        out[i]=(0, time.perf_counter()-t0); print(f"  req{i} FAILED: {str(e)[:70]}")
for n in (1, 2, 4):
    out={}; ths=[threading.Thread(target=one,args=(i,out,192)) for i in range(n)]
    t0=time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join()
    wall=time.perf_counter()-t0
    tot=sum(v[0] for v in out.values())
    per=[f"{v[0]/v[1]:.2f}" for v in out.values() if v[1]>0]
    print(f"concurrency {n}: {tot} tok in {wall:.1f}s = AGGREGATE {tot/wall:.2f} tok/s | per-stream [{' '.join(per)}]")
