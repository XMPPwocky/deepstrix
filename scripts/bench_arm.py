import json,time,urllib.request,sys
para=("The expert pager maintains a pool of resident expert weights keyed by layer and expert id. "
      "When a routed pick misses, the pager selects a victim slot by least-recently-used order and "
      "pages the expert in from disk, repacking MXFP4 nibbles on the way. ")
UNIQ=[0]
def req(nsec,mt=120):
    UNIQ[0]+=1
    # A unique LEADING marker so no prefix is shared with any earlier request:
    # an identical prompt hits the snapshot prefix cache and skips prefill entirely.
    body=f"Document {UNIQ[0]}-{sys.argv[1]}-{sys.argv[2]}. " + "".join(
        f"Section {i} of doc {UNIQ[0]}. {para}" for i in range(nsec))
    P={"model":"deepseek-v4.1-flash","max_tokens":mt,"temperature":0,
       "messages":[{"role":"user","content":body+"\n\nIn two sentences, summarise what the text above describes."}]}
    r=urllib.request.Request("http://127.0.0.1:18080/v1/chat/completions",
      data=json.dumps(P).encode(),headers={"Content-Type":"application/json"})
    t=time.time(); d=json.load(urllib.request.urlopen(r,timeout=1800)); e=time.time()-t
    return d['usage']['prompt_tokens'], d['usage']['completion_tokens'], e
arm=sys.argv[1]
for i in range(6): req(60,40)
print(f"[{arm}] warmed", flush=True)
for n in (140,200):
    ts=[];p=0
    for i in range(3):
        p,c,e=req(n,120); ts.append(e)
    ts.sort()
    print(f"[{arm}] prompt~{p} tok: median {ts[1]:.1f}s  (runs {' '.join(f'{x:.1f}' for x in ts)})", flush=True)
