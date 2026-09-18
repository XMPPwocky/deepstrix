import re,sys
SP="/tmp/claude-1000/-home-claude-code-deepstrix/e6bbeb19-ceab-4950-b666-df1b6c344139/scratchpad"
ansi=re.compile(r'\x1b\[[0-9;]*m')
lines=[ansi.sub('',l) for l in open(SP+"/"+sys.argv[1]) if "dspark.draft.split" in l]
use=lines[len(lines)//2:]; n=len(use); agg={}
def add(k,v): agg[k]=agg.get(k,0)+int(v)
for l in use:
    for k in ["l_hcmix_us","l_rms_us","l_attn_us","l_moe_us","l_hcpost_us"]:
        m=re.search(k+r"=(\d+)",l)
        if m: add(k,m.group(1))
    m=re.search(r'igpu_enqueue_ms="([\d.]+)"',l)
    if m: add("enq_us", int(float(m.group(1))*1000))
    for tag,pat in (("A",r'l_attn_split_us="([^"]+)"'),("K",r'l_kernel_split_us="([^"]+)"')):
        m=re.search(pat,l)
        if m:
            for kv in m.group(1).replace("|"," ").split():
                a,b=kv.split("="); add(f"{tag}.{a}",b)
BLK=["l_hcmix_us","l_rms_us","l_attn_us","l_moe_us","l_hcpost_us"]
tot=sum(agg.get(k,0) for k in BLK)
def row(lbl,v,den,ind=2):
    print(f"{' '*ind}{lbl:20s}{v/n/1000:8.2f} ms/step {100*v/den if den else 0:6.1f}%")
print(f"warm steps: {n}   drafter enqueue wall = {agg.get('enq_us',0)/n/1000:.2f} ms/step\n")
print("PER-STEP (3 drafter layers, DEVICE time):")
for k in BLK: row(k[2:-3],agg.get(k,0),tot)
print(f"  {'SUM':20s}{tot/n/1000:8.2f} ms/step\n")
at=agg.get("l_attn_us",0); mo=agg.get("l_moe_us",0)
print(f"ATTN {at/n/1000:.2f} ms ({100*at/tot:.0f}%):")
for k in ["A.qloop","A.kv","A.qa","A.outproj"]: row(k[2:],agg.get(k,0),at)
op=agg.get("A.outproj",0)
print("    outproj:")
for k in ["K.oquant","K.owa","K.owb"]: row(k[2:],agg.get(k,0),op,6)
print(f"\nMOE {mo/n/1000:.2f} ms ({100*mo/tot:.0f}%):")
acc=0
for k in ["K.mrouter","K.mtopk","K.mq8k","K.mgateup","K.mdown"]:
    row(k[2:],agg.get(k,0),mo); acc+=agg.get(k,0)
row("(shared+other)",mo-acc,mo)
