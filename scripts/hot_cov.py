import array, collections
N_LAYER, TOPK, N_EXPERT = 40, 6, 384
buf = array.array('H'); buf.frombytes(open('/home/claude-code/.cache/deepstrix/v41/expert_trace_v41_decode313.bin','rb').read())
per = N_LAYER*TOPK; ntok = len(buf)//per
hot=[]
for line in open('/home/claude-code/.cache/deepstrix/hot_experts.txt'):
    items=[x.split(':') for x in line.strip().split(',') if x]
    hot.append([(int(e),int(c)) for e,c in items])
print("hot_experts.txt layers", len(hot), "entries/layer min/max", min(len(h) for h in hot), max(len(h) for h in hot))
tot_counts=[sum(c for _,c in h) for h in hot]
print("total counts per layer (first 3):", tot_counts[:3])
# coverage of hot top-N on the trace, per layer group
freq=[collections.Counter() for _ in range(N_LAYER)]
for t in range(ntok):
    for l in range(N_LAYER):
        for k in range(TOPK):
            freq[l][buf[t*per+l*TOPK+k]]+=1
for N in (4,6,8,16,40,116,154,200,268):
    enc=dec=0; enc_n=dec_n=0
    per_layer=[]
    for l in range(N_LAYER):
        top=set(e for e,_ in hot[l][:N])
        n=sum(freq[l].values()); c=sum(v for e,v in freq[l].items() if e in top)
        per_layer.append(c/n)
        if l<20: enc+=c; enc_n+=n
        else: dec+=c; dec_n+=n
    print(f"hot top-{N:3d}: enc cov {enc/enc_n*100:5.1f}%  dec cov {dec/dec_n*100:5.1f}%  all {(enc+dec)/(enc_n+dec_n)*100:5.1f}%  min layer {min(per_layer)*100:.1f}")
# Zipf shape of hot_experts.txt itself (its own counts)
print("\nhot_experts.txt self-coverage (its own counts):")
for N in (6,40,116,154,268):
    cov=[sum(c for _,c in h[:N])/sum(c for _,c in h) for h in hot]
    print(f"  top-{N}: enc {sum(cov[:20])/20*100:.1f}%  dec {sum(cov[20:])/20*100:.1f}%")
