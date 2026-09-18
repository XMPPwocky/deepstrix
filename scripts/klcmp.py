import sys, struct, math
# engine logits: raw f32 little-endian
eng_path, oracle_pt = sys.argv[1], sys.argv[2]
raw = open(eng_path,'rb').read()
eng = list(struct.unpack(f'<{len(raw)//4}f', raw))
# oracle logits via torch
import torch
o = torch.load(oracle_pt, map_location='cpu')
orc = o.float().flatten().tolist()
n = min(len(eng), len(orc))
eng, orc = eng[:n], orc[:n]
def softmax_logZ(x):
    m = max(x); z = sum(math.exp(v-m) for v in x); return m, math.log(z)
em, elz = softmax_logZ(eng); om, olz = softmax_logZ(orc)
# KL(oracle || engine) = sum q*(log q - log p), q=oracle, p=engine
kl_oe = 0.0; kl_eo = 0.0
for i in range(n):
    lq = orc[i]-om-olz; q = math.exp(lq)
    lp = eng[i]-em-elz; p = math.exp(lp)
    if q>1e-12: kl_oe += q*(lq-lp)
    if p>1e-12: kl_eo += p*(lp-lq)
ea = max(range(n), key=lambda i: eng[i]); oa = max(range(n), key=lambda i: orc[i])
print(f'  KL(oracle||engine) = {kl_oe:.5f} nats')
print(f'  KL(engine||oracle) = {kl_eo:.5f} nats')
print(f'  argmax: oracle={oa}  engine={ea}  {"MATCH" if oa==ea else "DIFFER"}')
