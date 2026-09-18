import os, sys, time, torch
sys.path.insert(0, "/home/claude-code/deepstrix/scripts/v41_oracle")
torch.set_default_dtype(torch.bfloat16)
b, m, h, d, seqlen, win, nc = 3, 128, 64, 512, 2048, 128, 2048
q = torch.randn(b, m, h, d).to(torch.bfloat16); kv = torch.randn(b, seqlen+nc, d).to(torch.bfloat16)
sink = torch.randn(h).float(); scale = d ** -0.5
widx = torch.arange(win, dtype=torch.int32).view(1,1,-1).expand(b,m,-1).contiguous()
cidx = torch.where(torch.rand(b,m,nc) < 0.5, torch.arange(nc, dtype=torch.int32)+seqlen, torch.tensor(-1,dtype=torch.int32))
topk_idxs = torch.cat([widx, cidx], -1)
wk, wi = seqlen, win
T = {}
def tick(k, t0):
    T[k] = T.get(k, 0.0) + time.time() - t0
    return time.time()
t0 = time.time()
kvw = kv[:, :wk].float(); kvwf = kvw.reshape(b*wk, d); kvc = kv[:, wk:].float()
idxw = topk_idxs[..., :wi].long(); idxc = topk_idxs[..., wi:].long()
qf = q.float(); sinkv = sink.view(1,1,h,1)
out = torch.empty(b, m, h, d, dtype=q.dtype)
t0 = tick("setup", t0)
chunk = 27
for c0 in range(0, m, chunk):
    c1 = min(m, c0+chunk); c = c1-c0
    qc = qf[:, c0:c1]; iw = idxw[:, c0:c1]; vw = (iw >= 0).unsqueeze(2)
    flat = (iw.clamp_min(0) + (torch.arange(b)*wk).view(b,1,1)).reshape(-1)
    kvg = kvwf.index_select(0, flat).view(b, c, wi, d)
    t0 = tick("gather", t0)
    sw = torch.einsum("bmhd,bmtd->bmht", qc, kvg) * scale
    sw = sw.masked_fill(~vw, float("-inf"))
    t0 = tick("score_win", t0)
    ic = idxc[:, c0:c1]
    pos = torch.where(ic >= 0, ic - wk, torch.tensor(nc, dtype=torch.long))
    mc = torch.zeros(b, c, nc+1, dtype=torch.bool); mc.scatter_(-1, pos, True); mc = mc[..., :nc].unsqueeze(2)
    t0 = tick("mask", t0)
    sc = torch.einsum("bmhd,bnd->bmhn", qc, kvc) * scale
    t0 = tick("score_comp", t0)
    sc = sc.masked_fill(~mc, float("-inf"))
    t0 = tick("mfill", t0)
    mx = torch.maximum(sw.amax(-1, keepdim=True), sc.amax(-1, keepdim=True)).clamp_min(-1e30)
    t0 = tick("amax", t0)
    ew = torch.exp(sw - mx).masked_fill(~vw, 0.0)
    ec = torch.exp(sc - mx).masked_fill(~mc, 0.0)
    t0 = tick("exp", t0)
    den = ew.sum(-1, keepdim=True) + ec.sum(-1, keepdim=True) + torch.exp(sinkv - mx)
    pw, pc = ew/den, ec/den
    t0 = tick("norm", t0)
    out[:, c0:c1] = (torch.einsum("bmht,bmtd->bmhd", pw, kvg) + torch.einsum("bmhn,bnd->bmhd", pc, kvc)).to(q.dtype)
    t0 = tick("av", t0)
tot = sum(T.values())
for k, v in sorted(T.items(), key=lambda x: -x[1]):
    print(f"{k:12s} {v:7.2f}s  {100*v/tot:5.1f}%")
print(f"total {tot:.2f}s for m={m} -> {tot*2048/m:.0f}s/layer")
