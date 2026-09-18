import os, sys, time
sys.path.insert(0, "/home/claude-code/deepstrix/scripts/v41_oracle")
import torch
torch.set_default_dtype(torch.bfloat16); torch.manual_seed(0)
import oracle, model as ref, kernel as K, dense_index as di

def build(b, m, h, d, seqlen, win, nc, topk):
    q = torch.randn(b, m, h, d).to(torch.bfloat16)
    kv = torch.randn(b, seqlen + nc, d).to(torch.bfloat16)
    sink = torch.randn(h).float()
    widx = ref.get_window_topk_idxs(win, b, m, 0)           # [b,m,min(m,win)]
    vis = torch.arange(nc).unsqueeze(0) < (torch.arange(1, m + 1) // 2).unsqueeze(-1).clamp(max=nc)
    sel = vis.clone()
    # emulate a top-k: keep at most `topk` of the visible rows (the newest ones)
    cs = sel.int().cumsum(-1)
    tot = sel.int().sum(-1, keepdim=True)
    sel = sel & (cs > (tot - topk).clamp_min(0))
    sel = sel.unsqueeze(0).expand(b, -1, -1).contiguous()
    cidx = torch.where(sel, torch.arange(nc, dtype=torch.int32) + seqlen, torch.tensor(-1, dtype=torch.int32))
    return q, kv, sink, torch.cat([widx, cidx], -1), sel

b, m, h, d, seqlen, win, nc, topk = 2, 96, 8, 64, 96, 16, 40, 12
q, kv, sink, idx, sel = build(b, m, h, d, seqlen, win, nc, topk)
scale = d ** -0.5
o_ref = K.sparse_attn(q, kv, sink, idx, scale).float()
di._WIN[0], di._WIN[1] = seqlen, win
di._PROBE = None
o_split = di.sparse_attn_masked(q, kv, sink, idx, scale).float()
dm = (o_split - o_ref).abs().max().item()
print(f"split vs reference gather: max|d| {dm:.3e} rel {dm/o_ref.abs().max():.3e} bit-equal {(o_split==o_ref).float().mean()*100:.2f}%")
os.environ["V41_ATTN_CHUNK_ELEMS"] = "8192"
o2 = di.sparse_attn_masked(q, kv, sink, idx, scale).float()
print("chunking changes result by max", (o2-o_split).abs().max().item())
# ratio-0 layer (window only)
idx_w = idx[..., :win]
kv_w = kv[:, :seqlen]
o_ref_w = K.sparse_attn(q, kv_w, sink, idx_w, scale).float()
os.environ["V41_ATTN_CHUNK_ELEMS"] = str(64<<20)
o_w = di.sparse_attn_masked(q, kv_w, sink, idx_w, scale).float()
print(f"window-only: max|d| {(o_w-o_ref_w).abs().max():.3e} bit-equal {(o_w==o_ref_w).float().mean()*100:.2f}%")
# speed at realistic shape, ratio-1 layer at T=2048, B=3
b, m, h, d, seqlen, win, nc = 3, 128, 64, 512, 2048, 128, 2048
q = torch.randn(b, m, h, d).to(torch.bfloat16); kv = torch.randn(b, seqlen+nc, d).to(torch.bfloat16)
sink = torch.randn(h).float()
widx = torch.arange(win, dtype=torch.int32).view(1,1,-1).expand(b,m,-1).contiguous()
cidx = torch.where(torch.rand(b,m,nc) < 0.5, torch.arange(nc, dtype=torch.int32)+seqlen, torch.tensor(-1,dtype=torch.int32))
idx = torch.cat([widx, cidx], -1)
di._WIN[0], di._WIN[1] = seqlen, win
os.environ["V41_ATTN_CHUNK_ELEMS"] = "16777216"
t0=time.time(); di.sparse_attn_masked(q, kv, sink, idx, scale); dt=time.time()-t0
print(f"split path: {dt:.2f}s for m={m} -> {dt*2048/m:.0f}s per ratio-1 layer at T=2048, B=3")
