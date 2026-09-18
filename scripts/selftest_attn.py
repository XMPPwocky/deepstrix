import os, sys
sys.path.insert(0, "/home/claude-code/deepstrix/scripts/v41_oracle")
import torch
torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(0)
import oracle            # sets sys.path for the reference package
import model as ref
import kernel as K
import dense_index

b, m, h, d, n, topk = 2, 37, 8, 64, 300, 64
q = torch.randn(b, m, h, d).to(torch.bfloat16)
kv = torch.randn(b, n, d).to(torch.bfloat16)
sink = torch.randn(h).float()
scale = d ** -0.5

# a narrow selection with some -1 padding, sorted ascending like the reference
sel = torch.zeros(b, m, n, dtype=torch.bool)
for bb in range(b):
    for mm in range(m):
        k = min(topk, mm + 1)
        pick = torch.randperm(mm + 1)[:k]
        sel[bb, mm, pick] = True
narrow = torch.full((b, m, topk), -1, dtype=torch.int32)
for bb in range(b):
    for mm in range(m):
        idx = sel[bb, mm].nonzero().flatten()
        narrow[bb, mm, : idx.numel()] = idx.int()
wide = torch.where(sel, torch.arange(n, dtype=torch.int32), torch.tensor(-1, dtype=torch.int32))

o_ref = K.sparse_attn(q, kv, sink, narrow, scale).float()
o_gather_wide = K.sparse_attn(q, kv, sink, wide, scale).float()
o_mask_narrow = dense_index.sparse_attn_masked(q, kv, sink, narrow, scale).float()
o_mask_wide = dense_index.sparse_attn_masked(q, kv, sink, wide, scale).float()
den = o_ref.abs().max()
for name, o in (("gather(wide -1 padded)", o_gather_wide), ("masked(narrow)", o_mask_narrow), ("masked(wide)", o_mask_wide)):
    d_ = (o - o_ref).abs().max().item()
    print(f"{name:26s} max|d| {d_:.3e}  rel {d_/den:.3e}  bit-equal {(o==o_ref).float().mean()*100:.1f}%")

# chunking must be exact
os.environ["V41_ATTN_CHUNK_ELEMS"] = "4096"
o_small = dense_index.sparse_attn_masked(q, kv, sink, wide, scale).float()
print("chunked == unchunked:", bool((o_small == o_mask_wide).all()))
