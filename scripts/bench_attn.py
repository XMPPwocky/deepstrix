import os, sys, time
sys.path.insert(0, "/home/claude-code/deepstrix/scripts/v41_oracle")
import torch
torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(0)
import oracle
import model as ref
import kernel as K
import dense_index

# exactness of the chunked gather vs the shim, at a small but realistic shape
b, m, h, d, n = 2, 96, 64, 512, 200
q = torch.randn(b, m, h, d).to(torch.bfloat16); kv = torch.randn(b, n, d).to(torch.bfloat16)
sink = torch.randn(h).float(); scale = d ** -0.5
sel = torch.rand(b, m, n) < 0.4
wide = torch.where(sel, torch.arange(n, dtype=torch.int32), torch.tensor(-1, dtype=torch.int32))
o_ref = K.sparse_attn(q, kv, sink, wide, scale)
for ce in (1 << 30, 1 << 22, 1 << 20):
    os.environ["V41_ATTN_CHUNK_ELEMS"] = str(ce)
    o = dense_index.sparse_attn_gather_chunked(q, kv, sink, wide, scale)
    print(f"gather_chunked(budget={ce}) bit-equal {bool((o==o_ref).all())} max|d| {(o.float()-o_ref.float()).abs().max():.2e}")

# speed at the real shape of a ratio-1 layer at T=2048 (window 2048 + compress 2048)
b, m, h, d, n = 2, 256, 64, 512, 4096
q = torch.randn(b, m, h, d).to(torch.bfloat16); kv = torch.randn(b, n, d).to(torch.bfloat16)
sink = torch.randn(h).float()
wide = torch.where(torch.rand(b, m, n) < 0.5, torch.arange(n, dtype=torch.int32), torch.tensor(-1, dtype=torch.int32))
os.environ["V41_ATTN_CHUNK_ELEMS"] = str(64 << 20)
for name, fn in (("masked", dense_index.sparse_attn_masked), ("gather_chunked", dense_index.sparse_attn_gather_chunked)):
    t0 = time.time(); fn(q, kv, sink, wide, scale); dt = time.time() - t0
    print(f"{name:16s} {dt:.2f}s for m={m} -> {dt*2048/m:.1f}s per layer at T=2048")
