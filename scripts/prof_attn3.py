import time, torch
torch.set_default_dtype(torch.bfloat16)
b, m, h, d, wi, nc = 3, 128, 64, 512, 128, 2048
qf = torch.randn(b, m, h, d, dtype=torch.float32)
kvg = torch.randn(b, 27, wi, d)
kvc = torch.randn(b, nc, d)
def t(f, n=3):
    f(); t0=time.time()
    for _ in range(n): r=f()
    return (time.time()-t0)/n
qc_nc = qf[:, 10:37]
qc_c = qc_nc.contiguous()
print("qc slice contiguous?", qc_nc.is_contiguous())
for name, qc in (("noncontig", qc_nc), ("contig", qc_c)):
    print(f"{name:10s} win-score {t(lambda: torch.einsum('bmhd,bmtd->bmht', qc, kvg))*1000:8.1f} ms   "
          f"comp-score {t(lambda: torch.einsum('bmhd,bnd->bmhn', qc, kvc))*1000:8.1f} ms")
sw = torch.randn(b, 27, h, wi); pw = sw
print("av-window  noncontig-p", t(lambda: torch.einsum('bmht,bmtd->bmhd', pw, kvg))*1000, "ms")
# alternative formulation: reshape+bmm by hand
def win_score_bmm(qc, kvg):
    B, C = qc.shape[0], qc.shape[1]
    a = qc.reshape(B*C, h, d)
    bb = kvg.reshape(B*C, wi, d).transpose(1, 2)
    return torch.bmm(a, bb).view(B, C, h, wi)
print("win-score via bmm (contig q)", t(lambda: win_score_bmm(qc_c, kvg))*1000, "ms")
