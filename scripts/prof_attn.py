import sys, time, torch
torch.set_default_dtype(torch.bfloat16)
b, h, d, wi, nc, wk = 3, 64, 512, 128, 2048, 2048
for c in (27, 64, 128):
    qc = torch.randn(b, c, h, d)
    kvc = torch.randn(b, nc, d)
    kvg = torch.randn(b, c, wi, d)
    def t(f, n=3):
        f(); t0 = time.time()
        for _ in range(n): r = f()
        return (time.time() - t0) / n, r
    a, sc = t(lambda: torch.einsum("bmhd,bnd->bmhn", qc, kvc))
    fl = b*c*h*nc*d*2/1e9
    print(f"c={c:4d} score_compressed {a*1000:8.1f} ms  {fl/a:7.1f} GFLOP/s")
    a2, sw = t(lambda: torch.einsum("bmhd,bmtd->bmht", qc, kvg))
    fl2 = b*c*h*wi*d*2/1e9
    print(f"        score_window     {a2*1000:8.1f} ms  {fl2/a2:7.1f} GFLOP/s")
    a3, _ = t(lambda: torch.einsum("bmhn,bnd->bmhd", sc, kvc))
    print(f"        av_compressed    {a3*1000:8.1f} ms  {fl/a3:7.1f} GFLOP/s")
    a4, _ = t(lambda: torch.einsum("bmht,bmtd->bmhd", sw, kvg))
    print(f"        av_window        {a4*1000:8.1f} ms  {fl2/a4:7.1f} GFLOP/s")
    m = torch.rand(b, c, 1, nc) < 0.5
    a5, _ = t(lambda: torch.exp(sc.masked_fill(m, float("-inf")) - sc.amax(-1, keepdim=True)).masked_fill(m, 0.0))
    print(f"        softmax elemwise {a5*1000:8.1f} ms")
