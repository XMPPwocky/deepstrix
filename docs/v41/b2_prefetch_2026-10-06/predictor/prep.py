#!/usr/bin/env python3
"""Prep (torch, CPU, few threads): for every layer l in 0..38 of the agentic dump, the
mean-over-hc-copies residual after layer l (f16, the trained heads' input), layer (l+1)'s
look-ahead gate scores on it (raw sqrtsoftplus s and selection score s + bias, f16), and the
TRUE top-6 ids at layer l+1 (the dump's topk_ids). One file per quantity under <out>.

usage: prep.py <dump_dir> <out_dir>
"""
import os, sys, time, resource
import torch
torch.set_num_threads(4)
sys.path.insert(0, '/home/claude-code/deepstrix/.claude/worktrees/b2-prefetch/scripts/v41_oracle')
from loader import Checkpoint  # noqa: E402

dump, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
MODEL = os.path.expanduser('~/.cache/deepstrix/models/dsv4.1f')
L, E = 40, 384
ck = Checkpoint(MODEL)
t0 = time.time()
T = torch.load(os.path.join(dump, 'layer_00_topk_ids.pt')).reshape(-1, 6).shape[0]
X = torch.empty(L - 1, T, 5120, dtype=torch.bfloat16)  # bf16: late-layer residuals reach |x| ~3e5 (f16 overflows)
S_RAW = torch.empty(L - 1, T, E, dtype=torch.float16)
S_SEL = torch.empty(L - 1, T, E, dtype=torch.float16)
IDS = torch.empty(L - 1, T, 6, dtype=torch.int16)
BIAS = torch.empty(L, E, dtype=torch.float32)
for l in range(L):
    BIAS[l] = ck.get(f"layers.{l}.ffn.gate.bias").float()
for l in range(L - 1):
    Lt = l + 1
    x = torch.load(os.path.join(dump, f"layer_{l:02d}_residual.pt"))[0].mean(dim=1).float()  # [T, 5120]
    g = ck.get(f"layers.{Lt}.ffn.gate.weight").float()
    fn = ck.get(f"layers.{Lt}.ffn_norm.weight").float()
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-20) * fn
    s = torch.nn.functional.softplus(xn @ g.t()).sqrt()
    X[l] = x.bfloat16(); S_RAW[l] = s.half(); S_SEL[l] = (s + BIAS[Lt]).half()
    IDS[l] = torch.load(os.path.join(dump, f"layer_{Lt:02d}_topk_ids.pt")).reshape(-1, 6).to(torch.int16)
    del x, xn, s, g, fn
    if l % 10 == 9:
        print(f"layer {l} {time.time() - t0:.0f}s rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB", flush=True)
# gate weights + ffn_norm for all layers (the per-layer LoRA variant needs them; 40 x 384 x 5120 x 2 B = 157 MB)
torch.save({'X': X, 'S_RAW': S_RAW, 'S_SEL': S_SEL, 'IDS': IDS, 'BIAS': BIAS, 'T': T}, os.path.join(out, 'data.pt'))
print(f"saved T={T} {time.time() - t0:.0f}s rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB")
# sanity: predicted rank-1 recall of the true rank-1 at l+1 (should be ~0.71 as in Step 0 / 09-14)
hit = (S_SEL.float().argmax(-1) == IDS[:, :, 0].long()).float().mean().item()
print(f"sanity: top-1 recall of true rank-1 at l+1 = {hit:.3f}")
