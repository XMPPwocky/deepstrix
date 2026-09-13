"""Learned lookahead routers: for lookahead k, a probe per layer l predicts layer
(l+k)'s routed top-6 from the residual after layer l. Initialised from layer
(l+k)'s own router (the "apply the later router" shortcut) and regularised
toward it (ridge prior), trained with a multi-target softmax cross-entropy on
the true 6 experts. Two input variants: mean over the 4 hc copies (dim 5120)
and the 4 copies concatenated (dim 20480, lets the probe learn the collapse).
Split by position: train on the first 70 %, test on the last 30 % of the transcript.

  python3 route_probe_train.py ~/.cache/deepstrix/v41/agentic/main
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oracle  # noqa: E402
from loader import Checkpoint  # noqa: E402

torch.manual_seed(0)


def main(dump, ks=(2, 3, 5), steps=100, lam=1e-3, lr=2e-4, variants=("mean", "copies")):
    ck = Checkpoint(oracle.MODEL)
    L = 40
    res4 = [torch.load(os.path.join(dump, f"layer_{l:02d}_residual.pt"))[0].float() for l in range(L)]  # [T, 4, dim]
    ids = [torch.load(os.path.join(dump, f"layer_{l:02d}_topk_ids.pt")).reshape(-1, 6).long() for l in range(L)]
    gate = [ck.get(f"layers.{l}.ffn.gate.weight").float() for l in range(L)]
    bias = [ck.get(f"layers.{l}.ffn.gate.bias").float() for l in range(L)]
    fnorm = [ck.get(f"layers.{l}.ffn_norm.weight").float() for l in range(L)]
    T = res4[0].shape[0]
    ntr = int(0.7 * T)
    # expert frequency (train split) for the "rare half" metric
    from collections import Counter
    freq = [Counter(ids[l][:ntr].flatten().tolist()) for l in range(L)]

    def features(l, k, variant):
        x = res4[l]
        w = fnorm[l + k]
        if variant == "mean":
            xm = x.mean(1)
            return xm * torch.rsqrt(xm.pow(2).mean(-1, keepdim=True) + 1e-20) * w
        xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-20) * w  # norm each copy
        return xn.reshape(T, -1)

    def recall(scores, true, rare=None):
        pred = scores.topk(6, dim=-1).indices
        hit = (pred.unsqueeze(2) == true.unsqueeze(1)).any(1)  # [n, 6] per true expert
        if rare is None:
            return hit.float().mean().item()
        m = rare[true]
        return (hit.float() * m).sum().item() / max(m.sum().item(), 1)

    print(f"T={T} train {ntr} test {T - ntr}; steps={steps} lam={lam} lr={lr} balanced={os.environ.get('PROBE_BALANCED')=='1'}")
    for variant in variants:
        for k in ks:
            t0 = time.time()
            base = base_r = tr = tr_r = 0.0
            n = 0
            for l in range(L - k):
                X = features(l, k, variant)
                true = ids[l + k]
                Xtr, Xte, ytr, yte = X[:ntr], X[ntr:], true[:ntr], true[ntr:]
                W0 = gate[l + k] if variant == "mean" else gate[l + k].repeat(1, 4) / 4.0
                b0 = bias[l + k]
                W = W0.clone().requires_grad_(True)
                b = b0.clone().requires_grad_(True)
                temp = torch.tensor(1.0, requires_grad=True)
                opt = torch.optim.Adam([W, b, temp], lr=lr)
                # inverse-frequency weights (train split) so the tail is not traded for the head
                cnt = torch.bincount(ytr.flatten(), minlength=384).float()
                wts = (1.0 / (cnt + 1.0)) if os.environ.get("PROBE_BALANCED") == "1" else torch.ones(384)
                wts = wts / wts.mean()
                for _ in range(steps):
                    logits = (Xtr @ W.t()) * temp + b
                    lp = torch.log_softmax(logits, -1)
                    loss = -(lp.gather(1, ytr) * wts[ytr]).mean() + lam * (W - W0).pow(2).sum()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    thr = 0.5 * (len(freq[l + k]) and sorted(freq[l + k].values())[len(freq[l + k]) // 2])
                    rare = torch.tensor([freq[l + k].get(e, 0) <= thr for e in range(384)], dtype=torch.float32)
                    s0 = torch.nn.functional.softplus(Xte @ W0.t()).sqrt() + b0  # the untrained shortcut (selection rule)
                    s1 = (Xte @ W.t()) * temp + b
                    base += recall(s0, yte); base_r += recall(s0, yte, rare)
                    tr += recall(s1, yte); tr_r += recall(s1, yte, rare)
                    n += 1
            print(f"{variant:6} k={k}: recall@6 test  shortcut {base / n:.3f} → trained {tr / n:.3f}   |  on the rarer half of experts "
                  f"{base_r / n:.3f} → {tr_r / n:.3f}   ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ks = tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (2, 3, 5)
    variants = sys.argv[3].split(",") if len(sys.argv) > 3 else ["mean", "copies"]
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    lam = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
    main(sys.argv[1], ks=ks, steps=steps, variants=variants, lam=lam)
