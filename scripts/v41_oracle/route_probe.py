"""Hidden-state routing prediction: can layer l's residual predict layer l+k's
routed experts? Predictor = layer (l+k)'s OWN router applied to the residual
after layer l (mean over hc copies, then that layer's ffn_norm, then gate +
sqrtsoftplus + bias for selection). Measures top-6 recall of the true picks
(from the same dump's topk_ids) within the predicted top-6 / top-12, per
lookahead k, plus the precision that a prefetcher would pay for.

  python3 route_probe.py ~/.cache/deepstrix/v41/agentic/main
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oracle  # noqa: E402
from loader import Checkpoint  # noqa: E402


def main(dump, ks=(1, 2, 3, 5, 8)):
    ck = Checkpoint(oracle.MODEL)
    L = 40
    res = [torch.load(os.path.join(dump, f"layer_{l:02d}_residual.pt"))[0].mean(dim=1).float() for l in range(L)]  # [T, dim]
    ids = [torch.load(os.path.join(dump, f"layer_{l:02d}_topk_ids.pt")).reshape(-1, 6).long() for l in range(L)]
    gate = [ck.get(f"layers.{l}.ffn.gate.weight").float() for l in range(L)]        # [384, dim]
    bias = [ck.get(f"layers.{l}.ffn.gate.bias").float() for l in range(L)]
    fnorm = [ck.get(f"layers.{l}.ffn_norm.weight").float() for l in range(L)]
    T = res[0].shape[0]
    print(f"T={T}; predictor = gate_(l+k)(ffn_norm_(l+k)(mean_copies(residual_l)))")
    # Sanity k=0 with the true router input is unavailable (needs the post-attention
    # collapse); k=0 here = residual AFTER layer l applied to layer l's router —
    # a bound on how much the mean-copy shortcut itself costs.
    for k in (0,) + tuple(ks):
        rec6 = rec12 = n = 0
        for l in range(L - k):
            x = res[l]
            w = fnorm[l + k]
            xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-20) * w
            s = torch.nn.functional.softplus(xn @ gate[l + k].t()).sqrt() + bias[l + k]
            p12 = s.topk(12, dim=-1).indices
            p6 = p12[:, :6]
            true = ids[l + k]
            rec6 += (p6.unsqueeze(2) == true.unsqueeze(1)).any(1).float().sum().item()
            rec12 += (p12.unsqueeze(2) == true.unsqueeze(1)).any(1).float().sum().item()
            n += true.numel()
        print(f"k={k}: recall of true top-6 within predicted top-6 {rec6 / n:.3f}, within top-12 {rec12 / n:.3f}"
              f"  (random {6 / 384:.3f} / {12 / 384:.3f})")


if __name__ == "__main__":
    main(sys.argv[1])


def prefetch_economics(dump, frac=0.66, ks=(2, 3, 5), margins=(-1, -2, -3, 0.0, 0.05)):
    """Recall on MISSES (under a warm LRU at `frac` residency) and prefetch reads/token
    when prefetching predicted-non-resident experts with a score margin above the
    predicted 6th score."""
    from collections import OrderedDict
    ck = Checkpoint(oracle.MODEL)
    L = 40
    res = [torch.load(os.path.join(dump, f"layer_{l:02d}_residual.pt"))[0].mean(dim=1).float() for l in range(L)]
    ids = [torch.load(os.path.join(dump, f"layer_{l:02d}_topk_ids.pt")).reshape(-1, 6).long() for l in range(L)]
    gate = [ck.get(f"layers.{l}.ffn.gate.weight").float() for l in range(L)]
    bias = [ck.get(f"layers.{l}.ffn.gate.bias").float() for l in range(L)]
    fnorm = [ck.get(f"layers.{l}.ffn_norm.weight").float() for l in range(L)]
    T = res[0].shape[0]
    # predicted scores for every (l, k)
    scores = {}
    for k in ks:
        for l in range(L - k):
            x = res[l]
            xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-20) * fnorm[l + k]
            scores[(l, k)] = torch.nn.functional.softplus(xn @ gate[l + k].t()).sqrt() + bias[l + k]
    slots = int(frac * L * 384)
    lru = OrderedDict()
    stats = {(k, m): [0, 0, 0] for k in ks for m in margins}  # [misses predicted, prefetch reads, misses total]
    misses_total = 0
    for t in range(T):
        for l in range(L):
            resident_before = set(lru.keys())
            for e in ids[l][t].tolist():
                key = (l, e)
                miss = key not in lru
                misses_total += miss and t >= T // 2
                if key in lru:
                    lru.move_to_end(key)
                else:
                    lru[key] = None
                    if len(lru) > slots:
                        lru.popitem(last=False)
                if t < T // 2 or not miss:
                    continue
                # was this miss predicted k layers earlier?
                for k in ks:
                    if l - k < 0:
                        continue
                    s = scores[(l - k, k)][t]
                    top = s.topk(12).indices.tolist()
                    thr6 = s.topk(6).values[-1].item()
                    for m in margins:
                        pred = top[:int(-m)] if m < 0 else ([i for i in top if s[i].item() >= thr6 - m] if m > 0 else top[:6])
                        stats[(k, m)][0] += e in pred
                        stats[(k, m)][2] += 1
            # prefetch reads this layer would issue for layer l (predicted from l-k) that are non-resident
            if t >= T // 2:
                for k in ks:
                    if l - k < 0:
                        continue
                    s = scores[(l - k, k)][t]
                    top = s.topk(12).indices.tolist()
                    thr6 = s.topk(6).values[-1].item()
                    for m in margins:
                        pred = top[:int(-m)] if m < 0 else ([i for i in top if s[i].item() >= thr6 - m] if m > 0 else top[:6])
                        stats[(k, m)][1] += sum((l, e) not in resident_before for e in pred)
    warm = T - T // 2
    print(f"\nprefetch economics at {frac:.0%} residency (warm half, {warm} tokens; {misses_total / warm:.2f} misses/token):")
    print("  k  rule    recall-on-misses  prefetch reads/token  MB/token   (rule<0: top-|rule| ranks only; 0: top-6; >0: margin below 6th)")
    for k in ks:
        for m in margins:
            hit, reads, tot = stats[(k, m)]
            print(f"  {k}   {m:.2f}      {hit / max(tot, 1):.3f}            {reads / warm:6.1f}          {reads / warm * 18.8:7.0f}")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "--prefetch":
    prefetch_economics(sys.argv[1])
