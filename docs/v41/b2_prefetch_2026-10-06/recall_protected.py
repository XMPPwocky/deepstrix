#!/usr/bin/env python3
"""Job A (torch via nix-shell, 1 thread): recall of layer (l+k)'s TRUE rank-1 / rank-2
picks that are BOX-2-OWNED (and likely non-resident) within the look-ahead top-N
computed at layer l, on the 1,006-token agentic dump used by the 2026-09-14 study
(scripts/v41_oracle/route_probe.py: same predictor = layer (l+k)'s own gate on the
mean-over-hc-copies residual after layer l; k=0 is the shortcut's own bound).

Ownership = per-layer top-103 by decode pick count (hot_2337.json). Residency weight
p_miss(L, e) = box-2 LRU miss probability per (layer, expert) from trace_lru.py;
an expert never picked in the trace prefix counts as cold (p_miss = 1).

usage: recall_protected.py <dump_dir> <hot_json> <lru_json> <out_json>
"""
import json, os, sys, time, resource
import torch
torch.set_num_threads(1)
sys.path.insert(0, '/home/claude-code/deepstrix/.claude/worktrees/b2-prefetch/scripts/v41_oracle')
from loader import Checkpoint  # noqa: E402

dump, hot_json, lru_json, out = sys.argv[1:5]
MODEL = os.path.expanduser('~/.cache/deepstrix/models/dsv4.1f')
L, E, K = 40, 384, 103
hot = json.load(open(hot_json)); box1 = [set(hot['rank'][l][:K]) for l in range(L)]; del hot
lru = json.load(open(lru_json))
pm = [[(lru['pe_miss'][l][e] / lru['pe_pick'][l][e]) if lru['pe_pick'][l][e] else 1.0 for e in range(E)] for l in range(L)]
del lru
ck = Checkpoint(MODEL)
gcache = {}
def gate(l):
    if l not in gcache:
        gcache[l] = (ck.get(f"layers.{l}.ffn.gate.weight").float(), ck.get(f"layers.{l}.ffn.gate.bias").float(), ck.get(f"layers.{l}.ffn_norm.weight").float())
    return gcache[l]
def residual(l):
    return torch.load(os.path.join(dump, f"layer_{l:02d}_residual.pt"))[0].mean(dim=1).float()
def ids(l):
    return torch.load(os.path.join(dump, f"layer_{l:02d}_topk_ids.pt")).reshape(-1, 6).long()

NS = (1, 2, 3, 6, 8, 12)
KS = (0, 1, 2)
def bucket(Lt):
    return 'enc' if Lt < 20 else 'dec'
# acc[k][N][bucket][metric] -> [num, den]
acc = {k: {N: {b: {} for b in ('enc', 'dec', 'all')} for N in NS} for k in KS}
def add(k, N, Lt, key, num, den):
    for b in (bucket(Lt), 'all'):
        a = acc[k][N][b].setdefault(key, [0.0, 0.0]); a[0] += num; a[1] += den
t0 = time.time()
res_cache = {}
for l in range(L):
    x = residual(l)
    for k in KS:
        Lt = l + k
        if Lt >= L: continue
        g, bias, fn = gate(Lt)
        xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-20) * fn
        s = torch.nn.functional.softplus(xn @ g.t()).sqrt() + bias
        p12 = s.topk(12, dim=-1).indices.tolist()
        true = ids(Lt).tolist()
        T = len(true)
        for t in range(T):
            tr = true[t]; tr6 = set(tr)
            for N in NS:
                pred = p12[t][:N]; ps = set(pred)
                # recall of true picks by rank, box-2-owned only, weighted by p_miss and unweighted
                for r, e in enumerate(tr):
                    if e in box1[Lt]: continue
                    w = pm[Lt][e]; hit = 1.0 if e in ps else 0.0
                    tag = 'r%d' % (r + 1)
                    add(k, N, Lt, 'recall_w_' + tag, w * hit, w); add(k, N, Lt, 'recall_u_' + tag, hit, 1.0)
                    if r < 2: add(k, N, Lt, 'recall_w_r12', w * hit, w); add(k, N, Lt, 'recall_u_r12', hit, 1.0)
                    if r < 3: add(k, N, Lt, 'recall_w_r123', w * hit, w); add(k, N, Lt, 'recall_u_r123', hit, 1.0)
                    add(k, N, Lt, 'recall_w_all', w * hit, w); add(k, N, Lt, 'recall_u_all', hit, 1.0)
                    # cold-only (p_miss >= 0.5): the 09-14 "recall on misses" population
                    if w >= 0.5:
                        add(k, N, Lt, 'recall_cold_' + tag, hit, 1.0)
                        if r < 2: add(k, N, Lt, 'recall_cold_r12', hit, 1.0)
                        add(k, N, Lt, 'recall_cold_all', hit, 1.0)
                # hints: predicted top-N, box-2-owned, weighted by p_miss (= expected non-resident)
                hw = dw = p1 = p2 = 0.0; hc = dc = 0
                for e in pred:
                    if e in box1[Lt]: continue
                    w = pm[Lt][e]; hw += w
                    cold = w >= 0.5; hc += cold
                    if e in tr6:
                        dw += w; dc += cold
                        if e == tr[0]: p1 += w
                        if e in tr[:2]: p2 += w
                add(k, N, Lt, 'hints_per_row', hw, 1.0); add(k, N, Lt, 'hints_cold_per_row', hc, 1.0)
                add(k, N, Lt, 'prec_any', dw, hw); add(k, N, Lt, 'prec_cold_any', dc, hc)
                add(k, N, Lt, 'prec_prot1', p1, hw); add(k, N, Lt, 'prec_prot2', p2, hw)
    del x
    if l % 10 == 9:
        print(f"layer {l} done {time.time() - t0:.0f}s rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB", flush=True)

def val(k, N, b, key):
    a = acc[k][N][b].get(key); return (a[0] / a[1]) if a and a[1] else float('nan')
print(f"T={T} tokens; ownership top-{K}/layer; weights = LRU p_miss (cap {4000})")
for k in KS:
    print(f"\n== k={k} (layer l+{k}'s gate on the residual after layer l)")
    print("bucket N | recall_w r1 r2 r1-2 r1-3 all | recall_u r1 r2 r1-2 all | cold r1 r1-2 all | hints/row (w, cold) prec_any prec_prot1 prec_prot2")
    for b in ('enc', 'dec', 'all'):
        for N in NS:
            v = lambda key: val(k, N, b, key)
            print(f"{b:3s} {N:2d} | {v('recall_w_r1'):.3f} {v('recall_w_r2'):.3f} {v('recall_w_r12'):.3f} {v('recall_w_r123'):.3f} {v('recall_w_all'):.3f} | "
                  f"{v('recall_u_r1'):.3f} {v('recall_u_r2'):.3f} {v('recall_u_r12'):.3f} {v('recall_u_all'):.3f} | "
                  f"{v('recall_cold_r1'):.3f} {v('recall_cold_r12'):.3f} {v('recall_cold_all'):.3f} | "
                  f"{v('hints_per_row'):.3f} {v('hints_cold_per_row'):.3f} {v('prec_any'):.3f} {v('prec_prot1'):.3f} {v('prec_prot2'):.3f}")
# weighted population sizes
for k in KS:
    a = acc[k][6]['all'].get('recall_w_r1'); u = acc[k][6]['all'].get('recall_u_r1'); c = acc[k][6]['all'].get('recall_cold_r1')
    print(f"k={k}: box-2-owned rank-1 picks {u[1]:.0f} of {T * (L - k)} rows; expected non-resident (sum p_miss) {a[1]:.1f}; cold (p>=0.5) {c[1]:.0f}")
json.dump({k: {N: {b: {key: v for key, v in acc[k][N][b].items()} for b in acc[k][N]} for N in acc[k]} for k in acc}, open(out, 'w'))
print(f"done {time.time() - t0:.0f}s rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB")
