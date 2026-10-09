#!/usr/bin/env python3
"""Trained head vs margin heuristic for predicting the PROTECTED box-2 pick (true rank-1 at
layer l+1, box-2-owned, weighted by p_miss) from layer l's residual. CPU torch, 4 threads.

Rows = (l, t), l in 0..38 (target layer Lt = l+1), t over the 1,006 dump tokens.
Candidates per row = box-2-owned experts at Lt with a method score; volume = sum of p_miss over
emitted candidates (expected non-resident hint words), per 80 row-layers (= one 2-row step).
Precision / recall are p_miss-weighted against the protected set (true rank-1 at Lt, box-2-owned).
Held-out: 5 contiguous token blocks (fold = block), models trained on the other 4.

usage: study.py <data_dir> <hot_json> <lru_json> <freq_json> <out_json> [own=top103|ge272]
"""
import json, math, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4); torch.manual_seed(0)
data_dir, hot_json, lru_json, freq_json, out_path = sys.argv[1:6]
OWN = sys.argv[6] if len(sys.argv) > 6 else 'top103'
L, E, K1 = 40, 384, 103
D = torch.load(os.path.join(data_dir, 'data.pt'))
X, S_RAW, S_SEL, IDS, BIAS, T = D['X'], D['S_RAW'], D['S_SEL'], D['IDS'], D['BIAS'], D['T']
NL = L - 1
# ---------- ownership, residency prior, trace frequency
hot = json.load(open(hot_json))
box2 = torch.ones(L, E, dtype=torch.bool)
if OWN == 'top103':
    for l in range(L):
        box2[l, torch.tensor(hot['rank'][l][:K1])] = False
else:
    box2[:, :272] = False
lru = json.load(open(lru_json))
PM = torch.tensor([[(lru['pe_miss'][l][e] / lru['pe_pick'][l][e]) if lru['pe_pick'][l][e] else 1.0 for e in range(E)] for l in range(L)])
fq = json.load(open(freq_json))
R1 = torch.tensor(fq['r1'], dtype=torch.float); ANY = torch.tensor(fq['any'], dtype=torch.float)
F_R1 = (R1 + 0.5) / (R1.sum(1, keepdim=True) + 0.5 * E)          # P(e is rank-1 at layer) from the live trace
F_ANY = (ANY + 0.5) / (ANY.sum(1, keepdim=True) / 6 + 0.5 * E)   # P(e in top-6)
del lru, hot, fq
# ---------- row-level quantities
LT = torch.arange(NL).unsqueeze(1).expand(NL, T) + 1              # target layer per row [NL, T]
R1_TRUE = IDS[:, :, 0].long()                                      # true rank-1 at Lt
SEL = S_SEL.float(); RAW = S_RAW.float()
top6 = SEL.topk(6, dim=-1).indices                                 # predicted top-6 by selection score
raw6 = RAW.gather(-1, top6); sum6 = raw6.sum(-1)
MARGIN = (raw6[..., 0] - raw6[..., 1]) / sum6                      # the design's margin for the predicted rank-1
PRED1 = top6[..., 0]
prot_w = (PM[LT, R1_TRUE] * box2[LT, R1_TRUE]).float()             # protected mass per row
prot_u = box2[LT, R1_TRUE].float()
ROWS_PER_STEP = 80
VOLS = (1, 2, 3, 5, 8)
blocks = [list(range(i * T // 5, (i + 1) * T // 5)) for i in range(5)]
def bucket_of(Lt):
    return 0 if Lt <= 13 else (1 if Lt <= 26 else 2)
BUCKET = torch.tensor([bucket_of(l + 1) for l in range(NL)]).unsqueeze(1).expand(NL, T)

# ---------- evaluation
def evaluate(cand_e, cand_s, tok_idx, tag):
    """cand_e [NL, Tt, K] expert ids, cand_s [NL, Tt, K] scores (-inf = not emitted); tok_idx = held-out tokens."""
    Lt = LT[:, tok_idx]; r1 = R1_TRUE[:, tok_idx]; nrows = Lt.numel()
    own = box2[Lt.unsqueeze(-1), cand_e]
    s = torch.where(own, cand_s, torch.full_like(cand_s, -math.inf))
    w = PM[Lt.unsqueeze(-1), cand_e]; hit = (cand_e == r1.unsqueeze(-1)).float()
    bk = BUCKET[:, tok_idx].unsqueeze(-1).expand_as(cand_e)
    valid = torch.isfinite(s)
    s, w, hit, bk = s[valid], w[valid], hit[valid], bk[valid]
    order = s.argsort(descending=True); s, w, hit, bk = s[order], w[order], hit[order], bk[order]
    cw = w.cumsum(0); cwh = (w * hit).cumsum(0); cu = torch.arange(1, len(s) + 1).float(); cuh = hit.cumsum(0)
    P = prot_w[:, tok_idx].sum().item(); PU = prot_u[:, tok_idx].sum().item()
    res = {'tag': tag, 'prot_per_step': P / nrows * ROWS_PER_STEP, 'prot_u_per_step': PU / nrows * ROWS_PER_STEP,
           'n_cand': int(len(s)), 'at_vol': {}, 'at_thr': {}}
    for V in VOLS:
        target = V / ROWS_PER_STEP * nrows
        i = int(torch.searchsorted(cw, torch.tensor(target)).item())
        if i >= len(s): i = len(s) - 1
        if i < 0 or len(s) == 0: res['at_vol'][V] = None; continue
        thr = s[i].item()
        m = s >= thr
        ww, wh = w[m].sum().item(), (w[m] * hit[m]).sum().item(); uu, uh = m.sum().item(), hit[m].sum().item()
        per_b = {}
        for b in range(3):
            mb = m & (bk == b)
            per_b[b] = {'vol': w[mb].sum().item() / nrows * ROWS_PER_STEP, 'prec': (w[mb] * hit[mb]).sum().item() / max(1e-9, w[mb].sum().item()),
                        'recall': (w[mb] * hit[mb]).sum().item() / max(1e-9, prot_w[:, tok_idx][BUCKET[:, tok_idx] == b].sum().item())}
        res['at_vol'][V] = {'thr': thr, 'vol': ww / nrows * ROWS_PER_STEP, 'prec': wh / max(1e-9, ww), 'recall': wh / max(1e-9, P),
                            'vol_u': uu / nrows * ROWS_PER_STEP, 'prec_u': uh / max(1, uu), 'recall_u': uh / max(1e-9, PU), 'bucket': per_b}
    return res

def eval_thresholds(cand_e, cand_s, tok_idx, thrs):
    """fixed-threshold points (the design's margin buckets)."""
    Lt = LT[:, tok_idx]; r1 = R1_TRUE[:, tok_idx]; nrows = Lt.numel()
    own = box2[Lt.unsqueeze(-1), cand_e]
    w = PM[Lt.unsqueeze(-1), cand_e]; hit = (cand_e == r1.unsqueeze(-1)).float()
    P = prot_w[:, tok_idx].sum().item(); PU = prot_u[:, tok_idx].sum().item()
    out = {}
    for th in thrs:
        m = own & (cand_s >= th)
        ww, wh = w[m].sum().item(), (w[m] * hit[m]).sum().item(); uu, uh = m.sum().item(), hit[m].sum().item()
        out[th] = {'vol': ww / nrows * ROWS_PER_STEP, 'prec': wh / max(1e-9, ww), 'recall': wh / max(1e-9, P),
                   'vol_u': uu / nrows * ROWS_PER_STEP, 'prec_u': uh / max(1, uu), 'recall_u': uh / max(1e-9, PU)}
    return out

# ---------- zero-cost baselines (no training; evaluated on every fold's held-out block and pooled)
def base_margin(tok):
    return PRED1[:, tok].unsqueeze(-1), MARGIN[:, tok].unsqueeze(-1)
def base_weight3(tok):  # predicted routing weight over the predicted top-3
    e = top6[:, tok, :3]; return e, (raw6[:, tok, :3] / sum6[:, tok].unsqueeze(-1))
def base_freq(tok):
    e = PRED1[:, tok]; return e.unsqueeze(-1), F_R1[LT[:, tok], e].unsqueeze(-1)
def base_margin_x_freq(tok):
    e = PRED1[:, tok]; return e.unsqueeze(-1), (MARGIN[:, tok].clamp(min=0) * F_R1[LT[:, tok], e]).unsqueeze(-1)
def base_margin_freqgate(p):
    def f(tok):
        e = PRED1[:, tok]; m = MARGIN[:, tok].clone(); m[F_R1[LT[:, tok], e] <= p] = -math.inf
        return e.unsqueeze(-1), m.unsqueeze(-1)
    return f

# ---------- learned heads
KC = 4  # candidates per row from the predicted top-4 (LR) / head top-4
def lr_features(l_idx, tok):
    """per-candidate features over the predicted top-4 by selection score; returns feats [NL, Tt, 4, F], cand e."""
    sel = SEL[l_idx][:, tok]; raw = RAW[l_idx][:, tok]
    e4 = sel.topk(KC, dim=-1).indices; sel4 = sel.gather(-1, e4); raw4 = raw.gather(-1, e4)
    Lt = LT[l_idx][:, tok]
    s6 = sum6[l_idx][:, tok].unsqueeze(-1); mg = MARGIN[l_idx][:, tok].unsqueeze(-1)
    p8 = F.softmax(sel.topk(8, dim=-1).values * 8.0, dim=-1); ent = -(p8 * (p8 + 1e-9).log()).sum(-1, keepdim=True)
    f = [sel4 - sel4[..., :1], raw4 / s6, mg.expand_as(sel4), ent.expand_as(sel4), (sel4[..., :1] - sel4[..., 1:2]).expand_as(sel4),
         F_R1[Lt.unsqueeze(-1), e4].log(), F_ANY[Lt.unsqueeze(-1), e4].log(), BIAS[Lt.unsqueeze(-1), e4],
         (Lt.float() / L).unsqueeze(-1).expand_as(sel4)]
    f += [(torch.arange(KC) == k).float().expand_as(sel4) for k in range(KC)]
    f += [(Lt.unsqueeze(-1) <= 13).float().expand_as(sel4), ((Lt.unsqueeze(-1) > 13) & (Lt.unsqueeze(-1) <= 26)).float().expand_as(sel4)]
    return torch.stack(f, -1), e4

def train_lr(train_tok, epochs=60, wd=1e-3):
    l_idx = torch.arange(NL)
    f, e4 = lr_features(l_idx, train_tok); y = (e4 == R1_TRUE[:, train_tok].unsqueeze(-1)).float()
    f = f.reshape(-1, f.shape[-1]); y = y.reshape(-1)
    mu, sd = f.mean(0), f.std(0) + 1e-6
    w = nn.Linear(f.shape[-1], 1); opt = torch.optim.Adam(w.parameters(), lr=1e-2, weight_decay=wd)
    fn = (f - mu) / sd
    for ep in range(epochs):
        for i in torch.randperm(len(fn)).split(4096):
            opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(w(fn[i]).squeeze(-1), y[i]); loss.backward(); opt.step()
    def predict(tok):
        ff, ee = lr_features(l_idx, tok)
        with torch.no_grad(): s = w(((ff - mu) / sd)).squeeze(-1)
        return ee, s
    return predict

class LowRankHead(nn.Module):
    """logits = sel + gain_l * W_out(h), h = act(W_in x_rms + emb_l) [+ beta * log F_R1]. Shared across layers."""
    def __init__(self, r=64, hidden=0, use_freq=True, per_layer_lora=0, calib_bias=False):
        super().__init__()
        self.hidden = hidden; self.use_freq = use_freq; self.lora = per_layer_lora; self.r = r
        if r:
            self.w_in = nn.Linear(5120, r, bias=False); self.emb = nn.Embedding(NL, r)
            self.w_h = nn.Linear(r, hidden) if hidden else None
            self.w_out = nn.Linear(hidden or r, E); nn.init.zeros_(self.w_out.weight); nn.init.zeros_(self.w_out.bias)
        self.cb = nn.Parameter(torch.zeros(NL, E)) if calib_bias else None   # per-(layer, expert) correction of the look-ahead gate
        self.beta = nn.Parameter(torch.zeros(1)); self.tau = nn.Parameter(torch.ones(1))
        if per_layer_lora:
            self.A = nn.Parameter(torch.randn(NL, 5120, per_layer_lora) * 0.01); self.B = nn.Parameter(torch.zeros(NL, per_layer_lora, E))
    def forward(self, x, l, sel):
        xr = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        logits = sel * self.tau
        if self.r:
            h = self.w_in(xr) + self.emb(l)
            if self.w_h is not None: h = F.gelu(self.w_h(F.gelu(h)))
            logits = logits + self.w_out(h)
        if self.cb is not None: logits = logits + self.cb[l]
        if self.lora:
            logits = logits + torch.bmm(torch.bmm(xr.unsqueeze(1), self.A[l]), self.B[l]).squeeze(1)
        if self.use_freq: logits = logits + self.beta * F_R1[l + 1].log()
        return logits

def ce_on(m, tok):
    li = torch.arange(NL).unsqueeze(1).expand(NL, len(tok)).reshape(-1); ti = torch.tensor(tok).unsqueeze(0).expand(NL, -1).reshape(-1)
    tot = 0.0
    with torch.no_grad():
        for i in torch.arange(len(li)).split(2048):
            tot += F.cross_entropy(m(X[li[i], ti[i]].float(), li[i], SEL[li[i], ti[i]]), R1_TRUE[li[i], ti[i]], reduction='sum').item()
    return tot / len(li)

def train_head(train_tok, epochs=15, lr=1e-3, wd=1e-2, bs=512, early_stop=False, **kw):
    """early_stop: the last 25% of train_tok is a validation block; keep the epoch with the lowest validation CE."""
    m = LowRankHead(**kw); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    if early_stop:
        nv = len(train_tok) // 4; val_tok = train_tok[-nv:]; train_tok = train_tok[:-nv]
    li = torch.arange(NL).unsqueeze(1).expand(NL, len(train_tok)).reshape(-1)
    ti = torch.tensor(train_tok).unsqueeze(0).expand(NL, -1).reshape(-1)
    n = len(li); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs * ((n + bs - 1) // bs)))
    best = (ce_on(m, val_tok) if early_stop else None, {k: v.clone() for k, v in m.state_dict().items()}, -1); tot = float('nan')
    for ep in range(epochs):
        tot = 0.0
        for i in torch.randperm(n).split(bs):
            l, t = li[i], ti[i]
            x = X[l, t].float(); sel = SEL[l, t]; y = R1_TRUE[l, t]
            opt.zero_grad(); loss = F.cross_entropy(m(x, l, sel), y); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(i)
        if early_stop:
            v = ce_on(m, val_tok)
            if v < best[0]: best = (v, {k: v_.clone() for k, v_ in m.state_dict().items()}, ep)
    if early_stop:
        m.load_state_dict(best[1]); m.best_epoch = best[2]; m.val_ce = best[0]
    m.eval()
    def predict(tok):
        li2 = torch.arange(NL).unsqueeze(1).expand(NL, len(tok)).reshape(-1); ti2 = torch.tensor(tok).unsqueeze(0).expand(NL, -1).reshape(-1)
        out_e, out_s = [], []
        with torch.no_grad():
            for i in torch.arange(len(li2)).split(2048):
                lg = m(X[li2[i], ti2[i]].float(), li2[i], SEL[li2[i], ti2[i]])
                # margin-equivalent score in RAW-weight units (the design's (w1-w2)/sum6 on the head's logits):
                # raw_eq = lg/tau - bias; rank-1 candidate: (r1-r2)/sum6, rank-k: (rk-r1)/sum6 (negative).
                # With no training (tau=1, delta=0) this is exactly the margin heuristic.
                raw_eq = lg / m.tau - BIAS[li2[i] + 1]
                t6 = lg.topk(6, dim=-1).indices; r6 = raw_eq.gather(-1, t6); s6 = r6.sum(-1, keepdim=True).clamp(min=1e-3)
                e = t6[:, :KC]; rk = r6[:, :KC]
                v = torch.cat([((rk[:, :1] - rk[:, 1:2]) / s6), (rk[:, 1:] - rk[:, :1]) / s6], 1)
                out_e.append(e); out_s.append(v)
        return torch.cat(out_e).reshape(NL, len(tok), KC), torch.cat(out_s).reshape(NL, len(tok), KC), m
    return predict, tot / n

def nparams(m): return sum(p.numel() for p in m.parameters())

# ---------- run
t0 = time.time(); results = {'own': OWN, 'T': T, 'folds': {}, 'pooled': {}, 'learning_curve': {}, 'ceiling': {}}
def pooled_eval(fn, tag, thrs=None):
    """evaluate a method on every fold's held-out block; pool the candidate lists across folds."""
    ce, cs = [], []
    for k in range(5):
        e, s = fn(k, blocks[k]); ce.append(e); cs.append(s)
    e = torch.cat(ce, 1); s = torch.cat(cs, 1); tok = sum(blocks, [])
    r = evaluate(e, s, tok, tag)
    if thrs is not None: r['at_thr'] = eval_thresholds(e, s, tok, thrs)
    results['pooled'][tag] = r
    print(f"[{tag}] prot/step {r['prot_per_step']:.2f} (u {r['prot_u_per_step']:.1f}) " +
          ' | '.join(f"V{V}: p={a['prec']:.3f} r={a['recall']:.3f} (pu={a['prec_u']:.3f})" for V, a in r['at_vol'].items() if a), flush=True)
    return r

# ceiling: protected recall within predicted top-N by selection score (all tokens), weighted
tok_all = list(range(T))
for N in (1, 2, 3, 4, 6, 8, 12, 24):
    eN = SEL.topk(N, dim=-1).indices; hit = (eN == R1_TRUE.unsqueeze(-1)).any(-1).float()
    results['ceiling'][N] = {'recall_w': (hit * prot_w).sum().item() / prot_w.sum().item(), 'recall_u': (hit * prot_u).sum().item() / prot_u.sum().item(),
                             'bucket': {b: (hit * prot_w)[BUCKET == b].sum().item() / prot_w[BUCKET == b].sum().item() for b in range(3)}}
print('ceiling (protected recall within predicted top-N):', {N: round(v['recall_w'], 3) for N, v in results['ceiling'].items()})
print('protected per step: weighted %.2f, unweighted %.1f; predicted-rank-1 box-2 words/step (weighted) %.2f' % (
    prot_w.mean().item() * 80, prot_u.mean().item() * 80, (PM[LT, PRED1] * box2[LT, PRED1]).float().mean().item() * 80))

pooled_eval(lambda k, tok: base_margin(tok), 'margin', thrs=(0.0, 0.1, 0.2, 0.3))
r = results['pooled']['margin']['at_thr']
print('  margin buckets:', {th: f"vol={a['vol']:.2f} p={a['prec']:.3f} r={a['recall']:.3f} pu={a['prec_u']:.3f}" for th, a in r.items()})
pooled_eval(lambda k, tok: base_weight3(tok), 'pred_weight_top3')
pooled_eval(lambda k, tok: base_freq(tok), 'freq_r1')
pooled_eval(lambda k, tok: base_margin_x_freq(tok), 'margin_x_freq')
for p in (1e-4, 3e-4, 1e-3):
    pooled_eval(lambda k, tok, p=p: base_margin_freqgate(p)(tok), f'margin|freq>{p}')

# learned, 5-fold
def fold_train_tok(k): return sum((blocks[j] for j in range(5) if j != k), [])
lr_models = {k: train_lr(fold_train_tok(k)) for k in range(5)}
print(f'LR trained {time.time() - t0:.0f}s', flush=True)
pooled_eval(lambda k, tok: lr_models[k](tok), 'logreg_top4')

HEADS = {'init_margin_top4': dict(r=0, epochs=0, use_freq=False),             # untrained: = the margin heuristic (+ rank 2-4 candidates below it); sanity
         'calib_tau_freq': dict(r=0, epochs=8, lr=3e-3, wd=0.0),              # 2 scalars: temperature + trace-frequency prior weight
         'calib_bias': dict(r=0, epochs=8, lr=3e-3, wd=0.1, calib_bias=True, early_stop=True),  # + per-(layer, expert) bias (15K)
         'lowrank16': dict(r=16), 'lowrank64': dict(r=64),
         'lowrank64_es': dict(r=64, lr=3e-4, wd=0.1, epochs=12, early_stop=True),
         'lowrank64_nofreq': dict(r=64, use_freq=False),
         'mlp64x256': dict(r=64, hidden=256), 'mlp64x256_es': dict(r=64, hidden=256, lr=3e-4, wd=0.1, epochs=12, early_stop=True),
         'lowrank64+lora4': dict(r=64, per_layer_lora=4)}
if os.environ.get('SKIP_HEADS'): HEADS = {}
for name, kw in HEADS.items():
    preds = {}
    for k in range(5):
        preds[k], tl = train_head(fold_train_tok(k), **kw)
    _, _, m = preds[0]([0])
    es = f" best_epoch {m.best_epoch} val_ce {m.val_ce:.3f}" if hasattr(m, 'best_epoch') else ''
    print(f'{name}: params {nparams(m)} train loss {tl:.3f} tau {m.tau.item():.3f} beta {m.beta.item():.3f}{es} {time.time() - t0:.0f}s', flush=True)
    r = pooled_eval(lambda k, tok: preds[k](tok)[:2], name)
    r['params'] = nparams(m)
    # head's own ceiling: protected recall within its top-4 and top-1
    e = torch.cat([preds[k](blocks[k])[0] for k in range(5)], 1); r1 = R1_TRUE
    for N in (1, 4):
        hit = (e[..., :N] == r1.unsqueeze(-1)).any(-1).float()
        r[f'top{N}_recall_w'] = (hit * prot_w).sum().item() / prot_w.sum().item()

# learning curve: held-out = last block; train on prefixes of the first 80%
train80 = sum(blocks[:4], []); test = blocks[4]
for frac in (0.1, 0.25, 0.5, 1.0):
    n = int(len(train80) * frac); tt = train80[:n]
    row = {'tokens': n}
    pr = train_lr(tt); row['logreg'] = evaluate(*pr(test), test, 'lc')['at_vol']
    for name in ('calib_bias', 'lowrank64', 'lowrank64_es', 'mlp64x256'):
        if name not in HEADS: continue
        pr, _ = train_head(tt, **HEADS[name]); row[name] = evaluate(*pr(test)[:2], test, 'lc')['at_vol']
    row['margin'] = evaluate(*base_margin(test), test, 'lc')['at_vol']
    results['learning_curve'][frac] = row
    print(f"LC {n} tokens: " + ' | '.join(f"{nm}: V3 p={row[nm][3]['prec']:.3f} r={row[nm][3]['recall']:.3f} V5 p={row[nm][5]['prec']:.3f} r={row[nm][5]['recall']:.3f}" for nm in ('margin', 'logreg', 'calib_bias', 'lowrank64', 'lowrank64_es', 'mlp64x256') if nm in row and row[nm].get(3) and row[nm].get(5)), flush=True)
json.dump(results, open(out_path, 'w'), indent=1, default=str)
print(f'done {time.time() - t0:.0f}s')
