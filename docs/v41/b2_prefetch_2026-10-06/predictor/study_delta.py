#!/usr/bin/env python3
"""Router + delta heads, regularized toward the router ("regularize towards the router").
Every head equals the untrained one-layer-early gate at zero delta, so the margin heuristic is reproduced
exactly at lambda -> inf. Same data / folds / ownership / matched-volume protocol as study.py.

 (a) affine:   z = a_l * sel + b_{l,e};  raw' = raw + b/a            (385 params/layer; L2 on b, (a-1))
 (b) delta:    logit' = logit0 + xn A_l B_l; raw' = sqrt(softplus(logit')); z = raw' + bias   (L2 on ||A_l B_l||_F^2)
 (c) (b) + per-layer input scale g_l (5120, init 1):  logit' += (xr * fn * (g-1)) @ W^T      (L2 on ||g-1||^2)
 (d) the 15-feature logistic regression stacked on (a) / (b)
Objectives: 'full' = CE over 384 experts on the true rank-1 at l+1; 'box2' = CE over box-2-owned experts on rows
whose true rank-1 is box-2-owned (the protected-rank-1 objective).
Scoring for matched volume: the design's margin on the head's raw-equivalent weights (top-6 by z):
rank-1 candidate (r1-r2)/sum6, rank-k (rk-r1)/sum6.
usage: study_delta.py <data_dir> <hot_json> <lru_json> <freq_json> <out_json> [own=top103|ge272]
"""
import json, math, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4); torch.manual_seed(0)
data_dir, hot_json, lru_json, freq_json, out_path = sys.argv[1:6]
OWN = sys.argv[6] if len(sys.argv) > 6 else 'top103'
sys.path.insert(0, '/home/claude-code/deepstrix/.claude/worktrees/b2-prefetch/scripts/v41_oracle')
from loader import Checkpoint  # noqa: E402
L, E, K1, NL, KC = 40, 384, 103, 39, 4
D = torch.load(os.path.join(data_dir, 'data.pt')); D2 = torch.load(os.path.join(data_dir, 'data2.pt'))
S_RAW, S_SEL, IDS, BIAS, T = D['S_RAW'].float(), D['S_SEL'].float(), D['IDS'], D['BIAS'], D['T']
XN, LOGIT0, FN = D2['XN'], D2['LOGIT0'], D2['FN']; del D, D2
ck = Checkpoint(os.path.expanduser('~/.cache/deepstrix/models/dsv4.1f'))
W = torch.stack([ck.get(f"layers.{l}.ffn.gate.weight").float() for l in range(L)])  # [40, 384, 5120]
hot = json.load(open(hot_json)); box2 = torch.ones(L, E, dtype=torch.bool)
if OWN == 'top103':
    for l in range(L): box2[l, torch.tensor(hot['rank'][l][:K1])] = False
else: box2[:, :272] = False
lru = json.load(open(lru_json))
PM = torch.tensor([[(lru['pe_miss'][l][e] / lru['pe_pick'][l][e]) if lru['pe_pick'][l][e] else 1.0 for e in range(E)] for l in range(L)])
fq = json.load(open(freq_json)); R1c = torch.tensor(fq['r1'], dtype=torch.float); ANYc = torch.tensor(fq['any'], dtype=torch.float)
F_R1 = (R1c + 0.5) / (R1c.sum(1, keepdim=True) + 0.5 * E); F_ANY = (ANYc + 0.5) / (ANYc.sum(1, keepdim=True) / 6 + 0.5 * E)
del hot, lru, fq
LT = torch.arange(NL).unsqueeze(1).expand(NL, T) + 1; R1_TRUE = IDS[:, :, 0].long()
prot_w = (PM[LT, R1_TRUE] * box2[LT, R1_TRUE]).float(); prot_u = box2[LT, R1_TRUE].float()
ROWS_PER_STEP = 80; VOLS = (1, 2, 3, 5, 8)
blocks = [list(range(i * T // 5, (i + 1) * T // 5)) for i in range(5)]
BUCKET = torch.tensor([0 if l + 1 <= 13 else (1 if l + 1 <= 26 else 2) for l in range(NL)]).unsqueeze(1).expand(NL, T)

# ---------- evaluation (as study.py)
def evaluate(cand_e, cand_s, tok_idx):
    Lt = LT[:, tok_idx]; r1 = R1_TRUE[:, tok_idx]; nrows = Lt.numel()
    own = box2[Lt.unsqueeze(-1), cand_e]; s = torch.where(own, cand_s, torch.full_like(cand_s, -math.inf))
    w = PM[Lt.unsqueeze(-1), cand_e]; hit = (cand_e == r1.unsqueeze(-1)).float(); bk = BUCKET[:, tok_idx].unsqueeze(-1).expand_as(cand_e)
    valid = torch.isfinite(s); s, w, hit, bk = s[valid], w[valid], hit[valid], bk[valid]
    order = s.argsort(descending=True); s, w, hit, bk = s[order], w[order], hit[order], bk[order]
    cw = w.cumsum(0); P = prot_w[:, tok_idx].sum().item(); PU = prot_u[:, tok_idx].sum().item(); res = {}
    for V in VOLS:
        i = min(int(torch.searchsorted(cw, torch.tensor(V / ROWS_PER_STEP * nrows)).item()), len(s) - 1)
        m = s >= s[i]; ww, wh = w[m].sum().item(), (w[m] * hit[m]).sum().item(); uu, uh = m.sum().item(), hit[m].sum().item()
        res[V] = {'thr': s[i].item(), 'vol': ww / nrows * ROWS_PER_STEP, 'prec': wh / max(1e-9, ww), 'recall': wh / max(1e-9, P), 'prec_u': uh / max(1, uu), 'recall_u': uh / max(1e-9, PU),
                  'bucket': {b: {'vol': w[m & (bk == b)].sum().item() / nrows * ROWS_PER_STEP, 'prec': (w * hit)[m & (bk == b)].sum().item() / max(1e-9, w[m & (bk == b)].sum().item()),
                                 'recall': (w * hit)[m & (bk == b)].sum().item() / max(1e-9, prot_w[:, tok_idx][BUCKET[:, tok_idx] == b].sum().item())} for b in range(3)}}
    return res
def fmt(r): return ' | '.join(f"V{V}: p={a['prec']:.3f} r={a['recall']:.3f}" for V, a in r.items())
def margin_scores(z, raw):
    """z, raw: [N, E] -> candidates [N, 4] and margin-equivalent scores."""
    t6 = z.topk(6, dim=-1).indices; r6 = raw.gather(-1, t6); s6 = r6.sum(-1, keepdim=True).clamp(min=1e-3)
    return t6[:, :KC], torch.cat([(r6[:, :1] - r6[:, 1:2]) / s6, (r6[:, 1:KC] - r6[:, :1]) / s6], 1)

# ---------- heads (per-layer parameter tensors, one Adam per layer)
class Head:
    def __init__(self, kind, r=16, lam=0.0, obj='full', lr=None):
        self.kind, self.r, self.lam, self.obj = kind, r, lam, obj
        self.params = []
        for l in range(NL):
            p = {}
            if kind == 'affine':
                p['a'] = torch.ones(1, requires_grad=True); p['b'] = torch.zeros(E, requires_grad=True)
            else:
                p['A'] = (torch.randn(5120, r) * 0.02).requires_grad_(); p['B'] = torch.zeros(r, E, requires_grad=True)
                if kind == 'delta_g': p['g'] = torch.ones(5120, requires_grad=True)
            self.params.append(p)
        self.lr = lr or (1e-2 if kind == 'affine' else 1e-3)
    def forward(self, l, idx):
        """returns z (selection-score units), raw' for tokens idx at layer l."""
        p = self.params[l]
        if self.kind == 'affine':
            raw = S_RAW[l, idx]; z = p['a'] * (raw + BIAS[l + 1]) + p['b']; rawp = raw + p['b'] / p['a']
            return z, rawp
        xn = XN[l, idx].float(); logit = LOGIT0[l, idx] + (xn @ p['A']) @ p['B']
        if self.kind == 'delta_g': logit = logit + ((xn * FN[l + 1]) * (p['g'] - 1)) @ W[l + 1].t()
        rawp = F.softplus(logit).sqrt(); return rawp + BIAS[l + 1], rawp
    def penalty(self, l):
        p = self.params[l]
        if self.kind == 'affine': return (p['b'] ** 2).sum() + (p['a'] - 1) ** 2
        pen = ((p['A'] @ p['B']) ** 2).sum()
        if self.kind == 'delta_g': pen = pen + ((p['g'] - 1) ** 2).sum()
        return pen
    def train(self, tok, epochs=12, bs=128):
        if epochs == 0: return
        opts = [torch.optim.Adam(list(p.values()), lr=self.lr) for p in self.params]
        tok = torch.tensor(tok)
        for ep in range(epochs):
            for l in torch.randperm(NL).tolist():
                for idx in tok[torch.randperm(len(tok))].split(bs):
                    y = R1_TRUE[l, idx]
                    if self.obj == 'box2':
                        keep = box2[l + 1, y]
                        if keep.sum() == 0: continue
                        idx, y = idx[keep], y[keep]
                    z, _ = self.forward(l, idx)
                    if self.obj == 'box2': z = z.masked_fill(~box2[l + 1], -1e4)
                    loss = F.cross_entropy(z, y) + self.lam * self.penalty(l)
                    opts[l].zero_grad(); loss.backward(); opts[l].step()
    def predict(self, tok):
        """candidates + margin-eq scores [NL, Tt, 4]; also returns z, raw' [NL, Tt, E] for the stacked LR."""
        tok = torch.tensor(tok); ce, cs, zs, rs = [], [], [], []
        with torch.no_grad():
            for l in range(NL):
                z, rawp = self.forward(l, tok); e, s = margin_scores(z, rawp); ce.append(e); cs.append(s); zs.append(z); rs.append(rawp)
        return torch.stack(ce), torch.stack(cs), torch.stack(zs), torch.stack(rs)
    def nparams(self): return sum(v.numel() for p in self.params for v in p.values())

# ---------- stacked logistic regression (study.py's 15 features, on (z, raw') of a head)
def lr_features(Z, RAWP, tok):
    e4 = Z.topk(KC, dim=-1).indices; sel4 = Z.gather(-1, e4); raw4 = RAWP.gather(-1, e4)
    t6 = Z.topk(6, dim=-1).indices; r6 = RAWP.gather(-1, t6); s6 = r6.sum(-1, keepdim=True); mg = (r6[..., :1] - r6[..., 1:2]) / s6
    Lt = LT[:, tok]; p8 = F.softmax(Z.topk(8, dim=-1).values * 8.0, dim=-1); ent = -(p8 * (p8 + 1e-9).log()).sum(-1, keepdim=True)
    f = [sel4 - sel4[..., :1], raw4 / s6, mg.expand_as(sel4), ent.expand_as(sel4), (sel4[..., :1] - sel4[..., 1:2]).expand_as(sel4),
         F_R1[Lt.unsqueeze(-1), e4].log(), F_ANY[Lt.unsqueeze(-1), e4].log(), BIAS[Lt.unsqueeze(-1), e4], (Lt.float() / L).unsqueeze(-1).expand_as(sel4)]
    f += [(torch.arange(KC) == k).float().expand_as(sel4) for k in range(KC)]
    f += [(Lt.unsqueeze(-1) <= 13).float().expand_as(sel4), ((Lt.unsqueeze(-1) > 13) & (Lt.unsqueeze(-1) <= 26)).float().expand_as(sel4)]
    return torch.stack(f, -1), e4
def train_lr(Z, RAWP, tok, epochs=60):
    f, e4 = lr_features(Z, RAWP, tok); y = (e4 == R1_TRUE[:, tok].unsqueeze(-1)).float().reshape(-1); f = f.reshape(-1, f.shape[-1])
    mu, sd = f.mean(0), f.std(0) + 1e-6; fn = (f - mu) / sd; w = nn.Linear(f.shape[-1], 1); opt = torch.optim.Adam(w.parameters(), lr=1e-2, weight_decay=1e-3)
    for ep in range(epochs):
        for i in torch.randperm(len(fn)).split(4096):
            opt.zero_grad(); F.binary_cross_entropy_with_logits(w(fn[i]).squeeze(-1), y[i]).backward(); opt.step()
    def predict(Z2, RAWP2, tok2):
        ff, ee = lr_features(Z2, RAWP2, tok2)
        with torch.no_grad(): return ee, w((ff - mu) / sd).squeeze(-1)
    return predict

# ---------- run
t0 = time.time(); results = {'own': OWN, 'sweep': {}, 'stacked': {}, 'learning_curve': {}, 'buckets': {}}
def fold_train(k): return sum((blocks[j] for j in range(5) if j != k), [])
all_tok = sum(blocks, [])
def pooled(make, train=True, stacked=False):
    """train per fold, pool held-out candidates; returns eval dict (+ stacked-LR eval)."""
    ce, cs, ce2, cs2 = [], [], [], []
    for k in range(5):
        h = make(); h.train(fold_train(k)) if train else None
        e, s, z, r = h.predict(blocks[k]); ce.append(e); cs.append(s)
        if stacked:
            _, _, zt, rt = h.predict(fold_train(k)); lr = train_lr(zt, rt, fold_train(k)); e2, s2 = lr(z, r, blocks[k]); ce2.append(e2); cs2.append(s2)
    out = evaluate(torch.cat(ce, 1), torch.cat(cs, 1), all_tok)
    if stacked: out = (out, evaluate(torch.cat(ce2, 1), torch.cat(cs2, 1), all_tok))
    return out, h

base, _ = pooled(lambda: Head('affine', lam=0), train=False)
results['margin_baseline'] = base; print('[router, no delta = margin heuristic]', fmt(base), flush=True)
# fixed-threshold margin buckets for reference
m_all = ((S_RAW.gather(-1, S_SEL.topk(6, -1).indices)))
mg = (m_all[..., 0] - m_all[..., 1]) / m_all.sum(-1); p1 = S_SEL.argmax(-1); own1 = box2[LT, p1]; w1 = PM[LT, p1]; hit1 = (p1 == R1_TRUE).float()
for th in (0.0, 0.1, 0.2, 0.3):
    m = own1 & (mg >= th); print(f"  margin>={th}: vol={w1[m].sum().item() / LT.numel() * 80:.2f} p={(w1 * hit1)[m].sum().item() / w1[m].sum().item():.3f} r={(w1 * hit1)[m].sum().item() / prot_w.sum().item():.3f}")

SWEEP = {('affine', 0): [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0], ('delta', 4): [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0], ('delta', 16): [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
         ('delta_g', 16): [1e-2, 1e-1, 1.0, 10.0]}
best = {}
for (kind, r), lams in SWEEP.items():
    for obj in (('full', 'box2') if kind != 'delta_g' else ('full',)):
        for lam in lams:
            res, h = pooled(lambda: Head(kind, r=r, lam=lam, obj=obj))
            tag = f"{kind}{'' if kind == 'affine' else f'_r{r}'}|{obj}|lam={lam:g}"
            results['sweep'][tag] = res; print(f"[{tag}] params/layer {h.nparams() // NL} {time.time() - t0:.0f}s  {fmt(res)}", flush=True)
            key = (kind, r, obj); score = res[3]['prec'] + res[2]['prec']
            if key not in best or score > best[key][0]: best[key] = (score, lam)
print('best lambda per variant (by pooled held-out V2+V3 precision):', {f'{k[0]}_r{k[1]}|{k[2]}': v[1] for k, v in best.items()}, flush=True)

# (d) stacked LR on the best affine and best delta_r16 (full objective)
for kind, r in (('affine', 0), ('delta', 16)):
    lam = best[(kind, r, 'full')][1]
    (res, res_lr), h = pooled(lambda: Head(kind, r=r, lam=lam, obj='full'), stacked=True)
    tag = f"LR on {kind}_r{r}|full|lam={lam:g}"; results['stacked'][tag] = res_lr; print(f"[{tag}] {fmt(res_lr)}", flush=True)
    results['buckets'][f"{kind}_r{r}"] = {V: res[V]['bucket'] for V in (3, 5)}
results['buckets']['margin'] = {V: base[V]['bucket'] for V in (3, 5)}

# learning curve on the last block for each variant's best lambda (full objective)
train80 = sum(blocks[:4], []); test = blocks[4]
lc_base = evaluate(*Head('affine', lam=0).predict(test)[:2], test); results['learning_curve']['margin'] = lc_base
print(f"LC margin (last block): {fmt(lc_base)}")
for kind, r in (('affine', 0), ('delta', 4), ('delta', 16), ('delta_g', 16)):
    lam = best[(kind, r, 'full')][1]; row = {}
    for n in (80, 201, 402, 804):
        h = Head(kind, r=r, lam=lam, obj='full'); h.train(train80[:n]); row[n] = evaluate(*h.predict(test)[:2], test)
    results['learning_curve'][f"{kind}_r{r}|lam={lam:g}"] = row
    print(f"LC {kind}_r{r} lam={lam:g}: " + ' | '.join(f"{n}: V2 {row[n][2]['prec']:.3f} V3 {row[n][3]['prec']:.3f}/{row[n][3]['recall']:.3f} V5 {row[n][5]['prec']:.3f}" for n in row), flush=True)
json.dump(results, open(out_path, 'w'), indent=1, default=str); print(f'done {time.time() - t0:.0f}s')
