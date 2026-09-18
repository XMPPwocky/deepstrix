"""Engine-vs-CPU-oracle first-token logit comparison across a prompt-length sweep.

Reports, per prompt length:
  - engine-vs-oracle  KLD + relRMSE   (absolute correctness)
  - engine-vs-engine  KLD + relRMSE   (this length's own noise floor)
The second column is what makes the first interpretable: an engine-oracle
distance below the engine-engine distance says nothing.
"""
import os, sys, glob
import numpy as np, torch

SC = os.path.dirname(os.path.abspath(__file__))

def sm(x):
    x = x - x.max(); e = np.exp(x); return e / e.sum()

def kld(p_log_src, q_log_src):
    p, q = sm(p_log_src), sm(q_log_src)
    return float((p * (np.log(p + 1e-30) - np.log(q + 1e-30))).sum())

def rel(a, b):
    d = a - b
    return float(np.sqrt((d**2).mean()) / np.sqrt((a**2).mean()))

print(f"{'len':>6} {'ntok':>6} | {'engine-vs-ORACLE':>28} | {'engine-vs-engine (noise)':>28}")
print(f"{'':>6} {'':>6} | {'KLD':>12} {'relRMSE':>9} {'am':>4} | {'KLD':>12} {'relRMSE':>9} {'am':>4}")
for tag in ('p1','p2','p3','p4'):
    ea = f"{SC}/or_{tag}_a.bin"; eb = f"{SC}/or_{tag}_b.bin"
    op = f"{SC}/orc_{tag}/logits_last.pt"
    idf = f"{SC}/or_{tag}_a.bin.ids"
    if not os.path.exists(ea):
        continue
    ntok = len(open(idf).read().split(',')) if os.path.exists(idf) else -1
    A = np.fromfile(ea, dtype=np.float32).astype(np.float64)
    row = [f"{tag:>6} {ntok:>6} |"]
    if os.path.exists(op):
        O = torch.load(op, map_location='cpu').float().numpy().reshape(-1).astype(np.float64)
        n = min(len(A), len(O)); a, o = A[:n], O[:n]
        row.append(f"{kld(o,a):12.4e} {rel(o,a):9.3e} {'ok' if int(a.argmax())==int(o.argmax()) else 'DIFF':>4} |")
    else:
        row.append(f"{'(no oracle)':>12} {'':>9} {'':>4} |")
    if os.path.exists(eb):
        B = np.fromfile(eb, dtype=np.float32).astype(np.float64)
        n = min(len(A), len(B)); a, b = A[:n], B[:n]
        row.append(f"{kld(a,b):12.4e} {rel(a,b):9.3e} {'ok' if int(a.argmax())==int(b.argmax()) else 'DIFF':>4}")
    else:
        row.append(f"{'(no rep b)':>12}")
    print(' '.join(row))
