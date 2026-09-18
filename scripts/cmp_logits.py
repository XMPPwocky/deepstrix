#!/usr/bin/env python3
"""Compare two V41_PREFILL_LOGITS_DUMP files call by call (N_VOCAB=129280 f32 each)."""
import sys, struct, array
V = 129280
def load(p):
    a = array.array('f'); a.frombytes(open(p, 'rb').read()); return a
a, b = load(sys.argv[1]), load(sys.argv[2])
na, nb = len(a) // V, len(b) // V
print(f"calls: {na} vs {nb}")
for i in range(min(na, nb)):
    x, y = a[i*V:(i+1)*V], b[i*V:(i+1)*V]
    ax = max(range(V), key=lambda j: x[j]); ay = max(range(V), key=lambda j: y[j])
    md = max(abs(x[j]-y[j]) for j in range(V)); sc = max(abs(v) for v in x)
    same = x.tobytes() == y.tobytes()
    top5x = sorted(range(V), key=lambda j: -x[j])[:5]; top5y = sorted(range(V), key=lambda j: -y[j])[:5]
    print(f"call {i}: bit_identical={same} argmax {ax} vs {ay} max|d|={md:.4g} scale={sc:.4g} rel={md/sc:.3g} top5_overlap={len(set(top5x)&set(top5y))}")
