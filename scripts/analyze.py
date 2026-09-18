"""Final tables from the dspark_accept_<exp>.json records (pure json, no torch)."""
import json, sys, os, glob, math
K = 5
def stats(R):
    n = len(R)
    acc = [sum(r['hit'][k] for r in R) / n for k in range(K)]
    pref = [sum(all(r['hit'][:k + 1]) for r in R) / n for k in range(K)]
    cond = [pref[0]] + [pref[k] / pref[k - 1] if pref[k - 1] else float('nan') for k in range(1, K)]
    et = [1 + sum(pref[:k + 1]) for k in range(K)]
    se = math.sqrt(acc[0] * (1 - acc[0]) / n)
    out = dict(n=n, acc=acc, cond=cond, et=et, se1=se, top5=sum(r['top5_hit'] for r in R) / n)
    if 'rs' in R[0]:
        rs = [sum(r['rs'][k] for r in R) / n for k in range(K)]
        chain = [[math.prod(r['rs'][:k + 1]) for k in range(K)] for r in R]
        rsp = [sum(c[k] for c in chain) / n for k in range(K)]
        out.update(rs=rs, rs_et=[1 + sum(rsp[:k + 1]) for k in range(K)], ent=sum(r['ent'][0] for r in R) / n)
    return out
def fmt(s):
    line = f"{s['n']:4d} | {s['acc'][0]:.3f} ± {s['se1']:.3f} | " + " ".join(f"{x:.2f}" for x in s['cond'][1:]) + " | " + " ".join(f"{x:.2f}" for x in s['et']) + f" | {s['top5']:.2f}"
    if 'rs' in s:
        line += " | " + " ".join(f"{x:.2f}" for x in s['rs']) + " | " + " ".join(f"{x:.2f}" for x in s['rs_et'])
    return line
hdr = "steps | pos-1 greedy ± 1σ | cond p2 p3 p4 p5 | E[tok/step] K=1..5 | top-5 | RS acc pos1..5 | RS E[tok] K=1..5"
for dump, cut in [(a.split(':')[0], int(a.split(':')[1]) if ':' in a else None) for a in sys.argv[1:]]:
    print(f"\n=== {dump} (on-distribution cut: draft target ≤ {cut}) ===\n{'experiment':18s} | {hdr}")
    for f in sorted(glob.glob(os.path.join(dump, 'dspark_accept_*.json'))):
        name = os.path.basename(f)[len('dspark_accept_'):-5]
        R = json.load(open(f))['records']
        print(f"{name:18s} | {fmt(stats(R))}")
        if cut is not None:
            R1 = [r for r in R if r['i'] + 2 <= cut]
            R5 = [r for r in R if r['i'] + 6 <= cut]
            s1, s5 = stats(R1), stats(R5)
            print(f"{'  ' + name + ' ≤cut':18s} | {fmt(s5)}   [pos-1 on {s1['n']} steps: {s1['acc'][0]:.3f} ± {s1['se1']:.3f}]")
