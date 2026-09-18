import re, sys, collections
def load(path, markpath):
    mark = int(open(markpath).read().strip())
    rows = collections.defaultdict(list)
    toks = set()
    for i, line in enumerate(open(path, errors='ignore')):
        if i < mark: continue
        line = re.sub(r'\x1b\[[0-9;]*m', '', line)
        if 'het.stage' not in line: continue
        d = {m[0]: m[2] for m in re.findall(r"(\w+)=(\"?)([\w.\-/]+)\2", line)}
        try:
            pos = int(d['token_pos']); us = int(d['total_us'])
        except (KeyError, ValueError): continue
        toks.add(pos)
        rows[(d.get('device','?'), d.get('stage','?'))].append((pos, us))
    if not toks: return {}, 0
    lo = sorted(toks)[len(toks)//2]          # steady-state half only
    out = {}
    for k, v in rows.items():
        sel = [us for pos, us in v if pos >= lo]
        if sel: out[k] = sum(sel)/len(sel)/1000.0   # mean ms/token
    return out, len(toks)
A, na = load(sys.argv[1], sys.argv[1]+'.mark')
B, nb = load(sys.argv[2], sys.argv[2]+'.mark')
print(f"tokens profiled: A={na} B={nb}   (steady-state half only)\n")
keys = sorted(set(A)|set(B), key=lambda k: -(B.get(k,0)-A.get(k,0)))
print(f"{'device':>5} {'stage':<34} {'A ms':>8} {'B ms':>8} {'delta':>8}")
ta=tb=0.0
for k in keys:
    a, b = A.get(k,0.0), B.get(k,0.0)
    ta+=a; tb+=b
    if abs(b-a) < 0.15 and max(a,b) < 1.0: continue
    print(f"{k[0]:>5} {k[1]:<34} {a:8.2f} {b:8.2f} {b-a:+8.2f}")
print(f"{'':>5} {'— SUM of all stages —':<34} {ta:8.2f} {tb:8.2f} {tb-ta:+8.2f}")
