import math
from collections import defaultdict

NAN = float('nan')


def load(path, kinds=None):
    fields, recs, hdr = {}, defaultdict(list), None
    with open(path) as f:
        for line in f:
            if line.startswith('#F '):
                p = line.split()
                fields[p[1]] = p[2:]
                continue
            if line.startswith('#H '):
                import json
                hdr = json.loads(line[3:])
                continue
            k, rest = line.split(' ', 1)
            if kinds and k not in kinds:
                continue
            recs[k].append(dict(zip(fields[k], map(float, rest.split()))))
    return hdr, recs


def isnan(x):
    return isinstance(x, float) and math.isnan(x)


def pct(v, p):
    v = sorted(x for x in v if not isnan(x))
    if not v:
        return NAN
    i = (len(v) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)
