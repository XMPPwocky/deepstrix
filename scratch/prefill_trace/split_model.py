#!/usr/bin/env python3
"""Offline model of the long-job split variants on the REAL persisted pick distributions (expert_stats.json).
Per encoder layer, p_e = prefill bank count share; a 512-row request = 3072 picks; box 2's expected distinct experts
per request = sum_{e in box2} 1 - (1 - p_e)^3072, and its pick share = sum_{e in box2} p_e. Box 2 is weight-read bound
(~100 us per distinct expert, 10-10), so distinct experts per request is its cost.
Variants: head = decode-hot (top 115 by the DECODE bank, a stand-in for the warm hot set) or cold (hash / prefill prior);
filler (41/layer as in the 10-10 23:11 job, 64 = knob max) = hash or prefill prior."""
import json

D = json.load(open("/home/claude-code/.cache/deepstrix/expert_stats.json"))
NE, NL, HEAD, PICKS = 384, 20, 115, 3072
pre, dec = D["prefill"]["counts"], D["decode"]["counts"]


def mix(l, e):
    z = ((l << 20) | e) + 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def run(head_mode, filler_mode, filler):
    dist, share, n2 = 0.0, 0.0, 0
    for l in range(NL):
        c = pre[l * NE:(l + 1) * NE]
        tot = sum(c)
        p = [x / tot for x in c]
        hash_order = sorted(range(NE), key=lambda e: mix(l, e))
        prior_order = sorted(range(NE), key=lambda e: (-c[e], mix(l, e)))
        if head_mode == "decode":
            d = dec[l * NE:(l + 1) * NE]
            head = set(sorted(range(NE), key=lambda e: -d[e])[:HEAD])
        else:
            head = set((prior_order if head_mode == "prior" else hash_order)[:HEAD])
        b1 = set(head)
        for e in (prior_order if filler_mode == "prior" else hash_order):
            if len(b1) >= len(head) + filler:
                break
            b1.add(e)
        b2 = [e for e in range(NE) if e not in b1]
        n2 += len(b2)
        share += sum(p[e] for e in b2)
        dist += sum(1 - (1 - p[e]) ** PICKS for e in b2)
    return dist / NL, share / NL, n2


print(f"prefill bank {D['prefill']['tokens']:,} tokens; decode bank {D['decode']['tokens']:,}")
print(f"{'head':8} {'filler':7} {'n':>3} {'box-2 distinct/req':>19} {'box-2 pick share':>17} {'box-2 ids':>9}")
for head_mode, filler_mode, f in [("hash", "hash", 41), ("prior", "prior", 41), ("decode", "hash", 41), ("decode", "prior", 41),
                                  ("decode", "hash", 64), ("decode", "prior", 64)]:
    d, s, n2 = run(head_mode, filler_mode, f)
    print(f"{head_mode:8} {filler_mode:7} {f:3d} {d:19.1f} {s:17.3f} {n2:9d}")
