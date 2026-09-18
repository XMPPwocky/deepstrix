#!/usr/bin/env python3
"""Box 2's ACTUAL per-layer LRU geometry vs alternatives, on the real routing trace.

The global-pool curve says 4.9 misses/token at 60.7% residency. We measure 20.2.
The difference can only be WHERE the slots are, so model where they actually are:
box 2 runs an independent LRU per layer, sized by what it OWNS on that layer
(260 on encoder layers 0-19, 68 on decoder layers 20-39), and box 1 forwards it
~92% of picks (measured 260.5 of ~280 picks/token reach box 2).
"""
import sys, array, collections

N_LAYER, TOPK, N_EXPERT = 40, 6, 384
buf = array.array('H'); buf.frombytes(open(sys.argv[1], 'rb').read())
per_tok = N_LAYER * TOPK
ntok = len(buf) // per_tok
t0 = ntok // 2                                  # score the warm half only

picks = []
for t in range(ntok):
    rows = []
    for l in range(N_LAYER):
        seen = []
        for k in range(TOPK):
            e = buf[t*per_tok + l*TOPK + k]
            if e < N_EXPERT and e not in seen: seen.append(e)
        rows.append(seen)
    picks.append(rows)

def sim(slots_per_layer, label, glob=False):
    if glob:
        lru = collections.OrderedDict(); miss = 0; S = sum(slots_per_layer)
        for ti, tk in enumerate(picks):
            for l, row in enumerate(tk):
                for e in row:
                    key = (l, e)
                    if key in lru: lru.move_to_end(key)
                    else:
                        if ti >= t0: miss += 1
                        lru[key] = 1
                        if len(lru) > S: lru.popitem(last=False)
        return miss/(ntok-t0)
    lrus = [collections.OrderedDict() for _ in range(N_LAYER)]
    miss = 0
    for ti, tk in enumerate(picks):
        for l, row in enumerate(tk):
            S = slots_per_layer[l]
            lru = lrus[l]
            for e in row:
                if e in lru: lru.move_to_end(e)
                else:
                    if ti >= t0: miss += 1
                    lru[e] = 1
                    if len(lru) > S: lru.popitem(last=False)
    return miss/(ntok-t0)

TOTAL = 6560
today   = [260]*20 + [68]*20
uniform = [TOTAL//40]*40
# decode-weighted: encoder layers keep the prefill minimum (256), rest to decoders
enc_min = [256]*20 + [(TOTAL-256*20)//20]*20

print(f"{'allocation':>34} {'slots':>6} {'enc/dec':>9} {'miss/tok':>9} {'ms/tok @6.6':>12}")
for sl, name in [(today,"TODAY 260/68 (per-layer LRU)"),
                 (uniform,"uniform 164/layer"),
                 (enc_min,"encoder-min 256 / decoder 328"),
                 (today,"TODAY, but ONE GLOBAL pool")]:
    g = name.endswith("GLOBAL pool")
    m = sim(sl, name, glob=g)
    print(f"{name:>34} {sum(sl):6d} {sl[0]:4d}/{sl[-1]:<4d} {m:9.1f} {m*6.6:12.0f}")

# --- Two-tier: box 1's pool joins the decode cache ---
# Box 1 holds ~2688 slots (52 GB pool, 21 windows x stride 128) PINNED to
# prefill's encoder windows, so during decode they hold the wrong experts and
# serve a measured 0.5 of 6.5 picks/layer. Its iGPU and its (faster) NVMe are
# idle. What if that pool ran a decode LRU over the layers box 2 starves?
def sim2(b2, b1_slots, b1_layers):
    lru2 = [collections.OrderedDict() for _ in range(N_LAYER)]
    lru1 = [collections.OrderedDict() for _ in range(N_LAYER)]
    per1 = {l: b1_slots // max(len(b1_layers), 1) for l in b1_layers}
    miss = 0
    for ti, tk in enumerate(picks):
        for l, row in enumerate(tk):
            for e in row:
                if l in per1:
                    h = lru1[l]
                    if e in h:
                        h.move_to_end(e); continue
                a = lru2[l]
                if e in a:
                    a.move_to_end(e); continue
                if ti >= t0: miss += 1
                # fill on box 1 when it covers this layer, else box 2
                if l in per1:
                    h[e] = 1
                    if len(h) > per1[l]: h.popitem(last=False)
                else:
                    a[e] = 1
                    if len(a) > b2[l]: a.popitem(last=False)
    return miss/(ntok-t0)

print()
dec = list(range(20, 40))
alll = list(range(40))
for b1s, lays, name in [
    (2688, dec,  "+ box1 2688 slots over DECODER layers (134/layer)"),
    (2688, alll, "+ box1 2688 slots over ALL layers (67/layer)"),
    (2688, dec,  "+ box1 on decoders, box2 uniform 164"),
]:
    b2 = uniform if "uniform" in name else today
    m = sim2(b2, b1s, lays)
    print(f"{name:>52} {m:7.1f} miss/tok  {m*6.6:5.0f} ms")
