#!/usr/bin/env python3
"""Job B (pure Python): replay a V41_PICK_TRACE against a box-2 LRU to get
(a) box-2 miss rate by router RANK, (b) per-(layer, expert) miss probability for
box-2-owned picks (feeds Job A's weighting), (c) protected-miss hints per lane-layer.

Trace lines (forward_prefill.rs ~9099 / ~9201):
  P <layer> <b> <6 ids>        one row's picks AS RUN (post cache-prior), rank order
  O <layer> <b> <row> <6 ids>  the router's OWN picks for a row the prior changed
Ownership: box 1 = per-layer top-K by decode pick count (hot_2337.json from the
same trace, K=103 = V41_B1_HOT_PER_LAYER live); everything else is box 2's.
Box 2 = LRU of CAP slots over box-2-owned SENT ids, plus background admission
of the ids the prior displaced (V41_SUB_ADMIT; the TinyLFU gate is ignored).

usage: trace_lru.py <trace> <nbytes> <hot_json> <out_json> [cap=4000] [protect=1]
"""
import json, os, sys
from collections import OrderedDict

path, nbytes, hot_json, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
CAP = int(sys.argv[5]) if len(sys.argv) > 5 else 4000
PROTECT = int(sys.argv[6]) if len(sys.argv) > 6 else 1
L, E, K = 40, 384, 103
hot = json.load(open(hot_json))
box1 = [set(hot['rank'][l][:K]) for l in range(L)]
del hot

lru = OrderedDict()           # key (layer<<9|e) -> None
picks_rank = [0] * 6          # box-2-owned router picks by rank
miss_rank = [0] * 6
pe_pick = [[0] * E for _ in range(L)]
pe_miss = [[0] * E for _ in range(L)]
req_n = 0; req_any_prot_miss = 0; prot_miss_total = 0; prot2_miss_total = 0
req_miss_total = 0; reqs_with_miss = 0
hist_prot = {}                # protected misses per request
dist_rows = {}
nrows = 0

def flush(layer, b, rows, orig, warm):
    global req_n, req_any_prot_miss, prot_miss_total, prot2_miss_total, req_miss_total, reqs_with_miss
    if not rows: return
    sent = set(); router_rows = []
    for r, row in enumerate(rows):
        rr = orig.get(r, row)
        router_rows.append(rr)
        for e in row:
            if 0 <= e < E and e not in box1[layer]: sent.add(e)
    before = set(e for e in sent if (layer << 9 | e) in lru)
    # also residency of router picks that were displaced (not sent): check before inserts
    if warm:
        req_n += 1
        prot = 0; prot2 = 0; nmiss = 0; seen = set()
        for rr in router_rows:
            for k, e in enumerate(rr):
                if not (0 <= e < E) or e in box1[layer]: continue
                res = (layer << 9 | e) in lru
                picks_rank[k] += 1; pe_pick[layer][e] += 1
                if not res:
                    miss_rank[k] += 1; pe_miss[layer][e] += 1
                    if e not in seen:
                        seen.add(e); nmiss += 1
                        if k < PROTECT: prot += 1
                        if k < 2: prot2 += 1
        prot_miss_total += prot; prot2_miss_total += prot2; req_miss_total += nmiss
        if prot: req_any_prot_miss += 1
        if nmiss: reqs_with_miss += 1
        hist_prot[prot] = hist_prot.get(prot, 0) + 1
    # the request pages its sent ids
    for e in sent:
        key = layer << 9 | e
        if key in lru: lru.move_to_end(key)
        else:
            lru[key] = None
            if len(lru) > CAP: lru.popitem(last=False)
    # background admissions of displaced router picks (box-2-owned)
    for r, rr in enumerate(router_rows):
        if r not in orig: continue
        for e in rr:
            if 0 <= e < E and e not in box1[layer] and e not in sent:
                key = layer << 9 | e
                if key in lru: lru.move_to_end(key)
                else:
                    lru[key] = None
                    if len(lru) > CAP: lru.popitem(last=False)

fd = os.open(path, os.O_RDONLY); pos = 0; buf = b''; CH = 32 << 20
cur = None; rows = []; orig = {}
ngroups = 0; warm_at = None
# pass 0: estimate group count from bytes (~55 B/line, ~1.4 rows per group) -> warm after 15%
warm_groups = int(nbytes / 55 / 1.4 * 0.15)
while pos < nbytes:
    chunk = os.pread(fd, CH, pos)
    if not chunk: break
    os.posix_fadvise(fd, pos, len(chunk), os.POSIX_FADV_DONTNEED); pos += len(chunk)
    lines = (buf + chunk).split(b'\n'); buf = lines.pop()
    for raw in lines:
        c = raw[:1]
        if c == b'P':
            p = raw.split(); layer = int(p[1]); b = int(p[2])
            if b > 8: continue
            if cur is None or cur != (layer, b) or len(rows) >= b:
                if cur is not None:
                    flush(cur[0], cur[1], rows, orig, ngroups >= warm_groups); ngroups += 1
                cur = (layer, b); rows = []; orig = {}
            rows.append([int(x) for x in p[3:9]]); nrows += 1
        elif c == b'O':
            p = raw.split(); layer = int(p[1]); b = int(p[2])
            if b > 8 or cur != (layer, b): continue
            orig[int(p[3])] = [int(x) for x in p[4:10]]
if cur is not None:
    flush(cur[0], cur[1], rows, orig, ngroups >= warm_groups); ngroups += 1
os.close(fd)

tot_p = sum(picks_rank); tot_m = sum(miss_rank)
print(f"rows {nrows} groups {ngroups} warm groups {req_n} cap {CAP} protect {PROTECT}")
print("box-2-owned router picks by rank:", picks_rank, " share of all picks %.3f" % (tot_p / max(1, nrows * 6)))
print("miss rate by rank:", ["%.3f" % (miss_rank[k] / max(1, picks_rank[k])) for k in range(6)])
print("miss share by rank:", ["%.3f" % (miss_rank[k] / max(1, tot_m)) for k in range(6)])
print(f"misses per request {req_miss_total / max(1, req_n):.3f}; requests with >=1 miss {reqs_with_miss / max(1, req_n):.3f}")
print(f"protected (rank<={PROTECT}) misses per request {prot_miss_total / max(1, req_n):.3f}; rank<=2 misses per request {prot2_miss_total / max(1, req_n):.3f}; requests with a protected miss {req_any_prot_miss / max(1, req_n):.3f}")
print("protected misses per request hist:", dict(sorted(hist_prot.items())))
# per-(layer,e) miss prob for box-2-owned
pm = [[(pe_miss[l][e] / pe_pick[l][e]) if pe_pick[l][e] else None for e in range(E)] for l in range(L)]
# by layer bucket
for name, rng in [("enc 0-19", range(0, 20)), ("dec 20-39", range(20, 40))]:
    p = [0] * 6; m = [0] * 6
    for l in rng:
        for e in range(E):
            pass
    # recompute by rank needs per-layer rank counts: not tracked; report per-layer miss prob instead
    mp = sum(pe_miss[l][e] for l in rng for e in range(E)); pp = sum(pe_pick[l][e] for l in rng for e in range(E))
    print(f"{name}: box-2 pick miss rate {mp / max(1, pp):.3f} (picks {pp})")
json.dump({"cap": CAP, "protect": PROTECT, "picks_rank": picks_rank, "miss_rank": miss_rank, "req_n": req_n,
           "prot_miss_per_req": prot_miss_total / max(1, req_n), "prot2_miss_per_req": prot2_miss_total / max(1, req_n),
           "miss_per_req": req_miss_total / max(1, req_n), "pe_pick": pe_pick, "pe_miss": pe_miss, "K": K}, open(out, "w"))
