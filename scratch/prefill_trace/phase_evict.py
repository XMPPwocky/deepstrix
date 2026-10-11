#!/usr/bin/env python3
"""What a prefill phase evicts on box 2 (b2win span of b2_read, cut at R0 = replay start, box-2 wall ISO): encoder-layer
(< 20) loads and their distinct count, per-layer distinct; victims by group (encoder < 20 / decoder >= 20); for the
encoder victims, whether they were loaded EARLIER IN THIS PHASE (self-churn) or before it; and the peak number of
encoder layers 'in flight' (loaded in the phase and still resident) when the phase ends.
  phase_evict.py TXT SPAN0_ISO R0_ISO"""
import collections, datetime, sys
path, s0, r0 = sys.argv[1:4]
F, reads = {}, []
for line in open(path):
    if line.startswith("#F "):
        p = line.split()
        F[p[1]] = p[2:]
        continue
    if line.startswith("b2_read ") and "b2_read" in F:
        reads.append(dict(zip(F["b2_read"], (float(x) for x in line.split()[1:]))))
lo = int([l for l in open(path) if l.startswith("# ranges raw ")][0].split()[-1].split(":")[0])
iso = lambda s: datetime.datetime.fromisoformat(s).timestamp()
R0 = lo + (iso(r0) - iso(s0)) * 1e9
t = lambda r: r["t_read_start"] if r["t_read_start"] > 0 else r["t_pop"]
ph = sorted((r for r in reads if t(r) < R0), key=t)
loaded = set()
enc_loads = [r for r in ph if r["layer"] < 20]
dec_loads = [r for r in ph if r["layer"] >= 20]
vic = collections.Counter()
self_churn = 0
resident_from_phase = set()
for r in ph:
    k = (int(r["layer"]), int(r["expert"]))
    if r["victim_layer"] == r["victim_layer"] and r["victim_layer"] >= 0:
        vk = (int(r["victim_layer"]), int(r["victim_expert"]))
        g = "enc" if vk[0] < 20 else "dec"
        if g == "enc" and vk in resident_from_phase:
            g = "enc (loaded this phase)"
            self_churn += 1
        vic[g] += 1
        resident_from_phase.discard(vk)
    resident_from_phase.add(k)
per_layer = collections.Counter(int(r["layer"]) for r in {(int(x["layer"]), int(x["expert"])): x for x in enc_loads}.values())
print(f"phase reads: {len(ph)} (encoder layers {len(enc_loads)}, decoder layers {len(dec_loads)})")
print(f"distinct encoder experts loaded: {len({(int(r['layer']), int(r['expert'])) for r in enc_loads})}; per layer: {dict(sorted(per_layer.items()))}")
print("victims:", dict(vic))
