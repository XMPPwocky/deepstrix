#!/usr/bin/env python3
"""E8M0 range scan of every routed-expert scale tensor (w1/w2/w3) of V4.1-Flash, for the f16-LUT WMMA arm.

The arm folds the block scale into a perm LUT that is exact only for e8 in [LO, HI] (mid_f16.inc: wm_lut CLAMPS outside it,
it does not flush). Reads ONLY the scale tensors (~17 GB of ~290); run under `ionice -c3`. Prints per (layer, matrix) the
count of out-of-range bytes and a histogram of them, plus the global min/max.

  ionice -c3 python3 e8_scan.py [LO HI]  > out.txt
"""
import collections, json, struct, sys, time

S = "/persist/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277/"
LO, HI = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (114, 137)
INRANGE = bytes(range(LO, HI + 1))

wm = json.load(open(S + "model.safetensors.index.json"))["weight_map"]
by_file = collections.defaultdict(list)
for k, f in wm.items():
    if ".ffn.experts." in k and k.endswith(".scale"):
        by_file[f].append(k)

total = 0
bad = collections.Counter()                 # (layer, mat) -> count
hist = collections.Counter()                # e8 value -> count (out of range only)
gmin, gmax = 255, 0
t0 = time.time()
for i, (f, keys) in enumerate(sorted(by_file.items())):
    with open(S + f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        base = 8 + n
        for k in sorted(keys, key=lambda k: hdr[k]["data_offsets"][0]):
            a, b = hdr[k]["data_offsets"]
            assert hdr[k]["dtype"] == "F8_E8M0", (k, hdr[k]["dtype"])
            fh.seek(base + a)
            buf = fh.read(b - a)
            total += len(buf)
            gmin, gmax = min(gmin, min(buf)), max(gmax, max(buf))
            rest = buf.translate(None, INRANGE)
            if rest:
                p = k.split(".")
                bad[(int(p[1]), p[-2])] += len(rest)
                hist.update(rest)
    print(f"{time.strftime('%H:%M:%S')} file {i + 1}/{len(by_file)} {f}: {total / 1e9:.2f} GB scanned, "
          f"{sum(bad.values())} out of [{LO},{HI}] so far, min {gmin} max {gmax}", flush=True)

print(f"\nDONE {total} scale bytes in {time.time() - t0:.0f} s; global e8 min {gmin} max {gmax}; range [{LO},{HI}]")
print(f"out of range: {sum(bad.values())} ({sum(bad.values()) / max(total, 1):.3e})")
for v, c in sorted(hist.items()):
    print(f"  e8={v:3d} (2^{v - 127:+d}): {c}")
for (l, m), c in sorted(bad.items()):
    print(f"  layer {l:2d} {m}: {c}")
