#!/usr/bin/env python3
"""Find THIS prompt's records in a busy V41_PREFILL_LOGITS_DUMP (other traffic interleaves records; no timestamps).

Every record = N_VOCAB f32. A record is ours when its top-K overlaps the reference prompt's top-K (from a saved arm file,
e.g. the 10-10 a0.f32) in >= MIN tokens: other prompts' next-token distributions do not share this one's top-5.
Prints every record's index, top-5, overlap, and KLD vs the reference; writes ours (in order) to <out>/<label_i>.f32.

  dump_find.py DUMP REF.f32 OUT_DIR LABELS(comma list, in order) [start_record]
"""
import math, os, struct, sys

NV = 129280
REC = NV * 4
dump, ref_p, out, labels = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
start = int(sys.argv[5]) if len(sys.argv) > 5 else 0
K, MIN = 5, 3


def lsm(x):
    m = max(x)
    s = math.log(sum(math.exp(v - m) for v in x)) + m
    return [v - s for v in x]


def topk(x):
    return sorted(range(NV), key=lambda i: -x[i])[:K]


ref = struct.unpack(f"<{NV}f", open(ref_p, "rb").read(REC))
ref_top, ref_ls = set(topk(ref)), lsm(ref)
n = os.path.getsize(dump) // REC
ours = []
with open(dump, "rb") as f:
    f.seek(start * REC)
    for i in range(start, n):
        r = struct.unpack(f"<{NV}f", f.read(REC))
        t = topk(r)
        ov = len(ref_top & set(t))
        if ov >= MIN:
            ls = lsm(r)
            kl = sum(math.exp(a) * (a - b) for a, b in zip(ref_ls, ls))
            ours.append((i, r))
            print(f"rec {i:5d} OURS overlap {ov} top5 {t} KLD(ref||rec) {kl:.3e}")
        else:
            print(f"rec {i:5d} other overlap {ov} top5 {t}")
print(f"{n - start} records scanned, {len(ours)} ours; labels {labels}")
if len(ours) == len(labels):
    os.makedirs(out, exist_ok=True)
    for lab, (i, r) in zip(labels, ours):
        with open(os.path.join(out, f"{lab}.f32"), "wb") as g:
            g.write(struct.pack(f"<{NV}f", *r))
    print(f"wrote {len(ours)} records to {out}")
else:
    print("COUNT MISMATCH: not writing (check which records are ours)")
