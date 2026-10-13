import csv, sys, collections, statistics as st

d = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "summary"
K = []
with open(f"{d}/kt_kernel_trace.csv") as f:
    for r in csv.DictReader(f):
        K.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"], int(r["Queue_Id"]), int(r["Grid_Size_X"]) * int(r["Grid_Size_Y"]) * int(r["Grid_Size_Z"]) // max(1, int(r["Workgroup_Size_X"]) * int(r["Workgroup_Size_Y"]) * int(r["Workgroup_Size_Z"])), int(r["Workgroup_Size_X"]) * int(r["Workgroup_Size_Y"]) * int(r["Workgroup_Size_Z"])))
M = []
with open(f"{d}/kt_memory_copy_trace.csv") as f:
    for r in csv.DictReader(f):
        M.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), "COPY:" + r["Direction"].replace("MEMORY_COPY_", ""), -1, 0, 0))
labels = open(f"{d}/labels.txt").read().split("\n")
spinq = collections.Counter(k[3] for k in K if k[2].startswith("fma_f32_throughput_probe"))
q = spinq.most_common(1)[0][0]
ev = sorted([k for k in K if k[3] == q and not k[2].startswith("ceil_read")] + M)
spins = [i for i, k in enumerate(ev) if k[2].startswith("fma_f32_throughput_probe")]
groups = []
for j, i in enumerate(spins):
    end = spins[j + 1] if j + 1 < len(spins) else len(ev)
    groups.append((ev[i], ev[i + 1:end]))
groups = groups[-len(labels):]
assert len(groups) == len(labels), (len(groups), len(labels))


def short(n):
    n = n.split("(")[0]
    return n[:40]


if mode == "dump":
    want = sys.argv[3]
    shown = 0
    for (sp, g), lab in zip(groups, labels):
        if want in lab and shown < int(sys.argv[4] if len(sys.argv) > 4 else 1):
            shown += 1
            print("==", lab)
            t0 = sp[1]
            prev = sp[1]
            for k in g[:80]:
                print(f"  {short(k[2]):<42} start+{(k[0]-t0)/1e3:8.2f} dur {(k[1]-k[0])/1e3:7.2f} gap {(k[0]-prev)/1e3:6.2f} grid {k[4]}x{k[5]}")
                prev = k[1]
    sys.exit()

# ops: first event after the spin
ops = collections.defaultdict(list)
opsname = {}
chains = collections.defaultdict(lambda: collections.defaultdict(list))
for (sp, g), lab in zip(groups, labels):
    w = lab.split()
    if w[0] == "op":
        b, op, reg = w[1], w[2], w[3]
        if g:
            k = g[0]
            ops[(b, op, reg)].append(((k[1] - k[0]) / 1e3, (k[0] - sp[1]) / 1e3))
            opsname[(b, op)] = (short(k[2]), k[4], k[5])
    elif w[0] == "chain":
        b, cls = w[1], w[2]
        n = int(w[4].split("=")[1])
        # take events until n kernels (non-copy, non-copyBuffer) seen, keeping copies/blits in between
        sel = []
        nk = 0
        for k in g:
            sel.append(k)
            if not k[2].startswith("COPY") and not k[2].startswith("__amd_rocclr"):
                nk += 1
            if nk == n:
                break
        if nk < n:
            continue
        span = (sel[-1][1] - sp[1]) / 1e3  # spin end -> last kernel end
        busy = sum(k[1] - k[0] for k in sel) / 1e3
        kbusy = sum(k[1] - k[0] for k in sel if not k[2].startswith("COPY")) / 1e3
        first_gap = (sel[0][0] - sp[1]) / 1e3
        gaps = [(sel[i][0] - sel[i - 1][1]) / 1e3 for i in range(1, len(sel))]
        c = chains[(b, cls)]
        c["span"].append(span)
        c["busy"].append(busy)
        c["kbusy"].append(kbusy)
        c["n"].append(len(sel))
        c["gapsum"].append(sum(gaps))
        c["first"].append(first_gap)
        for i, k in enumerate(sel):
            c[f"d{i}:{short(k[2])}"].append((k[1] - k[0]) / 1e3)
            if i:
                c[f"g{i}"].append(gaps[i - 1])


def p(v, x):
    v = sorted(v)
    return v[min(len(v) - 1, int(round((len(v) - 1) * x)))]


print("== isolated ops (kernel duration from the trace, us): p50 / p90; launch latency after spin end p50")
for b in sorted({k[0] for k in ops}):
    print(f"-- {b}")
    for key in sorted([k for k in ops if k[0] == b], key=lambda k: (k[1], k[2])):
        if key[2] != "cold":
            continue
        cv = [x[0] for x in ops[key]]
        wv = [x[0] for x in ops.get((key[0], key[1], "warm"), [])]
        lat = [x[1] for x in ops[key]]
        nm = opsname.get((key[0], key[1]), ("?", 0, 0))
        print(f"  {key[1]:<12} cold {p(cv,.5):7.2f} / {p(cv,.9):7.2f}  warm {p(wv,.5) if wv else float('nan'):7.2f} / {p(wv,.9) if wv else float('nan'):7.2f}  lat {p(lat,.5):5.2f}  [{nm[0]} grid {nm[1]}x{nm[2]}]")

print("\n== chains (queued after spin): span = spin end -> last kernel end; busy = sum of event durations; gaps = span - busy")
for key in sorted(chains):
    c = chains[key]
    print(f"-- {key[0]} {key[1]}: n_samples {len(c['span'])}, events {p(c['n'],.5)}: span p50 {p(c['span'],.5):7.1f} p90 {p(c['span'],.9):7.1f} | busy p50 {p(c['busy'],.5):7.1f} (kernels {p(c['kbusy'],.5):7.1f}) | gaps sum p50 {p(c['gapsum'],.5):6.1f} + first {p(c['first'],.5):5.1f}")
    if mode == "detail":
        i = 0
        while True:
            ks = [k for k in c if k.startswith(f"d{i}:")]
            if not ks:
                break
            gi = c.get(f"g{i}", [0])
            print(f"     {ks[0][len(str(i))+2:]:<42} dur p50 {p(c[ks[0]],.5):7.2f}  gap-before p50 {p(gi,.5):5.2f}")
            i += 1
