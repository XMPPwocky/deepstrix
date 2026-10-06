#!/usr/bin/env python3
"""Today's SPECULATIVE box-2 reads (src=3: cache-prior admissions, the exact daemon
path a hint word would take) from the b2tail evtrace dumps: hint->pop queueing,
yield, read wall, hint->land, pause, drop rate, and the frame->hints-applied delay
of decode requests (where hint words are applied today)."""
import json, sys, statistics as st

def pct(v, p):
    if not v: return float('nan')
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]

def summ(name, v, scale=1e-3):
    v = [x * scale for x in v if x == x]
    if not v: print(f"  {name}: n=0"); return
    print(f"  {name}: n={len(v)} med {pct(v,.5):.3f} p10 {pct(v,.1):.3f} p90 {pct(v,.9):.3f} p99 {pct(v,.99):.3f} mean {st.mean(v):.3f}")

for path in sys.argv[1:]:
    d = json.load(open(path))
    rf = {k: i for i, k in enumerate(d['read_fields'])}
    qf = {k: i for i, k in enumerate(d['req_fields'])}
    reads = d.get('read') or []
    reqs = (d.get('req') or []) + (d.get('reservoir_decode') or [])
    print(f"== {path.split('/')[-1]}: reads {len(reads)} reqs {len(reqs)}")
    by_src = {}
    for r in reads: by_src.setdefault(int(r[rf['src']]), []).append(r)
    print("  reads by src:", {k: len(v) for k, v in sorted(by_src.items())})
    for src in (3, 2, 1, 0):
        rs = by_src.get(src, [])
        if not rs: continue
        print(f" src={src}")
        # stamps are in us? assume evtrace::now() units; print raw deltas scaled by 1e-3 (ns->us) -- detect
        t_hint = [r[rf['t_hint']] for r in rs]; t_pop = [r[rf['t_pop']] for r in rs]
        t_rs = [r[rf['t_read_start']] for r in rs]; t_re = [r[rf['t_read_end']] for r in rs]
        t_land = [r[rf['t_land_end']] for r in rs]; t_recv = [r[rf['t_recv']] for r in rs]
        sample = [a for a in (t_re[i] - t_rs[i] for i in range(len(rs))) if a == a]
        scale = 1e-3 if (sample and st.median(sample) > 1e5) else 1.0  # ns -> us if large
        unit = 'us'
        summ(f"hint->pop ({unit})", [t_pop[i] - t_hint[i] for i in range(len(rs))], scale)
        summ(f"pop->read_start (yield) ({unit})", [t_rs[i] - t_pop[i] for i in range(len(rs))], scale)
        summ(f"read wall ({unit})", [t_re[i] - t_rs[i] for i in range(len(rs))], scale)
        summ(f"read_end->recv (land poll) ({unit})", [t_recv[i] - t_re[i] for i in range(len(rs))], scale)
        summ(f"hint->land_end ({unit})", [t_land[i] - t_hint[i] for i in range(len(rs))], scale)
        summ("pause_ns (us)", [r[rf['pause_ns']] for r in rs], 1e-3)
        summ("yield_ns (us)", [r[rf['yield_ns']] for r in rs], 1e-3)
        w = [r[rf['wanted']] for r in rs]; b = [r[rf['blocked_on']] for r in rs]; ar = [r[rf['already_resident']] for r in rs]
        print(f"  wanted share {sum(1 for x in w if x==1)/len(w):.3f} blocked_on share {sum(1 for x in b if x==1)/len(b):.3f} already_resident share {sum(1 for x in ar if x==1)/len(ar):.3f}")
        rt = {}
        for r in rs: rt[r[rf['route']]] = rt.get(r[rf['route']], 0) + 1
        print("  route:", rt, " demand_reads_at_start>0 share", sum(1 for r in rs if r[rf['demand_reads_at_start']] and r[rf['demand_reads_at_start']] > 0) / len(rs))
    # decode requests: frame -> hints applied, dequeue delay, and pf drop counters
    dec = [q for q in reqs if q[qf['b']] <= 16]
    if dec:
        tf = qf['t_frame']; td = qf['t_dequeue']; th = qf['t_hints_end']
        sample = [q[td] - q[tf] for q in dec if q[td] == q[td] and q[tf] == q[tf]]
        scale = 1e-3 if (sample and st.median([abs(x) for x in sample]) > 1e4) else 1.0
        summ("decode frame->dequeue (us)", [q[td] - q[tf] for q in dec], scale)
        summ("decode frame->hints_end (us)", [q[th] - q[tf] for q in dec], scale)
        summ("decode dequeue->hints_end (us)", [q[th] - q[td] for q in dec], scale)
        npf = [q[qf['n_prefetch_words']] for q in dec]
        print(f"  decode reqs {len(dec)}; n_prefetch_words mean {st.mean(npf):.3f} share>0 {sum(1 for x in npf if x>0)/len(npf):.3f} max {max(npf)}")
        for k in ('pf_d_hinted', 'pf_d_admitted', 'pf_d_dropped', 'pf_d_waited', 'pf_d_promoted', 'pf_free_sets', 'pf_pending', 'pf_run_spec', 'pf_run_certain'):
            v = [q[qf[k]] for q in dec if q[qf[k]] == q[qf[k]]]
            if v: print(f"  {k}: sum {sum(v):.0f} mean {st.mean(v):.3f} max {max(v):.0f}")

print("\n== isolated speculative reads (src=3, run_spec_at_start==0, run_certain_at_start==0, pause_ns==0) vs certain early-page reads (src=1)")
for path in sys.argv[1:]:
    d = json.load(open(path))
    rf = {k: i for i, k in enumerate(d['read_fields'])}
    rs = [r for r in d['read'] if int(r[rf['src']]) == 3]
    import collections
    print("  run_spec_at_start dist:", dict(collections.Counter(r[rf['run_spec_at_start']] for r in rs)), " run_certain_at_start dist:", dict(collections.Counter(r[rf['run_certain_at_start']] for r in rs)))
    iso = [r for r in rs if r[rf['run_spec_at_start']] <= 1 and r[rf['run_certain_at_start']] == 0 and r[rf['pause_ns']] == 0 and r[rf['demand_reads_at_start']] == 0]
    summ("  isolated spec per-role wall r1 (us)", [r[rf['r1_end']] - r[rf['r1_start']] for r in iso], 1e-3)
    summ("  isolated spec per-role wall r2 (us)", [r[rf['r2_end']] - r[rf['r2_start']] for r in iso], 1e-3)
    iso_ids = set(id(r) for r in iso)
    busy = [r for r in rs if id(r) not in iso_ids]
    print(f" {path.split('/')[-1]}: spec {len(rs)} isolated {len(iso)} busy {len(busy)}")
    summ("  isolated spec read wall (us)", [r[rf['t_read_end']] - r[rf['t_read_start']] for r in iso], 1e-3)
    summ("  busy spec read wall (us)", [r[rf['t_read_end']] - r[rf['t_read_start']] for r in busy], 1e-3)
    summ("  isolated spec per-role wall r0 (us)", [r[rf['r0_end']] - r[rf['r0_start']] for r in iso], 1e-3)
    c1 = [r for r in d['read'] if int(r[rf['src']]) == 1]
    summ("  certain(src=1) per-role wall r0 (us)", [r[rf['r0_end']] - r[rf['r0_start']] for r in c1], 1e-3)
    summ("  spec chunk_n", [r[rf['chunk_n']] for r in rs], 1.0)
    w = [(r[rf['t_read_end']] - r[rf['t_read_start']]) / 1e3 for r in rs]
    for be in (2300, 3650, 4100, 5200):
        print(f"  spec reads with wall <= {be} us: {sum(1 for x in w if x <= be)/len(w):.3f}")
    qf = {k: i for i, k in enumerate(d['req_fields'])}
    req = [q for q in d['req'] if q[qf['b']] <= 16]
    summ("  hub-sampled decode frame->dequeue (us)", [q[qf['t_dequeue']] - q[qf['t_frame']] for q in req], 1e-3)
    summ("  hub-sampled decode dequeue->hints_end (us)", [q[qf['t_hints_end']] - q[qf['t_dequeue']] for q in req], 1e-3)
    summ("  hub-sampled decode frame->hints_end (us)", [q[qf['t_hints_end']] - q[qf['t_frame']] for q in req], 1e-3)
    npf = [q[qf['n_prefetch_words']] for q in req]
    print(f"  hub-sampled decode reqs {len(req)}: n_prefetch_words mean {st.mean(npf):.3f} share>0 {sum(1 for x in npf if x>0)/len(npf):.3f}; pf_d_dropped sum {sum(q[qf['pf_d_dropped']] for q in req):.0f} hinted sum {sum(q[qf['pf_d_hinted']] for q in req):.0f}")
