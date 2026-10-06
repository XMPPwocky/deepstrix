#!/usr/bin/env python3
"""Discrete-event pipeline simulator of one V4.1 arena decode step (ready-first lane driver,
forward_prefill.rs:4404-4668). Resources: DC = de.compute (single in-order FIFO shared by all lanes),
DX = de.xfer, IC = ie.compute (shared), IX = ie.xfer, box 2 (parallel server, empirical rtt),
HOST = one thread running the Chain/Route/Post state machine round-robin. Times in us.
"""
import random, statistics as st

class Stream:
    def __init__(self):
        self.free = 0.0
        self.busy = 0.0
    def enqueue(self, t_enq, dur, deps=()):
        start = max([t_enq, self.free] + list(deps))
        end = start + dur
        self.free = end
        self.busy += dur
        return start, end

def simulate(P, rng, trace=False):
    """P: parameter dict. Returns dict of per-step results (ms)."""
    n = P['n_lanes']; NL = 40
    DC, DX, IC, IX = Stream(), Stream(), Stream(), Stream()
    CS = Stream() if P.get('combine_sep') else DC
    combine_end = [0.0] * P['n_lanes']
    now = P['prologue']
    # per lane state
    ph = ['chain0'] * n
    layer = [0] * n
    sel_ready = [0.0] * n          # time the router readback is visible to the host
    reply_t = [None] * n           # box-2 reply arrival (None = no request)
    moe_end = [0.0] * n; pushback_end = [0.0] * n
    chain_end = [0.0] * n
    entered = [0] * n; routed = [0] * n
    ordered = P.get('ordered', False)
    engram_joined = False
    moe_launch_times = []; chain_wait = 0.0
    idle_poll = 0.0
    done = [False] * n
    lane_rows = P['lane_rows']
    rtt = P['rtt_samples']; p_idle = P['p_box2_idle']

    def may_enter(i, l):
        return (not ordered) or i == 0 or entered[i - 1] > l
    def may_route(i, l):
        return (not ordered) or i == 0 or routed[i - 1] > l

    def enqueue_chain(i, l):
        nonlocal now, engram_joined
        if l == 1 and P['engram_join'] > 0 and not engram_joined:
            now += P['engram_join']; engram_joined = True
        if l in (1, 14):
            now += P['engram_stage']
        now += P['h_chain_enq']
        dur = P['D_chain'] + (P['D_indexer'] if l in (2, 8, 14, 20, 24, 28, 32, 36) else 0.0) + (P['D_engram'] if l in (1, 14) else 0.0)
        s, e = DC.enqueue(now, dur, deps=(combine_end[i],))
        chain_end[i] = e
        sel_ready[i] = e + P['event_lat']
        entered[i] = l + 1

    for i in range(n):
        enqueue_chain(i, 0)
        ph[i] = 'route'
    while not all(done):
        progressed = False
        for i in range(n):
            if done[i]:
                continue
            l = layer[i]
            if ph[i] == 'chain':
                if may_enter(i, l):
                    enqueue_chain(i, l); ph[i] = 'route'; progressed = True
            elif ph[i] == 'route':
                if may_route(i, l) and now >= sel_ready[i]:
                    now += P['h_route'] + P['h_prep']
                    # box-2 submit
                    if rng.random() < p_idle:
                        reply_t[i] = None
                    else:
                        reply_t[i] = now + rng.choice(rtt)
                    # deferred shared expert on DC (after the submit)
                    DC.enqueue(now, P['D_shared'])
                    # peer push dGPU->iGPU on DX, after the chain
                    _, push_end = DX.enqueue(now, P['D_push'], deps=(chain_end[i],))
                    now += P['h_launch']
                    # iGPU MoE on IC
                    _, me = IC.enqueue(now, P['D_moe'] * P['moe_scale'][i], deps=(push_end,))
                    moe_end[i] = me; moe_launch_times.append((now, me))
                    _, pb = IX.enqueue(me, P['D_pushback'])
                    pushback_end[i] = pb
                    # optional dGPU hot-expert work (hot split) beside the MoE
                    if P.get('D_hot_dgpu', 0) > 0:
                        DC.enqueue(now, P['D_hot_dgpu'])
                    routed[i] = l + 1
                    ph[i] = 'post'; progressed = True
            elif ph[i] == 'post':
                r = reply_t[i]
                if r is None or now >= r:
                    now += P['h_post']
                    _, ce = CS.enqueue(now, P['D_combine'], deps=(pushback_end[i], (r or 0.0) + P['h2d_partial'], chain_end[i]))
                    combine_end[i] = ce
                    if l + 1 < NL:
                        layer[i] = l + 1
                        if may_enter(i, l + 1):
                            enqueue_chain(i, l + 1); ph[i] = 'route'
                        else:
                            ph[i] = 'chain'
                    else:
                        done[i] = True
                    progressed = True
        if not progressed:
            # nothing ready: spin one poll round
            nxt = []
            for i in range(n):
                if done[i]: continue
                if ph[i] == 'route': nxt.append(sel_ready[i])
                elif ph[i] == 'post' and reply_t[i] is not None: nxt.append(reply_t[i])
            step = P['poll']
            if nxt:
                tnext = min(nxt)
                if tnext > now:
                    # jump in poll quanta
                    k = int((tnext - now) // step) + 1
                    now += k * step
                    idle_poll += k * step
                    continue
            now += step; idle_poll += step
    fwd_sync = max(now, DC.free, IX.free)
    # epilogue: head on DC + host
    _, head_end = DC.enqueue(fwd_sync, P['D_head'])
    fwd_all = head_end + P['epilogue_host']
    # iGPU idle inside the loop
    ml = sorted(moe_launch_times)
    idle_in = 0.0; prev_end = None
    for (lt, me) in ml:
        start = me - P['D_moe']  # approx
        if prev_end is not None and lt > prev_end:
            idle_in += lt - prev_end
        prev_end = max(prev_end or 0, me)
    return {'fwd': fwd_sync / 1e3, 'fwd_all': fwd_all / 1e3, 'igpu_busy': IC.busy / 1e3, 'dgpu_busy': (DC.busy + DX.busy) / 1e3,
            'igpu_idle_inloop': idle_in / 1e3, 'first_launch': ml[0][0] / 1e3, 'last_moe_end': ml[-1][1] / 1e3}

def run(P, n=40, seed=1):
    rng = random.Random(seed)
    rs = [simulate(P, rng) for _ in range(n)]
    out = {}
    for k in rs[0]:
        out[k] = st.median(r[k] for r in rs)
    return out

if __name__ == '__main__':
    import json, sys
    P = json.load(open(sys.argv[1]))
    print(run(P))
