#!/usr/bin/env python3
"""Discrete-event model of ONE decode step (forward only), one or two lanes.

Mirrors the drivers in crates/v4flash-kernels/src/het/forward_prefill.rs:
  * n = 1: `forward_step_arena` (sequential: chain -> route [blocks on the router
    event] -> submit box 2 -> prep/launch iGPU MoE -> post [enqueue wait(moe) +
    combine.local, BLOCK on box 2's reply, H2D partial, combine.remote] -> next chain).
  * n >= 2: `forward_step_arena_ready_first`: one host thread polls the lanes in
    order 0..n-1 and runs whichever lane's next phase is ready -- Route(l) when its
    router event fired, Post(l) when its box-2 reply is in (NOT when its iGPU MoE is
    done: the dGPU stream waits for that on the device). Lane i>0 may enter layer
    l's chain only after lane i-1 has (ordered cut, one DSpark stream).

Resources (each an in-order queue, all times in ms from the first chain enqueue):
  host thread, dGPU compute stream (chain/mid/combine of BOTH lanes, in enqueue
  order: a post waiting on its iGPU event blocks everything queued behind it),
  dGPU xfer (activation push to the iGPU), iGPU compute (MoE incl. push back),
  box-2 server (compute serial; misses read on its disks, the request parks),
  the link (fixed latency by rows, half each way).

Per lane-layer input: LL(b, n1, d1, n2, d2, m2, m1):
  b rows; box-1 picks / distinct; box-2 picks / distinct / misses; box-1 misses.
"""
from dataclasses import dataclass, field


@dataclass
class Params:
    # dGPU per lane-layer (ms): chain up to the router event, the rest of the
    # pre-MoE work (shared expert, mix_ffn_late), combine; activation push (xfer).
    chain0: float = 0.5327
    chain_b: float = 0.01493
    mid0: float = 0.100
    mid_b: float = 0.0037
    comb_local: float = 0.0088
    comb_remote0: float = 0.0083
    comb_b: float = 0.0005
    push0: float = 0.050
    push_b: float = 0.002
    # iGPU MoE per lane-layer (ms), incl. its push back to the dGPU
    ig0: float = 0.10599
    ig_d: float = 0.07670
    ig_n: float = 0.00727
    ig_b: float = 0.02773
    # box 2 (us): server compute and link by rows; paging per missed expert
    s0: float = 114.9
    s_d: float = 101.1
    s_b: float = 15.4
    link_us: dict = field(default_factory=lambda: {1: 58, 2: 87, 3: 118, 4: 276, 5: 295, 6: 318, 7: 340, 8: 360})
    page_us: float = 3235.0
    b2_disks_parallel: int = 1
    # box-1 synchronous miss read (ms per expert, already parallel-amortized)
    b1_read_ms: float = 7.0
    # host costs (ms)
    h_r1: float = 0.020   # router event seen -> box-2 submit (readback, pick split, submit)
    h_r2: float = 0.060   # submit -> push/MoE enqueued (ensure, remap upload, launch)
    h_p1: float = 0.030   # post: enqueue wait(moe) + combine.local
    h_p2: float = 0.120   # post after the reply: H2D partial, combine.remote, next chain enqueue
    h_req: float = 0.0    # extra host per box-2 request (submit + reply handling)
    h_poll: float = 0.0   # ready-first only (n >= 2): extra host per route / post action
    h2d_drain: bool = True  # post's partial H2D is a blocking null-stream hipMemcpy
    post_wait_moe: bool = False  # what-if: ready-first posts only once the lane's iGPU MoE event fired too
    h_copy: float = 0.015   # that copy (20 KB/row f32) once the stream drained
    h_chain0: float = 0.05  # initial chain enqueue per lane
    # device-side gaps between a lane-layer's kernels (launch latency; occupancy
    # that the event-timed busy sums do not see), per lane-layer
    dg_gap: float = 0.0
    ig_gap: float = 0.0
    # scale knobs for sensitivity
    link_scale: float = 1.0
    link_add_us: float = 0.0
    b2_scale: float = 1.0

    def link(self, b):
        v = self.link_us.get(b)
        if v is None:
            v = self.link_us[max(self.link_us)] + 20 * (b - max(self.link_us))
        return (v * self.link_scale + self.link_add_us) / 1000.0


@dataclass
class LL:
    b: int
    n1: int
    d1: int
    n2: int
    d2: int
    m2: int = 0
    m1: int = 0
    # optional: pre-assigned replicated picks handled by the policy layer


def simulate(lanes, P, engram_join_ms=0.0, return_trace=False):
    """lanes: list (per lane) of 40 LL. Returns a metrics dict."""
    n = len(lanes)
    NL = len(lanes[0])
    host = 0.0
    dg_free = 0.0
    dx_free = 0.0
    ig_free = 0.0
    b2_free = 0.0
    disk_free = [0.0] * max(1, P.b2_disks_parallel)
    R = [[None] * NL for _ in range(n)]
    reply = [[None] * NL for _ in range(n)]
    moe = [[None] * NL for _ in range(n)]
    sub_t = [[None] * NL for _ in range(n)]
    entered = [0] * n
    phase = [('route', 0)] * n
    ig_busy = dg_busy = 0.0
    rtt_sum = srv_sum = 0.0
    wait_exposed = 0.0
    n_req = 0
    trace = [] if return_trace else None

    push_busy = [0.0]
    hb = [0.0]
    b2_busy = [0.0]
    # where lanes wait (ms summed over lane-layers): host busy with the other lane
    # when the router fired; MoE queued behind the other lane's; box-2 reply later
    # than the local MoE; host blocked in the post's H2D drain
    attr = {'route_lag': 0.0, 'moe_queue': 0.0, 'b2_beyond_moe': 0.0, 'drain_block': 0.0}
    wait_enter = [[None] * NL for _ in range(n)]
    wait_exit = [[None] * NL for _ in range(n)]

    def chain_dur(b):
        return P.chain0 + P.chain_b * b + P.dg_gap

    def mid_dur(b):
        return P.mid0 + P.mid_b * b

    def enqueue_chain(i, l, t):
        nonlocal dg_free, dg_busy
        b = lanes[i][l].b
        s = max(t, dg_free)
        R[i][l] = s + chain_dur(b)
        dg_free = R[i][l] + mid_dur(b)
        dg_busy += chain_dur(b) + mid_dur(b)
        entered[i] = l + 1

    # step start: every lane's layer-0 chain
    for i in range(n):
        host += P.h_chain0
        enqueue_chain(i, 0, host)

    def do_route(i, l):
        nonlocal host, dx_free, ig_free, b2_free, ig_busy, rtt_sum, srv_sum, n_req
        x = lanes[i][l]
        attr['route_lag'] += max(0.0, host - R[i][l])
        host = max(host, R[i][l]) + P.h_r1 + (P.h_poll if n > 1 else 0.0)
        hb[0] += P.h_r1 + P.h_r2 + (P.h_poll if n > 1 else 0.0) + (P.h_req if x.d2 > 0 else 0.0)
        if x.d2 > 0:
            ts = host
            sub_t[i][l] = ts
            lk = P.link(x.b)
            arr = ts + lk / 2
            comp = (P.s0 + P.s_d * x.d2 + P.s_b * x.b) * P.b2_scale / 1000.0
            st = max(arr, b2_free)
            done = st + comp
            b2_free = done
            b2_busy[0] += comp
            if x.m2 > 0:
                k = min(range(len(disk_free)), key=lambda j: disk_free[j])
                rs = max(arr, disk_free[k])
                re = rs + x.m2 * P.page_us / 1000.0
                disk_free[k] = re
                done = max(done, re + P.s_d * x.m2 / 1000.0)
            reply[i][l] = done + lk / 2
            rtt_sum += reply[i][l] - ts
            srv_sum += done - arr
            n_req += 1
            host += P.h_req
        host += P.h_r2
        if x.m1 > 0:
            host = max(host, ig_free)  # cross-lane iGPU drain before an evicting ensure
            host += x.m1 * P.b1_read_ms
        ps = max(host, dx_free, R[i][l])
        pe = ps + P.push0 + P.push_b * x.b
        dx_free = pe
        push_busy[0] += P.push0 + P.push_b * x.b
        dur = P.ig0 + (P.ig_d * x.d1 + P.ig_n * x.n1 if x.d1 > 0 else 0.0) + P.ig_b * x.b
        ms = max(pe, ig_free)
        attr['moe_queue'] += ms - pe
        moe[i][l] = ms + dur + P.ig_gap
        ig_free = moe[i][l]
        ig_busy += dur
        if reply[i][l] is not None:
            attr['b2_beyond_moe'] += max(0.0, reply[i][l] - moe[i][l])
        if trace is not None:
            trace.append(('route', i, l, R[i][l], host, ms, moe[i][l], reply[i][l]))

    def do_post(i, l):
        """Post(l): combine; then the chain of l+1 (if this lane may enter it)."""
        nonlocal host, dg_free, dg_busy, wait_exposed
        x = lanes[i][l]
        host += P.h_p1 + (P.h_poll if n > 1 else 0.0)
        hb[0] += P.h_p1 + P.h_p2 + (P.h_poll if n > 1 else 0.0) + (P.h_copy if reply[i][l] is not None else 0.0)
        s = max(host, dg_free, moe[i][l])
        dg_free = s + P.comb_local
        dg_busy += P.comb_local
        wait_enter[i][l] = host
        if reply[i][l] is not None and reply[i][l] > host:
            wait_exposed += reply[i][l] - host
            host = reply[i][l]
        wait_exit[i][l] = host
        if reply[i][l] is not None and P.h2d_drain:
            # The box-2 partial goes up with a SYNCHRONOUS hipMemcpy on the null
            # stream (`copy_from_host`), which waits for every blocking stream:
            # de.compute drains -- this lane's combine.local (gated on its iGPU
            # MoE) and everything the other lane queued before it.
            attr['drain_block'] += max(0.0, dg_free - host)
            host = max(host, dg_free)
            host += P.h_copy
        host += P.h_p2
        s = max(host, dg_free)
        c2 = P.comb_remote0 + P.comb_b * x.b
        dg_free = s + c2
        dg_busy += c2

    def may_enter(i, l):
        return i == 0 or entered[i - 1] > l

    done_lanes = 0
    guard = 0
    while done_lanes < n:
        guard += 1
        if guard > 100000:
            raise RuntimeError('DES stuck')
        progressed = False
        for i in range(n):
            kind, l = phase[i]
            if kind == 'done':
                continue
            if kind == 'route':
                if n == 1 or R[i][l] <= host:
                    do_route(i, l)
                    phase[i] = ('post', l)
                    progressed = True
                    if n == 1:
                        # sequential driver: post right away, blocking on the reply
                        kind, l = phase[i]
                    else:
                        continue
            if kind == 'post':
                rl = reply[i][l]
                moe_ok = not P.post_wait_moe or moe[i][l] <= host
                if n == 1 or ((rl is None or rl <= host) and moe_ok):
                    do_post(i, l)
                    progressed = True
                    if l + 1 >= NL:
                        phase[i] = ('done', l)
                        done_lanes += 1
                        continue
                    if engram_join_ms and i == 0 and l + 1 == 1:
                        host += engram_join_ms
                    if may_enter(i, l + 1):
                        enqueue_chain(i, l + 1, host)
                        phase[i] = ('route', l + 1)
                    else:
                        phase[i] = ('chain', l + 1)
                continue
            if kind == 'chain':
                if may_enter(i, l):
                    host += 0.01
                    enqueue_chain(i, l, host)
                    phase[i] = ('route', l)
                    progressed = True
        if not progressed:
            # jump to the next readiness time
            nxt = []
            for i in range(n):
                kind, l = phase[i]
                if kind == 'route':
                    nxt.append(R[i][l])
                elif kind == 'post':
                    rl = reply[i][l]
                    t_ready = rl if rl is not None else host
                    if P.post_wait_moe:
                        t_ready = max(t_ready, moe[i][l])
                    nxt.append(t_ready)
            t = min(nxt) if nxt else host
            host = max(host, t)
    end = max(dg_free, host)
    # dGPU busy as hub_step reports it: compute + the activation push (xfer stream);
    # the step-level stages (engram, head) are added by the reporter.
    out = {'fwd_core': end, 'igpu_busy': ig_busy, 'dgpu_busy': dg_busy + push_busy[0], 'rtt_sum': rtt_sum, 'srv_sum': srv_sum,
           'wait_exposed': wait_exposed, 'n_req': n_req}
    # host cadence like req_timeline.py --cadence (lane-layers with a request)
    per, s2w, ex2n = [], [], []
    for i in range(n):
        for l in range(NL):
            if sub_t[i][l] is None:
                continue
            s2w.append(wait_enter[i][l] - sub_t[i][l])
            if l + 1 < NL and sub_t[i][l + 1] is not None:
                per.append(sub_t[i][l + 1] - sub_t[i][l])
                ex2n.append(sub_t[i][l + 1] - wait_exit[i][l])
    out['cad'] = (per, s2w, ex2n)
    out['host_busy'] = hb[0]
    out['b2_busy'] = b2_busy[0]
    out['attr'] = attr
    if trace is not None:
        out['trace'] = trace
        out['sub_t'] = sub_t
        out['reply'] = reply
    return out
