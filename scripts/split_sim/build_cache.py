#!/usr/bin/env python3
"""Align the pick trace with hub_step records and write a compact replay cache.

  build_cache.py TRACE STEP.tsv OUT.pkl [--max-bytes N]
                 [--window FROM:TO --evt HUB.evt [--refresh REFRESH.txt]]

For each hub_step (in step-id order) the matching decode batches are consumed
from the trace: `lanes` lanes with lane_rows(rows, lanes) rows each, 40 layers
per lane. Lanes are assigned per batch by (b, next expected layer); lane A is
never behind lane B and, at equal layer, routes first (ready-first ordered cut).
Batches with b > 8 are prefill chunks: kept as (layer, distinct ids) between
steps for the optional pool-pollution model. A step whose batches do not match
its record is dropped (counted) and the reader resyncs on the next layer-0 batch.

--window FROM:TO (unix seconds; needs --evt for the clock): align from the
trace's start as always, but keep only the steps whose t_start is in
[FROM, TO) (and the prefill batches between them); records after TO are not
read. With --refresh, the ownership replica (simlib.HotSet) runs over EVERY
aligned step from the start, refreshing at the logged times; its state just
before the window goes into the cache as 'hs0' (consumers start their replica
from it: the live owned set is sticky over hours). The replica's `changed` per
refresh is compared with the log's `changed=` (`meta['refresh_check']`).

Output (pickle): {'steps': [StepRec dicts], 'prefill': [(step_index_before, layer, ids)],
'meta': {...}[, 'hs0': hot-set snapshot]}. StepRec: id, lanes, rows, lane_b (tuple),
router / ran: per lane a list of 40 tuples-of-rows (each row a 6-tuple of ids;
`router` shares `ran`'s tuple when the prior changed nothing), plus the hub_step
fields the calibration needs.
"""
import csv
import math
import os
import pickle
import sys

from picktrace import batches

STEP_FIELDS = ['t_start', 't_end', 'step', 'rows', 'live', 'lanes', 'fwd_ms', 'step_ms', 'dgpu_busy_ms', 'igpu_busy_ms',
               'i_pair_kwide', 'i_q2k_down', 'i_moe_group_builder', 'i_moe_work_items', 'i_q8k_quantize_post_iq2',
               'i_peer_push_ffn_moe', 'i_other', 'remote_wait_ms', 'remote_rtt_ms', 'remote_srv_ms', 'b2_page_ms',
               'b2_misses', 'b1_misses', 'b1_read_ms', 'b2_pinned', 'sub_picks_swapped', 'lh_engram_join',
               'lh_pager_block', 'd_output_proj', 'd_attn_compute', 'd_q_chain', 'd_shared_expert', 'd_mhc_pre_attn',
               'd_mhc_pre_ffn', 'd_mhc_mix_ffn_late', 'd_router', 'd_prefill_indexer', 'd_prefill_indexer_reuse',
               'd_peer_push_ffn_input_norm', 'd_kv_chain', 'd_rb_pack', 'd_head_batch', 'd_engram',
               'd_ffn_combine_local', 'd_ffn_combine_remote', 'd_kv_append_compressor_serial', 'd_mhc_post_attn',
               'pos_max', 'profiled', 'b2_paged_replies', 'b2_service_ms', 'fwd_all_ms', 'engram_ms', 'sample_ms',
               'b1_pf_admitted', 'b1_pf_admit_ms', 'lh_remote_upload', 'lh_remote_sync', 'sub_admits_queued']
N_LAYER = 40


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def lane_rows(b, n):
    return [b // n + (1 if i < b % n else 0) for i in range(n)]


class Reader:
    """Batch stream with one-batch pushback; diverts prefill batches."""

    def __init__(self, it, prefill_sink):
        self.it = it
        self.buf = []
        self.sink = prefill_sink
        self.sink_on = True
        self.n_steps_done = 0

    def next(self):
        while True:
            if self.buf:
                return self.buf.pop()
            bt = next(self.it, None)
            if bt is None:
                return None
            if bt.b > 8:
                if not self.sink_on:
                    continue
                ids = {}
                for row in bt.ran:
                    for e in row:
                        ids[e] = ids.get(e, 0) + 1
                self.sink.append((self.n_steps_done, bt.layer, bt.b, tuple(sorted(ids.items()))))
                continue
            return bt

    def push(self, bt):
        self.buf.append(bt)


def consume_step(rd, lanes, rows):
    """Try to read one step's batches. Returns (lanes' batch lists) or None,
    plus the batches consumed (for pushback on failure)."""
    lb = lane_rows(rows, lanes)
    nxt = [0] * lanes
    got = [[] for _ in range(lanes)]
    taken = []
    need = lanes * N_LAYER
    while sum(len(g) for g in got) < need:
        bt = rd.next()
        if bt is None:
            return None, taken
        taken.append(bt)
        cand = [i for i in range(lanes) if lb[i] == bt.b and nxt[i] == bt.layer]
        if not cand:
            return None, taken
        i = cand[0]
        got[i].append(bt)
        nxt[i] += 1
    return got, taken


def main():
    trace, step_tsv, out = sys.argv[1], sys.argv[2], sys.argv[3]
    a = sys.argv
    max_bytes = int(a[a.index('--max-bytes') + 1]) if '--max-bytes' in a else None
    win = None
    clk = None
    if '--window' in a:
        lo, hi = a[a.index('--window') + 1].split(':')
        win = (float(lo), float(hi))
    if '--evt' in a:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        from evt2perfetto import header_of
        import simlib
        h = header_of(a[a.index('--evt') + 1])[1]
        clk = simlib.Clock(h['t_mono_raw_at_open'], h['t_realtime_at_open'])
    hs = None
    rlog = []
    if '--refresh' in a:
        import simlib
        rlog = simlib.load_refresh_log(a[a.index('--refresh') + 1])
        hs = simlib.HotSet()
    if (win or hs is not None) and clk is None:
        raise SystemExit('--window / --refresh need --evt')
    # Records: a compact dict per step before the window, the full one inside it.
    rd_tsv = csv.reader(open(step_tsv), delimiter='\t')
    hdr = next(rd_tsv)
    ix = {k: i for i, k in enumerate(hdr)}
    recs = []
    n_tsv = 0
    for row in rd_tsv:
        n_tsv += 1

        def g(k):
            i = ix.get(k)
            return f(row[i]) if i is not None and i < len(row) else float('nan')

        t = g('t_start')
        if win is not None:
            tu = clk.to_unix(t)
            if tu >= win[1]:
                continue
            if tu < win[0]:
                recs.append({'step': g('step'), 'lanes': g('lanes'), 'rows': g('rows'), 'live': g('live'), 't_start': t,
                             '_pre': True})
                continue
        recs.append({k: g(k) for k in STEP_FIELDS})
    if any(recs[i]['step'] > recs[i + 1]['step'] for i in range(len(recs) - 1)):
        recs.sort(key=lambda r: r['step'])
    prefill = []
    rd = Reader(batches(trace, max_bytes=max_bytes), prefill)
    steps = []
    dropped = dropped_win = 0
    resyncs = 0
    foreign = 0
    n_pre = 0
    ri = 0
    rcheck = []
    snap = None
    for r in recs:
        pre = r.get('_pre', False)
        if hs is not None:
            tu = clk.to_unix(r['t_start'])
            if not pre and snap is None:
                # every refresh at or before the snapshot's t_unix is in it
                snap = hs.snapshot(rlog[ri - 1][0] if ri > 0 else -1.0)
            while ri < len(rlog) and rlog[ri][0] <= tu:
                ch = hs.refresh()
                rcheck.append((rlog[ri][0], ch, rlog[ri][1].get('changed')))
                ri += 1
        rd.sink_on = not pre
        if math.isnan(r['rows']) or math.isnan(r['lanes']):
            dropped += 1
            dropped_win += not pre
            continue
        lanes, rows = int(r['lanes']), int(r['rows'])
        if lanes < 1 or rows < 1:
            dropped += 1
            dropped_win += not pre
            continue
        skipped = 0
        got = None
        exhausted = False
        while True:
            got, taken = consume_step(rd, lanes, rows)
            if got is not None:
                break
            if not taken:
                exhausted = True
                break
            # Foreign batches (a step with no hub_step record, a small prefill
            # chunk): skip to the next layer-0 batch after the first one taken
            # and retry THIS record.
            resyncs += 1
            if '--debug' in a and resyncs <= 8:
                print(f"resync at step {int(r['step'])} live={int(r['live'])} lanes={lanes} rows={rows}: taken "
                      + ' '.join(f'{bt.layer}:{bt.b}' for bt in taken[:100]), file=sys.stderr)
            keep = None
            for j in range(1, len(taken)):
                if taken[j].layer == 0:
                    keep = j
                    break
            n_skip = len(taken) if keep is None else keep
            skipped += n_skip
            if keep is not None:
                for bt in reversed(taken[keep:]):
                    rd.push(bt)
            if skipped > 4000:
                break
        foreign += skipped
        if exhausted:
            break
        if got is None:
            dropped += 1
            dropped_win += not pre
            continue
        if hs is not None:
            for gl in got:
                for bt in gl:
                    for row in bt.rows('router'):
                        for e in row:
                            if 0 <= e < 384:
                                hs.note(bt.layer, e)
        if pre:
            n_pre += 1
            if n_pre % 50000 == 0:
                print(f'pre-window steps aligned {n_pre} (dropped {dropped}, resyncs {resyncs}, foreign {foreign})',
                      file=sys.stderr, flush=True)
            continue
        st = dict(r)
        st['lane_b'] = tuple(lane_rows(rows, lanes))
        st['ran'] = [[tuple(bt.ran) for bt in gl] for gl in got]
        # the router's rows: the SAME tuple as `ran` when the prior changed nothing
        st['router'] = [[(ran if bt.router is None else tuple(bt.router)) for bt, ran in zip(gl, rr)]
                        for gl, rr in zip(got, st['ran'])]
        st['n_prior_rows'] = sum(bt.n_changed for gl in got for bt in gl)
        steps.append(st)
        rd.n_steps_done = len(steps)
    meta = {'trace': trace, 'step_tsv': step_tsv, 'n_records': len(recs), 'n_tsv': n_tsv, 'n_steps': len(steps),
            'n_pre_window_aligned': n_pre, 'window': win, 'dropped': dropped, 'dropped_in_window': dropped_win,
            'resyncs': resyncs, 'foreign_batches': foreign, 'n_prefill_batches': len(prefill)}
    if rcheck:
        ok = sum(1 for _, x, y in rcheck if x is not None and y is not None and x == y)
        meta['refresh_check_exact'] = (ok, len(rcheck))
    meta['refresh_check'] = rcheck
    cache = {'steps': steps, 'prefill': prefill, 'meta': meta}
    if snap is not None:
        cache['hs0'] = snap
    with open(out, 'wb') as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print({k: v for k, v in meta.items() if k != 'refresh_check'}, file=sys.stderr)


if __name__ == '__main__':
    main()
