#!/usr/bin/env python3
"""Align the pick trace with hub_step records and write a compact replay cache.

  build_cache.py TRACE STEP.tsv OUT.pkl [--max-bytes N]

For each hub_step (in step-id order) the matching decode batches are consumed
from the trace: `lanes` lanes with lane_rows(rows, lanes) rows each, 40 layers
per lane. Lanes are assigned per batch by (b, next expected layer); lane A is
never behind lane B and, at equal layer, routes first (ready-first ordered cut).
Batches with b > 8 are prefill chunks: kept as (layer, distinct ids) between
steps for the optional pool-pollution model. A step whose batches do not match
its record is dropped (counted) and the reader resyncs on the next layer-0 batch.

Output (pickle): {'steps': [StepRec dicts], 'prefill': [(step_index_before, layer, ids)],
'meta': {...}}. StepRec: id, lanes, rows, lane_b (tuple), router / ran:
per lane a list of 40 tuples-of-rows (each row a 6-tuple of ids), plus the
hub_step fields the calibration needs.
"""
import csv
import math
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
               'pos_max']
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
        self.n_steps_done = 0

    def next(self):
        while True:
            if self.buf:
                return self.buf.pop()
            bt = next(self.it, None)
            if bt is None:
                return None
            if bt.b > 8:
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
    max_bytes = int(sys.argv[sys.argv.index('--max-bytes') + 1]) if '--max-bytes' in sys.argv else None
    recs = []
    for r in csv.DictReader(open(step_tsv), delimiter='\t'):
        recs.append({k: f(r.get(k)) for k in STEP_FIELDS})
    recs.sort(key=lambda r: r['step'])
    prefill = []
    rd = Reader(batches(trace, max_bytes=max_bytes), prefill)
    steps = []
    dropped = 0
    resyncs = 0
    foreign = 0
    for r in recs:
        lanes, rows = int(r['lanes']), int(r['rows'])
        if lanes < 1 or rows < 1 or math.isnan(r['rows']):
            dropped += 1
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
            if '--debug' in sys.argv and resyncs <= 8:
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
            continue
        st = dict(r)
        st['lane_b'] = tuple(lane_rows(rows, lanes))
        st['router'] = [[tuple(bt.rows('router')) for bt in g] for g in got]
        st['ran'] = [[tuple(bt.ran) for bt in g] for g in got]
        st['n_prior_rows'] = sum(bt.n_changed for g in got for bt in g)
        steps.append(st)
        rd.n_steps_done = len(steps)
    meta = {'trace': trace, 'step_tsv': step_tsv, 'n_records': len(recs), 'n_steps': len(steps),
            'dropped': dropped, 'resyncs': resyncs, 'foreign_batches': foreign, 'n_prefill_batches': len(prefill)}
    with open(out, 'wb') as fh:
        pickle.dump({'steps': steps, 'prefill': prefill, 'meta': meta}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(meta, file=sys.stderr)


if __name__ == '__main__':
    main()
