#!/usr/bin/env python3
"""Stream an hub .evt file and write compact TSVs of the kinds the split
simulator needs (hub_step: every field; hub_req: the decode-relevant fields).

  extract_evt.py HUB.evt OUTDIR [--req-max-b 16]

Never loads the file: `evt2perfetto.records` streams it in 64 MB chunks.
Output: OUTDIR/<stem>.hub_step.tsv, OUTDIR/<stem>.hub_req.tsv."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from evt2perfetto import records  # noqa: E402

REQ_FIELDS = ['t_submit', 't_submit_end', 't1', 't4', 't_wait_enter', 't_wait_exit', 't2_b2', 't3_b2', 'step', 'lane', 'layer', 'b',
              'seq', 'flags', 'partner', 'unmasked', 'n_picks', 'n_distinct', 'n_pred_miss', 'n_pred_incoming',
              'n_pred_pending', 'rtt_us', 'srv_us', 'page_us', 'compute_us', 'n_miss', 'miss_bits', 'blocked', 'step_rows',
              'n_held', 'n_surprise', 'n_paged', 'pinned', 'bytes_out', 'bytes_in']


def fmt(v):
    if isinstance(v, float):
        if math.isnan(v):
            return 'nan'
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f'{v:.6g}'
    return str(v)


def main():
    path, outdir = sys.argv[1], sys.argv[2]
    req_max_b = 16
    if '--req-max-b' in sys.argv:
        req_max_b = int(sys.argv[sys.argv.index('--req-max-b') + 1])
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.basename(path).rsplit('.', 1)[0]
    fs = open(os.path.join(outdir, stem + '.hub_step.tsv'), 'w')
    fr = open(os.path.join(outdir, stem + '.hub_req.tsv'), 'w')
    step_fields = None
    fr.write('\t'.join(REQ_FIELDS) + '\n')
    ns = nr = 0
    for _t, name, fields, vals in records(path):
        if name == 'hub_step':
            if step_fields is None:
                step_fields = list(fields)
                fs.write('\t'.join(step_fields) + '\n')
            d = dict(zip(fields, vals))
            fs.write('\t'.join(fmt(d.get(k, float('nan'))) for k in step_fields) + '\n')
            ns += 1
        elif name == 'hub_req':
            d = dict(zip(fields, vals))
            b = d.get('b', float('nan'))
            if b == b and b > req_max_b:
                continue
            fr.write('\t'.join(fmt(d.get(k, float('nan'))) for k in REQ_FIELDS) + '\n')
            nr += 1
    fs.close()
    fr.close()
    print(f'{stem}: hub_step {ns}, hub_req {nr}', file=sys.stderr)


if __name__ == '__main__':
    main()
