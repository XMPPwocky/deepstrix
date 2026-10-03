#!/usr/bin/env python3
"""Pick-trace reader (`V41_PICK_TRACE`, written by forward_prefill.rs
`pre_moe_route`).

Line kinds (crates/v4flash-kernels/src/het/forward_prefill.rs ~8560-8800):
  P <layer> <b> <ids x6>          one ROW of a lane-layer batch of b rows: the picks that
                                  RAN (after the cache-prior, before mode-2 substitution).
                                  b is the LANE's row count (lane_rows(rows, lanes)), not the step's.
  O <layer> <b> <row> <ids x6>    the ROUTER's own picks for row `row` of the batch just
                                  emitted, written only when the cache-prior (`V41_SUB=3`)
                                  changed that row. Follows the batch's P lines.
  C|c <layer> <b> <row> <from> <to>  one cache-prior swap (c = dry run).
  S|s ...                         mode-2 substitution swaps (not in this trace).
  A ...                           alternatives (not in this trace).
  D <layer> <ids>                 legacy decode path (forward_layer.rs; not in this trace).

A batch is b consecutive P lines with the same (layer, b). Lines are in emission
order: a one-lane step emits 40 batches (layers 0..39); a two-lane step 80, the
lanes interleaved per layer (lane A's batch for layer L before lane B's; A never
behind B -- `forward_step_arena_ready_first` orders the cut).

`batches(path)` yields Batch objects (layer, b, rows (router picks), ran (P picks),
n_prior_rows, offset). Streams the file; `max_bytes` cuts a growing file.
"""
import os


class Batch:
    __slots__ = ('layer', 'b', 'ran', 'router', 'n_changed', 'off')

    def __init__(self, layer, b, off):
        self.layer = layer
        self.b = b
        self.ran = []      # list of tuples (6 ids) as run (P lines)
        self.router = None  # list of tuples: router's own picks (O lines applied), lazily = ran
        self.n_changed = 0
        self.off = off

    def rows(self, which='router'):
        if which == 'ran' or self.router is None:
            return self.ran
        return self.router


def batches(path, max_bytes=None, start=0):
    """Stream batches. O lines are applied to the batch they follow."""
    size = os.path.getsize(path)
    if max_bytes is not None:
        size = min(size, max_bytes)
    cur = None
    last_done = None
    with open(path, 'rb') as fh:
        fh.seek(start)
        pos = start
        for raw in fh:
            pos += len(raw)
            if pos > size:
                break
            if not raw or raw[-1:] != b'\n':
                break
            k = raw[:1]
            if k == b'P':
                parts = raw.split()
                layer = int(parts[1])
                b = int(parts[2])
                ids = tuple(int(x) for x in parts[3:])
                if cur is not None and (cur.layer != layer or cur.b != b or len(cur.ran) >= cur.b):
                    if len(cur.ran) == cur.b:
                        last_done = cur
                        yield cur
                    cur = None
                if cur is None:
                    cur = Batch(layer, b, pos - len(raw))
                cur.ran.append(ids)
                if len(cur.ran) == cur.b:
                    last_done = cur
                    yield cur
                    cur = None
            elif k == b'O':
                parts = raw.split()
                layer, b, r = int(parts[1]), int(parts[2]), int(parts[3])
                ids = tuple(int(x) for x in parts[4:])
                tgt = last_done
                if tgt is not None and tgt.layer == layer and tgt.b == b and r < tgt.b:
                    if tgt.router is None:
                        tgt.router = list(tgt.ran)
                    tgt.router[r] = ids
                    tgt.n_changed += 1
            # C/c/S/s/A/D lines: not needed (P + O carry the picks)


if __name__ == '__main__':
    import sys
    n = 0
    for bt in batches(sys.argv[1], max_bytes=int(sys.argv[3]) if len(sys.argv) > 3 else None):
        if bt.b <= 6:
            print(n, bt.layer, bt.b, bt.n_changed)
        n += 1
        if n >= int(sys.argv[2]):
            break
