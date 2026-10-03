#!/usr/bin/env python3
"""List a perfetto JSON trace's tracks (pid/tid names) and the most common
slice names per track. Read-only helper for the split simulator."""
import gzip
import json
import sys
from collections import Counter, defaultdict


def main():
    path = sys.argv[1]
    with gzip.open(path, 'rt') as fh:
        d = json.load(fh)
    ev = d['traceEvents'] if isinstance(d, dict) else d
    names = {}
    pnames = {}
    cnt = defaultdict(Counter)
    for e in ev:
        if e.get('ph') == 'M':
            if e['name'] == 'thread_name':
                names[(e['pid'], e.get('tid'))] = e['args']['name']
            elif e['name'] == 'process_name':
                pnames[e['pid']] = e['args']['name']
            continue
        nm = e.get('name', '')
        key = nm.split(' ')[0] if ' ' in nm else nm
        cnt[(e.get('pid'), e.get('tid'))][(e.get('ph'), key)] += 1
    for k in sorted(cnt, key=lambda k: (k[0] or 0, k[1] or 0)):
        print(f'pid={k[0]} ({pnames.get(k[0])}) tid={k[1]} "{names.get(k)}"  total={sum(cnt[k].values())}')
        for (ph, nm), c in cnt[k].most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 12):
            print(f'      {ph} {nm:50s} {c}')


if __name__ == '__main__':
    main()
