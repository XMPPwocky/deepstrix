#!/usr/bin/env python3
"""List an .evt file's header (kinds + fields) and, with --count, record counts
per kind (streamed). Read-only helper for the split simulator."""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from evt2perfetto import header_of, records  # noqa: E402


def main():
    path = sys.argv[1]
    want = set(sys.argv[2].split(',')) if len(sys.argv) > 2 and sys.argv[2] != '--count' else None
    h = header_of(path)[1]
    print({k: v for k, v in h.items() if k not in ('kinds', 'strings')})
    for k in h['kinds']:
        if want is None or k['name'] in want:
            print(k['id'], k['name'], k['fields'])
    if '--count' in sys.argv:
        c = Counter()
        for _t, name, _f, _v in records(path):
            c[name] += 1
        for k, v in c.most_common():
            print(f'{k:20s} {v}')


if __name__ == '__main__':
    main()
