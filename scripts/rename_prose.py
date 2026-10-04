#!/usr/bin/env python3
"""Rewrite a phrase inside Rust COMMENTS only (never code, never strings).

    scripts/rename_prose.py REGEX REPLACEMENT [--apply] [--root crates]

Uses rename_idents.py's tokenizer, so `//`, `///`, `//!` and `/* */` are
comments and string literals are left alone (log text is an interface).
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rename_idents import rs_files, tokens  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('regex')
ap.add_argument('repl')
ap.add_argument('--apply', action='store_true')
ap.add_argument('--root', default='crates')
a = ap.parse_args()
rx = re.compile(a.regex)
total = 0
for p in sorted(rs_files(a.root)):
    src = open(p, encoding='utf-8').read()
    out, last, n = [], 0, 0
    for kind, s, e in tokens(src):
        if kind != 'comment':
            continue
        new, k = rx.subn(a.repl, src[s:e])
        if k:
            out.append(src[last:s])
            out.append(new)
            last = e
            n += k
    if n:
        out.append(src[last:])
        total += n
        print(f'{p}: {n}')
        if a.apply:
            open(p, 'w', encoding='utf-8').write(''.join(out))
print('total', total)
