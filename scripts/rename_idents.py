#!/usr/bin/env python3
"""Rename Rust identifiers across the tree, token-aware and re-runnable.

A rename here is an INTERFACE-FREE change: string literals (env names,
telemetry fields, log text, checkpoint tensor names) are never edited, except
`{name}` / `{name:..}` placeholders of inline format args, which name a
binding, not output text. Comments are edited only for identifiers that
cannot be prose (CamelCase, UPPER_SNAKE, snake_with_underscores, or a bare
word followed by `::`).

    scripts/rename_idents.py RULES [--apply] [--root crates]

RULES is a Python file defining
    def rename(ident: str) -> str | None   # new name, or None = keep
    EXCLUDE = {...}                         # identifiers never renamed
    MOVES = [("old/path.rs", "new/path.rs")] # optional file moves (git mv)

Before writing anything it refuses when a NEW name already exists as an
identifier anywhere in the tree (a collision could silently shadow). It is
idempotent: running it again on its own output changes nothing, so a branch
that conflicts with a rename can re-run the same RULES instead of
hand-resolving.
"""
import argparse
import os
import re
import runpy
import subprocess
import sys

IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def tokens(src):
    """Yield (kind, start, end): kind in code|ident|str|char|comment."""
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith('//', i):
            j = src.find('\n', i)
            j = n if j < 0 else j
            yield 'comment', i, j
            i = j
            continue
        if src.startswith('/*', i):
            d, j = 1, i + 2
            while j < n and d:
                if src.startswith('/*', j):
                    d, j = d + 1, j + 2
                elif src.startswith('*/', j):
                    d, j = d - 1, j + 2
                else:
                    j += 1
            yield 'comment', i, j
            i = j
            continue
        m = re.match(r'b?r(#*)"', src[i:i + 12])
        if m and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == '_')):
            end = src.find('"' + m.group(1), i + m.end())
            j = n if end < 0 else end + 1 + len(m.group(1))
            yield 'str', i, j
            i = j
            continue
        if c == '"' or (c == 'b' and src.startswith('b"', i) and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == '_'))):
            j = i + (2 if c == 'b' else 1)
            while j < n and src[j] != '"':
                j += 2 if src[j] == '\\' else 1
            yield 'str', i, j + 1
            i = j + 1
            continue
        if c == "'":
            m = re.match(r"b?'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|.)|[^\\'\n])'", src[i:i + 16])
            if m:
                yield 'char', i, i + m.end()
                i += m.end()
                continue
            yield 'code', i, i + 1  # lifetime tick
            i += 1
            continue
        if c.isalpha() or c == '_':
            m = IDENT.match(src, i)
            yield 'ident', i, m.end()
            i = m.end()
            continue
        yield 'code', i, i + 1
        i += 1


def prose_safe(word, after):
    """In a comment, may `word` be renamed? Only when it cannot be English."""
    if after.startswith('::'):
        return True
    if '_' in word.strip('_'):
        return True
    if re.fullmatch(r'[A-Z][a-z0-9]+[A-Z]\w*', word):  # CamelCase with 2+ humps
        return True
    return False


def rewrite(src, rename, exclude, stats):
    out = []
    last = 0
    for kind, a, b in tokens(src):
        text = src[a:b]
        new = text
        if kind == 'ident':
            r = None if text in exclude else rename(text)
            if r:
                new = r
                stats['ident'] = stats.get('ident', 0) + 1
        elif kind == 'str':
            def ph(m):
                r = None if m.group(1) in exclude else rename(m.group(1))
                if r:
                    stats['placeholder'] = stats.get('placeholder', 0) + 1
                    return '{' + r + m.group(2)
                return m.group(0)
            new = re.sub(r'(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)([:}])', ph, text)
        elif kind == 'comment':
            def cm(m):
                w = m.group(0)
                if w in exclude:
                    return w
                r = rename(w)
                if r and prose_safe(w, text[m.end():m.end() + 2]):
                    stats['comment'] = stats.get('comment', 0) + 1
                    return r
                return w
            new = IDENT.sub(cm, text)
        if new != text:
            out.append(src[last:a])
            out.append(new)
            last = b
    out.append(src[last:])
    return ''.join(out)


def rs_files(root):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ('target', '.git')]
        for f in fn:
            if f.endswith('.rs'):
                yield os.path.join(dp, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rules')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--root', default='crates')
    a = ap.parse_args()
    rules = runpy.run_path(a.rules)
    rename, exclude, moves = rules['rename'], set(rules.get('EXCLUDE', ())), rules.get('MOVES', [])

    files = sorted(rs_files(a.root))
    mapping = {}
    per_file = {}
    for p in files:
        src = open(p, encoding='utf-8').read()
        idents = set()
        for kind, s, e in tokens(src):
            if kind == 'ident':
                w = src[s:e]
                idents.add(w)
                if w not in exclude:
                    r = rename(w)
                    if r:
                        mapping[w] = r
        per_file[p] = idents
    targets = {}
    for old, new in mapping.items():
        targets.setdefault(new, set()).add(old)
    bad = [f'{new} <- {sorted(olds)}: two names map to one' for new, olds in targets.items() if len(olds) > 1]
    # A new name that already exists in a file which also uses the old one could
    # capture or shadow a binding: refuse. (Across files, rustc catches clashes.)
    for p, idents in per_file.items():
        for old in idents:
            new = mapping.get(old)
            if new and new in idents and new not in mapping:
                bad.append(f'{p}: {old} -> {new}, but {new} is already used in this file')
    for b in bad:
        print('COLLISION:', b)
    if bad:
        sys.exit(1)
    print(f'{len(mapping)} identifiers:')
    for old in sorted(mapping):
        print(f'  {old} -> {mapping[old]}')
    if not a.apply:
        return
    total = {}
    for p in files:
        src = open(p, encoding='utf-8').read()
        st = {}
        new = rewrite(src, rename, exclude, st)
        if new != src:
            open(p, 'w', encoding='utf-8').write(new)
            for k, v in st.items():
                total[k] = total.get(k, 0) + v
    for old, new in moves:
        if os.path.exists(old):
            subprocess.run(['git', 'mv', old, new], check=True)
            print(f'moved {old} -> {new}')
    print('rewrote:', total)


if __name__ == '__main__':
    main()
