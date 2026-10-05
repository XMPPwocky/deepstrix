#!/usr/bin/env python3
"""Ontology census: count canonical names and retired aliases across the tree.

docs/GLOSSARY.md is the vocabulary; scripts/ontology_terms.tsv is its
machine-readable half: one row per name, with the glossary entry it belongs
to, its kind, what the ratchet checks, an optional path filter and a regex.

    scripts/ontology_census.py                       # table for this tree
    scripts/ontology_census.py --tree ../other       # another checkout
    scripts/ontology_census.py --save base.tsv       # write the counts
    scripts/ontology_census.py --ratchet scripts/ontology_baseline.tsv
    scripts/ontology_census.py --show drafter        # list one entry's names

Kinds:
  canonical  the glossary name; reported so adoption is visible.
  alias      a retired name. The RATCHET fails when its count rises.
  pending    retired, but only a planned rename can remove it (crate and
             module names, which every new `use` repeats); reported only.
  homonym    one word, several glossary entries; reported only.

Ratchet metric (column `ratchet`):
  distinct   number of DIFFERENT identifiers matched. New code may still call
             an existing `MTP_BLOCK`; it may not mint `mtp_new_thing`.
  occ        number of occurrences (for literals such as `b > 16`).

Comments are stripped by default (`//` to end of line), so the census counts
code, string literals included (knob names live in strings).
Only the standard library; no build, no GPU.
"""
import argparse
import csv
import os
import re
import sys

EXTS = (".rs", ".hip", ".h", ".inc", ".cpp")
ROOT = "crates"


def load_terms(path):
    rows = []
    with open(path, newline="") as f:
        lines = [l for l in f if l.strip() and not l.startswith("#")]
    for r in csv.DictReader(lines, delimiter="\t"):
        r["rx"] = re.compile(r["regex"])
        r["scope_rx"] = None if r["scope"] in ("", "*") else re.compile(r["scope"])
        rows.append(r)
    return rows


def files(tree):
    for dp, dn, fn in os.walk(os.path.join(tree, ROOT)):
        dn[:] = sorted(d for d in dn if d not in ("target", ".git"))
        for name in sorted(fn):
            if os.path.splitext(name)[1] in EXTS:
                p = os.path.join(dp, name)
                yield os.path.relpath(p, tree), p


def scan(tree, terms, comments):
    occ = {t["name"]: 0 for t in terms}
    ids = {t["name"]: {} for t in terms}
    for rel, path in files(tree):
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if not comments:
            text = re.sub(r"//[^\n]*", "", text)
        for t in terms:
            if t["scope_rx"] and not t["scope_rx"].search(rel):
                continue
            for m in t["rx"].finditer(text):
                s = m.group(0)
                occ[t["name"]] += 1
                ids[t["name"]][s] = ids[t["name"]].get(s, 0) + 1
    return occ, ids


def metric(t, occ, ids):
    return len(ids[t["name"]]) if t["ratchet"] == "distinct" else occ[t["name"]]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default=os.path.dirname(here))
    ap.add_argument("--terms", default=os.path.join(here, "ontology_terms.tsv"))
    ap.add_argument("--comments", action="store_true", help="count inside // comments too")
    ap.add_argument("--save", metavar="TSV", help="write name<TAB>occ<TAB>distinct")
    ap.add_argument("--ratchet", metavar="TSV", help="exit 1 if an alias's ratchet metric rose above this baseline")
    ap.add_argument("--show", metavar="ENTRY", help="list the identifiers matched for one glossary entry")
    a = ap.parse_args()

    terms = load_terms(a.terms)
    occ, ids = scan(a.tree, terms, a.comments)

    if a.show:
        for t in terms:
            if t["entry"] == a.show:
                print(f"[{t['kind']}] {t['name']}")
                for s, n in sorted(ids[t["name"]].items(), key=lambda kv: -kv[1]):
                    print(f"  {n:6d}  {s}")
        return

    base = {}
    if a.ratchet:
        with open(a.ratchet) as f:
            for l in f:
                if l.strip() and not l.startswith("#"):
                    k, o, d = l.rstrip("\n").split("\t")
                    base[k] = (int(o), int(d))

    w = max(len(t["name"]) for t in terms)
    print(f"  {'kind':<9} {'name':<{w}} {'occ':>6} {'distinct':>8}")
    entry, grew = None, []
    for t in terms:
        if t["entry"] != entry:
            entry = t["entry"]
            print(f"[{entry}]")
        o, d = occ[t["name"]], len(ids[t["name"]])
        note = ""
        if t["name"] in base:
            bo, bd = base[t["name"]]
            note = f"  (was {bo} / {bd})" if (bo, bd) != (o, d) else ""
            if t["kind"] == "alias" and t["ratchet"] in ("distinct", "occ"):
                b = bd if t["ratchet"] == "distinct" else bo
                if metric(t, occ, ids) > b:
                    grew.append((t["name"], t["ratchet"], b, metric(t, occ, ids)))
        print(f"  {t['kind']:<9} {t['name']:<{w}} {o:>6} {d:>8}{note}")

    if a.save:
        with open(a.save, "w") as f:
            f.write("# name\tocc\tdistinct  (scripts/ontology_census.py --save)\n")
            for t in terms:
                f.write(f"{t['name']}\t{occ[t['name']]}\t{len(ids[t['name']])}\n")

    if grew:
        print("\nRATCHET: retired names gained ground (docs/GLOSSARY.md has the replacements):", file=sys.stderr)
        for k, m, b, n in grew:
            print(f"  {k}: {m} {b} -> {n}   (--show the entry to list them)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
